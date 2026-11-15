import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse,csv,json,time,fcntl,hashlib,shutil,traceback,math
from pathlib import Path
import numpy as np
import cv2
from data_adapters import UBFC2,VideoFrames
from oval_frontend import OvalFrontend
from run_t5_qa import read,save,sha,csv_write,merge
from t8_t1_loader import supported
from hr_baseline import pos,estimate_hr
from score_round1 import score

def predict(t,rgb,valid,start):
 tv,values,ok=merge(t,rgb,valid & np.isfinite(rgb).all(1))
 if not supported(tv,ok,start,start+10):raise ValueError('Image support/gap/ROI')
 q=start+np.arange(300)/30
 sampled=np.stack([np.interp(q,tv,values[:,c]) for c in range(3)],axis=1)
 return estimate_hr(pos(sampled,30),30,band=(.7,3.))

def masked(image,frontend):
 mask,points=frontend.step(image)
 return (np.array(cv2.mean(image,mask=mask)[:3]) if np.count_nonzero(mask)>=16 else np.full(3,np.nan)),points

def source_traces(record,indices,update):
 raw=[];small=[];pr=[];ps=[];selected=set(indices.tolist())
 with OvalFrontend() as full,OvalFrontend() as decimated:
  for i,image in enumerate(record.frames):
   value,points=masked(image,full);raw.append(value);pr.append(points)
   if i in selected:
    value,points=masked(image,decimated);small.append(value);ps.append(points)
   if i%500==0:update(i)
 assert len(raw)==len(record.timestamps) and len(small)==len(indices)
 return dict(native=(np.asarray(raw),np.asarray(pr)),raw10=(np.asarray(small),np.asarray(ps)))

def process(rid,a,fingerprint,update):
 dest=a.output/rid;dest.mkdir(exist_ok=True)
 old=a.t6/rid;canonical=read(old/'canonical_windows.json')
 reference=dest/'reference_signal.csv';shutil.copyfile(old/'reference_signal.csv',reference)
 record=UBFC2(a.root).load(rid)
 refhash=sha(reference);sourcehash=sha(record.frames.path)
 m10=np.load(a.t5/'c5_500k_10fps'/rid/'frame_mapping.npz');ix=m10['source_frame_indices'];tt=m10['timestamps_s']
 assert len(ix)>1 and np.all(np.diff(ix)>0) and ix[-1]<len(record.timestamps)
 np.testing.assert_allclose(record.timestamps[ix],tt,rtol=0,atol=1e-9)
 traces=source_traces(record,ix,update)
 times={'native':record.timestamps,'raw10':tt};indices={'native':np.arange(len(record.timestamps)),'raw10':ix}
 d=a.t5/'c5_500k_30fps'/rid;manifest=read(d/'manifest.json');ev=read(d/'evaluation.json')
 assert ev['recording_id']==rid and manifest['kind']=='c5' and manifest['fps_requested']==30 and manifest['bitrate_limit_kbps']==500
 assert manifest['bitrate_pass'];video=Path(manifest['video']);assert video.parent.resolve()==d.resolve()
 mapping=np.load(d/'frame_mapping.npz');ids=mapping['source_frame_indices'];t=mapping['timestamps_s']
 assert np.all(np.diff(ids)>0);np.testing.assert_allclose(record.timestamps[ids],t,rtol=0,atol=1e-9)
 values=[];points=[]
 with OvalFrontend() as frontend:
  for i,image in enumerate(VideoFrames(video,len(t))):
   value,pt=masked(image,frontend);values.append(value);points.append(pt)
   if i%500==0:update(i)
 traces['h26430']=(np.asarray(values),np.asarray(points));times['h26430']=t;indices['h26430']=ids
 with reference.open(newline='') as f:refs=list(csv.DictReader(f))
 validstarts={round(w['start_s'],9) for w in canonical['valid_windows']}
 metrics={};counts={};failures={}
 for name,(rgb,pts) in traces.items():
  valid=np.isfinite(rgb).all(1);np.savez_compressed(dest/(name+'_trace.npz'),rgb=rgb,oval_points=pts,t=times[name],valid=valid,source_indices=indices[name])
  out=[];fails=[]
  for ref in refs:
   start=float(ref['window_start_s']);hr=''
   try:
    if round(start,9) not in validstarts:raise ValueError('Canonical source invalid')
    hr=predict(times[name],rgb,valid,start)
   except ValueError as e:fails.append(dict(start_s=start,reason=str(e)))
   out.append(dict(subject_id=record.subject_id,recording_id=rid,window_start_s=start,hr_bpm=hr,abstain=hr==''))
  csv_write(dest/(name+'_output.csv'),out);metrics[name]=score(reference,dest/(name+'_output.csv'))
  failures[name]=fails;counts[name]=dict(frames=len(rgb),source_frames=len(record.timestamps),effective_fps=(len(rgb)-1)/(times[name][-1]-times[name][0]))
 shutil.copyfile(old/'control_output.csv',dest/'h26410_output.csv')
 assert sha(dest/'h26410_output.csv')==sha(old/'control_output.csv')
 metrics['h26410']=score(reference,dest/'h26410_output.csv')
 assert sha(reference)==refhash
 return dict(recording_id=rid,subject_id=record.subject_id,fingerprint=fingerprint,reference_sha256=refhash,source_sha256=sourcehash,h26430_video_sha256=sha(video),h26410_receipt_sha256=sha(old/'complete.json'),h26410_prediction_sha256=sha(old/'control_output.csv'),h26430_manifest=manifest,counts=counts,failures=failures,metrics=metrics)

def main(a):
 cv2.setNumThreads(1);os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[2]})
 a.output.mkdir(parents=True,exist_ok=True);lock=(a.output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 code=Path(__file__).parent;fp=read(code/'SHA256.json');assert all(sha(code/k)==v for k,v in fp.items())
 folds=read(code/'folds_frozen.json');ids=sorted(set(folds['folds'][0]['ubfc_train_subjects']+folds['folds'][0]['ubfc_val_subjects']))
 assert len(ids)==35 and set(ids).isdisjoint(folds['original_ubfc_test_subjects']);assert read(a.t6/'status.json')['state']=='complete'
 if (a.output/'fingerprint.json').exists():assert read(a.output/'fingerprint.json')==fp
 else:save(a.output/'fingerprint.json',fp)
 save(a.output/'cohort.json',dict(recordings=ids,expected_records=35,expected_windows=207,original7test_accessed=False))
 status=dict(state='running',pid=os.getpid(),completed=0,total=35,errors=[]);save(a.output/'status.json',status)
 for rid in ids:
  try:
   if (a.output/rid/'complete.json').exists():
    assert read(a.output/rid/'complete.json')['fingerprint']==fp;status['completed']+=1;continue
   if (a.output/rid).exists():raise RuntimeError('Preserved partial requires versioned continuation')
   assert sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())<190000000,'200MB output cap'
   def update(n):status.update(recording=rid,frame=n,updated_unix=time.time());save(a.output/'status.json',status)
   update(0);result=process(rid,a,fp,update);save(a.output/rid/'complete.json',result)
   status['completed']+=1;print(json.dumps(dict(recording=rid,completed=status['completed'])),flush=True)
   if a.limit and status['completed']>=a.limit:status['state']='smoke_complete';save(a.output/'status.json',status);return
  except Exception:
   status['state']='blocked_error';status['errors'].append(dict(recording=rid,error=traceback.format_exc()));save(a.output/'status.json',status);raise
 status['state']='complete';save(a.output/'status.json',status)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('root','t5','t6','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--limit',type=int,default=0);main(p.parse_args())
