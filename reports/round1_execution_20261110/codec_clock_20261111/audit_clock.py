"""Bounded CPU codec clock pilot. No training or holdout access."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,hashlib,subprocess,time,traceback,fcntl,argparse,csv,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
import cv2
cv2.setNumThreads(1)
P=Path(__file__).resolve().parent;E=P.parent;R=E.parent.parent
V=R/'reports/round1_v4_20261105';B=R/'reports/round1_bayes_20261108'
sys.path.insert(0,str(E));sys.path.insert(0,str(V/'btc_zeroshot'))
import preprocess as pre
from infer_dev import classical_functions,decode_wave,decode_model
from oval_crop import OvalCrop,crop_from_points
from oval_frontend import oval_mask
from t8_evaluate import decode_difference
from score_round1 import reference_hr
from polarity_guard import storage_guard
from ppg_polarity import adapt_cached_meta
NS,FNS=classical_functions()
SPLIT='97944f717b7fc166bb8a0e91c791c7ad6b7e451cffed564f40232dde3ec8e221'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def size(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file()) if Path(p).exists() else 0
def mem():
 return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
def guard(fresh=False):
 a=read(E/'A100_INVENTORY.json');idx=read(R/'t8_oval_20261031/training_index.json')
 if fresh:assert 0<=(datetime.now(timezone.utc)-datetime.fromisoformat(a['measured_utc'])).total_seconds()<3600,'FRESH_A100_REQUIRED'
 base=storage_guard(idx,R/'t8_oval_20261031',R/'t8_polarity_v3_20261102')['total_bytes']+size(V)+size('/data_pool/khangle/facevitals/reports/round1_v4_20261105')+a['a100_v4_bytes']+size(B)
 used=size(E);own=size(P);remaining=max(0,600_000_000-own)
 project=sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in [R,Path('/data_pool/khangle/facevitals')])+a['a100_bytes']
 q=dict(utc=now(),scope_actual=base+used,project_actual=project,audit_actual=own,additional_reserved=remaining,headroom=420_000_000,a100_sha=sha(E/'A100_INVENTORY.json'))
 assert own<600_000_000 and base+used+remaining+420_000_000<84_000_000_000,'STOP_STORAGE_SCOPE'
 assert project+remaining+420_000_000<500_000_000_000,'STOP_STORAGE_PROJECT'
 assert mem()>12_000_000_000,'STOP_RAM'
 return q
def binding():
 paths=[Path(__file__),P/'PROTOCOL.md',V/'btc_zeroshot/infer_dev.py',V/'btc_zeroshot/preprocess.py',Path('/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102/t8_evaluate.py'),Path('/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102/score_round1.py')]
 return {str(p):sha(p) for p in paths}|FNS
def prepare():
 assert not (P/'PLAN.json').exists(),'Preserve frozen plan'
 budget=guard(True);idx=read(R/'t8_oval_20261031/training_index.json')
 split=read(V/'btc_zeroshot/btc_split.json');assert sha(V/'btc_zeroshot/btc_split.json')==SPLIT
 pub=read(B/'public_comparison/MANIFEST.json')['records'];dev=read(V/'btc_zeroshot/dev_manifest.json')['videos']
 rank=lambda x:hashlib.sha256((x['dataset']+':'+x['recording']+':codec-clock-20261111').encode()).hexdigest()
 selected=[];excluded=[]
 for ds in ['UBFC2','MCD']:
  seen=set()
  for r in sorted([x for x in pub if x['dataset']==ds],key=rank):
   if r['person'] in seen:continue
   d=Path(idx['roots'][ds])/r['recording']/'native';c=read(d/'complete.json')
   with np.load(d/'crop/meta.npz') as z:span=float(z['original_t'][-1]-z['original_t'][0])
   if span<30.1:excluded.append(dict(dataset=ds,recording=r['recording'],reason='short_source'));continue
   source=str(Path('/data_pool/khangle/facevitals/datasets/UBFC-rPPG/DATASET_2')/r['recording']/'vid.avi') if ds=='UBFC2' else r['source']
   selected.append(dict(dataset=ds,recording=r['recording'],person=r['person'],fold=r['fold'],source=source,source_sha=c['source_sha256'],meta=str(d/'crop/meta.npz'),selection_rank=rank(r)));seen.add(r['person'])
   if len(seen)==4:break
  assert len(seen)==4
 seen=set()
 for group in ['Normal-Resting','Abnormal-Resting','Task','Walking']:
  def belongs(v):
   return v['task'].lower().startswith('task') if group=='Task' else v['task']=='Walking' if group=='Walking' else v['task']=='Resting' and v['group']==group.split('-')[0]
  candidates=[]
  for v in dev:
   assert split['people'][v['person']]['partition']=='dev'
   if not belongs(v):continue
   m=v['media'];eligible=float(m['seconds'])>=30.1 and 12<float(m['avg_fps'])<=60 and int(m['width'])*int(m['height'])<=1920*1440
   if not eligible:excluded.append(dict(dataset='BTC',recording=v['recording'],reason='metadata_duration_fps_or_size_bound'));continue
   candidates.append(dict(dataset='BTC',recording=v['recording'],person=v['person'],fold=v['fold'],source=v['path'],source_sha=v['sha256'],drive_id=v['drive_id'],group=group,media=m))
  r=next(x for x in sorted(candidates,key=rank) if x['person'] not in seen);r['selection_rank']=rank(r);selected.append(r);seen.add(r['person'])
 assert len(selected)==12
 save(P/'PLAN.json',dict(utc=now(),videos=selected,exclusions=excluded,split_sha=SPLIT,selection_uses_PR_or_prediction=False,bindings=binding(),synthetic=dict(scenes=[0,1],seeds=[0,1],pulse_bpm=[0,72,120,150,180]),fps=[8,10],bf=[0,3,4],seconds=30))
 save(P/'PREFLIGHT.json',budget);print(json.dumps(dict(budget=budget,selected=[(j['dataset'],j['recording']) for j in selected])))
def spectrum(x,fs):
 x=np.asarray(x,float);f,p=signal.periodogram(x,fs=fs,window='hann',nfft=8192,detrend='linear');band=(f>=.7)&(f<=3.)
 power=float(p[band].sum());sd=float(x.std());peak=float(60*f[band][np.argmax(p[band])]) if power>1e-30 else None
 return f,p,dict(raw_wave_sd=sd,band_power=power,peak_bpm=peak,near150_fraction=float(p[abs(f-2.5)<=.1].sum()/power) if power>1e-30 else None)
def metrics(traces,clock,fs,d):
 rows=[];arrays={};qall=np.arange(len(clock))/fs
 for roi,rgb in traces.items():
  for wi,start in enumerate([0,10,20]):
   use=(qall>=start)&(qall<start+10)
   for mode,t in [('sampled_clock',qall),('source_clock',np.asarray(clock))]:
    sl=(t>=start-.2)&(t<=start+10+.2)
    q=start+np.arange(300)/30
    reason=None
    if t[0]>q[0]+1e-8 or t[-1]<q[-1]-1e-8 or np.any(~np.isfinite(rgb[sl])) or np.max(np.diff(t[sl]))>.2+1e-8:reason='invalid_support'
    x=None if reason else np.stack([np.interp(q,t,rgb[:,c]) for c in range(3)],axis=1)
    for method in ['G','POS','CHROM']:
     row=dict(roi=roi,window=wi,start=start,clock=mode,method=method,prediction=None,reason=reason)
     if x is not None:
      w=-x[:,1] if method=='G' else NS[method.lower()](x,30)
      try:row['prediction']=decode_wave(w)
      except ValueError as e:row['reason']=str(e)
      f,p,stats=spectrum(w,30);row.update(stats);key=f'{roi}_{wi}_{mode}_{method}';arrays[key+'_f']=f;arrays[key+'_p']=p;arrays[key+'_w']=w
     rows.append(row)
   # Rate-native spectra, no interpolation; identical pipeline for all bf.
   if np.isfinite(rgb[use]).all():
    for method in ['G','POS','CHROM']:
     w=-rgb[use,1] if method=='G' else NS[method.lower()](rgb[use],fs)
     f,p,stats=spectrum(w,fs);rows.append(dict(roi=roi,window=wi,start=start,clock='native_rate',method=method,prediction=stats['peak_bpm'],reason=None,**stats));key=f'{roi}_{wi}_native_rate_{method}';arrays[key+'_f']=f;arrays[key+'_p']=p;arrays[key+'_w']=w
 np.savez_compressed(d/'spectra.npz',**arrays)
 return rows
def framemeta(path):
 return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time,pict_type,key_frame,pkt_size','-of','json',str(path)]))
def schedule(g,fs):
 ff=g['frames'];types=[x['pict_type'] for x in ff];ix=np.where(np.asarray(types)!='B')[0];delta=np.diff(ix);v,c=np.unique(delta,return_counts=True)
 period=float(np.median(delta)) if len(delta) else None;dominance=float(c.max()/c.sum()) if len(c) else None
 return dict(frame_counts={s:types.count(s) for s in set(types)},reference_spacing_counts={str(int(a)):int(b) for a,b in zip(v,c)},reference_spacing_median=period,dominance=dominance,cycle_bpm=60*fs/period if period and period>1 and dominance>=.8 else None,PTS=[float(x['best_effort_timestamp_time']) for x in ff])
def encode(frames,fs,bf,d):
 assert not (d/'video.mp4').exists(),'Partial video preserved, no auto retry'
 h,w=frames[0].shape[:2]
 cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fs),'-i','pipe:0','-an','-threads','1','-filter_threads','1','-c:v','libx264','-preset','medium','-crf','23','-maxrate','500k','-bufsize','1000k','-g','240','-keyint_min','240','-sc_threshold','0','-bf',str(bf),'-b_strategy','0','-x264-params','b-adapt=0:b-pyramid=none:scenecut=0','-pix_fmt','yuv420p','-movflags','+faststart',str(d/'video.mp4')]
 save(d/'COMMAND.json',dict(command=cmd,official_MATLAB_parity=False))
 hsh=hashlib.sha256()
 with (d/'encode.log').open('wb') as log:
  proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
  try:
   for x in frames:b=x.tobytes();hsh.update(b);proc.stdin.write(b)
   proc.stdin.close();assert proc.wait(timeout=600)==0,'encode_failed'
  finally:
   if proc.poll() is None:proc.kill();proc.wait()
 g=framemeta(d/'video.mp4');save(d/'GOP.json',g);sch=schedule(g,fs);n=len(frames)
 assert len(g['frames'])==n and np.allclose(sch['PTS'],np.arange(n)/fs,atol=2e-5),'ENCODE_PTS_MISMATCH'
 return dict(encoded_sha=sha(d/'video.mp4'),input_rgb_sha=hsh.hexdigest(),wholefile_kbps=(d/'video.mp4').stat().st_size*8/(n/fs)/1000,**sch)
def decode_frames(path,h,w):
 proc=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-i',str(path),'-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
 try:
  while True:
   raw=proc.stdout.read(h*w*3)
   if not raw:break
   assert len(raw)==h*w*3,'PARTIAL_FRAME';yield np.frombuffer(raw,np.uint8).reshape(h,w,3)
  assert proc.wait(timeout=60)==0
 finally:
  proc.stdout.close()
  if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
def trace(image,points):
 good,crop,_=crop_from_points(image,points);mask=oval_mask(image.shape[:2],points)>0
 if not good:return np.full(3,np.nan),np.full(3,np.nan)
 return image[mask].mean(0),crop.astype(float).mean((0,1))
def real(plan):
 for vi,j in enumerate(plan['videos']):
  root=P/'real'/j['dataset']/j['recording'];root.mkdir(parents=True,exist_ok=True);assert not (root/'COMPLETE.json').exists(),'No duplicate completed video'
  save(P/'REAL_STATUS.json',dict(state='running',pid=os.getpid(),index=vi,recording=j['recording'],utc=now()));guard();assert sha(j['source'])==j['source_sha']
  video=pre.Video(j['source'],root/'decode.log');t=video.t
  raw=None
  if j['dataset']!='BTC':
   with np.load(j['meta']) as z:raw={k:z[k].copy() for k in z.files}
   assert len(raw['original_t'])==len(t);t=raw['original_t'];assert abs(t[0])<.05
  # Physical first sample is time0 after the source's first timestamp.
  origin=float(t[0]);t=t-origin;assert t[-1]>=30
  selections={}
  for fs in plan['fps']:
   q=np.arange(fs*30+1)/fs;r=np.searchsorted(t,q).clip(1,len(t)-1);ix=np.where(abs(t[r]-q)<abs(t[r-1]-q),r,r-1);assert np.all(np.diff(ix)>0);selections[fs]=ix
  union=np.unique(np.concatenate(list(selections.values())));assert len(union)*video.width*video.height*3<8_000_000_000
  images={};points={};it=iter(video);det=OvalCrop() if raw is None else None;wanted=set(union.tolist())
  try:
   for k,img in enumerate(it):
    if k in wanted:
     images[k]=img.copy()
     if raw is None:good,small,box,pp=det.step(img);points[k]=pp.copy()
     else:
      c=int(np.argmin(abs(raw['t']-(t[k]+origin))));assert abs(raw['t'][c]-(t[k]+origin))<1e-5;points[k]=raw['oval_points'][c].copy()
    if k>=union[-1]:break
  finally:
   it.close()
   if det:det.close()
  assert len(images)==len(union)
  np.savez_compressed(root/'SOURCE_MAP.npz',source_t=t[union],source_indices=union,points=np.asarray([points[k] for k in union]))
  for fs,ix in selections.items():
   frames=[images[k] for k in ix];pp=[points[k] for k in ix];times=t[ix];q=np.arange(len(ix))/fs
   before=np.array([trace(im,p) for im,p in zip(frames,pp)]);dest=root/f'f{fs}_before';dest.mkdir();np.savez_compressed(dest/'traces.npz',skin=before[:,0],oval36=before[:,1],source_t=times,sampled_t=q,indices=ix)
   rows=metrics(dict(skin=before[:,0],oval36=before[:,1]),times,fs,dest);save(dest/'RESULT.json',dict(dataset=j['dataset'],recording=j['recording'],fs=fs,bf=None,rows=rows))
   inputhash=None
   for bf in plan['bf']:
    d=root/f'f{fs}_b{bf}';d.mkdir();a=encode(frames,fs,bf,d)
    if inputhash is None:inputhash=a['input_rgb_sha']
    assert a['input_rgb_sha']==inputhash
    after=np.array([trace(im,p) for im,p in zip(decode_frames(d/'video.mp4',video.height,video.width),pp)])
    assert len(after)==len(ix);np.savez_compressed(d/'traces.npz',skin=after[:,0],oval36=after[:,1],source_t=times,sampled_t=q)
    rows=metrics(dict(skin=after[:,0],oval36=after[:,1]),times,fs,d)
    save(d/'RESULT.json',dict(dataset=j['dataset'],recording=j['recording'],person=j['person'],fs=fs,bf=bf,source_sha=j['source_sha'],bitrate_pass=a['wholefile_kbps']<=500,max_timing_error_s=float(max(abs(times-q))),roi_fixed=True,face_valid_fraction=float(np.isfinite(after[:,0]).all(1).mean()),rows=rows,**a))
    save(P/'REAL_STATUS.json',dict(state='running',pid=os.getpid(),index=vi,recording=j['recording'],last=f'f{fs}_b{bf}',completed_encodes=len(list((P/'real').glob('*/*/f*_b*/RESULT.json'))),utc=now()))
   guard()
  save(root/'COMPLETE.json',dict(utc=now(),source_sha=j['source_sha'],n=len(union),origin_s=origin));del images,frames
 save(P/'REAL_COMPLETE.json',dict(utc=now(),videos=len(plan['videos']),encodes=72,holdout_evaluations=0))
def synthetic_frames(scene,seed,bpm,fs):
 h,w=120,160;rng=np.random.default_rng(seed+scene*1000);tex=gaussian_filter(rng.normal(size=(h,w)),2);tex=tex/tex.std()*5
 base=np.array([180.,130.,110.])*(1 if scene==0 else .5)+tex[:,:,None]
 for k in range(30*fs+1):
  t=k/fs;dx=.8*np.sin(2*np.pi*.73*t);dy=.8*np.cos(2*np.pi*.41*t)
  im=cv2.warpAffine(base,np.float32([[1,0,dx],[0,1,dy]]),(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
  # Continuous-time pulse/light; common noise at shared times across fps and pulse conditions.
  im=im*(1+.02*np.sin(2*np.pi*.13*t))
  if bpm:im=im+1.5*np.array([.2,1.,.1])*np.sin(2*np.pi*bpm/60*t)
  noise=np.random.default_rng(seed+scene*100000+int(round(t*120))*1009).normal(0,1.5,im.shape)
  yield np.clip(np.rint(im+noise),0,255).astype(np.uint8)
def synth(plan):
 yy,xx=np.ogrid[:120,:160];mask=((xx-80)/55)**2+((yy-60)/45)**2<=1
 for scene in [0,1]:
  for seed in [0,1]:
   for bpm in [0,72,120,150,180]:
    for fs in [8,10]:
     root=P/'synthetic'/f'scene{scene}_seed{seed}_pulse{bpm}_f{fs}';root.mkdir(parents=True);frames=list(synthetic_frames(scene,seed,bpm,fs));before=np.array([im[mask].mean(0) for im in frames]);clock=np.arange(len(frames))/fs
     d=root/'before';d.mkdir();np.savez_compressed(d/'traces.npz',skin=before,t=clock);rows=metrics(dict(skin=before),clock,fs,d);save(d/'RESULT.json',dict(scene=scene,seed=seed,pulse_bpm=bpm,fs=fs,bf=None,rows=rows))
     inputhash=None
     for bf in [0,3,4]:
      d=root/f'b{bf}';d.mkdir();a=encode(frames,fs,bf,d)
      if inputhash is None:inputhash=a['input_rgb_sha']
      assert a['input_rgb_sha']==inputhash
      after=np.array([im[mask].mean(0) for im in decode_frames(d/'video.mp4',120,160)]);assert len(after)==len(frames)
      np.savez_compressed(d/'traces.npz',skin=after,t=clock);rows=metrics(dict(skin=after),clock,fs,d)
      save(d/'RESULT.json',dict(scene=scene,seed=seed,pulse_bpm=bpm,fs=fs,bf=bf,rows=rows,bitrate_pass=a['wholefile_kbps']<=500,**a))
      save(P/'SYNTH_STATUS.json',dict(state='running',pid=os.getpid(),scene=scene,seed=seed,bpm=bpm,fs=fs,bf=bf,completed_encodes=len(list((P/'synthetic').glob('*/b*/RESULT.json'))),utc=now()))
     guard()
 save(P/'SYNTH_COMPLETE.json',dict(utc=now(),encodes=120))
def audit():
 paths=[E/'common.py',E/'model_eval.py',V/'btc_zeroshot/infer_dev.py',Path('/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102/t8_evaluate.py'),Path('/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102/score_round1.py')]
 evidence=[]
 for p in paths:evidence.append(dict(path=str(p),sha256=sha(p),lines=[dict(line=i+1,text=s) for i,s in enumerate(p.read_text().splitlines()) if any(k in s for k in ['cumsum','decode_difference','decode_model','[.7,3','std() < 1e-10','std()<1e-10'])]))
 tt=np.arange(301)/30;y=np.sin(2*np.pi*1.2*tt)+.6*np.sin(2*np.pi*2.4*tt);dy=np.diff(y)
 def result(x):
  try:return dict(hr=decode_difference(x),std=float(np.std(x)),reason=None)
  except ValueError as e:return dict(hr=None,std=float(np.std(x)),reason=str(e))
 controls=dict(zero=result(np.zeros(300)),constant=result(np.ones(300)*15.33),difference=result(dy),tiny_sine=result(1e-12*dy),near_threshold_sine=result(1e-9*dy),direct_difference_peak=decode_wave(dy),reconstructed_peak=decode_difference(dy))
 assert controls['zero']['hr'] is None and controls['constant']['hr'] is None and controls['tiny_sine']['hr'] is None and abs(controls['reconstructed_peak']-72)<1
 static={}
 for p in (E/'sweep').glob('fold*/STATIC_EPOCH*.json'):static[str(p)]=read(p)
 save(P/'READOUT_AUDIT.json',dict(utc=now(),evidence=evidence,synthetic_controls=controls,existing_static_receipts=static,band_hz=[.7,3],no_frozen_code_change=True,conclusion='Cumsum present in inspected current model evaluation paths; tiny-signal argmax remains amplitude-dependent. Not a full remote runtime dependency audit of every historical run.'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','audit','real','synthetic']);a=ap.parse_args();P.mkdir(exist_ok=True)
 lease=(P/(a.mode+'.lock')).open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if a.mode=='prepare':prepare();return
 plan=read(P/'PLAN.json');assert plan['bindings']==binding(),'FROZEN_CODE_MISMATCH';assert sha(V/'btc_zeroshot/btc_split.json')==SPLIT
 save(P/(a.mode+'_LAUNCH.json'),dict(utc=now(),pid=os.getpid(),guard=guard(True),plan_sha=sha(P/'PLAN.json')))
 if a.mode=='audit':audit()
 elif a.mode=='real':real(plan)
 else:synth(plan)
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(P/(sys.argv[1]+'_ERROR.json'),dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
