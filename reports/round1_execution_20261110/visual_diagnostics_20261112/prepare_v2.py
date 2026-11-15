import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='3',LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='swrast',GALLIUM_DRIVER='llvmpipe',EGL_PLATFORM='surfaceless')
from vis_common import *
import numpy as np,cv2,csv,argparse,traceback
cv2.setNumThreads(1)
from scipy.interpolate import interp1d
from scipy import signal
from oval_crop import OvalCrop
from oval_frontend import OVAL
from t8_t1_loader import supported
from t8_evaluate import reference_hr
sys.path.insert(0,str(BACK));from data_adapters import MCD
sys.path.insert(0,str(OLD));from preprocess import Video
from pipeline import pulse

def movie(path):
 b=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(path),'-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']);return np.frombuffer(b,np.uint8).reshape(-1,36,36,3)
def unique(t,arr):
 u,inv=np.unique(t,return_inverse=True)
 if len(u)==len(t):return u,arr
 a=np.zeros((len(u),)+arr.shape[1:],np.float64);np.add.at(a,inv,arr);a/=np.bincount(inv).reshape((-1,)+(1,)*(arr.ndim-1));return u,a
def source(r,d):
 assert sha(r['source'])==r['source_sha256'],'SOURCE_HASH'
 if r['group']=='MCD Iriun':
  db=d/'db.csv'
  with db.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r['raw_row']));w.writeheader();w.writerow(r['raw_row'])
  rec=MCD(R/'datasets/MCD-rPPG',R/'mapping_audit_20261030',db_path=db).load(r['recording']);assert rec.metadata['ppg_polarity']['raw_to_canonical_sign']==-1
  detector=OvalCrop();rgb=[];boxes=[];points=[];valid=[];ts=[];it=iter(rec.frames)
  try:
   for i,frame in enumerate(it):
    if rec.timestamps[i]>31:break
    ok,small,box,pt=detector.step(frame);rgb.append(small);boxes.append(box);points.append(pt);valid.append(ok);ts.append(rec.timestamps[i])
  finally:it.close();detector.close()
  t=np.array(ts);rgb=np.array(rgb);ot=t.copy();t,rgb=unique(t,rgb);val=np.array([all(np.asarray(valid)[ot==s]) for s in t]);ppg=rec.labels['ppg'];pt,pv=unique(ppg.timestamps,ppg.values)
  return rgb,t,val,np.array(boxes),ot,np.array(points),pt,pv,np.isfinite(pv),{'source_polarity':rec.metadata['ppg_polarity'],'offset_seconds':0}
 c=Path(r['cache']);m=dict(np.load(c/'meta.npz'));n=np.searchsorted(m['t'],31,side='right');t=m['t'][:n];valid=m['valid'][:n]
 if r['group'].startswith('BTC'):
  receipt=read(c/'COMPLETE.json');assert sha(c/'crop.mkv')==receipt['crop_sha256'] and sha(c/'meta.npz')==receipt['meta_sha256'];rgb=movie(c/'crop.mkv')[:n];ot=m['t'];points=None;pt=np.array([]);pv=np.array([]);pvalid=np.array([],bool)
 else:
  receipt=read(c.parent/'complete.json');cf=receipt['cache_files']
  # Cache receipt may key by basename or relative crop path.
  for name in ['rgb.npy','meta.npz']:
   expected=cf.get(name,cf.get('crop/'+name));assert expected and sha(c/name)==expected
  rgb=np.asarray(np.load(c/'rgb.npy',mmap_mode='r')[:n]);ot=m['original_t'];points=m['oval_points'];pt=m['ppg_t'];pv=m['ppg']*r['ppg_sign'];pvalid=m['ppg_valid']
 return rgb,t,valid,m['boxes'],ot,points,pt,pv,pvalid,{'cache_receipt_sha256':sha(c/'COMPLETE.json' if r['group'].startswith('BTC') else c.parent/'complete.json'),'offset_seconds':0}

def regions_and_evm(r,d,t,boxes,ot,points,start):
 import mediapipe as mp
 # Canonical BTC colour/rotation loader; public streams are ordinary SDR.
 # Actual per-frame cache clock, not container playback clock, indexes the decoded frame ordinal.
 from types import SimpleNamespace
 stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',r['source']]))['streams'][0]
 rotation=next((float(s['rotation']) for s in stream.get('side_data_list',[]) if 'rotation' in s),0);width=int(stream['width']);height=int(stream['height'])
 if abs(rotation)%180==90:width,height=height,width
 elif rotation%180:raise ValueError('unsupported_rotation')
 hdr=stream.get('color_transfer') in ['arib-std-b67','smpte2084'];vf='format=rgb24'
 if hdr:
  assert stream.get('color_primaries')=='bt2020' and stream.get('color_space') in ['bt2020nc','bt2020c'];vf='zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0:peak=10,zscale=t=bt709:m=bt709:r=full,format=rgb24'
 v=SimpleNamespace(t=ot,width=width,height=height,vf=vf);scale=min(1,640/max(v.width,v.height));ww=max(2,round(v.width*scale/2)*2);hh=max(2,round(v.height*scale/2)*2);sx=ww/v.width;sy=hh/v.height
 targets=np.linspace(start,start+10,8);sampleidx=np.unique([int(np.argmin(abs(v.t-s))) for s in targets]);last=np.searchsorted(v.t,30.2,side='right') if r['evm'] else max(sampleidx)+1
 filt=v.vf+f',scale={ww}:{hh}:flags=area'
 if not r['evm']:filt+=',select='+ '+'.join(f'eq(n\\,{j})' for j in sampleidx)
 cmd=['ffmpeg','-v','error','-threads','1','-i',r['source'],'-frames:v',str(last if r['evm'] else len(sampleidx)),'-an','-sn','-filter_threads','1','-vf',filt,'-vsync','0','-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
 mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=False,min_detection_confidence=.5)
 masks=[];rawframes=[];evm=[];evmt=[];positions=[];fn=0
 eye1=[33,160,158,133,153,144];eye2=[362,385,387,263,373,380];mouth=[61,40,37,0,267,270,291,321,314,17,84,91]
 with (d/'raw_decode.log').open('wb') as log:
  proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log)
  try:
   while True:
    raw=proc.stdout.read(ww*hh*3)
    if not raw:break
    assert len(raw)==ww*hh*3
    fi=fn if r['evm'] else int(sampleidx[fn]);fn+=1
    if fi>=last:break
    frame=np.frombuffer(raw,np.uint8).reshape(hh,ww,3);tt=float(v.t[fi]);bi=int(np.argmin(abs(ot-tt)));box=boxes[bi].astype(float)
    if box[2]<=box[0] or box[3]<=box[1]:continue
    x0,y0,x1,y1=box;bb=np.rint(box*np.array([sx,sy,sx,sy])).astype(int);bb[[0,2]]=np.clip(bb[[0,2]],0,ww);bb[[1,3]]=np.clip(bb[[1,3]],0,hh)
    if bb[2]<=bb[0] or bb[3]<=bb[1]:continue
    roi=cv2.resize(frame[bb[1]:bb[3],bb[0]:bb[2]],(256,256),interpolation=cv2.INTER_AREA)
    if r['evm']:
     z=roi.astype(np.float32)/255
     for _ in range(4):z=cv2.pyrDown(z)
     evm.append(z[:,:,1]);evmt.append(tt)
    if fi not in sampleidx:continue
    lm=mesh.process(frame)
    if not lm.multi_face_landmarks:continue
    xyz=np.array([(p.x*v.width,p.y*v.height) for p in lm.multi_face_landmarks[0].landmark]);coords=(xyz-[x0,y0])/[x1-x0,y1-y0]*36
    ovalpts=coords[OVAL] if points is None else (points[bi]-[x0,y0])/[x1-x0,y1-y0]*36
    if not np.isfinite(ovalpts).all():continue
    oval=np.zeros((36,36),np.uint8);cv2.fillConvexPoly(oval,cv2.convexHull(np.rint(ovalpts).astype(np.int32)),1)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5));inside=cv2.erode(oval,kernel);outside=cv2.dilate(oval,kernel);edge=(outside-inside).astype(bool);eyes=np.zeros_like(oval)
    for poly in [eye1,eye2,mouth]:cv2.fillConvexPoly(eyes,cv2.convexHull(np.rint(coords[poly]).astype(np.int32)),1)
    eyes=eyes.astype(bool)&inside.astype(bool);skin=inside.astype(bool)&~eyes;background=~outside.astype(bool)
    mask=np.stack([skin,edge,eyes,background]);assert np.all(mask.sum(0)==1);masks.append(mask);rawframes.append(cv2.resize(roi,(36,36)));positions.append(tt)
  finally:
   proc.stdout.close()
   if proc.poll() is None:proc.terminate()
   proc.wait();mesh.close()
 region=np.mean(masks,axis=0) if masks else np.full((4,36,36),np.nan)
 result={'region_samples':len(masks),'sample_times':positions,'raw_scale':[ww,hh],'source_geometry':[v.width,v.height],'raw_command':cmd,'region_order':['skin_proxy','oval_edge','eyes_mouth','exterior']}
 np.savez_compressed(d/'geometry.npz',regions=region.astype(np.float32),raw_samples=np.array(rawframes),sample_times=positions)
 if r['evm'] and len(evm)>100:
  evmt=np.array(evmt);evm=np.array(evm);q=np.arange(max(0,evmt[0]),min(30,evmt[-1]),1/30);z=interp1d(evmt,evm,axis=0)(q);f=signal.sosfiltfilt(signal.butter(3,[.7,3],fs=30,btype='bandpass',output='sos'),z,axis=0);np.savez_compressed(d/'evm.npz',t=q,yt=50*f[:,:,8],raw_yt=z[:,:,8])
 return result

def prepare(r):
 d=P/'data'/r['key'];assert not d.exists(),'Partial exists; inspect before continuation';d.mkdir(parents=True)
 rgb,t,valid,boxes,ot,points,pt,pv,pvalid,receipt=source(r,d)
 starts=[s for s in r.get('starts',[float(max(0,t[0]))+j*10 for j in range(3)]) if s+10<=min(30.2,t[-1]) and supported(t,valid,s,s+10)]
 if not starts:raise ValueError('No supported 10s image window within first30s')
 starts=starts[:3];clips=[];prs=[];ppgs=[]
 for s in starts:
  q=s+np.arange(301)/30;j=np.searchsorted(t,q,side='right').clip(1,len(t)-1);w=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32);clip=np.asarray(rgb[j-1],np.float32)*(1-w[:,None,None,None])+np.asarray(rgb[j],np.float32)*w[:,None,None,None];clips.append(clip)
  y=np.full(300,np.nan);pr=float(r.get('pr',np.nan))
  if len(pt) and supported(pt,pvalid,s,s+10):
   y=np.interp(q[:-1],pt,pv);lo=max(0,np.searchsorted(pt,s,side='right')-1);hi=min(len(pt),np.searchsorted(pt,s+10,side='left')+1);pr=float(reference_hr(dict(window_start_s=float(s),ppg_timestamps_json=json.dumps(pt[lo:hi].tolist()),ppg_values_json=json.dumps(pv[lo:hi].tolist()))))
  prs.append(pr);ppgs.append(y)
 np.savez_compressed(d/'clips.npz',rgb=np.array(clips,dtype=np.float32),starts=starts,pr=prs,ppg=ppgs)
 geo=regions_and_evm(r,d,t,boxes,ot,points,starts[0]);save(d/'READY.json',dict(utc=now(),source_sha256=r['source_sha256'],clips_sha256=sha(d/'clips.npz'),geometry_sha256=sha(d/'geometry.npz'),starts=starts,pr=[float(x) if np.isfinite(x) else None for x in prs],valid_fraction=float(np.mean(valid)),receipt=receipt,geometry=geo))

def main(shard):
 lease=lock(f'prepare{shard}.lock');assert not (P/f'PREP{shard}_V2_START.json').exists();save(P/f'PREP{shard}_V2_START.json',dict(pid=os.getpid(),utc=now(),budget=budget(fresh=True)));rows=manifest()[shard::2];errors=[]
 for i,r in enumerate(rows):
  budget();save(P/f'PREP{shard}_STATUS.json',dict(pid=os.getpid(),completed=i,total=len(rows),current=r['key'],utc=now()))
  try:
   old=P/'data'/r['key']/'READY.json'
   if old.exists():
    q=read(old);assert sha(old.parent/'clips.npz')==q['clips_sha256'] and sha(old.parent/'geometry.npz')==q['geometry_sha256'];assert r['group'].startswith('BTC'),'Only verified BTC from first attempt can be reused; public source-clock fix must be applied';continue
   prepare(r)
  except Exception as e:errors.append(r['key']);save(P/'data'/r['key']/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()))
 save(P/f'PREP{shard}_COMPLETE.json',dict(utc=now(),total=len(rows),errors=errors))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--shard',type=int,required=True);main(a.parse_args().shard)
