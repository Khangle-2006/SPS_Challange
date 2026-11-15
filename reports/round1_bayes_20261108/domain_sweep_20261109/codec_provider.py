"""Full-frame encoding before a fixed, source-landmark oval; bounded RAM cache only."""
import os,sys,json,hashlib,subprocess,math,time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import cv2
V3=Path('/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102')
sys.path.insert(0,str(V3))
from t8_t1_loader import supported
from ppg_polarity import adapt_cached_meta
sys.path.insert(0,'/data_pool/khangle/facevitals/code/t8_oval_20261031')
from oval_crop import crop_from_points
cv2.setNumThreads(1)
ROOT=Path('/mnt/samsung_t5/khangle/facevitals')

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def source_path(ds,rid,receipt):
 if ds=='UBFC2':return Path('/data_pool/khangle/facevitals/datasets/UBFC-rPPG/DATASET_2')/rid/'vid.avi'
 return Path(receipt['source']) if 'source' in receipt else None

class Provider:
 def __init__(self,index,bpp_bounds,max_bytes=8_000_000_000):
  self.index=index;self.bounds=bpp_bounds;self.max_bytes=max_bytes;self.cache=OrderedDict();self.bytes=0;self.sources={};self.meta={};self.receipts={}
  self.manifest=read(ROOT/'reports/round1_bayes_20261108/public_comparison/MANIFEST.json')
 def metadata(self,ds,rid):
  if (ds,rid) not in self.meta:
   with np.load(Path(self.index['roots'][ds])/rid/'native/crop/meta.npz') as a:self.meta[ds,rid]=({k:a[k].copy() for k in a.files},adapt_cached_meta(a,ds,source_convention='raw_device')[0])
  return self.meta[ds,rid]
 def source_selection(self,ds,rid,start,recipe):
  raw,meta=self.metadata(ds,rid);t=raw['original_t'];end=min(start+12.01,t[-1])
  i0=max(0,int(np.searchsorted(t,start,side='right'))-1);i1=min(len(t)-1,int(np.searchsorted(t,end)))
  indices=np.arange(i0,i1+1);times=t[indices];fs=(len(times)-1)/(times[-1]-times[0])
  if recipe=='c5_proxy':indices=indices[::max(1,math.ceil(fs/10))];times=t[indices]
  ct=raw['t'];nearest=np.searchsorted(ct,times).clip(0,len(ct)-1);assert np.max(abs(ct[nearest]-times))<1e-6
  valid=raw['valid'][nearest].astype(bool);unique,inv=np.unique(times,return_inverse=True)
  if len(unique)!=len(times):valid=np.array([valid[inv==j].all() for j in range(len(unique))]);times=unique
  return indices,times,valid,meta
 def feasible_rate(self,ds,rid,start,recipe):
  _,t,v,m=self.source_selection(ds,rid,start,recipe)
  def okay(rate):return supported(t,v,start,start+10*rate) and supported(m['ppg_t'],m['ppg_valid'],start,start+10*rate)
  if not okay(.8):return None
  if okay(1.2):return [.8,1.2]
  lo,hi=.8,1.2
  for _ in range(32):
   mid=(lo+hi)/2
   if okay(mid):lo=mid
   else:hi=mid
  return [.8,lo]
 def prepare(self,ds,rid,start,recipe,lowres,fold):
  key=(ds,rid,float(start),recipe,bool(lowres),fold)
  if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
  folder=Path(self.index['roots'][ds])/rid/'native/crop'
  raw,meta=self.metadata(ds,rid);t=raw['original_t'];ct=raw['t'];end=min(start+12.01,t[-1])
  i0=max(0,int(np.searchsorted(t,start,side='right'))-1);i1=min(len(t)-1,int(np.searchsorted(t,end)))
  rec=read(folder.parent/'complete.json')
  source=next(Path(x['source']) for x in self.manifest['records'] if x['dataset']==ds and x['recording']==rid)
  if ds=='UBFC2':source=source_path(ds,rid,rec)
  if str(source) not in self.sources:
   assert sha(source)==rec['source_sha256'],'raw source SHA mismatch'
   probe=read_probe(source);self.sources[str(source)]=(int(probe['width']),int(probe['height']))
  w,h=self.sources[str(source)];scale=min(1.,848/w,480/h) if lowres else 1.
  w2=max(2,int(w*scale)//2*2);h2=max(2,int(h*scale)//2*2)
  # Source frame ordinals are authoritative; no seeking by approximate container time.
  vf=f'select=between(n\\,{i0}\\,{i1})'
  if (w2,h2)!=(w,h):vf+=f',scale={w2}:{h2}:flags=area'
  cmd=['ffmpeg','-v','error','-threads','2','-i',str(source),'-vf',vf,'-vsync','0','-frames:v',str(i1-i0+1),'-threads','2','-filter_threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
  started=time.monotonic();data=subprocess.check_output(cmd);assert len(data)==(i1-i0+1)*w2*h2*3,'source ordinal count mismatch'
  frames=np.frombuffer(data,np.uint8).reshape(-1,h2,w2,3);times=t[i0:i1+1];indices=np.arange(i0,i1+1)
  duration=float(times[-1]-times[0]);fs=(len(times)-1)/duration
  if recipe=='c5_proxy':
   step=max(1,math.ceil(fs/10));sel=np.arange(0,len(frames),step);frames=frames[sel];times=times[sel];indices=indices[sel];duration=float(times[-1]-times[0]);fs=(len(times)-1)/duration
   assert fs<=10+1e-8
  target=450000 if recipe=='c5_proxy' else float(np.mean(self.bounds))*w2*h2*fs
  # Deterministic target in train-fold quartile interval; no reference PR used.
  if recipe!='c5_proxy':
   rng=np.random.default_rng(int(hashlib.sha256(repr(key).encode()).hexdigest()[:12],16));target=float(rng.uniform(*self.bounds))*w2*h2*fs
  enc=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w2}x{h2}','-r',f'{fs:.12f}','-i','pipe:0','-an','-threads','2','-filter_threads','1','-c:v','libx265' if recipe=='hevc' else 'libx264','-preset','veryfast','-pix_fmt','yuv420p','-b:v',str(int(target)),'-maxrate',str(int(target)),'-bufsize',str(int(2*target)),'-g',str(max(1,round(2*fs)))]
  if recipe=='hevc':enc+=['-x265-params','pools=1:frame-threads=1:log-level=error']
  else:enc+=['-keyint_min',str(max(1,round(2*fs))),'-sc_threshold','0']
  enc+=['-movflags','frag_keyframe+empty_moov','-f','mp4','pipe:1']
  compressed=subprocess.run(enc,input=frames.tobytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
  kbps=len(compressed)*8/(len(frames)/fs)/1000
  if recipe=='c5_proxy' and kbps>500:raise RuntimeError(f'C5 proxy bitrate failed: {kbps}; no automatic rate retuning')
  decoded=subprocess.run(['ffmpeg','-v','error','-threads','2','-i','pipe:0','-threads','2','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],input=compressed,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
  assert len(decoded)==len(frames)*w2*h2*3,'encoded frame count mismatch'
  decoded=np.frombuffer(decoded,np.uint8).reshape(-1,h2,w2,3)
  points=raw['oval_points'];nearest=np.searchsorted(ct,t[indices]).clip(0,len(ct)-1);assert np.max(abs(ct[nearest]-t[indices]))<1e-6
  crops=[];valid=[]
  for image,j in zip(decoded,nearest):
   ok,crop,_=crop_from_points(image,points[j]*np.array([w2/w,h2/h]));crops.append(crop);valid.append(bool(ok and raw['valid'][j]))
  rgb=np.asarray(crops);valid=np.asarray(valid)
  # Average duplicate timestamps exactly before interpolation.
  unique,inv=np.unique(times,return_inverse=True)
  if len(unique)!=len(times):
   sums=np.zeros((len(unique),36,36,3),np.float64);np.add.at(sums,inv,rgb);counts=np.bincount(inv);rgb=(sums/counts[:,None,None,None]).astype(np.float32);valid=np.array([valid[inv==j].all() for j in range(len(unique))]);times=unique
  artifact=[]
  if recipe=='hevc' or recipe=='h264' or recipe=='c5_proxy':
   prob=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=pict_type','-of','json','pipe:0'],input=compressed,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
   types=[z['pict_type'] for z in json.loads(prob.stdout)['frames']];ii=np.where(np.array(types)=='I')[0]
   period=float(np.median(np.diff(ii))) if len(ii)>1 else None
   if period:
    for k in range(1,7):
     hz=abs((k*fs/period+fs/2)%fs-fs/2)
     if .7<=hz<=3:artifact.append(float(hz))
  receipt=dict(dataset=ds,recording=rid,start=start,recipe=recipe,requested_lowres=lowres,actual_resized=(w2,h2)!=(w,h),source_size=[w,h],encoded_size=[w2,h2],target_bpp=target/(w2*h2*fs),whole_file_kbps=kbps,encoded_frames=len(frames),analysis_source_times=[float(times[0]),float(times[-1])],fs=fs,artifact_hz=artifact,source_sha256=rec['source_sha256'],encoded_sha256=hashlib.sha256(compressed).hexdigest(),elapsed_seconds=time.monotonic()-started,official_MATLAB_parity=False,tracking='frozen native oval points transformed with resize',source_decode_command=cmd,encode_command=enc)
  value=(times,rgb,valid,meta,receipt);needed=rgb.nbytes+times.nbytes+valid.nbytes
  while self.cache and self.bytes+needed>self.max_bytes:
   _,old=self.cache.popitem(last=False);self.bytes-=old[1].nbytes+old[0].nbytes+old[2].nbytes
  assert needed<=self.max_bytes;self.cache[key]=value;self.bytes+=needed;self.receipts[repr(key)]=receipt
  return value
 def clip(self,ds,rid,start,recipe,lowres,fold,rate):
  times,rgb,valid,meta,receipt=self.prepare(ds,rid,start,recipe,lowres,fold)
  q=start+np.arange(301)/30*rate
  if not supported(times,valid,q[0],q[-1]):raise ValueError('Augmented image clock/support')
  if not supported(meta['ppg_t'],meta['ppg_valid'],q[0],q[-1]):raise ValueError('Augmented PPG clock/support')
  j=np.searchsorted(times,q,side='right').clip(1,len(times)-1);w=((q-times[j-1])/(times[j]-times[j-1])).astype(np.float32)
  x=rgb[j-1].astype(np.float32)*(1-w[:,None,None,None])+rgb[j].astype(np.float32)*w[:,None,None,None]
  assert x.std()>1e-6;x=(x-x.mean())/x.std();y=np.interp(q,meta['ppg_t'],meta['ppg']).astype(np.float32);dy=np.diff(y);assert dy.std()>1e-8;dy/=dy.std()
  return x.transpose(0,3,1,2).copy(),dy,receipt
def read_probe(p):return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(p)]))['streams'][0]
