"""CPU streaming organiser substitute: frozen oval36 POS + trace-branch ROI cache."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import csv,fcntl,hashlib,json,sys,time
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np,mediapipe as mp
BASE=Path('/data/khangle/facevitals');ROOT=BASE/'reports/round1_v4_20261105';ENC=ROOT/'organiser_mf';OUT=ROOT/'organiser_evaluation'
sys.path.insert(0,str(BASE/'code/t8_oval_20261031'))
from oval_crop import crop_from_points
from oval_frontend import OVAL,oval_mask
from t8_t1_loader import supported
sys.path.insert(0,str(BASE/'code/t9_fps_codec_v2_20261031'))
from score_round1 import reference_hr
from hr_baseline import pos
from frozen_t3 import polygons_from_landmarks,rasterize_means
assert mp.__version__=='0.10.21';cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Path(__import__('score_round1').__file__))==json.loads((Path(__file__).parent/'V3_FROZEN_SHA256.json').read_text())['score_round1.py']
def save(p,x):t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
OUT.mkdir(exist_ok=True);lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
refs=json.loads((ROOT/'UBFC_C5_FROZEN_POS.json').read_text());by=defaultdict(list)
for r in refs:by[r['recording_id']].append(r)
assert len(by)==35 and len(refs)==207
def process(sid):
 d=ENC/sid;receipt=json.loads((d/'receipt.json').read_text());video=d/'organiser_mf.mp4';assert sha(video)==receipt['video_sha256']
 z=np.load(d/'frame_mapping.npz');t=z['timestamps_s'];assert len(t)==receipt['selected_frames']
 rgb36=[];regions=[];valid=[];areas=[];cap=cv2.VideoCapture(str(video))
 with mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=False,min_detection_confidence=.5,min_tracking_confidence=.5) as mesh:
  while True:
   ok,bgr=cap.read()
   if not ok:break
   image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=image.shape[:2];result=mesh.process(image)
   xy=np.full((468,2),np.nan,np.float32)
   if result.multi_face_landmarks:xy=np.asarray([(p.x*w,p.y*h) for p in result.multi_face_landmarks[0].landmark],np.float32)
   oval=xy[OVAL];good,crop,_=crop_from_points(image,oval);mask=oval_mask((h,w),oval)
   m=np.full((5,3),np.nan,np.float32);a=np.zeros(5,int)
   if good:
    m[0]=cv2.mean(image,mask=mask)[:3];a[0]=np.count_nonzero(mask)
    polys,small=polygons_from_landmarks(xy);m[1:],_,a[1:]=rasterize_means(image,polys)
   rgb36.append(crop.astype(np.float32).mean((0,1)) if good else np.full(3,np.nan));regions.append(m);valid.append(good);areas.append(a)
 cap.release();assert len(rgb36)==len(t),'Decoded frame count mismatch'
 dest=OUT/sid;dest.mkdir(exist_ok=False)
 rgb36=np.asarray(rgb36);regions=np.asarray(regions);valid=np.asarray(valid)
 np.savez_compressed(dest/'traces.npz',t=t,oval36_rgb=rgb36,region_rgb=regions,oval_valid=valid,region_valid=np.isfinite(regions).all(2),areas=np.asarray(areas),source_indices=z['source_indices'])
 # Same duplicate-time merge as preparation, without bridging invalid support.
 ts,inv=np.unique(t,return_inverse=True);count=np.bincount(inv);x=np.stack([np.bincount(inv,weights=np.nan_to_num(rgb36[:,j]))/count for j in range(3)],1)
 v=np.bincount(inv,weights=valid.astype(float))==count
 rows=[]
 for r in by[sid]:
  hr=None;reason='';start=float(r['window_start_s'])
  try:
   if not r['accepted']:raise ValueError('frozen_common_support_rejected')
   if not supported(ts,v,start,start+10):raise ValueError('organiser_variant_support')
   q=start+np.arange(300)/30;sample=np.stack([np.interp(q,ts,x[:,j]) for j in range(3)],1);wave=pos(sample,30)
   hr=reference_hr(dict(window_start_s=0.,ppg_timestamps_json=json.dumps((np.arange(301)/30).tolist()),ppg_values_json=json.dumps(np.r_[wave,wave[-1]].tolist())))
  except ValueError as exc:reason=str(exc)
  rows.append(dict(person=r['subject_id'],recording=sid,start=start,reference_eligible=r['reference_eligible'],reference_hr_bpm=r['reference_hr_bpm'],x264_hr_bpm=r['prediction_hr_bpm'],x264_accepted=r['accepted'],organiser_hr_bpm=hr,organiser_accepted=hr is not None,reason=reason))
 save(dest/'windows.json',rows);save(dest/'receipt.json',dict(source_video_sha256=receipt['video_sha256'],trace_sha256=sha(dest/'traces.npz'),frame_mapping_sha256=sha(d/'frame_mapping.npz'),T3_code_sha256=sha(Path(__file__).parent/'frozen_t3.py'),scorer_sha256=sha(Path(__import__('score_round1').__file__)),oval_rgb_for_evaluation='spatial mean of frozen oval36 INTER_AREA black-outside crop',trace_channels='full-resolution oval + 4 T3 shrunken polygons RGB; no affine',new_RGB_cache=False))
 print('EVALUATED',sid,flush=True)
done=set();start=time.time()
while len(done)<35:
 for sid in sorted(by,key=lambda s:int(s[7:])):
  if sid not in done and (ENC/sid/'receipt.json').exists():
   process(sid);done.add(sid);save(OUT/'STATUS.json',dict(state='running',completed=sorted(done),total=35))
 state=json.loads((ENC/'STATUS.json').read_text()) if (ENC/'STATUS.json').exists() else {}
 if state.get('state')=='error':raise RuntimeError('Encoder stopped: '+state.get('error',''))
 if time.time()-start>24*3600:raise RuntimeError('Encoder wait exceeded24h; preserve partial')
 if len(done)<35:time.sleep(30)
rows=[r for sid in sorted(by) for r in json.loads((OUT/sid/'windows.json').read_text())];person=defaultdict(list)
for r in rows:
 if r['x264_accepted'] and r['organiser_accepted']:person[r['person']].append((abs(r['organiser_hr_bpm']-r['reference_hr_bpm']),abs(r['x264_hr_bpm']-r['reference_hr_bpm'])))
assert len(person)==35,'Undefined person metric retained; no silent person dropping'
a=np.array([np.mean(v,axis=0) for _,v in sorted(person.items())]);delta=a[:,0]-a[:,1];rng=np.random.default_rng(0);boot=delta[rng.integers(35,size=(10000,35))].mean(1)
result=dict(organiser_MAE=float(a[:,0].mean()),x264_MAE_common_windows=float(a[:,1].mean()),delta_organiser_minus_x264=float(delta.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),paired_persons=35,paired_windows=sum(len(v) for v in person.values()),candidates=len(rows),reference_eligible=sum(r['reference_eligible'] for r in rows),organiser_accepted=sum(r['organiser_accepted'] for r in rows),fixed_abs_delta_ge2=bool(abs(delta.mean())>=2),bootstrap_seed=0,resamples=10000,substitution='Windows MediaFoundation Quality, not proven MATLAB parity',interpretation='Fixed domain-shift diagnostic; delta>=2 does not by itself establish a unique causal explanation.',trace_branch_include_variant=True)
save(OUT/'SUMMARY.json',result);save(OUT/'STATUS.json',dict(state='complete',completed=sorted(done),total=35));print('COMPLETE',json.dumps(result),flush=True)
