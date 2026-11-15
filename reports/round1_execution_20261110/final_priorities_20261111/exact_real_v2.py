"""Operational V4: preserve audited MCD retained ordinal prefix and resume only unencoded sources."""
from audit_clock import *
import audit_clock as base
from run_v2 import metrics_v2 as metrics

def validate_mapping(j,raw,container_t):
 m=read(Path(j['meta']).parent/'manifest.json')['metadata'];n=len(raw['original_t'])
 assert len(raw['oval_points'])==n
 assert np.all(np.diff(raw['original_t'])>=0)
 if j['dataset']=='MCD':assert np.array_equal(raw['t'],raw['original_t'])
 else:assert np.array_equal(raw['t'],np.unique(raw['original_t']))
 if j['dataset']=='MCD':
  assert m['mapping']=='row i = decoded frame i; final decoded frame and excess rows dropped'
  assert m['dropped_final_decoded_frames']==1 and len(container_t)==n+1
  assert m['source_video_sha256']==j['source_sha']
  assert m['clock']=='actual metadata and raw contact PPG wallclock'
 else:assert len(container_t)==n
 return dict(dataset=j['dataset'],recording=j['recording'],container_frames=len(container_t),retained_frames=n,dropped_tail=len(container_t)-n,source_clock='frozen original_t',cached_mapping=m.get('mapping'),landmark_mapping='original decoded frame ordinal, before timestamp deduplication',meta_sha=sha(j['meta']),manifest_sha=sha(Path(j['meta']).parent/'manifest.json'),clock_origin=m.get('clock_origin_microseconds'),source_sha=j['source_sha'])

def real(plan):
 for vi,j in enumerate(plan['videos']):
  root=P/'real'/j['dataset']/j['recording'];root.mkdir(parents=True,exist_ok=True)
  if (root/'EXCLUDED.json').exists():
   e=read(root/'EXCLUDED.json');assert e['source_sha']==j['source_sha'] and e['reason']=='source_clock_gap_repeated_selection';continue
  if (root/'COMPLETE.json').exists():
   assert read(root/'COMPLETE.json')['source_sha']==j['source_sha']
   for fs in plan['fps']:
    for bf in plan['bf']:
     d=root/f'f{fs}_b{bf}';a=read(d/'RESULT.json');assert sha(d/'video.mp4')==a['encoded_sha']
   continue
  save(P/'REAL_STATUS.json',dict(state='running',pid=os.getpid(),index=vi,recording=j['recording'],utc=now()));guard();assert sha(j['source'])==j['source_sha']
  video=pre.Video(j['source'],root/'decode.log');t=video.t
  raw=None
  if j['dataset']!='BTC':
   with np.load(j['meta']) as z:raw={k:z[k].copy() for k in z.files}
   mapping=validate_mapping(j,raw,t);save(root/'MAPPING_V3.json',mapping);t=raw['original_t'];assert abs(t[0])<.05
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
      c=k;assert abs(raw['original_t'][c]-(t[k]+origin))<1e-5;points[k]=raw['oval_points'][c].copy()
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
 save(P/'REAL_COMPLETE.json',dict(utc=now(),planned_videos=len(plan['videos']),videos=11,excluded_videos=1,planned_encodes=72,encodes=66,holdout_evaluations=0))