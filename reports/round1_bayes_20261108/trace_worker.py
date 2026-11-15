"""Native-resolution ROI extraction. Kumar2015 motivates multiple regions;
Tarassenko2014 motivates a non-skin background control. No affine/36px images.
Landmark polygons reused from versioned frozen_t3.py; no result/label-dependent ROI.
"""
from common import *
os.environ.update(LIBGL_ALWAYS_SOFTWARE='1',EGL_PLATFORM='surfaceless',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
import numpy as np,fcntl,traceback,cv2
sys.path.insert(0,str(OLD));from preprocess import Video
import mediapipe as mp
cv2.setNumThreads(1)
ROI=[[109,10,338,337,151,108],[50,101,205,187,123],[280,330,425,411,352]]
def skin(rgb):
 ycc=cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb);cr=ycc[:,:,1];cb=ycc[:,:,2]
 return (cr>=133)&(cr<=173)&(cb>=77)&(cb<=127)
def region_mean(frame,poly,background=False):
 h,w=frame.shape[:2];p=np.rint(poly).astype(int);lo=np.maximum(p.min(0),0);hi=np.minimum(p.max(0)+1,[w,h]);x0,y0=lo;x1,y1=hi
 if x1<=x0 or y1<=y0:return np.full(3,np.nan),0,0
 patch=frame[y0:y1,x0:x1];mask=np.zeros(patch.shape[:2],np.uint8);cv2.fillPoly(mask,[p-lo],1);area=int(mask.sum())
 sm=skin(patch);valid=(mask>0)&(np.all(patch>10,axis=2))&(np.all(patch<250,axis=2))&(~sm if background else sm)
 n=int(valid.sum());return (patch[valid].mean(0) if n>=max(25,.05*area) else np.full(3,np.nan)),n,area
def extract(path,dest,analysis_t=None):
 dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():
  a=read(dest/'COMPLETE.json');assert sha(dest/'trace.npz')==a['trace_sha256'];return a
 if (dest/'trace.npz').exists():raise RuntimeError('Partial trace: inspect before resume')
 source=Video(path,dest/'decode.log');n=len(source.t);means=np.full((n,4,3),np.nan,np.float32);face=np.zeros(n,bool);pixels=np.zeros((n,4),np.int32);areas=pixels.copy();boxes=np.full((n,4),np.nan,np.float32)
 detector=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=False,min_detection_confidence=.5,min_tracking_confidence=.5)
 try:
  for i,frame in enumerate(source):
   found=detector.process(frame)
   if not found.multi_face_landmarks:continue
   pts=np.array([[p.x*source.width,p.y*source.height] for p in found.multi_face_landmarks[0].landmark]);face[i]=True
   lo=pts.min(0);hi=pts.max(0);boxes[i]=[*lo,*hi];ww,hh=hi-lo
   # Above forehead when >=10px tall; otherwise larger side strip, excluding face bbox.
   x0,y0=lo;x1,y1=hi;topheight=min(.4*hh,y0)
   if topheight>=10:bg=np.array([[x0,y0-topheight],[x1,y0-topheight],[x1,y0-2],[x0,y0-2]])
   elif x0>source.width-x1:bg=np.array([[max(0,x0-.4*ww),y0],[x0-2,y0],[x0-2,y1],[max(0,x0-.4*ww),y1]])
   else:bg=np.array([[x1+2,y0],[min(source.width-1,x1+.4*ww),y0],[min(source.width-1,x1+.4*ww),y1],[x1+2,y1]])
   polygons=[pts[idx].mean(0)+.88*(pts[idx]-pts[idx].mean(0)) for idx in ROI]+[bg]
   for j,poly in enumerate(polygons):means[i,j],pixels[i,j],areas[i,j]=region_mean(frame,poly,j==3)
   if i==min(30,n-1):
    # Small reproducible QA annotation, not an analysis resize.
    panel=frame.copy()
    for j,p in enumerate(polygons):cv2.polylines(panel,[np.rint(p).astype(np.int32)],True,[(255,0,0),(0,255,0),(0,0,255),(255,255,0)][j],3)
    scale=min(1,800/source.width);cv2.imwrite(str(dest/'ROI_QA.jpg'),cv2.cvtColor(cv2.resize(panel,None,fx=scale,fy=scale),cv2.COLOR_RGB2BGR))
 finally:detector.close()
 t=source.t if analysis_t is None else np.asarray(analysis_t,float)
 assert len(t)==n and np.all(np.diff(t)>0),'analysis/source frame-clock mapping mismatch'
 np.savez_compressed(dest/'trace.npz',t=t,container_t=source.t,rgb=means,face=face,pixels=pixels,areas=areas,boxes=boxes)
 a=dict(frames=n,face_fraction=float(face.mean()),valid_roi_fraction=np.mean(np.isfinite(means).all(2),axis=0).tolist(),source=str(path),stream=source.stream,hdr=source.hdr,decode_command=source.command,trace_sha256=sha(dest/'trace.npz'),code_sha256=sha(__file__),loader_sha256=sha(OLD/'preprocess.py'))
 save(dest/'COMPLETE.json',a);return a
def main():
 lock=(OUT/'trace_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prereg();save(OUT/'trace_PREFLIGHT.json',guard())
 vs=dev_manifest();byrec={v['recording']:v for v in vs};audit=OLD/'diagnosis205_20261108'
 audit_keys=set(read(audit/'COHORT.json')['recordings'])
 top=[r['recording'] for r in rows(audit/'TOP20.csv')];base=[r for r in rows(audit/'BASELINE_VIDEO.csv') if r['recording'] in audit_keys and r['prediction'] and r['method']=='POS' and r['condition']=='native']
 worst=[r['recording'] for r in sorted(base,key=lambda r:(-abs(float(r['prediction'])-float(r['truth'])),r['recording']))[:20]]
 order=list(dict.fromkeys(top+worst+[v['recording'] for v in vs]));save(OUT/'TRACE_ORDER.json',dict(order=order,background_top20=top,background_worst20=worst,selection_uses_old_diagnostic_errors=True,all_dev_still_included=True))
 for j,rec in enumerate(order):
  v=byrec[rec]
  for cond in ['native','c5']:
   guard(False);p=Path(v['path']) if cond=='native' else OLD/'dev'/v['drive_id']/'c5_encoded/video.mp4';d=OUT/'traces'/rec/cond
   if not p.exists():save(d/'EXCLUDED.json',dict(reason='no_existing_c5',recording=rec));continue
   save(OUT/'trace_STATUS.json',dict(state='extracting',recording=rec,condition=cond,video_index=j+1,total=221,pid=os.getpid(),utc=now()))
   if cond=='native':assert sha(p)==v['sha256']
   clock=None
   if cond=='c5':
    with np.load(OLD/'dev'/v['drive_id']/'c5_encoded/frame_mapping.npz') as mapping:clock=mapping['timestamps_s']
   extract(p,d,clock)
 save(OUT/'trace_COMPLETE.json',dict(utc=now(),videos=221,holdout_evaluations=0))
if __name__=='__main__':
 try:main()
 except Exception as e:save(OUT/'trace_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=now()));raise
