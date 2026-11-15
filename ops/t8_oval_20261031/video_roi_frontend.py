"""Image-only causal Haar frontend shared by control, evaluation and timing."""
import math,time
from pathlib import Path
import cv2
import numpy as np
class Tracker:
    def __init__(self,fps,detector=None):
        self.period=max(1,round(fps/2));self.last=None;self.misses=0;self.attempts=0;self.successes=0
        self.detector=detector if detector is not None else cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        if hasattr(self.detector,'empty') and self.detector.empty():raise RuntimeError('Missing Haar detector')
    def needs_detection(self,i):return i%self.period==0 or self.last is None
    def step(self,i,width,height,gray=None):
        fresh=False
        if self.needs_detection(i):
            if gray is None:raise ValueError('Required detector input missing')
            faces=self.detector.detectMultiScale(gray,1.1,5,minSize=(30,30));self.attempts+=1
            if len(faces):
                if self.last is None:face=max(faces,key=lambda b:int(b[2])*int(b[3]))
                else:
                    lx,ly,lw,lh=self.last
                    face=min(faces,key=lambda b:(2*b[0]+b[2]-lx-lw/2)**2+(2*b[1]+b[3]-ly-lh/2)**2)
                self.last=tuple(int(v)*2 for v in face);self.misses=0;self.successes+=1;fresh=True
            else:self.misses+=1
        if self.last is None or self.misses>4:return False,np.zeros(4,dtype=np.int32),fresh
        x,y,bw,bh=self.last
        box=np.array([max(0,int(x+.15*bw)),max(0,int(y+.1*bh)),min(width,int(x+.85*bw)),min(height,int(y+.85*bh))],dtype=np.int32)
        valid=bool(box[2]>box[0] and box[3]>box[1]);return valid,box if valid else np.zeros(4,dtype=np.int32),fresh
def process_video(path,timestamps,capture_grays=False):
    """No reference boxes, physiological labels or cross-condition images accepted."""
    cv2.setNumThreads(1);timestamps=np.asarray(timestamps);n=len(timestamps)
    assert n>1 and np.all(np.diff(timestamps)>=0)
    start=time.perf_counter();cap=cv2.VideoCapture(str(path));opened=cap.isOpened()
    fps=float(cap.get(cv2.CAP_PROP_FPS));declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crops=np.zeros((n,36,36,3),np.uint8);boxes=np.zeros((n,4),np.int32);usable=np.zeros(n,bool);fresh=np.zeros(n,bool)
    grays=[];gray_indices=[];decoded=0;decode_s=0.;detect_s=0.;crop_s=0.;extra=False;dims=None
    tracker=Tracker(fps) if opened and 1<fps<300 else None
    if tracker is not None:
        while True:
            t=time.perf_counter();ok,frame=cap.read();decode_s+=time.perf_counter()-t
            if not ok:break
            if decoded>=n:extra=True;decoded+=1;break
            i=decoded;decoded+=1;h,w=frame.shape[:2];dims=[w,h]
            t=time.perf_counter();gray=None
            if tracker.needs_detection(i):
                gray=cv2.cvtColor(cv2.resize(frame,(w//2,h//2)),cv2.COLOR_BGR2GRAY)
                if capture_grays:grays.append(gray.copy());gray_indices.append(i)
            valid,box,is_fresh=tracker.step(i,w,h,gray);detect_s+=time.perf_counter()-t
            boxes[i]=box;usable[i]=valid;fresh[i]=is_fresh
            if valid:
                t=time.perf_counter();x1,y1,x2,y2=box
                crops[i]=cv2.cvtColor(cv2.resize(frame[y1:y2,x1:x2],(36,36),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                crop_s+=time.perf_counter()-t
    cap.release()
    # Extra frames make alignment ambiguous: fail the recording, never truncate silently.
    if extra:usable[:]=False;crops[:]=0
    merge_start=time.perf_counter();unique,inverse,counts=np.unique(timestamps,return_inverse=True,return_counts=True)
    merged=np.zeros((len(unique),36,36,3),np.float32);np.add.at(merged,inverse,crops);merged/=counts[:,None,None,None]
    valid_unique=np.ones(len(unique),bool);np.logical_and.at(valid_unique,inverse,usable)
    merge_s=time.perf_counter()-merge_start
    report={'opened':opened,'container_fps':fps,'declared_frames':declared,'expected_frames':n,'decoded_frames':decoded,
       'decode_complete':bool(decoded==n and not extra),'extra_frames':extra,'dimensions':dims,
       'detector_period_frames':tracker.period if tracker else None,'detection_attempts':tracker.attempts if tracker else 0,
       'fresh_detection_successes':tracker.successes if tracker else 0,'usable_ROI_frames':int(usable.sum()),'unusable_ROI_frames':int((~usable).sum()),
       'decode_s':decode_s,'detector_resize_gray_tracking_s':detect_s,'crop_resize_rgb_s':crop_s,'timestamp_duplicate_merge_s':merge_s,
       'frontend_wall_s':time.perf_counter()-start,'failure_policy':'No native-image fallback. No initial face or >4 consecutive scheduled misses makes frame unusable; later reacquisition allowed. Missing tail unusable; extra frames invalidate whole recording.'}
    result={'rgb':merged.astype(np.float16),'t':unique,'valid':valid_unique,'boxes':boxes,'original_valid':usable,'fresh':fresh,'report':report}
    if capture_grays:
        result.update(gray_frames=np.asarray(grays),gray_indices=np.asarray(gray_indices),frame_dimensions=np.asarray(dims),fps=fps)
    return result
def clip_tensor(raw,t,valid,start_s):
    """301 samples at30Hz; fail if any positive-weight interpolation support is invalid."""
    q=start_s+np.arange(301)/30
    if q[0]<t[0] or q[-1]>t[-1]:return None,301
    j=np.clip(np.searchsorted(t,q,side='right'),1,len(t)-1);w=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)
    good=((w>=1)|valid[j-1])&((w<=0)|valid[j]);bad=int((~good).sum())
    if bad:return None,bad
    ww=w[:,None,None,None];x=raw[j-1].astype(np.float32)*(1-ww)+raw[j].astype(np.float32)*ww
    sd=float(x.std())
    if sd<1e-6 or not np.isfinite(x).all():return None,0
    x=(x-float(x.mean()))/sd
    return x.transpose(0,3,1,2).copy(),0
