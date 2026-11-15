"""Synthetic missing-face and resampling-support tests; no validation tuning."""
import numpy as np
from video_roi_frontend import Tracker,clip_tensor
class Fake:
    def __init__(self,answers):self.answers=iter(answers)
    def detectMultiScale(self,*args,**kwargs):return next(self.answers)
def run():
    face=np.array([[20,20,80,80]]);empty=np.empty((0,4),int);gray=np.zeros((100,150),np.uint8)
    tr=Tracker(30,Fake([empty,face]))
    assert not tr.step(0,300,200,gray)[0]
    assert tr.needs_detection(1) and tr.step(1,300,200,gray)[0]
    tr=Tracker(30,Fake([face]+[empty]*5+[face]));valid=[]
    for i in range(91):valid.append(tr.step(i,300,200,gray if tr.needs_detection(i) else None)[0])
    assert all(valid[:75]) and not any(valid[75:90]) and valid[90]
    t=np.arange(0,10.1,1/30);raw=np.zeros((len(t),36,36,3),np.float16)
    raw[:]=np.arange(len(t))[:,None,None,None];good=np.ones(len(t),bool)
    assert clip_tensor(raw,t,good,0)[0] is not None
    good[150]=False;assert clip_tensor(raw,t,good,0)[0] is None
    # A zero-weight interpolation neighbor must not invalidate an exact sample.
    good[:]=True;good[301]=False;assert clip_tensor(raw,t,good,0)[0] is not None
    return {'passed':True,'initial_miss_reacquisition':True,'four_misses_tolerated_fifth_invalid':True,'later_reacquisition':True,
       'invalid_interpolation_support_rejected':True,'zero_weight_invalid_neighbor_not_used':True}
if __name__=='__main__':print(run(),flush=True)
