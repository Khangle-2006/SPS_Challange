"""Window selection on real shared clock support; never bridge missing samples."""
import numpy as np


def _validated(t, valid):
    t=np.asarray(t,float)
    if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t)<0):
        raise ValueError('Invalid timestamps')
    v=np.ones(len(t),bool) if valid is None else np.asarray(valid,bool)
    if v.shape!=t.shape:raise ValueError('Validity/time shape mismatch')
    return t,v


def invalid_intervals(t, valid, max_gap_s=.2):
    t,v=_validated(t,valid)
    bad=[]
    for i in np.flatnonzero(~v):
        bad.append((t[max(0,i-1)],t[min(len(t)-1,i+1)],'invalid_sample'))
    for i in np.flatnonzero(np.diff(t)>max_gap_s+1e-12):
        bad.append((t[i],t[i+1],'gap_support'))
    return bad


def supported_windows(video_t, ppg_t, video_valid=None, ppg_valid=None,
                      duration_s=10., stride_s=10., max_gap_s=.2):
    vt,vv=_validated(video_t,video_valid);pt,pv=_validated(ppg_t,ppg_valid)
    if min(duration_s,stride_s,max_gap_s)<=0:raise ValueError('Positive timing parameters required')
    lo=max(vt[0],pt[0]);hi=min(vt[-1],pt[-1])
    n=max(0,int(np.floor((hi-lo-duration_s+1e-9)/stride_s))+1) if hi>=lo else 0
    intervals=[(a,b,'video_'+reason) for a,b,reason in invalid_intervals(vt,vv,max_gap_s)]
    intervals += [(a,b,'ppg_'+reason) for a,b,reason in invalid_intervals(pt,pv,max_gap_s)]
    accepted=[];rejected=[]
    for i in range(n):
        start=float(lo+i*stride_s);end=start+duration_s
        reasons=sorted({reason for a,b,reason in intervals if a<end-1e-10 and b>start+1e-10})
        # An invalid sample exactly at a queried endpoint is also unavailable.
        for label,t,v in [('video',vt,vv),('ppg',pt,pv)]:
            if np.any((~v)&(t>=start-1e-10)&(t<=end+1e-10)):
                reasons=sorted(set(reasons)|{label+'_invalid_sample'})
        row={'index':i,'start_s':start,'end_s':end,'reasons':reasons}
        (rejected if reasons else accepted).append(row)
    legacy=max(0,int(np.floor((vt[-1]-vt[0]-duration_s+1e-9)/stride_s))+1)
    return {'coverage_start_s':float(lo),'coverage_end_s':float(hi),
            'candidate_windows':n,'legacy_video_only_windows':legacy,
            'support_boundary_count_difference':legacy-n,
            'valid_windows':accepted,'rejected_windows':rejected,
            'parameters':{'duration_s':duration_s,'stride_s':stride_s,'max_gap_s':max_gap_s}}


def recording_windows(recording, video_valid=None, **kwargs):
    ppg=recording.require_synchronized_ppg()
    if video_valid is None:video_valid=recording.metadata.get('frame_valid')
    return supported_windows(recording.timestamps,ppg.timestamps,video_valid,ppg.valid_mask,**kwargs)
