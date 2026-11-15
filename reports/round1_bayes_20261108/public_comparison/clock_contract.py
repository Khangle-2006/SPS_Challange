"""Frozen ordinal-clock mapping, no label-dependent offset or time reversal."""
import numpy as np
def align(original_t, canonical_t, rgb, allowed_tail_drop=0):
    t=np.asarray(original_t,float);q=np.asarray(canonical_t,float);x=np.asarray(rgb)
    assert len(x)-len(t)==allowed_tail_drop, 'undeclared decoded tail mismatch'
    assert np.isfinite(t).all() and np.all(np.diff(t)>=0), 'unordered clock'
    x=x[:len(t)];u,inv=np.unique(t,return_inverse=True)
    assert len(u)==len(q) and np.allclose(u,q,atol=1e-10,rtol=0), 'frozen canonical clock mismatch'
    if len(u)==len(t):return q,x
    out=np.full((len(u),)+x.shape[1:],np.nan)
    for i in range(len(u)):
        block=x[inv==i];finite=np.isfinite(block);den=finite.sum(0)
        np.divide(np.where(finite,block,0).sum(0),den,out=out[i],where=den>0)
    return q,out

def window(t,rgb,start,duration=10.):
    end=start+duration;t=np.asarray(t);x=np.asarray(rgb)
    assert t[0]<=start and t[-1]>=end, 'no extrapolation'
    # Endpoint support is checked per ROI; missing data remain NaN for native_grid.
    inside=(t>start)&(t<end);values=[]
    for bound in (start,end):
        j=np.searchsorted(t,bound)
        if j<len(t) and abs(t[j]-bound)<1e-10:values.append(x[j]);continue
        assert 0<j<len(t) and t[j]-t[j-1]<=.2, 'boundary gap'
        w=(bound-t[j-1])/(t[j]-t[j-1]);values.append(x[j-1]*(1-w)+x[j]*w)
    return np.r_[start,t[inside],end],np.concatenate([values[0][None],x[inside],values[1][None]])
