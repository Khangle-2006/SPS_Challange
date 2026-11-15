"""EfficientPhys clip inputs using T1's separate image and contact-PPG clocks."""
from pathlib import Path
import numpy as np
from window_support import invalid_intervals

def supported(t,valid,start,end,gap=.2):
    t=np.asarray(t,float);valid=np.asarray(valid,bool)
    if len(t)<2 or np.any(np.diff(t)<=0):return False
    if start<t[0]-1e-9 or end>t[-1]+1e-9:return False
    if any(a<end-1e-10 and b>start+1e-10 for a,b,_ in invalid_intervals(t,valid,gap)):return False
    return not np.any((~valid)&(t>=start-1e-10)&(t<=end+1e-10))

def clip_arrays(rgb,t,valid,ppg,ppg_t,ppg_valid,start):
    q=start+np.arange(301)/30
    if not supported(t,valid,q[0],q[-1]):raise ValueError('Image clock/support/gap')
    if not supported(ppg_t,ppg_valid,q[0],q[-1]):raise ValueError('PPG clock/support/gap')
    j=np.searchsorted(t,q,side='right').clip(1,len(t)-1)
    w=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)
    x=np.asarray(rgb[j-1],np.float32)*(1-w[:,None,None,None])+np.asarray(rgb[j],np.float32)*w[:,None,None,None]
    if not np.isfinite(x).all() or x.std()<1e-6:raise ValueError('Invalid/constant image clip')
    x=(x-x.mean())/x.std()
    y=np.interp(q,ppg_t,ppg).astype(np.float32)
    dy=np.diff(y)
    if not np.isfinite(dy).all() or dy.std()<1e-8:raise ValueError('Invalid/constant waveform target')
    dy/=dy.std()
    return x.transpose(0,3,1,2).copy(),dy,y[:300].copy()

def load_clip(folder,start):
    folder=Path(folder)
    with np.load(folder/'meta.npz') as m:
        return clip_arrays(np.load(folder/'rgb.npy',mmap_mode='r'),m['t'],m['valid'],m['ppg'],m['ppg_t'],m['ppg_valid'],start)

def tests():
    t=np.arange(601)/30;pt=.4+np.arange(2000)/100
    rgb=np.stack([np.broadcast_to(np.array([80,120,140])+4*np.sin(2*np.pi*1.2*s),(2,2,3)) for s in t]).astype(np.float32)
    ppg=np.sin(2*np.pi*1.2*pt+.3)
    x,dy,ref=clip_arrays(rgb,t,np.ones(len(t),bool),ppg,pt,np.ones(len(pt),bool),1.)
    q=1+np.arange(301)/30;truth=np.interp(q,pt,ppg).astype(np.float32)
    assert x.shape==(301,3,2,2) and dy.shape==(300,) and ref.shape==(300,)
    np.testing.assert_allclose(ref,truth[:300],atol=1e-7)
    expected=np.diff(truth);expected/=expected.std();np.testing.assert_allclose(dy,expected,atol=1e-7)
    # Different clock length/origin cannot be silently forced onto the video clock.
    assert len(t)!=len(pt) and pt[0]!=t[0]
    valid=np.ones(len(pt),bool);valid[400]=False
    try:clip_arrays(rgb,t,np.ones(len(t),bool),ppg,pt,valid,1.)
    except ValueError:pass
    else:raise AssertionError('Invalid PPG was bridged')
    keep=(pt<4)|(pt>4.4)
    try:clip_arrays(rgb,t,np.ones(len(t),bool),ppg[keep],pt[keep],np.ones(keep.sum(),bool),1.)
    except ValueError:pass
    else:raise AssertionError('PPG gap was bridged')
    try:clip_arrays(rgb,t,np.ones(len(t),bool),ppg,pt,np.ones(len(pt),bool),0.)
    except ValueError:pass
    else:raise AssertionError('PPG extrapolated')
    print('PASS independent clocks, derivative alignment, missing/gap refusal, no extrapolation')

if __name__=='__main__':tests()
