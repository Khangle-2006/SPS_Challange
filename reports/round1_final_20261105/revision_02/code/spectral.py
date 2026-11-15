"""Fixed label-free spectral SNR; no threshold fitting."""
import numpy as np
from scipy import signal,sparse
from scipy.sparse.linalg import spsolve

def snr(wave):
    w=np.asarray(wave,float)
    if w.shape!=(300,) or not np.isfinite(w).all() or w.std()<1e-10:return None
    n=len(w);d=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')
    trend=spsolve(sparse.eye(n,format='csc')+100**2*(d.T@d),w)
    y=signal.sosfiltfilt(signal.butter(1,[.7,3],fs=30,btype='bandpass',output='sos'),w-trend)
    f,p=signal.periodogram(y,fs=30,window='hann',nfft=4096,detrend=False)
    keep=(f>=.7)&(f<=3);f,p=f[keep],p[keep]
    if not np.isfinite(p).all() or p.sum()<=1e-30:return None
    peak=f[np.argmax(p)];near=np.abs(f-peak)<=.1+1e-12
    return float(p[near].sum()/max(p[~near].sum(),np.finfo(float).tiny))
