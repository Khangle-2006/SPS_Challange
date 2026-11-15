"""Versioned fixed readouts. Never edits the original scorer/reference files."""
import numpy as np
from scipy import signal,sparse,ndimage
from scipy.sparse.linalg import factorized
N=300;FS=30.;D=sparse.diags([np.ones(N-2),-2*np.ones(N-2),np.ones(N-2)],[0,1,2],shape=(N-2,N),format='csc');TREND=factorized(sparse.eye(N,format='csc')+100**2*(D.T@D))
def spectra(d,low=.7):
 d=np.asarray(d,float)
 if d.shape!=(300,) or not np.isfinite(d).all() or d.std()<1e-10:raise ValueError('invalid_prediction_waveform')
 c=np.cumsum(d);z=c-TREND(c);z=signal.sosfiltfilt(signal.butter(1,[low,3.],fs=30,btype='bandpass',output='sos'),z)
 f,pd=signal.periodogram(d,fs=30,window='hann',nfft=4096,detrend=False);_,pc=signal.periodogram(c,fs=30,window='hann',nfft=4096,detrend=False);_,pf=signal.periodogram(z,fs=30,window='hann',nfft=4096,detrend=False)
 return f,pd,pc,pf
def decode(d,low=.7,mode='power'):
 f,pd,pc,p=spectra(d,low);sel=(f>=low)&(f<=3);q=p.copy()
 if mode=='prominence':q=p/(ndimage.gaussian_filter1d(p,.15/(f[1]-f[0]),mode='reflect')+1e-12*max(p.max(),1e-30))
 elif mode=='tilt':q=p*4*np.sin(np.pi*f/FS)**2
 elif mode!='power':raise ValueError(mode)
 if not np.isfinite(q[sel]).all() or p[sel].sum()<=1e-30:raise ValueError('invalid_reference_spectrum')
 return float(60*f[sel][np.argmax(q[sel])])
def all_decodes(d):
 out={}
 for low in [.7,.6]:
  for mode in ['power','prominence','tilt']:
   try:out[f'{mode}_{low}']=decode(d,low,mode)
   except ValueError:out[f'{mode}_{low}']=None
 return out
