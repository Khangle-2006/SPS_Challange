"""Fixed equal-weight, unit-band-power spectral fusion; no labels or calibration."""
import numpy as np
from scipy import signal,sparse
from scipy.sparse.linalg import spsolve
from hr_decoder import BAND_HZ
FS=30.;NFFT=4096;LOW,HIGH=BAND_HZ;LAMBDA=100.
def normalized_psd(wave,band=BAND_HZ):
    w=np.asarray(wave,dtype=float)
    if w.ndim!=1 or len(w)<FS*5 or not np.isfinite(w).all() or w.std()<1e-10:
        raise ValueError('Invalid, short, nonfinite or flat waveform')
    n=len(w);d=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')
    trend=spsolve(sparse.eye(n,format='csc')+LAMBDA**2*(d.T@d),w)
    low,high=band
    filtered=signal.sosfiltfilt(signal.butter(1,[low,high],btype='bandpass',fs=FS,output='sos'),w-trend)
    f,p=signal.periodogram(filtered,fs=FS,window='hann',nfft=NFFT,detrend=False)
    selected=(f>=low)&(f<=high);power=p[selected];total=power.sum()
    if not np.isfinite(power).all() or not np.isfinite(total) or total<=1e-30:
        raise ValueError('Invalid spectral power')
    return f[selected]*60,power/total
def summarize_three(waves):
    waves=np.asarray(waves)
    if waves.shape!=(3,300):raise ValueError('Expected exactly three 300-sample waveforms')
    psds=[];single=[]
    for w in waves:
        f,p=normalized_psd(w);psds.append(p);single.append(float(f[np.argmax(p)]))
    powers=np.stack(psds);fused=powers.mean(axis=0)
    return {'single_hr':single,'median_hr':float(np.median(single)),
            'psd_mean_hr':float(f[np.argmax(fused)]),'psd':powers,'bpm':f,'fused_psd':fused}
