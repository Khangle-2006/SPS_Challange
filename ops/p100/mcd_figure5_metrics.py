"""Paper-aligned diagnostic kernels; unspecified settings are frozen in protocol."""
import numpy as np
from scipy.signal import cheby2,sosfiltfilt,find_peaks
from scipy.stats import theilslopes

def pos_unfiltered(rgb,fs):
    rgb=np.asarray(rgb,float);length=int(np.ceil(1.6*fs));out=np.zeros(len(rgb))
    if rgb.ndim!=2 or rgb.shape[1]!=3 or not np.isfinite(rgb).all():raise ValueError('Invalid RGB')
    for n in range(length,len(rgb)):
        c=rgb[n-length:n];mean=c.mean(0)
        if np.any(abs(mean)<1e-12):raise ValueError('Zero RGB mean')
        c=(c/mean).T;s1=c[1]-c[2];s2=-2*c[0]+c[1]+c[2]
        if np.std(s2)<1e-12:raise ValueError('Degenerate POS window')
        h=s1+np.std(s1)/np.std(s2)*s2;out[n-length:n]+=h-h.mean()
    return out

def filtered(x,fs,rs=30):
    x=np.asarray(x,float)
    if not np.isfinite(x).all() or np.std(x)<1e-12:raise ValueError('Nonfinite/constant waveform')
    return sosfiltfilt(cheby2(4,rs,[.4,8],btype='bandpass',fs=fs,output='sos'),x)

def lag_metrics(pos,ppg,fs):
    pos=filtered(pos,fs);ppg=filtered(ppg,fs)
    if len(pos)!=len(ppg):raise ValueError('Lengths differ')
    margin=int(np.ceil(2*fs))+20
    core=np.arange(margin,len(pos)-margin)
    if len(core)<fs*10:raise ValueError('Insufficient common valid support')
    shifts=np.arange(-20,21);corr=[]
    ref=ppg[core];ref=ref-ref.mean();den=np.linalg.norm(ref)
    for s in shifts:
        x=pos[core+s];x=x-x.mean();scale=np.linalg.norm(x)*den
        if scale<1e-12:raise ValueError('Degenerate correlation')
        corr.append(float(np.dot(x,ref)/scale))
    corr=np.array(corr);candidates=np.flatnonzero(corr>=corr.max()-1e-12)
    best=min(candidates,key=lambda j:(abs(int(shifts[j])),int(shifts[j])))
    peaks=find_peaks(corr)[0].tolist()
    if corr[0]>corr[1]:peaks.append(0)
    if corr[-1]>corr[-2]:peaks.append(40)
    secondary=[j for j in peaks if abs(int(shifts[j]-shifts[best]))>=int(np.ceil(.25*fs))]
    second=max((float(corr[j]) for j in secondary),default=None)
    return {'lag_frames':int(shifts[best]),'lag_seconds':float(shifts[best]/fs),
        'pearson_max':float(corr[best]),'pearson_zero':float(corr[20]),
        'search_boundary':best in (0,40),'weak_peak':bool(corr[best]<.3),
        'secondary_peak':second,'ambiguous_peak':bool(second is not None and second>=.95*corr[best]),
        'shifts_frames':shifts.tolist(),'pearson_curve':corr.tolist(),
        'positive_lag_convention':'POS(t+lag) vs PPG(t): positive means POS delayed',
        'filter_rs_db':30,'filter_mode':'sosfiltfilt','fs':fs}

def drift_metrics(midpoints_s,lags_s):
    t=np.asarray(midpoints_s,float);y=np.asarray(lags_s,float)
    if len(t)!=len(y) or len(t)<3:return {'slope_ms_per_min':None,'reason':'fewer_than_three_windows'}
    if not np.all(np.diff(t)>0) or not np.isfinite(y).all():raise ValueError('Invalid time/lag series')
    return {'slope_ms_per_min':float(theilslopes(y,t).slope*60000),
            'first_last_delta_ms':float(1000*(y[-1]-y[0])),
            'lag_range_ms':float(1000*np.ptp(y)),'windows':len(y)}
