"""Public spectral fusion + offline tracking from b04, with explicit abstention.
Inputs are cached 300-frame CNN differences and POS/CHROM waves. No fitting here.
Weights/sigma/tau must come from the other four person-folds in b04.
"""
import numpy as np
from scipy import signal,sparse
from scipy.sparse.linalg import factorized
N=300
D=sparse.diags([np.ones(N-2),-2*np.ones(N-2),np.ones(N-2)],[0,1,2],shape=(N-2,N),format='csc')
TREND=factorized(sparse.eye(N,format='csc')+100**2*(D.T@D))
SOS=signal.butter(1,[.7,3],btype='bandpass',fs=30,output='sos')
F=np.fft.rfftfreq(4096,1/30);BAND=(F>=.7)&(F<=3);BPM=60*F[BAND]

def spectrum(w):
    w=np.asarray(w,float)
    if w.shape!=(N,) or not np.isfinite(w).all():return None
    y=signal.sosfiltfilt(SOS,w-TREND(w))
    _,p=signal.periodogram(y,fs=30,window='hann',nfft=4096,detrend=False)
    return p[BAND]

def wave_spectra(difference,pos,chrom):
    waves=[np.asarray(w,float) for w in (difference,pos,chrom)]
    # A static model can emit a DC difference; its integral is a ramp, not pulse.
    if any(w.shape!=(N,) or not np.isfinite(w).all() for w in waves):return None
    if all(np.std(w)<1e-8 for w in waves):return None
    return [spectrum(np.cumsum(waves[0])),spectrum(waves[1]),spectrum(waves[2])]

def decode(spectra,weights,starts,tracking=None):
    """spectra [windows,3,bins]; None rows abstain and break temporal tracks."""
    weights=np.asarray(weights,float);starts=np.asarray(starts,float)
    assert weights.shape==(3,) and np.all(weights>=0) and weights.max()>0
    assert len(spectra)==len(starts) and np.all(np.diff(starts)>0)
    emissions=[]
    for row in spectra:
        if row is None:emissions.append(None);continue
        row=np.asarray(row,float)
        if row.shape!=(3,len(BPM)) or not np.isfinite(row).all() or np.any(row<0):emissions.append(None);continue
        log=np.log(row+1e-30);sd=log.std(1)
        # Do not let flat/zero evidence select the first frequency via argmax.
        if np.any(sd[weights>0]<1e-12) or np.any(row.sum(1)[weights>0]<=0):emissions.append(None);continue
        z=np.divide(log-log.mean(1,keepdims=True),sd[:,None],out=np.zeros_like(log),where=sd[:,None]>0)
        emissions.append(weights@z)
    pred=[None]*len(starts);run=[]
    def finish(ids):
        if not ids:return
        E=np.array([emissions[i] for i in ids])
        if tracking is None:path=E.argmax(1)
        else:
            sigma,tau=tracking;assert sigma>0 and tau>0
            pen=-(BPM[:,None]-BPM[None,:])**2/(2*sigma**2)
            score=tau*E[0].copy();back=np.zeros(E.shape,int)
            for t in range(1,len(E)):
                m=score[:,None]+pen;back[t]=m.argmax(0);score=m.max(0)+tau*E[t]
            path=np.zeros(len(E),int);path[-1]=score.argmax()
            for t in range(len(E)-1,0,-1):path[t-1]=back[t,path[t]]
        for i,j in zip(ids,path):pred[i]=float(BPM[j])
    for i,e in enumerate(emissions):
        if e is None or (run and abs(starts[i]-starts[run[-1]]-10)>=.6):finish(run);run=[]
        if e is not None:run.append(i)
    finish(run)
    return dict(predictions=pred,abstain=[p is None for p in pred],offline_noncausal=tracking is not None)
