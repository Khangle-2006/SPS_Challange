"""Original source layout, router features and prediction."""
import numpy as np
import reader_v4 as legacy
from reader_v4 import MODELS
from btc_spectral_math import GRID, zlog
from . import calibrate
from . import router_fit as rr

def sources(cp,e):
    return np.stack([cp[:,0],cp[:,1],*[e[m] for m in MODELS]],axis=1)

def evidence(s):
    joint=legacy.evidence(s[:,0],s[:,1],{m:s[:,i+2] for i,m in enumerate(MODELS)})
    return np.concatenate([joint[:,None,:],zlog(s)],axis=1)

def features(s,pred,low,task):
    peaks=GRID[np.argmax(s,axis=-1)]
    mass=np.sum(s*(abs(GRID-peaks[:,:,None])<=3),axis=-1)
    entropy=-np.sum(s*np.log(s+1e-30),axis=-1)/np.log(len(GRID))
    disp=np.stack([np.median(abs(peaks[:,i,None]-np.delete(peaks,i,axis=1)),axis=1) for i in range(5)],axis=1)
    return np.stack([mass,entropy,disp,abs(pred[:,1:]-pred[:,:1]),np.broadcast_to(low[:,None],peaks.shape),np.broadcast_to(task[:,None],peaks.shape)],axis=2)

def route(models,s,low,task):
    z=evidence(s)
    all_p=np.column_stack([calibrate.predict(m,z[:,j],low,task) for j,m in enumerate(models["readers"])])
    router=[(np.asarray(m),np.asarray(sd),float(b),np.asarray(c)) for m,sd,b,c in models["router"]]
    index=rr.choose(router,features(s,all_p,low,task))
    return all_p[np.arange(len(z)),index],index
