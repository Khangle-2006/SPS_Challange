"""Original person-weighted source router fit and decision rule."""
import numpy as np
import reader_models_fixed as rm

def fit_router(features, gain, person):
    """One fixed penalty, separate expected excess-loss functions for five sources."""
    weight=rm.pw(person)
    weight=weight/weight.sum()
    fits=[]
    for i in range(5):
        x=features[:,i,:]
        mean=weight@x
        sd=np.sqrt(weight@(x-mean)**2)
        sd=np.where(sd>1e-10,sd,1.)
        x=(x-mean)/sd
        b=weight@gain[:,i]
        coef=np.linalg.solve(x.T@(weight[:,None]*x)+.25*np.eye(x.shape[1]),x.T@(weight*(gain[:,i]-b)))
        fits.append((mean,sd,b,coef))
    return fits

def choose(fits,features):
    expected=np.column_stack([(features[:,i,:]-m)@ (c/s)+b for i,(m,s,b,c) in enumerate(fits)])
    return np.argmin(np.column_stack([np.zeros(len(features)),expected]),axis=1)
