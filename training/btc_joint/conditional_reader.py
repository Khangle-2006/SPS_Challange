"""Original conditional anchor; the joint recipe selects shift10_median only."""
import numpy as np
from scipy.special import softmax
import reader_models_fixed as base
import reader_v4 as legacy

def evidence(chrom,pos,experts):
    """Each input N x 277; experts is keyed by the three frozen open-rppg names."""
    return legacy.evidence(np.asarray(chrom),np.asarray(pos),experts)

def posterior_median(z,prior,beta):
    q=softmax(np.log(prior+1e-12)+np.asarray(beta).reshape(-1,1)*z,axis=1)
    return base.GRID[np.minimum((q.cumsum(1)<.5).sum(1),len(base.GRID)-1)]

def median_offsets(y,pred,low,task,person):
    off=np.zeros(4);cell=2*low+task
    w=base.pw(person)
    for k in range(4):
        ix=np.flatnonzero(cell==k)
        if len(ix)>=base.MIN_N:
            err=y[ix]-pred[ix];order=np.argsort(err);ww=w[ix][order]
            off[k]=err[order[np.searchsorted(np.cumsum(ww),ww.sum()*.5)]]
    return off

def child_density(y,per,mask,parent):
    n=len(np.unique(per[mask]))
    if n==0:return parent
    p=base.kde(y[mask],base.pw(per[mask]))
    return (n*p+24*parent)/(n+24)

def conditional_density(y,per,low,task):
    pooled=base.kde(y,base.pw(per));out=[]
    for k in (0,1):
        is_class=low==k
        parent=child_density(y,per,is_class,pooled)
        for is_task in (0,1):
            out.append(child_density(y,per,is_class&(task==is_task),parent))
    return pooled,np.asarray(out)

def fit(method,native_z,shift_z,y,person,low,task):
    low=np.asarray(low,int);task=np.asarray(task,int)
    cell=2*low+task;z=shift_z if method=='hier_shift_train' else native_z
    old=legacy.fitG(native_z,y,person,low,low,task)
    original_beta=np.array([old['beta'][c] for c in range(2)])
    original_prior=old['prior']
    if method.startswith('hier'):
        _,priors=conditional_density(y,person,low,task)
        beta=np.zeros(2)
        for k in (0,1):
            mask=low==k
            if mask.sum()<base.MIN_N:mask=np.ones(len(y),bool)
            w=base.pw(person[mask])
            losses=[np.average(abs(y[mask]-posterior_median(z[mask],priors[cell[mask]],b)),weights=w) for b in base.BETAS]
            beta[k]=base.BETAS[np.argmin(losses)]
        raw=posterior_median(z,priors[cell],beta[low])
        off=median_offsets(y,raw,low,task,person) if method!='hier' else np.zeros(4)
    else:
        beta=original_beta
        raw=posterior_median(z,original_prior,beta[low])
        off=median_offsets(y,raw,low,task,person)
        if method=='shift10_median':
            priors=np.repeat(original_prior[None],4,axis=0)
        else:
            exponent={'density_ratio_half':.5,'density_ratio_full':1}[method]
            pool,priors=conditional_density(y-off[cell],person,low,task)
            ratio=(priors+1e-12)/(pool+1e-12)
            priors=original_prior[None]*np.power(ratio,exponent)
            priors/=priors.sum(1,keepdims=True)
    return dict(method=method,train_trace_view='shift10' if method=='hier_shift_train' else 'native',
                inference_trace_view='shift10',prior=priors.tolist(),beta=beta.tolist(),offset=off.tolist())

def predict(model,shift_z,low,task):
    low=np.asarray(low,int);task=np.asarray(task,int)
    z=np.atleast_2d(shift_z);cell=2*low+task
    assert z.shape==(len(cell),len(base.GRID))
    prior=np.asarray(model['prior'])[cell];beta=np.asarray(model['beta'])[low]
    return posterior_median(z,prior,beta)+np.asarray(model['offset'])[cell]
