"""Original posterior calibration; the joint recipe selects median_offset only."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from . import fusion
import reader_models_fixed as base
import reader_v4 as legacy

def weighted_median(values,weights):
    ix=np.argsort(values)
    return float(values[ix[np.searchsorted(weights[ix].cumsum(),weights.sum()/2)]])

def determine_offsets(name,pred,y,people,low,task):
    weights=base.pw(people)
    cell=low*2+task.astype(int)
    offsets=[]
    for k in range(4):
        ix=np.flatnonzero(cell==k)
        if ix.size<base.MIN_N:
            offsets.append(0.)
            continue
        resid=y[ix]-pred[ix];n=len(np.unique(people[ix]))
        value=weighted_median(resid,weights[ix]) if name in ('median_offset','shrink_median') else float(np.average(resid,weights=weights[ix]))
        offsets.append(value*n/(n+12) if name.startswith('shrink') else value)
    return np.asarray(offsets)

def posterior(z,low,task,m):
    p=np.array(m['priors'])[task.astype(int)]
    return fusion.posterior(np.log(p+1e-12)+np.asarray(m['beta'])[low,None]*z)

def distribution(z,low,task,m):
    q=posterior(z,low,task,m)
    if m['method']=='smooth_median':
        q=gaussian_filter1d(q,10,axis=1,mode='constant')
        q/=q.sum(axis=1,keepdims=True)
    return q

def fit_beta_varpriors(z,y,per,prior_rows,mask,method):
    ids=np.unique(per[mask]);weights=base.pw(per[mask]);loss=[]
    for b in base.BETAS:
        q=fusion.posterior(np.log(prior_rows[mask]+1e-12)+b*z[mask])
        if method=='smooth_median':
            q=gaussian_filter1d(q,10,axis=1,mode='constant');q/=q.sum(axis=1,keepdims=True)
        ae=abs(fusion.median_prob(q)-y[mask])
        if method=='cross_beta':
            loss.append([ae[per[mask]==p].mean() for p in ids])
        else:
            loss.append([np.average(ae,weights=weights)])
    losses=np.array(loss);means=losses.mean(axis=1);best=np.argmin(means)
    if method=='cross_beta':
        diff=losses-losses[best]
        # Paired bootstrap SE of the mean difference in each class.
        ix=np.random.default_rng(0).integers(0,len(ids),(1000,len(ids)))
        se=diff[:,ix].mean(axis=2).std(axis=1)
        best=np.flatnonzero(means-means[best]<=se)[0]
    return base.BETAS[best]

def fit(name,z,y,per,low,task,fold):
    if name=='bagged':
        rng=np.random.default_rng(0);ids=np.unique(per);models=[]
        for _ in range(32):
            sample=rng.choice(ids,len(ids),replace=True)
            indexes=[np.flatnonzero(per==p) for p in sample]
            bootstrap_people=np.repeat(np.arange(len(sample)),[len(i) for i in indexes])
            ix=np.concatenate(indexes)
            models.append(fit('r1',z[ix],y[ix],bootstrap_people,low[ix],task[ix],fold[ix]))
        return {'method':'bagged','models':models}
    original=legacy.fitG(z,y,per,low,low,task)
    priors=np.repeat(original['prior'][None],2,axis=0)
    beta=np.array([original['beta'][i] for i in (0,1)])
    if name=='activity_prior':
        for t in (0,1):
            ix=task==t;n=len(np.unique(per[ix]))
            if n:
                p=base.kde(y[ix],base.pw(per[ix]))
                priors[t]=(n*p+12*priors[t])/(n+12)
    if name in ('activity_prior','cross_beta','smooth_median'):
        prior_rows=priors[task.astype(int)]
        if name=='cross_beta':
            for f in np.unique(fold):
                mask=fold!=f
                prior_rows[~mask]=base.kde(y[mask],base.pw(per[mask]))
        for k in (0,1):
            subset=low==k
            if subset.sum()<base.MIN_N:
                subset=np.ones(len(low),bool)
            beta[k]=fit_beta_varpriors(z,y,per,prior_rows,subset,name)
    result={'method':name,'beta':beta,'priors':priors,'offsets':np.zeros(4)}
    if name!='activity_prior':
        result['offsets']=determine_offsets(name,fusion.median_prob(distribution(z,low,task,result)),y,per,low,task)
    return result

def predict(model,z,low,task):
    cell=2*low+task.astype(int)
    if model['method']!='bagged':
        return fusion.median_prob(distribution(z,low,task,model))+np.array(model['offsets'])[cell]
    # Crossing of mixture CDF interpolated on a 0.5-bpm output grid that
    # extends beyond the spectral grid to include all offsets.
    grid=np.arange(0,240.01,.5)
    mix=np.zeros((len(z),len(grid)))
    for m in model['models']:
        offsets=np.asarray(m['offsets'])[cell]
        C=posterior(z,low,task,m).cumsum(axis=1)
        for i in range(len(z)):
            ix=np.searchsorted(fusion.GRID,grid-offsets[i],side='right')-1
            mix[i]+=np.where(ix>=0,C[i,np.clip(ix,0,len(fusion.GRID)-1)],0)
    return grid[(mix/len(model['models'])<.5).sum(axis=1)]
