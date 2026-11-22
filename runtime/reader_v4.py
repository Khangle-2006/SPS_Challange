"""Reader v4: fixed equal evidence fusion and original reader_v3 fit; no tuning."""
import numpy as np
import reader_models_fixed as rm
from btc_spectral_math import GRID,zlog
MODELS=("FacePhys.rlap","ME-chunk.pure","ME-flow.pure")

def pmedv(Z, prior, b):
    lp = np.log(prior + 1e-12)[None] + b * Z; q = np.exp(lp - lp.max(1, keepdims=True)); q /= q.sum(1, keepdims=True); return GRID[np.minimum((np.cumsum(q, 1) < .5).sum(1), len(GRID) - 1)]

def fit_beta(Z, L, pr, prior):
    w = rm.pw(pr); return rm.BETAS[int(np.argmin([np.sum(w * np.abs(pmedv(Z, prior, b) - L)) for b in rm.BETAS]))]

def fitG(Z, L, pr, cb, co, tk):
    """reader_v3 with separate class vectors for beta (cb) and for the offsets (co); cb == co == pixel class is reader_v3 itself."""
    prior = rm.kde(L, rm.pw(pr)); gb = fit_beta(Z, L, pr, prior); beta = {0: gb, 1: gb}
    for c in (0, 1):
        i = np.flatnonzero(cb == c)
        if len(i) >= rm.MIN_N: beta[c] = fit_beta(Z[i], L[i], pr[i], prior)
    m = dict(prior=prior, beta=beta, off={k: 0. for k in range(4)}); r = L - predG(m, Z, cb, co, tk); w = rm.pw(pr); cell = co * 2 + tk.astype(int)
    for k in range(4):
        s = cell == k
        if s.sum() >= rm.MIN_N: m['off'][k] = float(np.sum(w[s] * r[s]) / np.sum(w[s]))
    return m

def predG(m, Z, cb, co, tk):
    p = np.zeros(len(Z))
    for c in (0, 1):
        i = np.flatnonzero(cb == c)
        if len(i): p[i] = pmedv(Z[i], m['prior'], m['beta'][c])
    return p + np.array([m['off'][k] for k in co * 2 + tk.astype(int)])

def evidence(C,P,model_spectra):
    Zc=(zlog(C)+zlog(P))/2
    return (2*Zc+sum(zlog(model_spectra[m]) for m in MODELS))/5

def crossval(Ztrain,Ztest,labels,persons,folds,low,task,subset):
    predictions=np.full(len(labels),np.nan);fits={}
    for f in sorted(set(folds)):
        tr=np.flatnonzero((folds!=f)&subset);te=np.flatnonzero((folds==f)&subset)
        if not len(te):continue
        model=fitG(Ztrain[tr],labels[tr],persons[tr],low[tr],low[tr],task[tr]);fits[int(f)]=model
        predictions[te]=predG(model,Ztest[te],low[te],low[te],task[te])
    return predictions,fits
