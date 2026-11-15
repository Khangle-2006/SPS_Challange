"""Paired HR errors with random-intercept repeated-measure limits of agreement."""
import numpy as np


def compute_metrics(predicted, reference, subjects, abstain=None):
    p=np.asarray(predicted,float); r=np.asarray(reference,float); s=np.asarray(subjects,str)
    if p.ndim!=1 or p.shape!=r.shape or p.shape!=s.shape:
        raise ValueError('Paired vectors of equal size required')
    a=np.zeros(len(p),bool) if abstain is None else np.asarray(abstain,bool)
    if a.shape!=p.shape:raise ValueError('Abstain shape mismatch')
    ok=np.isfinite(p)&np.isfinite(r)&~a
    m={'mae':'NA','rmse':'NA','pearson_r':'NA','bland_altman_bias':'NA',
       'loa_lower':'NA','loa_upper':'NA','coverage':float(ok.mean()) if len(p) else 0.,
       'n_windows':len(p),'n_evaluated':int(ok.sum()),'n_subjects':len(set(s[ok])),
       'loa_method':'Bland-Altman2007 paired changing quantity; random-intercept ANOVA; window-weighted bias',
       'loa_assumptions':'constant within-subject variance; independent residuals within subject; approximately normal differences',
       'loa_status':'insufficient subjects or repeated observations'}
    if not ok.any():return m
    p,r,s=p[ok],r[ok],s[ok]; d=p-r; n=len(d); groups=np.unique(s); k=len(groups)
    bias=float(d.mean())
    m.update(mae=float(np.abs(d).mean()),rmse=float(np.sqrt(np.mean(d*d))),bland_altman_bias=bias)
    if n>1 and p.std()>0 and r.std()>0:m['pearson_r']=float(np.corrcoef(p,r)[0,1])
    if k<2 or n<=k:return m
    sizes=np.asarray([sum(s==g) for g in groups],float)
    means=np.asarray([d[s==g].mean() for g in groups])
    msw=sum(float(np.sum((d[s==g]-mu)**2)) for g,mu in zip(groups,means))/(n-k)
    msb=float(np.sum(sizes*(means-bias)**2))/(k-1)
    divisor=float((n-np.sum(sizes**2)/n)/(k-1))
    raw_between=(msb-msw)/divisor; between=max(raw_between,0.)
    sd=float(np.sqrt(between+msw))
    m.update(loa_lower=bias-1.96*sd,loa_upper=bias+1.96*sd,
             between_subject_variance=between,within_subject_variance=msw,
             between_variance_truncated=bool(raw_between<0),loa_status='estimated; assumptions not automatically verified')
    return m
