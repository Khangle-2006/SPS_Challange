"""Frozen integrated signed Pearson + bounded non-wrapping lag loss."""
import torch
def correlations(p,y,max_lag=6):
    assert p.ndim==y.ndim==2 and p.shape==y.shape and p.shape[1]>2*max_lag
    values=[]
    for lag in range(-max_lag,max_lag+1):
        a,b=(p[:,:-lag],y[:,lag:]) if lag>0 else (p[:,-lag:],y[:,:lag]) if lag<0 else (p,y)
        a=a-a.mean(-1,keepdim=True); b=b-b.mean(-1,keepdim=True)
        # Clamp before sqrt as well: finite derivative at a collapsed prediction.
        den=(a.square().sum(-1)*b.square().sum(-1)).clamp_min(1e-16).sqrt()
        values.append(((a*b).sum(-1)/den).clamp(-1,1))
    return torch.stack(values,dim=-1)
def phase_loss(difference,target_difference):
    p=difference.cumsum(-1);y=target_difference.cumsum(-1)
    r=correlations(p,y)
    return .5*(1-r[:,6])+.5*(1-r.max(-1).values)
def grouped_draws(pool,seed,fold,epoch,steps,batch_size,draw):
    assert batch_size in (4,8)
    for first in range(0,steps,batch_size//2):
        pairs=[draw(pool,seed,fold,epoch,s,'joint') for s in range(first,min(first+batch_size//2,steps))]
        picks=[p for pair in pairs for p in pair]
        assert [p[0] for p in picks]==['UBFC2','MCD']*len(pairs)
        yield pairs,picks
