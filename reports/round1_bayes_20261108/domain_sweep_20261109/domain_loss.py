"""Predeclared spectral objectives; no validation labels, hard-coded device GPU, or I/O."""
import torch

def spectrum(d, fs=30.):
    # No bandpass here: bandwidth loss must see out-of-band energy.
    w=(d-d.mean(-1,keepdim=True)).cumsum(-1)
    t=torch.linspace(-1,1,w.shape[-1],device=w.device,dtype=w.dtype)
    w=w-w.mean(-1,keepdim=True)
    w=w-(w*t).sum(-1,keepdim=True)/t.square().sum()*t
    p=torch.fft.rfft(w*torch.hann_window(w.shape[-1],device=w.device,dtype=w.dtype),n=2048).abs().square()
    f=torch.fft.rfftfreq(2048,1/fs,device=w.device)
    return f,p

def weak_ce(d, pr, sigma_bpm=5.):
    f,p=spectrum(d);mask=(f>=.7)&(f<=3.)
    if not torch.isfinite(pr).all() or not ((pr>=42)&(pr<=180)).all():
        raise ValueError('PR outside spectral support: do not clip to boundary')
    prob=(p[:,mask]+1e-10);prob=prob/prob.sum(-1,keepdim=True)
    target=torch.exp(-.5*((f[mask][None]*60-pr[:,None])/sigma_bpm)**2)
    target=target/target.sum(-1,keepdim=True)
    return -(target*prob.log()).sum(-1).mean()

def sinc_loss(d, artifact_hz=None):
    f,p=spectrum(d);band=(f>=.7)&(f<=3.);ac=f>0
    total=p[:,ac].sum(-1).clamp_min(1e-10);inside=p[:,band]
    bw=(p[:,ac&~band].sum(-1)/total).mean()
    power=inside.sum(-1,keepdim=True).clamp_min(1e-10)
    prob=inside/power;fb=f[band];peak=fb[inside.detach().argmax(-1)]
    sparse=(prob*((fb[None]-peak[:,None]).abs()>.1)).sum(-1).mean()
    # CDF distance from uniform, as in SiNC; prevents one-frequency batch collapse.
    mean=prob.mean(0);uniform=torch.ones_like(mean)/len(mean)
    variance=(mean.cumsum(0)-uniform.cumsum(0)).square().mean()
    codec=d.sum()*0
    if artifact_hz is not None:
        if len(artifact_hz)!=len(d):raise ValueError('One measured artifact list per clip required')
        penalties=[]
        for i,centers in enumerate(artifact_hz):
            hit=torch.zeros_like(fb,dtype=torch.bool)
            for hz in centers:
                if .7<=hz<=3:hit|=(fb-hz).abs()<=.05
            penalties.append(prob[i,hit].sum())
        codec=torch.stack(penalties).mean()
    return bw+sparse+variance+.1*codec,dict(bandwidth=bw,sparsity=sparse,variance=variance,codec=codec)
