"""Fixed experimental PSD supervision, not a reproduction of PhysFormer loss.

L = MSE(forward differences) + 0.1 KL(target PSD || predicted PSD).
Reconstruct waveform before FFT; use the inference estimator's linear filter,
periodic Hann window, 4096 FFT, 45-180 bpm support. No labels at inference.
"""
import numpy as np
import torch
from scipy import signal
from scipy.linalg import solve

WEIGHT=.1
class SpectralObjective(torch.nn.Module):
    def __init__(self):
        super().__init__();n=300
        identity=np.eye(n);d=np.diff(identity,n=2,axis=0)
        detrend=identity-solve(identity+100**2*(d.T@d),identity,assume_a='pos')
        operator=signal.sosfiltfilt(signal.butter(1,[.75,3.],btype='bandpass',fs=30,output='sos'),detrend,axis=0).copy()
        self.register_buffer('operator',torch.tensor(operator,dtype=torch.float32))
        self.register_buffer('window',torch.hann_window(n,periodic=True))
        f=torch.fft.rfftfreq(4096,d=1/30)
        self.register_buffer('band',(f>=.75)&(f<=3))
    @staticmethod
    def reconstruct(dy):return torch.cat([torch.zeros_like(dy[:,:1]),torch.cumsum(dy[:,:-1],dim=-1)],dim=-1)
    def distribution(self,dy):
        wave=self.reconstruct(dy)
        filtered=wave@self.operator.T
        power=torch.fft.rfft(filtered*self.window,n=4096,dim=-1).abs().square()[:,self.band]
        power=power/power.sum(dim=-1,keepdim=True).clamp_min(1e-20)
        power=power+1e-6
        return power/power.sum(dim=-1,keepdim=True)
    def forward(self,pred,target):
        mse=(pred-target).square().mean()
        p=self.distribution(pred);q=self.distribution(target).detach()
        kl=(q*(q.log()-p.log())).sum(dim=-1).mean()
        return mse+WEIGHT*kl,mse,kl
