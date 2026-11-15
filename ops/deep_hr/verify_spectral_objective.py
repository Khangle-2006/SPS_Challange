"""Signal and gradient checks before the fixed six-run experiment."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy import signal
from hr_baseline import detrend
from spectral_objective import SpectralObjective
torch.set_num_threads(2)
loss=SpectralObjective().cuda();t=torch.arange(301,device='cuda')/30
def dy(bpm,phase=0):return torch.diff(torch.sin(2*torch.pi*bpm/60*t+phase))[None]
y=dy(126);correct=dy(126,.6);wrong=dy(65)
same=float(loss(y,y)[2]);phase=float(loss(correct,y)[2]);bad=float(loss(wrong,y)[2])
assert abs(same)<1e-6 and phase<bad and bad>1
mixed=(.2*y+wrong).detach().requires_grad_(True)
v=loss(mixed,y)[0];v.backward();assert torch.isfinite(mixed.grad).all() and mixed.grad.norm()>0
before=float(v)
after=float(loss(mixed.detach()-.001*mixed.grad,y)[0]);assert after<before
z=torch.zeros_like(y,requires_grad=True);v=loss(z,y)[0];v.backward();assert torch.isfinite(v) and torch.isfinite(z.grad).all()
wave=loss.reconstruct(correct).detach().cpu().numpy()[0]
f,p=signal.periodogram(signal.sosfiltfilt(signal.butter(1,[.75,3],btype='bandpass',fs=30,output='sos'),detrend(wave)),fs=30,window='hann',nfft=4096,detrend=False)
b=(f>=.75)&(f<=3);p=p[b];p/=p.sum();p+=1e-6;p/=p.sum()
err=float(np.max(np.abs(loss.distribution(correct).detach().cpu().numpy()[0]-p)));assert err<1e-5
report={'passed':True,'same_signal_KL':same,'phase_shifted_KL':phase,'wrong_frequency_KL':bad,
        'gradient_step_loss_before':before,'gradient_step_loss_after':after,'max_PSD_difference_vs_inference_numpy':err,
        'flat_input_finite':True,'note':'Checks establish implementation behavior, not clinical or generalization effectiveness.'}
out=Path('/data_pool/khangle/facevitals/results/spectral_ablation_v1');out.mkdir(parents=True,exist_ok=True)
(out/'loss_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
