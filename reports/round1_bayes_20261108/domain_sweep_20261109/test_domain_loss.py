import torch
from domain_loss import weak_ce,sinc_loss
torch.set_num_threads(1)
t=torch.arange(301,dtype=torch.float64)/30
def d(hz):return torch.diff(torch.sin(2*torch.pi*hz*t))
x=torch.stack([d(1.2),d(1.5)]).requires_grad_()
truth=torch.tensor([72.,90.],dtype=torch.float64)
assert weak_ce(x,truth)<weak_ce(x,torch.tensor([120.,150.],dtype=torch.float64))
loss,parts=sinc_loss(x,[[],[]]);loss.backward();assert torch.isfinite(x.grad).all() and x.grad.abs().max()>0
_,good=sinc_loss(torch.stack([d(1.2)]*8));_,bad=sinc_loss(torch.stack([d(4.)]*8));assert bad['bandwidth']>good['bandwidth']+.5
_,collapsed=sinc_loss(torch.stack([d(1.2)]*8));_,diverse=sinc_loss(torch.stack([d(float(z)) for z in torch.linspace(.8,2.9,8)]));assert diverse['variance']<collapsed['variance']
base,_=sinc_loss(torch.stack([d(1.2)]*2),[[],[]]);penalized,_=sinc_loss(torch.stack([d(1.2)]*2),[[1.2],[1.2]]);assert penalized>base
for hz in [40.,190.]:
 try:weak_ce(x,torch.tensor([hz,80.]))
 except ValueError:pass
 else:raise AssertionError('Out-of-band weak label was accepted')
print('PASS weak-label alignment, finite nonzero gradient, bandwidth, batch collapse, measured-codec penalty, outside-band refusal')
