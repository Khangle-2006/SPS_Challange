import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from adabn import setup_bn
from common import save,P,sha
torch.set_num_threads(1)
m=torch.nn.Sequential(torch.nn.BatchNorm2d(3),torch.nn.Dropout(.5),torch.nn.Conv2d(3,2,1)).eval()
parameters={n:p.detach().clone() for n,p in m.named_parameters()};b=setup_bn(m)
assert not m[1].training and m[0].training
x=torch.arange(96.).reshape(2,3,4,4)
with torch.no_grad():m(x)
assert torch.allclose(m[0].running_mean,x.mean((0,2,3)))
assert torch.allclose(m[0].running_var,x.var((0,2,3),unbiased=True))
with torch.no_grad():m(x+10)
assert torch.allclose(m[0].running_mean,x.mean((0,2,3))+5)
assert all(torch.equal(parameters[n],p) for n,p in m.named_parameters())
save(P/'ADABN_CPU_TEST.json',dict(passed=True,tests=['BN reset/mean/variance','cumulative average','dropout remains eval','parameters bitwise unchanged'],adabn_sha256=sha(P/'adabn.py'),common_sha256=sha(P/'common.py')))
print('PASS AdaBN CPU regression')
