import json, platform, time
from pathlib import Path
import numpy as np
import torch, torchvision
root=Path("/data_pool/khangle/facevitals")
torch.set_num_threads(2)
assert torch.cuda.is_available()
device=torch.device("cuda:0")
model=torch.nn.Sequential(torch.nn.Conv2d(3,16,3,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(16,1)).to(device)
x=torch.from_numpy(np.random.default_rng(2027).normal(size=(4,3,32,32)).astype(np.float32)).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
optimizer.zero_grad(); loss=model(x).square().mean(); loss.backward(); optimizer.step(); torch.cuda.synchronize()
assert torch.isfinite(loss)
result={"status":"passed","python":platform.python_version(),"torch":torch.__version__,"torchvision":torchvision.__version__,"numpy":np.__version__,"cuda_runtime":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"compute_capability":list(torch.cuda.get_device_capability(0)),"compiled_architectures":torch.cuda.get_arch_list(),"forward_backward_optimizer_step":True,"numpy_to_cuda":True,"loss":loss.item(),"max_allocated_bytes":torch.cuda.max_memory_allocated(),"visible_physical_gpu":"1","timestamp_unix":time.time()}
(root/"logs/gpu_smoke_test.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

