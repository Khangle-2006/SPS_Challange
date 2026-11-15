import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time,platform
from pathlib import Path
import numpy as np
import torch
root,code=map(Path,sys.argv[1:]);sys.path.insert(0,str(code));from EfficientPhys_fixed import EfficientPhys
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(2027)
rows=[]
for size in (36,72):
    model=EfficientPhys(frame_depth=10,img_size=size).eval();x=torch.randn(1,301,3,size,size)
    with torch.inference_mode():
        for _ in range(2):model(x)
        ts=[]
        for _ in range(5):
            t=time.perf_counter();y=model(x);ts.append((time.perf_counter()-t)*1000)
    rows.append(dict(size=size,parameters=sum(p.numel() for p in model.parameters()),median_ms_per_clip=float(np.median(ts)),mean_ms_per_clip=float(np.mean(ts)),samples_ms=ts,shape=list(x.shape)))
out=dict(device='CPU',threads=1,synthetic_random_inputs=True,trained72=False,torch=torch.__version__,platform=platform.platform(),measurements=rows,note='Forward-only timing, no decoding/ROI; synthetic72 weights are not a trained branch or quality result.')
(root/'BRANCH72_BENCHMARK.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
