import model_entry_v2
m=model_entry_v2.model_maps
import torch,time,subprocess
b=m.budget(fresh=True);samples=[]
for _ in range(3):
 line=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0];idx,free=map(int,line.split(','));assert idx==0 and free>=6000;samples.append(free);time.sleep(2)
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35,0)
common=m.model(0,'cuda:0');r=m.manifest()[0];m.run(r,common,m.P/'MODEL_PREFLIGHT');m.save(m.P/'MODEL_VALIDATION.json',dict(passed=True,utc=m.now(),budget=b,gpu_free=samples,case=r['key'],result=m.read(m.P/'MODEL_PREFLIGHT/COMPLETE.json'),code_sha256=m.sha(m.P/'model_maps.py')));print('PASS gradient, finite difference, 36 occlusions, static abstention, common features')
