import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from vis_common import *
import numpy as np,time,traceback
sys.path.insert(0,str(R/'reports/round1_v4_20261105/code'))
from EfficientPhys_fixed import EfficientPhys
from replicate_tsm import use_replicate_padding
from t8_evaluate import decode_difference
import torch

def model(f,device):
 p=checkpoint(f);assert sha(p)==read(P/'FROZEN.json')['checkpoints'][str(f)]['sha256']
 m=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36));s=torch.load(p,map_location='cpu',weights_only=True);m.load_state_dict(s['model'],strict=True);m.eval();return m.to(device)
def normalize(a):
 a=a.astype(np.float32);return ((a-a.mean())/max(float(a.std()),1e-6)).transpose(0,3,1,2).copy()
def power(d,f):
 y=d.double().cumsum(-1);n=y.shape[-1];t=torch.arange(n,device=y.device,dtype=torch.float64)/30;c=t-t.mean();y=y-y.mean(-1,keepdim=True);y=y-(y*c).sum(-1,keepdim=True)/c.square().sum()*c;win=torch.hann_window(n,periodic=True,device=y.device,dtype=torch.float64);phase=2*np.pi*f*t
 return ((y*win*phase.cos()).sum(-1).square()+(y*win*phase.sin()).sum(-1).square())/win.sum().square()
def run(r,common,out_override=None):
 d=P/'data'/r['key'];out=out_override or P/'model'/r['key'];assert not out.exists();out.mkdir(parents=True);ready=read(d/'READY.json');assert sha(d/'clips.npz')==ready['clips_sha256'];a=dict(np.load(d/'clips.npz'));m=model(r['fold'],'cuda:0');waves=[];freq=[];features=[]
 hidden=[];hook=common.final_dense_1.register_forward_hook(lambda mod,inp,out:hidden.append(out.detach().tanh()))
 try:
  for raw in a['rgb']:
   x=torch.from_numpy(normalize(raw)[None]).to('cuda:0')
   with torch.no_grad():wave=m(x)[0].cpu().numpy();common(x);features.append(hidden.pop().mean(0).cpu().numpy())
   waves.append(wave)
   try:freq.append(float(decode_difference(wave)))
   except ValueError:freq.append(np.nan)
 finally:hook.remove()
 x=torch.from_numpy(normalize(a['rgb'][0])[None]).to('cuda:0');static=x[:,:1].repeat(1,301,1,1,1)
 with torch.no_grad():sw=m(static)[0].cpu().numpy()
 try:shr=decode_difference(sw)
 except ValueError:shr=None
 assert shr is None and np.std(sw.astype(np.float64))<1e-10,'STATIC_MODEL_REGRESSION'
 grad=np.full((36,36),np.nan);occ=np.full((6,6),np.nan);fd=None;baseline=None
 if np.isfinite(freq[0]):
  x.requires_grad_(True);p=power(m(x),freq[0]/60).sum();g=torch.autograd.grad(p,x)[0];baseline=float(p.detach());grad=g.abs().sum((0,1,2)).detach().cpu().numpy()
  # Fixed deterministic direction and step, real model finite-difference audit.
  direction=g.detach()/max(float(g.norm()),1e-30);eps=.01
  with torch.no_grad():fdv=float((power(m(x+eps*direction),freq[0]/60)-power(m(x-eps*direction),freq[0]/60))/(2*eps));ad=float((g*direction).sum())
  fd=dict(analytical=ad,finite_difference=fdv,relative_error=abs(fdv-ad)/max(abs(ad),1e-15),step=eps)
  assert fd['relative_error']<.15 or abs(fdv-ad)<1e-8,'GRADIENT_FINITE_DIFFERENCE'
  with torch.no_grad():
   for j in range(6):
    for k in range(6):
     z=x.detach().clone();z[:,:, :,j*6:(j+1)*6,k*6:(k+1)*6]=0;occ[j,k]=baseline-float(power(m(z),freq[0]/60))
  m.zero_grad(set_to_none=True)
 geo=dict(np.load(d/'geometry.npz'));regional={}
 for name,heat in [('gradient',grad),('occlusion',np.repeat(np.repeat(abs(occ),6,0),6,1))]:
  total=np.nansum(heat)
  if total>0 and np.isfinite(geo['regions']).all():regional[name]=dict(zip(['skin_proxy','oval_edge','eyes_mouth','exterior'],[float(np.sum(heat*mask)/total) for mask in geo['regions']]))
 np.savez_compressed(out/'outputs.npz',difference=waves,predicted_hr=freq,features_common=features,gradient=grad,occlusion_signed=occ,mean_rgb=a['rgb'][0].mean(0),starts=a['starts'],pr=a['pr'],ppg=a['ppg'])
 save(out/'COMPLETE.json',dict(utc=now(),checkpoint_sha256=sha(checkpoint(r['fold'])),common_feature_checkpoint_sha256=sha(checkpoint(0)),input_sha256=sha(d/'clips.npz'),outputs_sha256=sha(out/'outputs.npz'),regional=regional,region_area=np.mean(geo['regions'],axis=(1,2)).tolist() if np.isfinite(geo['regions']).all() else None,static_hr=shr,static_std=float(np.std(sw.astype(np.float64))),baseline_power=baseline,finite_difference=fd,predicted_hr=[float(v) if np.isfinite(v) else None for v in freq],peak_memory_bytes=torch.cuda.max_memory_allocated()))
 del x,m;torch.cuda.empty_cache()
def main():
 lease=lock('model.lock');assert not (P/'MODEL_START.json').exists();b=budget(fresh=True);samples=[]
 for _ in range(3):
  line=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0];idx,free=map(int,line.split(','));assert idx==0 and free>=6000;samples.append(free);time.sleep(2)
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35,0)
 save(P/'MODEL_START.json',dict(utc=now(),pid=os.getpid(),budget=b,free_MiB=samples,code_sha256=sha(__file__),no_training=True));rows=manifest();done=set();errors=[];common=model(0,'cuda:0');started=time.time()
 while len(done)<len(rows):
  assert time.time()-started<4*3600,'FOUR_HOUR_DIAGNOSTIC_BOUND'
  for r in rows:
   if r['key'] in done:continue
   d=P/'data'/r['key']
   if (d/'ERROR.json').exists():done.add(r['key']);errors.append(dict(key=r['key'],stage='prepare'));continue
   if not (d/'READY.json').exists():continue
   save(P/'MODEL_STATUS.json',dict(utc=now(),pid=os.getpid(),completed=len(done),current=r['key']));budget()
   try:run(r,common)
   except Exception as e:errors.append(dict(key=r['key'],stage='model'));save(P/'model'/r['key']/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()))
   done.add(r['key'])
  if len(done)<len(rows):time.sleep(10)
 save(P/'MODEL_COMPLETE.json',dict(utc=now(),total=len(done),errors=errors,no_training=True))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'MODEL_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()));raise
