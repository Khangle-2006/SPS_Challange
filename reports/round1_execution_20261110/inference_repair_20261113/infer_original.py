import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from common_repair import *
from readout import *
from t8_t1_loader import supported
from t8_evaluate import decode_difference
from EfficientPhys_fixed import EfficientPhys
from replicate_tsm import use_replicate_padding
import torch,time,traceback
CONDITIONS=['native','c5','c5prime_default','c5prime_bf0']
def inventory():
 out=[];frozen=read(P/'FROZEN.json')['padding_checkpoints']
 for f in range(5):
  a=frozen[str(f)];assert sha(a['path'])==a['sha256'];out.append(dict(name=f'padding_f{f}',kind='padding',fold=f,path=a['path'],sha256=a['sha256']))
  for seed in [2027,2028,2029]:
   p=R/'t8_polarity_v3_20261102'/f'fold{f}/seed{seed}/joint/final.pt';s=read(p.parent/'status.json');assert s['state']=='complete' and s['completed_epochs']==30;h=sha(p);assert h==s['checkpoint_sha256'];out.append(dict(name=f'v3_f{f}_s{seed}',kind='v3',fold=f,seed=seed,path=str(p),sha256=h))
 return out
def load_models(inv):
 models={}
 for e in inv:
  m=EfficientPhys(frame_depth=10,img_size=36)
  if e['kind']=='padding':m=use_replicate_padding(m)
  m.load_state_dict(torch.load(e['path'],map_location='cpu',weights_only=True)['model'],strict=True);models[e['name']]=m.eval().cuda()
 return models
def chosen(j,inv):return [e for e in inv if (e['kind']=='padding' and e['fold']==j['fold']) or (e['kind']=='v3' and (j['dataset']=='BTC' or (e['fold']==j['fold'] and e['seed']==2027)))]
def load_crop(j,condition):
 if condition.startswith('c5prime'):
  root=E/'c5prime'/j['dataset']/j['recording']/condition.split('_',1)[1];a=read(root/'COMPLETE.json')
  if a['state']!='complete' or not a['bitrate_pass']:raise ValueError('C5prime_source_ineligible')
  d=root/'crop'
 else:d=Path(j['crop']) if condition=='native' else Path(j['crop']).parent.parent/('c5' if j['dataset']=='BTC' else 'f10_b500')/'crop'
 if j['dataset']=='BTC' and condition=='c5':d=O/'dev'/j['drive_id']/'c5'
 m=dict(np.load(d/'meta.npz'));t=m['t'];valid=m['valid']
 if (d/'rgb.npy').exists():
  a=read(d.parent/'complete.json');cache=a.get('cache_files',{});
  for n in ['rgb.npy','meta.npz']:
   h=cache.get(n,cache.get('crop/'+n));assert h and sha(d/n)==h
  rgb=np.asarray(np.load(d/'rgb.npy',mmap_mode='r'))
 else:
  a=read(d/'COMPLETE.json');assert sha(d/'crop.mkv')==a['crop_sha256'] and sha(d/'meta.npz')==a['meta_sha256']
  raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(d/'crop.mkv'),'-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']);assert len(raw)==len(t)*3888;rgb=np.frombuffer(raw,np.uint8).reshape(-1,36,36,3)
 return rgb,t,valid,m
def clip(rgb,t,valid,start):
 q=start+np.arange(301)/30
 if not supported(t,valid,q[0],q[-1]):raise ValueError('image_support_gap_or_tracking_failure')
 ix=np.searchsorted(t,q,side='right').clip(1,len(t)-1);w=((q-t[ix-1])/(t[ix]-t[ix-1])).astype(np.float32);x=np.asarray(rgb[ix-1],np.float32)*(1-w[:,None,None,None])+np.asarray(rgb[ix],np.float32)*w[:,None,None,None]
 if not np.isfinite(x).all() or x.std()<1e-6:raise ValueError('constant_or_nonfinite_input')
 return ((x-x.mean())/x.std()).transpose(0,3,1,2).copy(),x
def evaluate(j,c,models,inv,rgb,t,valid,meta,out):
 out.mkdir(parents=True,exist_ok=False);rows=[];waves=[];keys=[];static=[]
 for wi,w in enumerate(j['windows']):
  reason=None;x=None
  try:
   if not w['eligible']:raise ValueError('reference_ineligible')
   x,raw=clip(rgb,t,valid,w['start'])
   if j['dataset']!='BTC' and c in ['native','c5'] and not supported(meta['ppg_t'],meta['ppg_valid'],w['start'],w['start']+10):raise ValueError('ppg_support_gap')
  except ValueError as e:reason=str(e)
  for e in chosen(j,inv):
   ans={f'{mode}_{low}':None for low in [.7,.6] for mode in ['power','prominence','tilt']}
   if x is not None:
    with torch.inference_mode():wave=models[e['name']](torch.from_numpy(x[None]).cuda())[0].cpu().numpy()
    ans=all_decodes(wave)
    try:old=decode_difference(wave)
    except ValueError:old=None
    assert ans['power_0.7']==old
    keys.append([e['name'],str(w['start'])]);waves.append(wave)
    if not any(s['model']==e['name'] for s in static):
     sx=np.repeat(x[:1],301,axis=0)
     with torch.inference_mode():sw=models[e['name']](torch.from_numpy(sx[None]).cuda())[0].double().cpu().numpy()
     ss=all_decodes(sw);passed=all(v is None for v in ss.values());static.append(dict(model=e['name'],std=float(sw.std()),passed=passed,readouts=ss))
     if e['kind']=='padding':assert passed,'STATIC_PADDING_FAILED'
   rows.append(dict(dataset=j['dataset'],recording=j['recording'],person=j['person'],fold=j['fold'],condition=c,model=e['name'],input='original',start=w['start'],truth=w['truth'],reference_eligible=w['eligible'],reason=reason,readouts=ans))
 np.savez_compressed(out/'waveforms.npz',difference=np.asarray(waves),keys=np.asarray(keys));save(out/'PREDICTIONS.json',rows);save(out/'STATIC.json',static);save(out/'COMPLETE.json',dict(utc=now(),wave_sha256=sha(out/'waveforms.npz'),rows_sha256=sha(out/'PREDICTIONS.json'),rows=len(rows),accepted=sum(r['readouts']['power_0.7'] is not None for r in rows)))
def setup():
 budget(True);samples=[]
 for _ in range(3):
  row=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0];idx,free=map(int,row.split(','));assert idx==0 and free>=6000;samples.append(free);time.sleep(2)
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35,0);return samples
def main():
 lease=lock('inference_gpu.lock');assert not (P/'ORIGINAL_START.json').exists();samples=setup();inv=inventory();save(P/'MODEL_INVENTORY.json',inv);models=load_models(inv);save(P/'ORIGINAL_START.json',dict(pid=os.getpid(),utc=now(),free_MiB=samples,no_training=True,code=sha(__file__)))
 done=0
 for j in manifest():
  for c in CONDITIONS:
   save(P/'ORIGINAL_STATUS.json',dict(utc=now(),completed=done,total=372*4,current=j['key'],condition=c));out=P/'original'/j['key']/c
   try:rgb,t,valid,meta=load_crop(j,c)
   except ValueError as e:save(out/'EXCLUDED.json',dict(reason=str(e)));done+=1;continue
   evaluate(j,c,models,inv,rgb,t,valid,meta,out);done+=1
  budget()
 save(P/'ORIGINAL_COMPLETE.json',dict(utc=now(),completed=done,no_training=True))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'ORIGINAL_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()));raise
