"""Frozen padding checkpoint, same cached tensors, batch placement/companions/permutation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from common_causal import *
from infer_original import clip,EfficientPhys,use_replicate_padding
from crop_io import cached
import torch,numpy as np,traceback,inspect
sys.path.insert(0,str(C/'supplied/FaceVitals_BTC_Causal_Audit_V1'))
from compare_stage_exports import compare

def buffers(model):return hashlib.sha256(b''.join(n.encode()+arrsha(v.detach().cpu().numpy()).encode() for n,v in model.named_buffers())).hexdigest()

def select():
 jobs=[j for j in repair.manifest() if j['dataset']=='BTC'];chosen=[];people=set()
 for fold in range(5):
  for group in ['Normal','Abnormal']:
   count=0
   for j in sorted(jobs,key=lambda r:r['key']):
    if j['fold']==fold and j['metadata']['group']==group and j['person'] not in people:
     chosen.append(j);people.add(j['person']);count+=1
     if count==2:break
   assert count==2
 assert len(chosen)==20;return chosen

def main():
 lease=lock('batch.lock');assert not (C/'BATCH_START.json').exists();check_deadline();b=budget(True);samples=[]
 for _ in range(3):
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.total','--format=csv,noheader,nounits'],text=True).splitlines();idx,free,total=map(int,rows[0].split(','));assert idx==0 and free>=6000;samples.append(free);time.sleep(2)
 # Small read-only audit shares GPU0 with the existing inference job; never interrupts it.
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.08,0)
 jobs=select();save(C/'BATCH_SELECTION.json',[dict(key=j['key'],person=j['person'],fold=j['fold'],group=j['metadata']['group']) for j in jobs]);save(C/'BATCH_START.json',dict(utc=now(),pid=os.getpid(),code=sha(__file__),budget=b,free_MiB=samples,device='cuda:0',dtype='float32',AMP=False,per_process_memory_fraction=.08,existing_AB_untouched=True));xs=[];starts=[]
 for j in jobs:
  rgb,m,_=cached(j,'native');found=False
  for w in j['windows']:
   if not w['eligible']:continue
   try:x,raw=clip(rgb,m['t'],m['valid'],w['start']);found=True;break
   except ValueError:continue
  assert found,'PRESELECTED_CASE_NO_SUPPORTED_WINDOW';xs.append(x);starts.append(w['start'])
 inv=read(P/'MODEL_INVENTORY.json');records=[];max_delta=0.;max_repeat=0.;max_peer=0.;flags=[]
 for i,j in enumerate(jobs):
  check_deadline();save(C/'BATCH_STATUS.json',dict(utc=now(),completed=i,total=20,current=j['key']));entry=next(e for e in inv if e['kind']=='padding' and e['fold']==j['fold']);assert sha(entry['path'])==entry['sha256'];model=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36));state=torch.load(entry['path'],map_location='cpu',weights_only=True)['model'];model.load_state_dict(state,strict=True);model=model.eval().cuda();initial=buffers(model)
  settings=[]
  for name,layer in model.named_modules():
   if isinstance(layer,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout)) or hasattr(layer,'n_segment'):settings.append(dict(name=name,class_name=type(layer).__name__,training=layer.training,track_running_stats=getattr(layer,'track_running_stats',None),n_segment=getattr(layer,'n_segment',None)))
  assert not model.training and all(not s['training'] for s in settings)
  out=C/'batch'/j['key'];out.mkdir(parents=True);np.save(out/'input.npy',xs[i]);runs=[];waves=[];shape_log=[]
  def forward(ids,label):
   model.load_state_dict(state,strict=True);assert buffers(model)==initial;positions=[n for n,k in enumerate(ids) if k==i];x=torch.from_numpy(np.stack([xs[k] for k in ids])).cuda();hook=model.TSM_1.register_forward_pre_hook(lambda module,args:shape_log.append(dict(run=label,input=list(args[0].shape),segments=int(args[0].shape[0]//module.n_segment),segment_length=module.n_segment,clip_length=300)))
   with torch.inference_mode():y=model(x).cpu().numpy()
   hook.remove();assert y.shape==(len(ids),300);assert buffers(model)==initial;assert all(np.array_equal(x[n].cpu().numpy(),xs[k]) for n,k in enumerate(ids));del x
   return y,positions
  # Both repeated-single and repeated-homogeneous-B3 controls precede companion trials.
  alone=[];same=[]
  for repeat in range(3):alone.append(forward([i],f'alone_{repeat}')[0][0]);same.append(forward([i,i,i],f'homogeneous_{repeat}')[0])
  ref=alone[0];repeat=max(float(abs(a-ref).max()) for a in alone);repeat=max(repeat,max(float(abs(a-same[0]).max()) for a in same));scale=max(1.,float(abs(ref).max()));atol=10*repeat+32*np.finfo(np.float32).eps*scale;rtol=32*np.finfo(np.float32).eps;max_repeat=max(max_repeat,repeat)
  context=dict(checkpoint=entry,buffer_hash_before=initial,buffer_hash_after=buffers(model),BN_dropout_TSM=settings,model_training=False,device='cuda:0',dtype='float32',AMP=False,readout='unchanged production',source_code=inspect.getfile(EfficientPhys),source_sha256=sha(inspect.getfile(EfficientPhys)),padding_source_sha256=sha(inspect.getfile(use_replicate_padding)))
  contract=dict(schema_version=1,comparison_kind='batch_invariance',case_id=j['key']+'@'+str(starts[i]),common_input_sha256=arrsha(xs[i]),checkpoint_sha256=entry['sha256'],pipeline_contract_id='exact_cached_301_TCHW_float32_eval_padding_fixed',model_mode='eval_frozen')
  stages=[dict(name='model_input',path='input.npy',axis_order='TCHW',units='per_clip_global_zscore_RGB',atol=0.,rtol=0.),dict(name='d_hat',path='wave.npy',axis_order='T_minus_1',units='trained_standardized_difference',atol=float(atol),rtol=float(rtol))]
  def export(label,wave,ids,pos):
   np.save(out/(label+'_wave.npy'),wave);spec=[dict(s) for s in stages];spec[1]['path']=label+'_wave.npy';path=out/(label+'_manifest.json');save(path,dict(**contract,stages=spec,run_context=dict(**context,case_ids=[jobs[k]['key'] for k in ids],target_position=pos)))
   return path
  left=export('alone',ref,[i],0);companions=[[(i+1)%20,(i+7)%20],[(i+4)%20,(i+11)%20],[(i+9)%20,(i+17)%20]]
  for ci,others in enumerate(companions):
   for pos in range(3):
    ids=others.copy();ids.insert(pos,i);label=f'companions{ci}_position{pos}';y,positions=forward(ids,label);wave=y[pos];right=export(label,wave,ids,pos);result=compare(left,right);save(out/(label+'_comparison.json'),result);delta=float(abs(wave-ref).max());max_delta=max(max_delta,delta)
    perm=[2,0,1];pids=[ids[k] for k in perm];py,_=forward(pids,label+'_permuted');unsorted=np.empty_like(py)
    for output_index,input_index in enumerate(perm):unsorted[input_index]=py[output_index]
    assert all(jobs[pids[k]]['key']==jobs[ids[perm[k]]]['key'] for k in range(3));permutation_delta=float(abs(unsorted-y).max());max_peer=max(max_peer,permutation_delta);permuted=export(label+'_permuted',unsorted[pos],pids,pids.index(i));pr=compare(left,permuted);save(out/(label+'_permuted_comparison.json'),pr)
    runs.append(dict(companion_set=ci,position=pos,max_abs_difference=delta,permutation_max_all_outputs=permutation_delta,first_recorded_divergence=result['first_recorded_divergence'],permuted_first=pr['first_recorded_divergence']));waves.extend([wave,unsorted[pos]])
  # Same-size companion effects distinguish B1-vs-B3 arithmetic from cross-clip coupling.
  peer=[]
  for pos in range(3):
   a=[waves[2*(ci*3+pos)] for ci in range(3)];peer.append(max(float(abs(q-a[0]).max()) for q in a));max_peer=max(max_peer,max(peer))
  np.savez_compressed(out/'controls_and_trials.npz',alone=alone,homogeneous=same,trials=waves)
  record=dict(key=j['key'],fold=j['fold'],checkpoint_sha256=entry['sha256'],repeated_run_max=repeat,atol=float(atol),rtol=float(rtol),max_abs_difference=max(r['max_abs_difference'] for r in runs),max_companion_same_position=max(peer),runs=runs,settings=settings,shapes=shape_log,buffers_unchanged=buffers(model)==initial,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
  save(out/'COMPLETE.json',record);records.append(record);del model;torch.cuda.empty_cache();budget(False)
 save(C/'BATCH_COMPLETE.json',dict(utc=now(),cases=20,records=records,max_waveform_abs_difference=max_delta,max_repeated_run_difference=max_repeat,max_companion_or_permutation_difference=max_peer,comparison_tool_sha256=sha(C/'supplied/FaceVitals_BTC_Causal_Audit_V1/compare_stage_exports.py'),no_training=True,holdout_evaluations=0,interpretation='A divergence beyond numeric control requires layer localization; passing is limited to these tensors/checkpoints.'))
if __name__=='__main__':
 try:main()
 except Exception as e:save(C/'BATCH_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
