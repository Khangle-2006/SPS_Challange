"""Distinct production callers, identical cached decoded frames/PTS, fixed state."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
from common_causal import *
from common_repair import O,manifest
import numpy as np,torch,traceback,inspect
from crop_io import cached
from t8_t1_loader import clip_arrays
from polarity_loader import load_canonical_clip
sys.path.insert(0,str(O));from infer_dev import image_clip,decode_model
from infer_original import clip as repair_clip
os.environ['CUDA_VISIBLE_DEVICES']=''
from EfficientPhys_fixed import EfficientPhys
from replicate_tsm import use_replicate_padding
from t8_evaluate import decode_difference
sys.path.insert(0,str(C/'supplied/FaceVitals_BTC_Causal_Audit_V1'));from compare_stage_exports import compare

def main():
 lease=lock('cross_route.lock');assert not (C/'ROUTE_STARTED.json').exists();check_deadline();budget(True);torch.set_num_threads(2);torch.set_num_interop_threads(1)
 save(C/'ROUTE_STARTED.json',dict(utc=now(),pid=os.getpid(),code_sha256=sha(__file__)))
 selected=read(C/'ladder_preflight/SELECTION.json');cases=[next(c for c in selected if c['job']['dataset']==d) for d in ['UBFC2','MCD']];btc=next(j for j in manifest() if j['dataset']=='BTC');cases.append(dict(job=btc,window=btc['windows'][0]));outcomes=[]
 for case in cases:
  j=case['job'];w=case['window'];rgb,meta,d=cached(j,'native');a=read(P/'FROZEN.json')['padding_checkpoints'][str(j['fold'])];assert sha(a['path'])==a['sha256'];model=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36));model.load_state_dict(torch.load(a['path'],map_location='cpu',weights_only=True)['model'],strict=True);model.eval()
  if j['dataset']!='BTC':left,dy,ref,norm=load_canonical_clip(d,w['start'],j['dataset']);left_name='public_load_canonical_clip'
  else:left,_=repair_clip(rgb,meta['t'],meta['valid'],w['start']);left_name='repair_clip'
  right=image_clip(rgb,meta['t'],meta['valid'],w['start'],10)
  root=C/'routes'/j['key'];root.mkdir(parents=True,exist_ok=False);routes=[]
  for name,x in [(left_name,left),('BTC_image_clip',right)]:
   out=root/name;out.mkdir();np.save(out/'frames.npy',rgb);np.save(out/'pts.npy',meta['t']);np.save(out/'tensor.npy',x)
   with torch.inference_mode():wave=model(torch.from_numpy(x[None]))[0].numpy()
   np.save(out/'wave.npy',wave);hr=decode_difference(wave) if name==left_name else decode_model(wave);np.save(out/'hr.npy',np.array([hr]))
   specs=[dict(name=k,path=k+'.npy',axis_order=axis,units=u,atol=tol,rtol=0.) for k,axis,u,tol in [('frames','THWC','RGB8',0.),('pts','T','seconds',0.),('tensor','TCHW','zscore',0.),('wave','T','difference',2e-6),('hr','scalar','bpm',1e-8)]]
   m=dict(schema_version=1,comparison_kind='route_equivalence',case_id=j['key'],common_input_sha256=arrsha(rgb),checkpoint_sha256=a['sha256'],pipeline_contract_id='same_cached_decoded_oval_frames_and_source_PTS_10s',model_mode='eval_frozen',stages=specs,caller=name)
   save(out/'manifest.json',m);routes.append(out/'manifest.json')
  result=compare(*routes);save(root/'COMPARISON.json',result);outcomes.append(dict(key=j['key'],**result))
  if j['dataset']!='BTC':
   np.savez_compressed(root/'TARGET.npz',normalized_difference=dy,reference_300=ref)
   save(root/'TARGET.json',dict(normalization=norm,reference_clock='independent contact PPG; unchanged source support',target_hr=decode_difference(dy),truth=w['truth'],N=301,N_minus_1=300))
 # Reuse existing real V5 offset/sign/target validation; never invoke training/backward again.
 V5=E/'v5_multiview_20261112';receipts=[]
 for f in sorted(V5.glob('VALIDATION_FOLD*.json')):
  v=read(f);receipts.append(dict(path=str(f),sha256=sha(f),passed=v.get('passed'),tests=v.get('tests')))
 save(C/'V5_TARGET_EVIDENCE.json',dict(utc=now(),receipts=receipts,source_sha256=sha(V5/'validate_v2.py'),scope='Actual V5 independent PPG timestamps + frozen per-session offsets and sign, as already verified before training; no offset0 visualization substituted; no validation/backward rerun.'))
 save(C/'ROUTE_COMPLETE.json',dict(utc=now(),cases=outcomes,scope='Downstream cached decoded oval frames and source PTS only; separate ladder native-route comparison covers upstream decode/ROI.',functions={str(Path(inspect.getfile(fn))):sha(inspect.getfile(fn)) for fn in [image_clip,repair_clip,load_canonical_clip,decode_model,decode_difference]},no_training=True,holdout_evaluations=0))
if __name__=='__main__':
 try:main()
 except Exception as e:save(C/'ROUTE_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
