"""Compare recorded upstream arrays, including rejected support. No inference.

Explicitly descriptive: public legacy crops and BTC raw-video route do not have
an asserted identical ROI contract. No temporal shift/PR-based alignment is fit.
"""
from common_causal import *
from crop_io import cached
import numpy as np
from compare_interventions import compare

def manifest_for(folder, case, filename='FRONTEND.npz', label='ladder'):
 j=case['job']; archive=folder/filename
 with np.load(archive,allow_pickle=False) as a:
  np.save(folder/(label+'_mean_RGB.npy'),a['rgb'].astype(np.float64).mean((1,2)))
 mask=dict(path=filename,key='valid')
 specs=[dict(name='display_PTS',path=filename,key='t',axis_order='T',units='seconds',atol=1e-6,rtol=0.),
        dict(name='face_valid',path=filename,key='valid',axis_order='T',units='boolean',atol=0.,rtol=0.),
        dict(name='oval_RGB8',path=filename,key='rgb',axis_order='THWC',units='RGB8',atol=0.,rtol=0.,valid_mask=mask),
        dict(name='RGB_before_normalization',path=label+'_mean_RGB.npy',axis_order='TC',units='RGB8',atol=0.,rtol=0.,valid_mask=mask)]
 ck=read(P/'FROZEN.json')['padding_checkpoints'][str(j['fold'])]
 m=dict(schema_version=1,comparison_kind='controlled_degradation',case_id=j['key'],common_input_sha256=j['source_sha256'],checkpoint_sha256=ck['sha256'],pipeline_contract_id='descriptive_same_source_upstream_arrays_no_equivalence_assertion',model_mode='eval_frozen',stages=specs,route_label=label)
 dest=folder/(label+'_FRONTEND_MANIFEST.json');save(dest,m);return dest

def safe_compare(left,right,dest,extra):
 try:result=compare(left,right)
 except Exception as e:
  # Preserve a rejected/invalid comparison, never replace it with a passing one.
  result=dict(status='INVALID_COMPARISON',error=str(e),first_recorded_divergence=None)
 save(dest,dict(**result,**extra))

def run():
 try:lease=lock('frontend_comparison.lock')
 except BlockingIOError:return
 if not (C/'LADDER_SELECTION.json').exists():return
 for case in read(C/'LADDER_SELECTION.json'):
  j=case['job'];root=C/'physical_v3/ladder'/j['key'];native=root/'s0_native'
  if not (native/'COMPLETE.json').exists():continue
  nm=native/'ladder_FRONTEND_MANIFEST.json'
  if not nm.exists():nm=manifest_for(native,case)
  out=C/'upstream_routes_physical'/j['key'];done=out/'COMPLETE.json'
  if not done.exists():
   budget(True);out.mkdir(parents=True,exist_ok=True)
   rgb,m,crop=cached(j,'native')
   with np.load(native/'FRONTEND.npz',allow_pickle=False) as a:n=len(a['t'])
   # Same prefix by frame index; any PTS/length discrepancy remains observable.
   np.savez_compressed(out/'LEGACY.npz',t=m['t'][:n],valid=np.asarray(m['valid'][:n],bool),rgb=rgb[:n])
   lm=manifest_for(out,case,'LEGACY.npz','legacy_public_cache')
   safe_compare(lm,nm,out/'COMPARISON.json',dict(scope='Legacy public cache versus actual BTC raw-video frontend, same source prefix by index; no fitted timestamp alignment, no inference rerun.',cache_path=str(crop),source_sha256=j['source_sha256'],legacy_frames_exported=min(n,len(rgb)),raw_route_frames=n))
   save(done,dict(utc=now(),comparison_sha256=sha(out/'COMPARISON.json'),legacy_export_sha256=sha(out/'LEGACY.npz'),native_frontend_sha256=sha(native/'FRONTEND.npz')))
  for stage in sorted(root.iterdir()):
   if not stage.is_dir() or stage==native or not (stage/'COMPLETE.json').exists():continue
   dest=stage/'FRONTEND_COMPARE_TO_S0.json'
   if dest.exists():continue
   sm=manifest_for(stage,case)
   safe_compare(nm,sm,dest,dict(stage_status=read(stage/'COMPLETE.json')['status'],scope='Pre-model arrays and explicit valid support. Invalid windows were not fed to the model; no fabricated tensor/waveform.',expected_transform_differences=True))
 receipts=list((C/'upstream_routes_physical').glob('*/COMPLETE.json'))
 save(C/'FRONTEND_COMPARISON_STATUS.json',dict(utc=now(),native_routes=len(receipts),total=10,stage_comparisons=len(list((C/'physical_v3/ladder').glob('*/*/FRONTEND_COMPARE_TO_S0.json'))),no_inference=True))
 if len(receipts)==10 and (C/'physical_v3/LADDER_COMPLETE.json').exists():
  save(C/'FRONTEND_COMPARISON_COMPLETE.json',dict(utc=now(),native_routes=10,stage_comparisons=len(list((C/'physical_v3/ladder').glob('*/*/FRONTEND_COMPARE_TO_S0.json'))),no_inference=True))

if __name__=='__main__':run()
