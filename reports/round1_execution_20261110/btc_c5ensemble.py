"""Frozen 15-checkpoint V3/V4 comparators on new BTC C5-prime, CPU only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,traceback
import numpy as np
from common import *
from evaluate_c5 import loadcrop
from infer_dev import image_clip,decode_model,model_inventory
from EfficientPhys_fixed import EfficientPhys
def main(shard,count):
 import torch
 torch.set_num_threads(1);lease=lock(f'btc_ensemble_{shard}.lock');guard(True)
 data=BTCData(0);rows=sorted(data.rows,key=lambda v:v['recording']);assigned=[v for i,v in enumerate(rows) if i%count==shard]
 inventory=[e for e in model_inventory() if e['branch'] in ['v3_joint','v4_repaired']];assert len(inventory)==30
 models={}
 for e in inventory:
  assert sha(e['path'])==e['sha256'];m=EfficientPhys(frame_depth=10,img_size=36);m.load_state_dict(torch.load(e['path'],map_location='cpu',weights_only=False)['model']);models[e['name']]=m.eval()
 while True:
  progressed=False;done=0
  for v in assigned:
   for variant in ['default','bf0']:
    d=P/'c5prime/BTC'/v['recording']/variant;out=P/'btc_c5ensemble'/v['recording']/(variant+'.json')
    if out.exists():done+=1;continue
    if not (d/'COMPLETE.json').exists():continue
    a=read(d/'COMPLETE.json');pred=[]
    if a['state']=='complete':
     rgb,t,valid,mean=loadcrop(d)
     for start in sorted({w['start'] for w in data.candidates(v,'native') if w['scheme']=='10s'}):
      reason=None;x=None
      try:
       if not a['bitrate_pass']:raise ValueError('bitrate_above_500_no_retuning')
       x=image_clip(rgb,t,valid,start,10)
      except ValueError as e:reason=str(e)
      for e in inventory:
       value=None;why=reason
       if x is not None:
        try:
         with torch.inference_mode():wave=models[e['name']](torch.from_numpy(x[None]))[0].numpy()
         value=decode_model(wave)
        except ValueError as error:why=str(error)
       pred.append(dict(recording=v['recording'],person=v['person'],fold=v['fold'],condition='c5prime_'+variant,method=e['name'],branch=e['branch'],start=start,truth=v['pr_bpm'],prediction=value,reason=why))
    save(out,dict(rows=pred,state=a['state'],source_receipt_sha256=sha(d/'COMPLETE.json'),models=inventory,code_sha256=sha(__file__),holdout_evaluations=0));done+=1;progressed=True
    save(P/f'btc_c5ensemble/SHARD{shard}_STATUS.json',dict(completed=done,total=len(assigned)*2,last=v['recording'],utc=now(),pid=os.getpid()))
  if done==len(assigned)*2:save(P/f'btc_c5ensemble/SHARD{shard}_COMPLETE.json',dict(utc=now(),files=done));break
  if not progressed:time.sleep(20)
if __name__=='__main__':
 shard,count=map(int,sys.argv[1:])
 try:main(shard,count)
 except Exception as e:save(P/f'btc_c5ensemble/SHARD{shard}_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
