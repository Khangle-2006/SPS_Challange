"""Read completed BTC V4 inference; no new inference or checkpoint selection."""
from common import *
from evaluate import old_method
assert read(OLD/'INFERENCE_STATUS.json')['state']=='complete'
entries=[x for x in read(OLD/'CHECKPOINTS.json') if x['branch']=='v4_repaired']
assert len(entries)==15
out=[];missing=[]
for v in dev_manifest():
 for cond in ['native','c5']:
  base=OLD/'predictions'/v['drive_id']/cond
  for e in entries:
   p=base/(e['name']+'.json')
   if not p.exists():continue
   a=read(p);assert a['checkpoint']['sha256']==e['sha256'] and a['split_sha256']==SPLIT
   assert a['cache_sha256']==sha(OLD/'dev'/v['drive_id']/cond/'COMPLETE.json')
  value=old_method(v,cond,'v4_repaired')
  if value is None:missing.append(dict(recording=v['recording'],condition=cond));continue
  out.append(dict(method='V4_original_padding_median15',condition=cond,recording=v['recording'],person=v['person'],fold=v['fold'],truth=v['pr_bpm'],prediction=value))
csvout(OUT/'BTC_V4_SUPPLEMENT.csv',out)
save(OUT/'BTC_V4_SUPPLEMENT_RECEIPT.json',dict(checkpoints=entries,checkpoint_count=15,rows=len(out),missing=missing,holdout_evaluations=0,static_gate='FAIL original-padding V4, no promotion',utc=now()))
