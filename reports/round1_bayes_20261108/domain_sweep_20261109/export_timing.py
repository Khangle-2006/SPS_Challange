"""Read existing predictions for fixed 218-video native cohort; no inference/holdout."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;B=P.parent;OLD=B.parent/'round1_v4_20261105/btc_zeroshot'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OLD/'btc_split.json')=='97944f717b7fc166bb8a0e91c791c7ad6b7e451cffed564f40232dde3ec8e221'
rows=list(csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open()));names=['constant','POS_old_argmax','CHROM_old','V3_median15','UBFC_median15']
keys=set.intersection(*({r['recording'] for r in rows if r['condition']=='native' and r['method']==n} for n in names));assert len(keys)==218
manifest={v['recording']:v for v in read(OLD/'dev_manifest.json')['videos']};out=[];inputs={}
for rec in sorted(keys):
 v=manifest[rec];folder=OLD/'predictions'/v['drive_id']/'native';cf=folder/'classical.json';a=read(cf);inputs[str(cf)]=sha(cf)
 starts=sorted({r['start'] for r in a['rows'] if r['scheme']=='10s'})
 for method,branch in [('POS',None),('CHROM',None),('UBFC','ubfc_only'),('V3','v3_joint'),('V4_old_padding','v4_repaired')]:
  if branch is None:lookup={r['start']:r['prediction'] for r in a['rows'] if r['scheme']=='10s' and r['method']==method}
  else:
   files=sorted(folder.glob(branch+'_f*_s*.json'));assert len(files)==15,(rec,branch,len(files));values={s:[] for s in starts}
   for f in files:
    data=read(f);inputs[str(f)]=sha(f)
    assert data['split_sha256']==sha(OLD/'btc_split.json')
    for r in data['rows']:
     if r['scheme']=='10s':values[r['start']].append(r['prediction'])
   lookup={s:float(np.median(x)) if len(x)==15 and all(y is not None for y in x) else None for s,x in values.items()}
  for position,ix in [('first',0),('middle',len(starts)//2),('last',len(starts)-1)]:
   s=starts[ix] if starts else None
   out.append(dict(recording=rec,person=v['person'],fold=v['fold'],group=v['group'],task=v['task'],truth=v['pr_bpm'],method=method,position=position,start=s,prediction=lookup.get(s) if len(starts)>=3 else None,reason='fewer_than_three_nonoverlap_candidates' if len(starts)<3 else 'missing_prediction' if lookup.get(s) is None else None))
 with_constant=next(r for r in rows if r['recording']==rec and r['condition']=='native' and r['method']=='constant')
 for position,ix in [('first',0),('middle',len(starts)//2),('last',len(starts)-1)]:
  out.append(dict(recording=rec,person=v['person'],fold=v['fold'],group=v['group'],task=v['task'],truth=v['pr_bpm'],method='constant',position=position,start=starts[ix] if starts else None,prediction=float(with_constant['prediction']) if len(starts)>=3 else None,reason='fewer_than_three_nonoverlap_candidates' if len(starts)<3 else None))
assert not (P/'TIMING_WINDOWS.json').exists(),'Preserve prior timing export'
(P/'TIMING_WINDOWS.json').write_text(json.dumps(dict(rows=out,cohort=sorted(keys),input_hashes=inputs,holdout_evaluations=0),allow_nan=False))
print('Exported',len(out),'rows for',len(keys),'dev videos; no inference.')
