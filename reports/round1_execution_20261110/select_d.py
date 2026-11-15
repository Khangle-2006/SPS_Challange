"""One declared dev-only family selection after all five C attempts finish."""
import numpy as np
from common import *
assert (P/'SWEEP_STATUS.json').exists() and read(P/'SWEEP_STATUS.json')['state']=='complete'
assert not (P/'D_SELECTION.json').exists(),'Selection already frozen'
plan=read(P/'PLAN.json');families={};files={};status={}
for family,base,name in [('initial_padding_fixed',P/'sweep','BEFORE_NATIVE.json'),('AdaBN',P/'adabn','NATIVE.json'),('five_prespecified_fold_recipes',P/'sweep','NATIVE.json')]:
 rr=[];ff=[];valid=True
 for fold in range(5):
  d=base/f'fold{fold}';f=d/name
  if not f.exists():valid=False;break
  if family=='five_prespecified_fold_recipes':
   st=read(d/'status.json')
   if st['state']!='complete' or not st['static_pass']:valid=False;break
  if family=='AdaBN' and not (d/'COMPLETE.json').exists():valid=False;break
  rr+=read(f);ff.append(str(f))
 if valid:families[family]=rr;files[family]=ff
 assert family!='initial_padding_fixed' or valid,'All five initial comparisons required'
def videos(rows):
 out={}
 for rid in {r['recording'] for r in rows}:
  a=[r for r in rows if r['recording']==rid and r['prediction'] is not None]
  if a:out[rid]=dict(a[0],prediction=float(np.median([r['prediction'] for r in a])))
 return out
vv={f:videos(r) for f,r in families.items()};keys=sorted(set.intersection(*(set(x) for x in vv.values())));assert keys
scores={}
for f,rows in vv.items():
 people=sorted({rows[k]['person'] for k in keys});scores[f]=float(np.mean([np.mean([abs(rows[k]['prediction']-rows[k]['truth']) for k in keys if rows[k]['person']==person]) for person in people]))
chosen=min(scores,key=lambda f:(scores[f],f));models=[]
for run in plan['runs']:
 fold=run['fold']
 path=Path(run['checkpoint']) if chosen=='initial_padding_fixed' else P/('adabn' if chosen=='AdaBN' else 'sweep')/f'fold{fold}'/'final.pt'
 models.append(dict(fold=fold,path=str(path),sha256=sha(path),initial_run=run))
save(P/'D_SELECTION.json',dict(utc=now(),family=chosen,scores=scores,common_videos=len(keys),models=models,input_hashes={f:sha(f) for ff in files.values() for f in ff},scope='exploratory dev selection; mixed five-fold recipes are not a causal recipe comparison or an independently validated deployment ensemble',holdout_evaluations=0))
print(chosen,scores)
