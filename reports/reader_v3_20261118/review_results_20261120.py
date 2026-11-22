"""Independent local arithmetic review of completed frozen-model evaluations."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.stats import spearmanr
P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def belongs(r,g):
 if g=='all':return True
 if g=='Task':return r['task'].lower().startswith('task')
 if g=='Walking':return r['task']=='Walking'
 return r['task']=='Resting' and r['group']==g.split('-')[0]
def close(a,b):assert np.allclose(a,b,atol=1e-10,rtol=1e-10),(a,b)
def main():
 c=P/'conditions_20261120';u=P/'public_20261120';s=P/'static_20261120'
 assert all(read(d/'COMPLETE.json')['passed'] for d in [c,u,s])
 rows=read(c/'PREDICTIONS.json');table=read(c/'METRICS.json');checked=0
 for cond in sorted({r['condition'] for r in rows}):
  for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
   sub=[r for r in rows if r['condition']==cond and belongs(r,group)]
   people=sorted({r['person'] for r in sub});methods=sorted({r['method'] for r in sub})
   ids={m:{r['recording'] for r in sub if r['method']==m} for m in methods}
   assert all(v==ids[methods[0]] for v in ids.values())
   ee={};bb={};rho={}
   for m in methods:
    rr=[r for r in sub if r['method']==m];assert len(rr)==len(ids[m])
    per=[[r for r in rr if r['person']==x] for x in people]
    ee[m]=np.array([np.mean([abs(r['prediction']-r['truth']) for r in v]) for v in per])
    bb[m]=np.array([np.mean([r['prediction']-r['truth'] for r in v]) for v in per])
    yp=[np.mean([r['prediction'] for r in v]) for v in per];yt=[np.mean([r['truth'] for r in v]) for v in per]
    rho[m]=float(spearmanr(yp,yt).statistic) if np.std(yp)>0 and np.std(yt)>0 else None
   ix=np.random.default_rng(0).integers(len(people),size=(10000,len(people)))
   ci=lambda x:np.quantile(x[ix].mean(1),[.025,.975])
   for r in [x for x in table if x['condition']==cond and x['group']==group]:
    m=r['method'];close(r['MAE'],ee[m].mean());close(r['CI95'],ci(ee[m]));close(r['bias'],bb[m].mean())
    if rho[m] is not None:close(r['Spearman'],rho[m])
    for key,ref in [('constant','constant'),('calibrated','calibrated'),('native','reader_native')]:
     delta=ee[m]-ee[ref];close(r['delta_'+key],delta.mean());close(r['delta_'+key+'_CI'],ci(delta))
    checked+=1
 proof=read(c/'SOURCE_CLASS_BINDING.json');seen={}
 for r in proof:
  assert r['source_class']==('low' if r['source_width']*r['source_height']<1800000 else 'high')
  key=(r['source_width'],r['source_height'],r['source_class']);assert r['recording'] not in seen or seen[r['recording']]==key
  seen[r['recording']]=key
 static=read(s/'RESULTS.json');assert len(static['cases'])==25 and all(r['pass_static'] for r in static['cases'])
 # Confirm public metric arithmetic using saved predictions; no fitting or inference.
 with np.load(u/'PREDICTIONS.npz',allow_pickle=False) as z:
  with np.load(P/'source/b04_public_predictions.npz',allow_pickle=False) as original:
   for key in ['fusion','tracked','truth','dataset','person','recording','start']:
    assert np.array_equal(z[key],original[key]),('Supplied b04 saved output mismatch',key)
   assert np.array_equal(z['condition'],original['cond'])
  public=[]
  for r in read(u/'METRICS.json'):
   if r['method']=='padding':continue
   mask=(z['condition']==r['condition'])&(z['dataset']==r['dataset']);per=z['person'][mask]
   pr=z['fusion' if r['method']=='fusion' else 'tracked'][mask];gt=z['truth'][mask]
   err=np.array([np.mean(abs(pr[per==p]-gt[per==p])) for p in sorted(set(per))]);close(err.mean(),r['MAE'])
   ix=np.random.default_rng(0).integers(len(err),size=(10000,len(err)));close(np.quantile(err[ix].mean(1),[.025,.975]),r['CI']);public.append(r)
 files=[d/f for d in [c,u,s] for f in ['COMPLETE.json','METRICS.json','RESULTS.json','PREDICTIONS.json','PREDICTIONS.npz','FITS.json'] if (d/f).exists()]
 result=dict(passed=True,condition_metric_rows=checked,public_metric_rows=len(public),public_saved_source_predictions_exact_match=True,source_class_records=len(seen),static_cases=25,static_max_difference_std=max(r['wave_std'] for r in static['cases']),static_nonfinite_chrom=sum(not r['chrom_finite'] for r in static['cases']),hashes={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},holdout_evaluations=0)
 (P/'REVIEW_20261120.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='hashes'}))
if __name__=='__main__':main()
