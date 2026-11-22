import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import sys,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent/'reader_v1_20261116';E=P.parent/'round1_execution_20261110'
sys.path[:0]=[str(R),str(E)]
import btc_posterior_reader as ref
from report_metrics import metric,belongs
def read(p):return json.loads(p.read_text())
def fit(zs,ys,people):
 prior,beta=ref.fit(zs,ys,people);w=ref.person_weights(people);w/=w.sum()
 base=float(np.sum(w*np.abs(np.array([ref.posterior_median(z,prior,0) for z in zs])-ys)));best=float(np.sum(w*np.abs(np.array([ref.posterior_median(z,prior,beta) for z in zs])-ys)))
 return prior,beta if base-best>=.5 else 0,dict(candidate_beta=beta,train_MAE_beta0=base,train_MAE_best=best,train_gain=base-best)
def evaluate(condition,records,zs):
 old=read(R/'PREDICTIONS.json');look={(r['recording'],r['method']):r for r in old if r['condition']=='native'};ys=np.array([r['truth'] for r in records]);folds=np.array([r['fold'] for r in records]);people=[r['person'] for r in records];rows=[];fits=[]
 for f in range(5):
  tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f);prior,beta,details=fit([zs[i] for i in tr],ys[tr],[people[i] for i in tr]);fits.append(dict(fold=f,beta=beta,**details))
  for i in te:
   r=records[i];base=dict(r,condition=condition);rows.append(dict(base,method='reader_gated',prediction=ref.posterior_median(zs[i],prior,beta),abstain=zs[i] is None,beta=beta))
   for m,source in [('constant','constant'),('calibrated','calibrated'),('reader_native','reader_v1')]:rows.append(dict(base,method=m,prediction=look[r['recording'],source]['prediction']))
 metrics=[]
 for g in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
  a=[r for r in rows if belongs(r,g)];metrics.extend(dict(r,condition=condition,group=g) for r in metric(a,['constant','calibrated','reader_native','reader_gated']))
  chosen=[r for r in a if r['method']=='reader_gated'];ps=sorted({r['person'] for r in chosen})
  if not ps:continue
  d=np.array([np.mean([abs(r['prediction']-r['truth'])-abs(look[r['recording'],'reader_v1']['prediction']-r['truth']) for r in chosen if r['person']==p]) for p in ps]);ix=np.random.default_rng(0).integers(len(ps),size=(10000,len(ps)))
  for q in metrics:
   if q['group']==g:q.update(delta_native=float(d.mean()),delta_native_CI=np.quantile(d[ix].mean(1),[.025,.975]).tolist())
 return dict(condition=condition,fits=fits,rows=rows,metrics=metrics)
def main():
 old=read(R/'PREDICTIONS.json');results=[]
 for c in ['native','c5','c5prime_default','c5prime_bf0']:
  records={r['recording']:r for r in old if r['condition']==c and r['method']=='reader_v1'}
  with np.load(R/(c+'_spectra.npz')) as a:rr=[records[str(k)] for k in a['recording']];zs=[None if b else z for z,b in zip(a['Z'],a['abstain'])]
  result=evaluate(c,rr,zs);results.append(result)
  if c=='native':
   assert all(r['prediction']==records[r['recording']]['prediction'] for r in result['rows'] if r['method']=='reader_gated'),'Native result changed; report, do not retune'
 (P/'GATED_RESULTS.json').write_text(json.dumps(results,indent=2))
 lines=['# Reader gated: beta train improvement >=0.5 bpm','Prior and beta are refit in each condition on the other four person folds. beta0 uses the KDE prior median, which need not equal the raw-label constant. Native calibrated comparator remains historical native inference. No other parameters changed.','', '|Condition|Group|Method|Videos/people|MAE|Bias|Spearman|Delta constant [CI95]|','|---|---|---|---:|---:|---:|---:|---|']
 for a in results:
  for r in a['metrics']:lines.append(f"|{a['condition']}|{r['group']}|{r['method']}|{r['videos']}/{r['people']}|{r['MAE']:.4f}|{r['bias']:.4f}|{r['Spearman']}|{r['delta_constant']:.4f} {r['delta_constant_CI']}|")
 (P/'GATED_REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
