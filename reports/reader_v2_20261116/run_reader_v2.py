import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import json,sys,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent/'reader_v1_20261116'
sys.path[:0]=[str(R),str(P.parent/'round1_execution_20261110')]
import btc_posterior_reader as ref
from report_metrics import metric,belongs
def save(n,x):(P/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def fit(z,y,persons,task):
 prior,b=ref.fit(z,y,persons);w=ref.person_weights(persons);w/=w.sum()
 pred=np.array([ref.posterior_median(a,prior,b) for a in z]);base=np.array([ref.posterior_median(a,prior,0) for a in z])
 gain=float(np.sum(w*(abs(base-y)-abs(pred-y))))
 if gain<.5:b=0;pred=base
 offsets=[]
 for flag in [False,True]:
  ix=np.flatnonzero(task==flag);assert len(ix)>0
  ww=ref.person_weights([persons[i] for i in ix]);offsets.append(float(np.sum(ww*(y[ix]-pred[ix]))/ww.sum()))
 return dict(prior=prior.tolist(),beta=b,offset_rest=offsets[0],offset_task=offsets[1],train_gain=gain)
def predict(z,m,task):return ref.posterior_median(z,np.array(m['prior']),m['beta'])+m['offset_task' if task else 'offset_rest']
def contrast(rows,m,refmethod,g):
 a={r['recording']:r for r in rows if r['method']==m and belongs(r,g)};b={r['recording']:r for r in rows if r['method']==refmethod and belongs(r,g)};keys=sorted(a.keys()&b.keys());ps=sorted({a[k]['person'] for k in keys})
 d=np.array([np.mean([abs(a[k]['prediction']-a[k]['truth'])-abs(b[k]['prediction']-b[k]['truth']) for k in keys if a[k]['person']==p]) for p in ps]);ix=np.random.default_rng(0).integers(len(ps),size=(10000,len(ps)));return float(d.mean()),np.quantile(d[ix].mean(1),[.025,.975]).tolist()
def main():
 assert json.loads((P/'SCHEMA_CHECK.json').read_text())['available']
 old=json.loads((R/'PREDICTIONS.json').read_text());lookup={(r['recording'],r['method']):r for r in old if r['condition']=='native'}
 with np.load(R/'native_spectra.npz') as a:records=[lookup[str(k),'reader_v1'] for k in a['recording']];zs=[None if b else z for z,b in zip(a['Z'],a['abstain'])]
 assert len(records)==218 and len({r['person'] for r in records})==158
 y=np.array([r['truth'] for r in records]);f=np.array([r['fold'] for r in records]);people=[r['person'] for r in records];task=np.array([r['task'].lower().startswith('task') for r in records])
 def run(rep=None,kind=None):
  out=np.zeros(len(y));models=[];fixed=0
  for fold in range(5):
   tr=np.flatnonzero(f!=fold);te=np.flatnonzero(f==fold);assert not ({people[i] for i in tr}&{people[i] for i in te})
   yy=y.copy();tt=task.copy();rng=np.random.default_rng(np.random.SeedSequence([0,rep or 0,fold]))
   if kind=='label':
    groups={p:sorted([i for i in tr if people[i]==p],key=lambda i:records[i]['recording']) for p in sorted({people[i] for i in tr})};strata={}
    for p,ids in groups.items():strata.setdefault(len(ids),[]).append(p)
    for n,ps in sorted(strata.items()):
     for recipient,donor in zip(ps,rng.permutation(ps)):
      yy[groups[recipient]]=y[groups[donor]];fixed+=recipient==donor
   if kind=='task':tt[tr]=rng.permutation(task[tr]);tt[te]=rng.permutation(task[te])
   m=fit([zs[i] for i in tr],yy[tr],[people[i] for i in tr],tt[tr]);models.append(dict(fold=fold,training_people=len({people[i] for i in tr}),**m))
   for i in te:out[i]=predict(zs[i],m,tt[i])
  return out,models,fixed
 pred,models,_=run();rows=[]
 for i,r in enumerate(records):
  rows.append(dict(r,method='reader_v2',prediction=float(pred[i])))
  rows.extend(lookup[r['recording'],m] for m in ['constant','calibrated','reader_v1'])
 metrics=[]
 for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
  for q in metric([r for r in rows if belongs(r,group)],['constant','calibrated','reader_v1','reader_v2']):
   d,ci=contrast(rows,q['method'],'reader_v1',group);metrics.append(dict(q,group=group,delta_reader_v1=d,delta_reader_v1_CI=ci))
 save('FITS.json',models);save('PREDICTIONS.json',rows);save('METRICS.json',metrics)
 # Refit on all dev only for a deployable artifact/port, never for reported OOF.
 model=fit(zs,y,people,task);model.update(grid_bpm=ref.GRID.tolist(),fitted_on='218 dev only; OOF metrics use separate fold models');save('reader_model_dev218_v2.json',model)
 nulls={}
 for kind in ['label','task']:
  trials=[]
  for rep in range(20):
   pp,mm,fixed=run(rep,kind);mae=float(ref.person_table(pp,y,people)[:,0].mean());trials.append(dict(rep=rep,MAE=mae,fixed_label_person_blocks=fixed,betas=[m['beta'] for m in mm]));save('STATUS.json',dict(stage=kind,completed=rep+1))
  vals=[t['MAE'] for t in trials];nulls[kind]=dict(runs=trials,mean=float(np.mean(vals)),p5_p95=np.quantile(vals,[.05,.95]).tolist(),range=[min(vals),max(vals)])
 save('SHUFFLES.json',nulls)
 lines=['# reader_v2 — BTC native dev', 'Train-only offsets; no holdout. All numbers person-first, paired bootstrap10000 seed0. Constant and calibrated are saved identical-video baselines. No extra variants.','', '|Group|Method|Video/people|MAE|Bias|Spearman|Delta constant CI95|Delta calibrated CI95|Delta reader_v1 CI95|','|---|---|---:|---:|---:|---:|---|---|---|']
 for q in metrics:lines.append(f"|{q['group']}|{q['method']}|{q['videos']}/{q['people']}|{q['MAE']:.5f}|{q['bias']:.5f}|{q['Spearman']}|{q['delta_constant']:.5f} {q['delta_constant_CI']}|{q['delta_calibrated']:.5f} {q['delta_calibrated_CI']}|{q['delta_reader_v1']:.5f} {q['delta_reader_v1_CI']}|")
 lines+=['','## Negative controls (actual results, not enforced expectations)',json.dumps({k:{a:v for a,v in x.items() if a!='runs'} for k,x in nulls.items()},indent=2),'','Static/Octave receipts are separate; do not claim complete before they pass.']
 (P/'REPORT.md').write_text('\n'.join(lines));save('EVALUATION_COMPLETE.json',dict(passed=True,holdout_evaluations=0,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 print(json.dumps([m for m in metrics if m['group']=='all' and m['method']=='reader_v2']))
if __name__=='__main__':main()
