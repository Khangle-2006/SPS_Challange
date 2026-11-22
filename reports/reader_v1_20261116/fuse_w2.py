import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,time,json,fcntl,hashlib,traceback
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;W=P.parent/'w2_btc_20261116';E=P.parent/'round1_execution_20261110'
sys.path[:0]=[str(W),str(E)]
import train_w2 as w2
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import reader_v1 as api
from report_metrics import metric,belongs
ref=api.ref
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
 lock=(P/'fusion.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while not (W/'TRAINING_TERMINAL.json').exists():
  if (W/'TRAIN_ERROR.json').exists():save(P/'FUSION_BLOCKED.json',dict(reason='W2 training error; no restart'));return
  time.sleep(45)
 if not all((W/'runs'/f'fold{f}/COMPLETE.json').exists() for f in range(5)):
  save(P/'FUSION_BLOCKED.json',dict(reason='Not all five W2 folds valid and complete'));return
 while not (P/'EVALUATION_COMPLETE.json').exists():
  if (P/'ERROR.json').exists():return
  time.sleep(30)
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 plan=read(E/'PLAN_ELIGIBLE.json');rows=[];zs=[];specs=[]
 for fold in range(5):
  data=w2.RateBTC(fold);run=next(r for r in plan['runs'] if r['fold']==fold);model=w2.make_model(run,'cpu');ck=W/'runs'/f'fold{fold}/final.pt';assert w2.sha(ck)==read(ck.parent/'COMPLETE.json')['checkpoint_sha256'];model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model']);model.eval()
  for v in data.rows:
   if v['fold']!=fold:continue
   rgb,t,valid=data.load(v,'native');powers=[];starts=[]
   for start in np.arange(0,max(0.,t[-1]-10+1e-8),5):
    if not ref.supported(t,valid,start,start+10):continue
    x=data.clip(v,float(start))
    with torch.inference_mode():diff=model(torch.from_numpy(x[None]))[0].numpy()
    wave=np.cumsum(diff.astype(float));assert len(wave)==300
    p=ref.window_spectrum(wave)
    if np.isfinite(p).all() and p.sum()>0:powers.append(p);starts.append(float(start))
   avg=np.mean(powers,axis=0) if powers else np.zeros(len(ref.GRID));log=np.log(avg+1e-30);z=(log-log.mean())/log.std() if powers and log.std()>0 else None
   rows.append(v);zs.append(z);specs.append(avg)
   save(P/'FUSION_STATUS.json',dict(videos=len(rows),fold=fold))
 assert len(rows)==218 and len({r['person'] for r in rows})==158
 np.savez_compressed(P/'W2_SPECTRA.npz',grid_bpm=ref.GRID,recording=np.array([r['recording'] for r in rows]),spectrum=np.array(specs),abstain=np.array([z is None for z in zs]))
 with np.load(P/'native_spectra.npz') as a:two={str(r):(None if ab else z) for r,z,ab in zip(a['recording'],a['Z'],a['abstain'])}
 twoz=[two[r['recording']] for r in rows];three=[(2*a+b)/3 if a is not None and b is not None else None for a,b in zip(twoz,zs)]
 truth=np.array([r['pr_bpm'] for r in rows]);folds=np.array([r['fold'] for r in rows]);people=[r['person'] for r in rows];out=[];fits=[]
 for name,values in [('reader_2spectra',twoz),('reader_3spectra',three)]:
  for f in range(5):
   tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f);prior,beta=ref.fit([values[i] for i in tr],truth[tr],[people[i] for i in tr]);fits.append(dict(method=name,fold=f,beta=beta))
   for i in te:out.append(dict(recording=rows[i]['recording'],person=people[i],fold=f,group=rows[i]['group'],task=rows[i]['task'],truth=float(truth[i]),method=name,prediction=ref.posterior_median(values[i],prior,beta),abstain=values[i] is None))
 baseline=[r for r in read(P/'PREDICTIONS.json') if r['condition']=='native' and r['method'] in ['constant','calibrated']];out+=baseline;metrics=[]
 for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
  rr=[r for r in out if belongs(r,group)];metrics+= [dict(q,group=group) for q in metric(rr,['constant','calibrated','reader_2spectra','reader_3spectra'])]
  a={r['recording']:r for r in rr if r['method']=='reader_2spectra'};b={r['recording']:r for r in rr if r['method']=='reader_3spectra'};ps=sorted({r['person'] for r in a.values()})
  d=np.array([np.mean([abs(b[k]['prediction']-r['truth'])-abs(r['prediction']-r['truth']) for k,r in a.items() if r['person']==p]) for p in ps]);ix=np.random.default_rng(0).integers(len(ps),size=(10000,len(ps)))
  metrics.append(dict(group=group,comparison='3minus2',delta=float(d.mean()),CI95=np.quantile(d[ix].mean(1),[.025,.975]).tolist()))
 save(P/'FUSION_RESULTS.json',dict(rows=metrics,fits=fits,holdout_evaluations=0,limitation='exploratory cross-fitting of OOF CNN spectra; not fully nested training of the whole fusion pipeline'));save(P/'FUSION_PREDICTIONS.json',out)
 lines=['# Reader: two vs three spectra','218 dev videos/158 people; holdout closed. Person-first paired bootstrap10000 seed0. Exploratory OOF-feature cross-fitting, not a fully nested evaluation of the entire learned pipeline.','', '|Group|Method|MAE|Bias|Spearman|Delta calibrated [CI95]|','|---|---|---:|---:|---:|---|']
 for r in metrics:
  if 'method' in r:lines.append(f"|{r['group']}|{r['method']}|{r['MAE']:.4f}|{r['bias']:.4f}|{r['Spearman']}|{r['delta_calibrated']:.4f} {r['delta_calibrated_CI']}|")
 lines+=['','## Paired three minus two spectra']
 for r in metrics:
  if 'comparison' in r:lines.append(f"{r['group']}: {r['delta']:+.4f} bpm; CI95 {r['CI95']}")
 (P/'FUSION_REPORT.md').write_text('\n'.join(lines));save(P/'FUSION_COMPLETE.json',dict(videos=218,people=158))
if __name__=='__main__':
 try:main()
 except Exception:save(P/'FUSION_ERROR.json',dict(traceback=traceback.format_exc()));raise
