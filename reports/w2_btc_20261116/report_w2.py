"""Read-only evaluation of completed W2 runs; no scoring-based queue control."""
import sys,json,csv,time,fcntl,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;E=P.parent/'round1_execution_20261110';B=P.parent/'round1_bayes_20261108';V=P.parent/'round1_v4_20261105/btc_zeroshot'
sys.path.insert(0,str(E))
from report_metrics import metric,belongs
def read(p):return json.loads(p.read_text())
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def report():
 dev=read(V/'dev_manifest.json')['videos'];meta={r['recording']:r for r in dev}
 prior=list(csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open()))
 cal={r['recording']:float(r['prediction']) for r in prior if r['condition']=='native' and r['method']=='calibrated_median_POS_CHROM_V3'}
 constants={r['recording']:float(r['prediction']) for r in prior if r['condition']=='native' and r['method']=='constant'}
 classical=list(csv.DictReader((V/'VIDEO_PREDICTIONS.csv').open()))
 cp={(r['method'],r['recording']):float(r['prediction']) for r in classical if r['condition']=='native' and r['scheme']=='10s' and r['method'] in ['POS','CHROM'] and r['prediction']}
 sets={};completed=[];rejected=[]
 for f in range(5):
  d=P/'runs'/f'fold{f}'
  if (d/'COMPLETE.json').exists():
   assert read(d/'COMPLETE.json')['static_pass'];sets[f'fold{f}']=read(d/'NATIVE.json');completed.append(f)
  if (d/'REJECTED_STATIC.json').exists():rejected.append(f)
 if (P/'runs/fold2_shuffle/COMPLETE.json').exists():sets['fold2_shuffle']=read(P/'runs/fold2_shuffle/NATIVE.json')
 full=len(completed)==5
 if completed:sets['pooled_full' if full else 'pooled_partial']=[r for f in completed for r in sets[f'fold{f}']]
 results=[];predictions=[];fits=[]
 for target,rr in sets.items():
  methods=['constant','calibrated','W2'];rows=[]
  for r in rr:
   assert r['person']==meta[r['recording']]['person'] and r['truth']==meta[r['recording']]['pr_bpm']
   constant=constants[r['recording']]
   rows.extend([dict(r,method='W2'),dict(r,method='constant',prediction=constant),dict(r,method='calibrated',prediction=cal[r['recording']])])
  if target=='fold2_shuffle' and 'fold2' in sets:
   rows.extend(dict(r,method='W2_true') for r in sets['fold2']);methods.append('W2_true')
  if target=='pooled_full':
   assert len(rr)==218 and len({r['person'] for r in rr})==158
   if all(r['prediction'] is not None for r in rr):
    for kind in ['W2_affine','median_POS_CHROM_W2_affine']:
     def feature(r):return float(r['prediction']) if kind=='W2_affine' else float(np.median([cp['POS',r['recording']],cp['CHROM',r['recording']],r['prediction']]))
     for fold in range(5):
      train=[r for r in rr if r['fold']!=fold];test=[r for r in rr if r['fold']==fold]
      assert not ({r['person'] for r in train}&{r['person'] for r in test})
      counts={p:sum(r['person']==p for r in train) for p in {r['person'] for r in train}}
      w=np.sqrt([1/counts[r['person']] for r in train]);x=np.array([[feature(r),1.] for r in train]);y=np.array([r['truth'] for r in train]);coef=np.linalg.lstsq(x*w[:,None],y*w,rcond=None)[0]
      fits.append(dict(method=kind,fold=fold,a=float(coef[0]),b=float(coef[1]),train_people=len(counts)))
      rows.extend(dict(r,method=kind,prediction=float(coef[0]*feature(r)+coef[1])) for r in test)
     methods.append(kind)
  for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
   selected=[r for r in rows if belongs(r,group)]
   for q in metric(selected,methods):
    pp=[r['prediction'] for r in selected if r['method']==q['method'] and r['prediction'] is not None]
    q.update(target=target,group=group,prediction_SD=float(np.std(pp)) if pp else None,requested_videos=sum(belongs(r,group) for r in rr))
    if target=='fold2_shuffle' and 'W2_true' in methods:
     # CI of true minus shuffled, whole-person paired resampling.
     look={r['recording']:r for r in sets['fold2'] if r['prediction'] is not None};pairs=[r for r in rr if r['prediction'] is not None and r['recording'] in look and belongs(r,group)]
     persons=sorted({r['person'] for r in pairs})
     if persons:
      delta=np.array([np.mean([abs(look[r['recording']]['prediction']-r['truth'])-abs(r['prediction']-r['truth']) for r in pairs if r['person']==p]) for p in persons]);ix=np.random.default_rng(0).integers(len(persons),size=(10000,len(persons)));q['true_minus_shuffle']=float(delta.mean());q['true_minus_shuffle_CI']=np.quantile(delta[ix].mean(1),[.025,.975]).tolist()
    results.append(q)
  predictions.extend(dict(r,target=target) for r in rows)
 gates=[]
 if full:
  for r in results:
   if r['target']=='pooled_full' and r['group']=='all' and r['method'] not in ['constant','calibrated']:
    gates.append(dict(method=r['method'],passed=r['videos']==218 and r['MAE']<=10 and r['delta_calibrated']<=-1 and r['delta_calibrated_CI'][1]<0))
 save(P/'RESULTS.json',dict(complete_folds=completed,rejected_static=rejected,full=full,rows=results,gates=gates,holdout_evaluations=0))
 save(P/'PREDICTIONS.json',predictions);save(P/'CALIBRATION_FITS.json',fits)
 lines=['# W2: measured BTC dev results','',f'Completed folds: {completed}; static-rejected folds: {rejected}. Full OOF: {full}.','No full-cohort claim until all five valid folds and 218 accepted videos. Person-first metrics; paired bootstrap 10,000, seed 0.','Calibration is exploratory cross-fitting over OOF base predictions, not a fully nested estimate of the entire model-selection procedure.','', '| Scope | Group | Method | Videos/people | MAE | Bias | Spearman | Video SD | Delta calibrated [95% CI] |','|---|---|---|---:|---:|---:|---:|---:|---|']
 for r in results:lines.append(f"|{r['target']}|{r['group']}|{r['method']}|{r['videos']}/{r['people']}|{r['MAE']:.4f}|{r['bias']:.4f}|{r['Spearman']}|{r['prediction_SD']:.4f}|{r['delta_calibrated']:.4f} {r['delta_calibrated_CI']}|")
 lines+=['','## Training diagnostics','Fixed 40 training videos are diagnostic, not used for checkpoint selection. Shuffled-run train MAE is evaluated against original PR; weak CE uses shuffled training labels.']
 for d in sorted((P/'runs').glob('*')):
  if (d/'history.json').exists():
   lines+=['',d.name,'','| Epoch | Public phase loss | BTC weak CE | Train40 MAE | Static pass |','|---:|---:|---:|---:|---|']
   for h in read(d/'history.json'):lines.append(f"|{h['epoch']}|{h['phase_loss']:.5f}|{h['weak_ce']:.5f}|{h['train40']['MAE']}|{h['static_pass']}|")
 (P/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 if '--watch' in sys.argv:
  lock=(P/'report.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   report()
   if (P/'TRAINING_TERMINAL.json').exists() or (P/'TRAIN_ERROR.json').exists():break
   time.sleep(45)
 else:report()
