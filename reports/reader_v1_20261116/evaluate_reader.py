import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import sys,json,csv,hashlib,fcntl,traceback
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;E=R/'round1_execution_20261110';V=R/'round1_v4_20261105/btc_zeroshot';W=R/'w2_btc_20261116'
sys.path.insert(0,str(E))
from report_metrics import metric,belongs
import reader_v1 as api
ref=api.ref
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 lock=(P/'evaluate.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(P/'REPRODUCTION.json')['passed']
 mapping=read(W/'PRIVATE_EXPORT_MAPPING.json')['video'];dev=[v for v in read(V/'dev_manifest.json')['videos'] if v['recording'] in mapping];assert len(dev)==218
 prior=list(csv.DictReader((R/'round1_bayes_20261108/VIDEO_PREDICTIONS.csv').open()))
 baseline={m:{r['recording']:float(r['prediction']) for r in prior if r['condition']=='native' and r['method']==m} for m in ['constant','calibrated_median_POS_CHROM_V3']}
 allpred=[];allmetrics=[];allfits=[];excluded=[];native_z=None
 for condition in ['native','c5','c5prime_default','c5prime_bf0']:
  zs=[];vs=[]
  for v in dev:
   if condition.startswith('c5prime'):
    root=E/'c5prime/BTC'/v['recording']/condition.split('_')[-1];receipt=read(root/'COMPLETE.json')
    if receipt['state']!='complete' or not receipt['bitrate_pass']:
     excluded.append(dict(recording=v['recording'],condition=condition,reason=receipt.get('reason','bitrate_above500')));continue
    d=root/'crop'
   else:d=V/'dev'/v['drive_id']/condition
   if not (d/'meta.npz').exists():excluded.append(dict(recording=v['recording'],condition=condition,reason='meta_unavailable'));continue
   receipt=read(d/'COMPLETE.json');assert sha(d/'meta.npz')==receipt['meta_sha256']
   with np.load(d/'meta.npz') as a:z=api.scores(a['t'],a['valid'],a['rgb_mean'])
   zs.append(z);vs.append(v)
  np.savez_compressed(P/(condition+'_spectra.npz'),Z=np.array([z if z is not None else np.zeros(len(ref.GRID)) for z in zs]),abstain=np.array([z is None for z in zs]),recording=np.array([v['recording'] for v in vs]))
  if condition=='native':native_z=zs;assert len(vs)==218
  folds=np.array([v['fold'] for v in vs]);truth=np.array([v['pr_bpm'] for v in vs]);persons=[v['person'] for v in vs];rows=[]
  for fold in range(5):
   tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);assert not ({persons[i] for i in tr}&{persons[i] for i in te})
   pr,b=ref.fit([zs[i] for i in tr],truth[tr],[persons[i] for i in tr]);allfits.append(dict(condition=condition,fold=fold,beta=b,prior=pr.tolist()))
   for i in te:
    v=vs[i];base=dict(recording=v['recording'],person=v['person'],fold=fold,group=v['group'],task=v['task'],truth=float(truth[i]),condition=condition)
    rows.append(dict(base,method='reader_v1',prediction=ref.posterior_median(zs[i],pr,b),abstain=zs[i] is None))
    for name,m in [('constant','constant'),('calibrated','calibrated_median_POS_CHROM_V3')]:rows.append(dict(base,method=name,prediction=baseline[m][v['recording']]))
  for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
   allmetrics.extend(dict(q,condition=condition,group=group) for q in metric([r for r in rows if belongs(r,group)],['constant','calibrated','reader_v1']))
  allpred+=rows;save(P/'STATUS.json',dict(condition=condition,videos=len(vs)))
 save(P/'PREDICTIONS.json',allpred);save(P/'METRICS.json',allmetrics);save(P/'FITS.json',allfits);save(P/'EXCLUSIONS.json',excluded)
 # Repeated single RGB frame is exactly the mean trace of a repeated ROI frame.
 model=read(P/'reader_model_dev218.json');static=[]
 for v in dev[:20]:
  with np.load(V/'dev'/v['drive_id']/'native/meta.npz') as a:
   t=np.arange(901)/30;rgb=np.tile(a['rgb_mean'][0],(len(t),1));p=api.predict(t,np.ones(len(t),bool),rgb,model)
  assert p['abstain'] and p['prediction']==ref.posterior_median(None,np.array(model['prior']),model['beta']);static.append(p)
 save(P/'STATIC.json',dict(passed=True,cases=static))
 # Whole-person block permutation preserves the number of labels per person.
 truth=np.array([v['pr_bpm'] for v in dev]);folds=np.array([v['fold'] for v in dev]);people=[v['person'] for v in dev];null=[]
 for rep in range(20):
  pred=np.zeros(218);fixed=[]
  for fold in range(5):
   tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);groups={p:sorted([i for i in tr if people[i]==p],key=lambda i:dev[i]['recording']) for p in sorted({people[i] for i in tr})};strata={}
   for p,ids in groups.items():strata.setdefault(len(ids),[]).append(p)
   y=truth.copy();rng=np.random.default_rng(np.random.SeedSequence([0,rep,fold]))
   for n,ps in sorted(strata.items()):
    for recipient,donor in zip(ps,rng.permutation(ps)):
     y[groups[recipient]]=truth[groups[donor]]
     if recipient==donor:fixed.append(recipient)
   pr,b=ref.fit([native_z[i] for i in tr],y[tr],[people[i] for i in tr])
   for i in te:pred[i]=ref.posterior_median(native_z[i],pr,b)
  mae=float(ref.person_table(pred,truth,people)[:,0].mean());null.append(dict(repetition=rep,MAE=mae,fixed_person_fold_blocks=len(fixed)))
 save(P/'SHUFFLE20.json',dict(runs=null,mean_MAE=float(np.mean([n['MAE'] for n in null])),range_MAE=[min(n['MAE'] for n in null),max(n['MAE'] for n in null)],policy='train-only whole-person blocks within identical video-count strata; original validation labels'))
 lines=['# reader_v1: BTC dev','Native calibrated baseline reused on identical videos in every condition; it is not a compressed-input calibrated estimator. C5-prime is an ffmpeg proxy, not MATLAB parity. Missing media/failed bitrate are excluded explicitly; valid media with no valid spectrum abstains to prior.','', '|Condition|Group|Method|Videos/people|MAE|Bias|Spearman|Delta calibrated [CI95]|','|---|---|---|---:|---:|---:|---:|---|']
 for q in allmetrics:lines.append(f"|{q['condition']}|{q['group']}|{q['method']}|{q['videos']}/{q['people']}|{q['MAE']:.4f}|{q['bias']:.4f}|{q['Spearman']}|{q['delta_calibrated']:.4f} {q['delta_calibrated_CI']}|")
 lines+=['',f'Static:20/20 PASS. Shuffle20 meanMAE={np.mean([n["MAE"] for n in null]):.4f}. No tuning to force null to12.', 'Holdout evaluations:0. W2-spectrum fusion and Octave parity are separate pending artifacts.']
 (P/'REPORT.md').write_text('\n'.join(lines));save(P/'EVALUATION_COMPLETE.json',dict(passed=True,holdout_evaluations=0))
if __name__=='__main__':
 try:main()
 except Exception:save(P/'ERROR.json',dict(traceback=traceback.format_exc()));raise
