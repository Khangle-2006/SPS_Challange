"""Five coefficients, outer-person exclusion at every learned stage, no holdout."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,csv
import numpy as np
from scipy import signal
from common import *
sys.path.insert(0,str(V/'btc_zeroshot'))
from infer_dev import classical_functions,decode_wave,supported
from report_metrics import metric,belongs

def main():
 lease=lock('d_stack.lock');guard(True);assert not (P/'D_STACK_COMPLETE.json').exists()
 source=B/'domain_sweep_20261109';complete=read(source/'PBV_STACK_COMPLETE.json')
 for file,h in complete['input_hashes'].items():assert sha(file)==h,('Frozen PBV input changed',file)
 signatures={tuple(r['excluded_folds']):r for r in read(source/'PBV_SIGNATURES.json')}
 excluded={(r['recording'],r.get('start')) for r in read(source/'PBV_EXCLUSIONS.json')}
 dev=read(V/'btc_zeroshot/dev_manifest.json')['videos'];ns,fnsha=classical_functions();windows=[]
 sos=signal.butter(1,[.7,3],btype='bandpass',fs=30,output='sos')
 for v in dev:
  d=V/'btc_zeroshot/dev'/v['drive_id'];meta=d/'native/meta.npz'
  if (v['recording'],None) in excluded:continue
  with np.load(meta) as z:t=z['t'].copy();valid=z['valid'].copy();rgb=z['rgb_mean'].copy()
  old=read(V/'btc_zeroshot/predictions'/v['drive_id']/'native/classical.json')['rows']
  for start in sorted({r['start'] for r in old if r['scheme']=='10s'}):
   if (v['recording'],start) in excluded:continue
   assert supported(t,valid,start,start+10)
   q=start+np.arange(300)/30;x=np.stack([np.interp(q,t,rgb[:,k]) for k in range(3)],1);cn=signal.sosfiltfilt(sos,x/x.mean(0)-1,axis=0);pos=ns['pos'](x,30)
   f,power=signal.periodogram(pos,fs=30,window='hann',nfft=4096,detrend='linear');band=(f>=.7)&(f<=3);peak=f[band][power[band].argmax()];hit=band&(abs(f-peak)<=.1);snr=float(10*np.log10(max(float(power[hit].sum()),1e-30)/max(float(power[band&~hit].sum()),1e-30)))
   pair={m:next(r['prediction'] for r in old if r['method']==m and r['scheme']=='10s' and r['start']==start) for m in ['POS','CHROM']}
   windows.append(dict(v=v,start=start,cn=cn,snr=snr,**pair))
 oldrows=list(csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open()));cal={r['recording']:float(r['prediction']) for r in oldrows if r['condition']=='native' and r['method']=='calibrated_median_POS_CHROM_V3'}
 result=[];fits=[];inputhash={}
 for outer in range(5):
  d=P/'D'/f'fold{outer}';assert (d/'COMPLETE.json').exists();pred=read(d/'WINDOWS.json');inputhash[str(d/'WINDOWS.json')]=sha(d/'WINDOWS.json')
  nn={};disjoint={}
  for rid in {r['recording'] for r in pred}:
   rr=[r for r in pred if r['recording']==rid and r['prediction'] is not None]
   if rr:nn[rid]=float(np.median([r['prediction'] for r in rr]))
   rr=[r for r in rr if r['also_disjoint']]
   if rr:disjoint[rid]=float(np.median([r['prediction'] for r in rr]))
  video_features=[]
  for v in dev:
   rid=v['recording'];excluded_folds=tuple(sorted({outer,v['fold']}));sig=signatures[excluded_folds];assert set(sig['training_people']).isdisjoint(x['person'] for x in dev if x['fold'] in excluded_folds)
   s=np.array(sig['signature']);rr=[]
   for w in windows:
    if w['v']['recording']!=rid:continue
    cn=w['cn'];cov=cn.T@cn/len(cn);cov+=np.eye(3)*max(float(np.trace(cov)),1e-12)*1e-6
    try:p=decode_wave(cn@np.linalg.solve(cov,s))
    except ValueError:continue
    rr.append(dict(PBV=p,snr=w['snr'],POS=w['POS'],CHROM=w['CHROM']))
   if not rr or rid not in nn:continue
   values={k:float(np.median([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None for k in ['POS','CHROM','PBV','snr']}
   if any(x is None or not np.isfinite(x) for x in values.values()):continue
   values.update(classical_pair=float(np.median([values['POS'],values['CHROM']])),new_model=nn[rid])
   video_features.append(dict(v,truth=v['pr_bpm'],**values))
  train=[r for r in video_features if r['fold']!=outer];val=[r for r in video_features if r['fold']==outer]
  assert {r['person'] for r in train}.isdisjoint(r['person'] for r in val)
  keys=['classical_pair','PBV','new_model','snr'];x=np.array([[r[k] for k in keys] for r in train]);y=np.array([r['truth'] for r in train]);counts={p:sum(r['person']==p for r in train) for p in {r['person'] for r in train}};w=np.array([1/counts[r['person']] for r in train]);mu=np.average(x,axis=0,weights=w);sd=np.maximum(np.sqrt(np.average((x-mu)**2,axis=0,weights=w)),1e-8)
  design=np.c_[np.ones(len(x)),(x-mu)/sd];coef=np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=1e-10)[0];assert len(coef)==5
  fits.append(dict(fold=outer,features=keys,parameters_including_intercept=5,coef=coef.tolist(),mean=mu.tolist(),std=sd.tolist(),training_people=sorted(counts),validation_people=sorted({r['person'] for r in val}),PBV_excludes_outer_and_inner=True,neural_model_excludes_outer=True,neural_train_features_are_in_sample=True))
  constant=float(np.median([v['pr_bpm'] for v in dev if v['fold']!=outer]))
  for r in val:
   stack=float(np.r_[1,(np.array([r[k] for k in keys])-mu)/sd]@coef);rid=r['recording']
   for method,value in [('constant',constant),('calibrated',cal.get(rid)),('model_stride1',nn[rid]),('model_disjoint',disjoint.get(rid)),('stack5_new',stack)]:
    result.append(dict(recording=rid,person=r['person'],fold=outer,group=r['group'],task=r['task'],truth=r['truth'],method=method,prediction=value,condition='native'))
 methods=['constant','calibrated','model_stride1','model_disjoint','stack5_new'];metrics=[]
 for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
  metrics.extend(dict(group=group,**m) for m in metric([r for r in result if belongs(r,group)],methods))
 save(P/'D_STACK_FITS.json',fits);save(P/'D_VIDEO_PREDICTIONS.json',result);save(P/'D_METRICS.json',metrics)
 save(P/'D_STACK_COMPLETE.json',dict(utc=now(),input_hashes=inputhash,PBV_signature_sha256=sha(source/'PBV_SIGNATURES.json'),PBV_exclusions_sha256=sha(source/'PBV_EXCLUSIONS.json'),calibrated_baseline_sha256=sha(B/'VIDEO_PREDICTIONS.csv'),selection_sha256=sha(P/'D_SELECTION.json'),holdout_evaluations=0))
 print('D_COMPLETE',[(r['method'],r['MAE']) for r in metrics if r['group']=='all'])
if __name__=='__main__':main()
