"""Person-held-out CPU evaluation. Results are NOT a release or holdout authorization."""
from common import *
from pipeline import trace_candidates,fit,predict
import numpy as np,time,fcntl,traceback
from collections import defaultdict
from scipy.stats import pearsonr
from codec_audit import stream_probe,analyze_frames
ABLATIONS=['none','no_background','no_codec','no_viterbi','constant_epsilon','no_prior','K1']
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
 if isinstance(x,list):return [serial(v) for v in x]
 return x
def summarize(rr,other=None):
 if other is not None:
  by={r['recording']:r for r in other};rr=[r for r in rr if r['recording'] in by]
 per=defaultdict(list)
 for r in rr:per[r['person']].append(r)
 if not per:return dict(people=0,videos=0)
 values=[];truths=[];preds=[]
 for p in sorted(per):
  rs=per[p];e=np.array([r['prediction']-r['truth'] for r in rs]);v=np.array([np.abs(e).mean(),np.sqrt(np.mean(e*e)),e.mean()])
  if other is not None:
   o=np.array([by[r['recording']]['prediction']-r['truth'] for r in rs]);v-=np.array([np.abs(o).mean(),np.sqrt(np.mean(o*o)),o.mean()])
  values.append(v);truths.append(np.mean([r['truth'] for r in rs]));preds.append(np.mean([r['prediction'] for r in rs]))
 values=np.array(values);ix=np.random.default_rng(0).integers(0,len(values),(10000,len(values)));ci=np.quantile(values[ix].mean(1),[.025,.975],axis=0).T
 ans=dict(people=len(per),videos=len(rr),MAE=float(values[:,0].mean()),RMSE_mean_person=float(values[:,1].mean()),bias=float(values[:,2].mean()),CI95=ci.tolist())
 if other is None:
  ans['Pearson_person_means']=float(pearsonr(truths,preds).statistic) if len(per)>2 and np.ptp(truths)>1e-10 and np.ptp(preds)>1e-10 else None
  ans['LoA_person_mean_errors']=(values[:,2].mean()+np.array([-1.96,1.96])*values[:,2].std(ddof=1)).tolist() if len(per)>1 else None
  if ans['Pearson_person_means'] is not None:
   xx=np.array(truths)[ix];yy=np.array(preds)[ix];xx-=xx.mean(1,keepdims=True);yy-=yy.mean(1,keepdims=True);den=np.sqrt(np.sum(xx*xx,axis=1)*np.sum(yy*yy,axis=1));valid=den>1e-12;rho=np.sum(xx[valid]*yy[valid],axis=1)/den[valid]
   ans['CI95_Pearson']=np.quantile(rho,[.025,.975]).tolist() if len(rho) else None;ans['Pearson_bootstrap_valid']=int(valid.sum())
  if len(per)>1:
   eb=values[:,2][ix];means=eb.mean(1);sd=eb.std(1,ddof=1);ans['CI95_LoA_lower']=np.quantile(means-1.96*sd,[.025,.975]).tolist();ans['CI95_LoA_upper']=np.quantile(means+1.96*sd,[.025,.975]).tolist()
 return ans
def old_method(v,cond,method):
 p=OLD/'predictions'/v['drive_id']/cond/'classical.json'
 if method in ['POS','CHROM']:
  if not p.exists():return None
  a=[r['prediction'] for r in read(p)['rows'] if r['method']==method and r['scheme']=='10s' and r['prediction'] is not None]
 else:
  files=list(p.parent.glob(method+'_f*_s*.json'))
  if len(files)!=15:return None
  win=defaultdict(list)
  for fp in files:
   a_=read(fp);assert a_['split_sha256']==SPLIT
   for r in a_['rows']:
    if r['scheme']=='10s':win[r['start']].append(r['prediction'])
  a=[float(np.median(x)) for x in win.values() if len(x)==15 and all(y is not None for y in x)]
 return float(np.median(a)) if a else None
def main():
 lock=(OUT/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prereg();save(OUT/'evaluation_PREFLIGHT.json',guard())
 assert not (OUT/'EVALUATION_COMPLETE.json').exists(),'Do not rerun completed evaluation'
 videos=dev_manifest();features=OUT/'features';features.mkdir(exist_ok=True);done=set()
 while len(done)<len(videos)*2:
  progress=False
  for v in videos:
   rec=v['recording']
   for cond in ['native','c5']:
    key=(rec,cond)
    if key in done:continue
    d=OUT/'traces'/rec/cond;cp=d/'COMPLETE.json'
    if (d/'EXCLUDED.json').exists():done.add(key);continue
    if not cp.exists():continue
    if cond=='native':
     gp=OUT/'native_gop'/(rec+'.json')
     if gp.exists():g=read(gp)
     elif rec not in read(OLD/'diagnosis205_20261108/COHORT.json')['recordings']:
      g=analyze_frames(stream_probe(v['path'],True)['frames']);save(features/(rec+'_native_GOP.json'),g)
     else:continue
    else:
     gp=features/(rec+'_c5_GOP.json')
     if gp.exists():g=read(gp)
     else:g=analyze_frames(stream_probe(OLD/'dev'/v['drive_id']/'c5_encoded/video.mp4',True)['frames']);save(gp,g)
    assert sha(d/'trace.npz')==read(cp)['trace_sha256']
    with np.load(d/'trace.npz') as z:t=z['t'];rgb=z['rgb'];face=z['face']
    for abl in ABLATIONS:
     p=features/(rec+'_'+cond+'_'+abl+'.json')
     if p.exists():continue
     a=trace_candidates(t,rgb,[x['hz'] for x in g['artifact_frequencies']],abl)
     save(p,dict(recording=rec,person=v['person'],fold=v['fold'],truth=v['pr_bpm'],duration=float(t[-1]-t[0]),face_fraction=float(face.mean()),condition=cond,ablation=abl,trace_sha256=sha(d/'trace.npz'),pipeline_sha256=sha(OUT/'pipeline.py'),**a))
    done.add(key);progress=True;save(OUT/'evaluation_STATUS.json',dict(state='features',completed=len(done),total=442,pid=os.getpid(),utc=now()))
  if not progress:
   if (OUT/'trace_ERROR.json').exists():raise RuntimeError('Trace worker stopped; preserve features, no automatic restart')
   time.sleep(15)
 allrows=[];fits=[]
 for cond in ['native','c5']:
  for abl in ABLATIONS:
   records=[read(features/(v['recording']+'_'+cond+'_'+abl+'.json')) for v in videos if (features/(v['recording']+'_'+cond+'_'+abl+'.json')).exists()]
   records=[r for r in records if r['duration']>=10]
   for fold in range(5):
    save(OUT/'evaluation_STATUS.json',dict(state='fitting',completed_fits=len(fits),total_fits=70,condition=cond,ablation=abl,fold=fold,pid=os.getpid(),utc=now()))
    train=[r for r in records if r['fold']!=fold];val=[r for r in records if r['fold']==fold];model=fit(train,abl)
    assert set(model['training_people']).isdisjoint(r['person'] for r in val)
    assert model['ablation']==abl
    fits.append(dict(serial(model),condition=cond,fold=fold))
    for r in val:
     ans=predict(r['candidates'],model);allrows.append(dict(method='Bayes_'+abl,condition=cond,recording=r['recording'],person=r['person'],fold=fold,truth=r['truth'],prediction=ans['hr'],lower90=ans['lower90'],upper90=ans['upper90'],pulse_observed=ans['pulse_observed'],sqi=float(np.mean([c['sqi'] for c in ans['contributors']])) if ans['contributors'] else None))
     if abl=='none':save(OUT/'predictions'/r['recording']/(cond+'.json'),dict(fold=fold,training_people=model['training_people'],features_sha256=sha(features/(r['recording']+'_'+cond+'_none.json')),pipeline_sha256=sha(OUT/'pipeline.py'),candidates=r['candidates'],**serial(ans)))
  for v in videos:
   p=features/(v['recording']+'_'+cond+'_none.json')
   if not p.exists() or read(p)['duration']<10:continue
   const=float(np.median([x['pr_bpm'] for x in videos if x['fold']!=v['fold']]))
   for method,value in [('constant',const),('POS_old_argmax',old_method(v,cond,'POS')),('CHROM_old',old_method(v,cond,'CHROM')),('V3_median15',old_method(v,cond,'v3_joint')),('UBFC_median15',old_method(v,cond,'ubfc_only'))]:
    if value is not None:allrows.append(dict(method=method,condition=cond,recording=v['recording'],person=v['person'],fold=v['fold'],truth=v['pr_bpm'],prediction=value,lower90=None,upper90=None,pulse_observed=None,sqi=None))
 # Refit the previously successful linear median baseline on this exact dev cohort,
 # keeping every validation person out of both calibration coefficients.
 calibration_fits=[]
 for cond in ['native','c5']:
  required=['POS_old_argmax','CHROM_old','V3_median15'];lookup={m:{r['recording']:r for r in allrows if r['method']==m and r['condition']==cond} for m in required}
  keys=set.intersection(*(set(lookup[m]) for m in required));median=[dict(lookup[required[0]][k],prediction=float(np.median([lookup[m][k]['prediction'] for m in required]))) for k in sorted(keys)]
  for fold in range(5):
   tr=[r for r in median if r['fold']!=fold];va=[r for r in median if r['fold']==fold]
   if len({r['person'] for r in tr})<2:continue
   assert not {r['person'] for r in tr}&{r['person'] for r in va}
   counts={p:sum(r['person']==p for r in tr) for p in {r['person'] for r in tr}};w=np.sqrt([1/counts[r['person']] for r in tr]);x=np.array([[r['prediction'],1] for r in tr]);y=np.array([r['truth'] for r in tr]);coef=np.linalg.lstsq(x*w[:,None],y*w,rcond=None)[0]
   calibration_fits.append(dict(condition=cond,fold=fold,a=float(coef[0]),b=float(coef[1]),train_people=sorted(counts)))
   allrows.extend(dict(r,method='calibrated_median_POS_CHROM_V3',prediction=float(coef[0]*r['prediction']+coef[1])) for r in va)
 save(OUT/'CALIBRATION_FITS.json',calibration_fits);save(OUT/'FITS.json',fits);csvout(OUT/'VIDEO_PREDICTIONS.csv',allrows)
 by={v['recording']:v for v in videos};summary=[]
 for cond in ['native','c5']:
  subsets={'all':list(by)}
  for v in videos:
   task=v['task'].lower();activity=v['group']+'-Resting' if task=='resting' else 'Walking' if task=='walking' else 'Task'
   for group in [activity,'skin='+v['skin'],'lighting='+v['lighting'],'resolution='+v['media']['width']+'x'+v['media']['height'],'HDR='+str(v['media']['transfer'] in ['arib-std-b67','smpte2084']),'rotation='+str(v['media']['rotation'])]:subsets.setdefault(group,[]).append(v['recording'])
  for group,keys in subsets.items():
   baseline=[r for r in allrows if r['method']=='constant' and r['condition']==cond and r['recording'] in keys];pos=[r for r in allrows if r['method']=='POS_old_argmax' and r['condition']==cond and r['recording'] in keys]
   common={r['recording'] for r in baseline}&{r['recording'] for r in pos}
   for method in sorted({r['method'] for r in allrows}):
    rr=[r for r in allrows if r['method']==method and r['condition']==cond and r['recording'] in common]
    if rr:summary.append(dict(group=group,method=method,condition=cond,**summarize(rr),delta_constant=summarize(rr,baseline)))
 save(OUT/'METRICS.json',summary)
 cover=[]
 for cond in ['native','c5']:
  per=defaultdict(list)
  for r in allrows:
   if r['method']=='Bayes_none' and r['condition']==cond:per[r['person']].append(r['lower90']<=r['truth']<=r['upper90'])
  x=np.array([np.mean(v) for v in per.values()]);ix=np.random.default_rng(0).integers(0,len(x),(10000,len(x)));cover.append(dict(condition=cond,people=len(x),coverage90=float(x.mean()),CI95=np.quantile(x[ix].mean(1),[.025,.975]).tolist()))
 save(OUT/'INTERVAL_COVERAGE.json',cover)
 ablation_rows=[]
 for cond in ['native','c5']:
  full=[r for r in allrows if r['method']=='Bayes_none' and r['condition']==cond]
  for abl in ABLATIONS[1:]:ablation_rows.append(dict(condition=cond,ablation=abl,delta_vs_full=summarize([r for r in allrows if r['method']=='Bayes_'+abl and r['condition']==cond],full)))
 save(OUT/'ABLATION.json',ablation_rows)
 qc=[]
 for v in videos:
  p=OUT/'traces'/v['recording']/'native/COMPLETE.json'
  if p.exists():a=read(p);qc.append(dict(recording=v['recording'],face_fraction=a['face_fraction'],roi_valid=a['valid_roi_fraction']))
 save(OUT/'G1.json',dict(passed=len(qc)==221 and sum(r['face_fraction']>=.95 for r in qc)/221>=.95,completed_videos=len(qc),videos_face95=sum(r['face_fraction']>=.95 for r in qc),denominator=221,decode_errors=0 if len(qc)==221 else None,details=qc))
 mono=[]
 for cond in ['native','c5']:
  rr=[r for r in allrows if r['method']=='Bayes_none' and r['condition']==cond and r['sqi'] is not None]
  edges=np.quantile([r['sqi'] for r in rr],[.25,.5,.75]);vals=[]
  for i in range(4):
   subset=[r for r in rr if np.searchsorted(edges,r['sqi'])==i];stats_=summarize(subset);mono.append(dict(condition=cond,quartile=i+1,**stats_));vals.append(stats_.get('MAE'))
  save(OUT/('SQI_VALIDITY_'+cond+'.json'),dict(monotone_nonincreasing=all(a is not None and b is not None and b<=a for a,b in zip(vals,vals[1:])),MAE_quartiles=vals,action='If false, SQI-dependent version not accepted; report preregistered constant-epsilon ablation, do not refit thresholds on dev validation'))
 save(OUT/'SQI_QUARTILES.json',mono)
 primary=next(x for x in summary if x['method']=='Bayes_none' and x['condition']=='native' and x['group']=='all')
 save(OUT/'G2.json',dict(passed=primary['delta_constant']['CI95'][0][1]<0,matched_only=True,metrics=primary,holdout_evaluations=0,automatic_promotion=False))
 save(OUT/'EVALUATION_COMPLETE.json',dict(utc=now(),scope='BTC native and OLD C5 only; C5prime subject to G4; UBFC/MCD comparisons still required; no automatic promotion',holdout_evaluations=0))
 save(OUT/'evaluation_STATUS.json',dict(state='complete',completed_fits=len(fits),total_fits=70,pid=os.getpid(),utc=now()))
if __name__=='__main__':
 try:main()
 except Exception as e:save(OUT/'evaluation_ERROR.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc()));raise
