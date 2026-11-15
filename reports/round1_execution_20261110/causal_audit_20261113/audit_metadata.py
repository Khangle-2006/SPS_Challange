"""Dev-only PTS, association and lineage audit. No inference or fitting."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common_causal import *
from common_repair import manifest,O,B
import numpy as np,csv,traceback,re
from collections import defaultdict
from scipy.stats import pearsonr,spearmanr
from t8_t1_loader import supported
from crop_io import crop_path

def csvout(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def associations():
 from report_repair import stats
 jobs={j['recording']:j for j in manifest() if j['dataset']=='BTC'}
 rows=list(csv.DictReader((P/'report_original/BTC_VIDEOS.csv').open()))
 for r in rows:r.update(prediction=float(r['prediction']),truth=float(r['truth']))
 baseline=list(csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open()));bases={}
 for method in ['constant','calibrated_median_POS_CHROM_V3']:
  bases[method]={r['recording']:dict(r,prediction=float(r['prediction']),truth=float(r['truth'])) for r in baseline if r['method']==method and r['condition']=='native'}
 buckets=defaultdict(list)
 for r in rows:buckets[tuple(r[k] for k in ['condition','input','method','readout'])].append(r)
 out=[]
 for key,rr in buckets.items():
  for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
   def member(r):
    m=jobs[r['recording']]['metadata'];g=m['group']+'-Resting' if m['task'].lower()=='resting' else 'Walking' if m['task'].lower()=='walking' else 'Task';return group=='all' or g==group
   selected=[r for r in rr if member(r) and all(r['recording'] in b for b in bases.values())]
   if not selected:continue
   for name,rs in [('candidate',selected)]+[(n,[b[r['recording']] for r in selected]) for n,b in bases.items()]:
    persons=defaultdict(list)
    for r in rs:persons[r['person']].append(r)
    y=np.array([np.mean([r['truth'] for r in persons[p]]) for p in sorted(persons)]);v=np.array([np.mean([r['prediction'] for r in persons[p]]) for p in sorted(persons)])
    ok=len(y)>2 and np.ptp(y)>0 and np.ptp(v)>0
    out.append(dict(zip(['condition','input','candidate','readout'],key),group=group,row=name,**stats(rs),Pearson_person_means=float(pearsonr(y,v).statistic) if ok else None,Spearman_person_means=float(spearmanr(y,v).statistic) if ok else None,baseline_condition='native saved, matched videos' if name!='candidate' else key[0]))
 csvout(C/'ASSOCIATIONS.csv',out);save(C/'ASSOCIATIONS_COMPLETE.json',dict(utc=now(),rows=len(out),source_sha256=sha(P/'report_original/BTC_VIDEOS.csv'),no_fitting=True))

def main():
 lease=lock('metadata.lock');assert not (C/'METADATA_STARTED.json').exists();budget(True);check_deadline()
 save(C/'METADATA_STARTED.json',dict(pid=os.getpid(),utc=now(),code_sha256=sha(__file__)))
 jobs=[j for j in manifest() if j['dataset']=='BTC'];assert len(jobs)==218 and len({j['person'] for j in jobs})==158
 split=read(O/'btc_split.json');dev=read(O/'dev_manifest.json');assert dev['split_sha256']==sha(O/'btc_split.json')
 lookup={j['recording']:j for j in dev['videos']};issues=[];lineage=[];ptrows=[]
 for n,j in enumerate(jobs):
  check_deadline();save(C/'METADATA_STATUS.json',dict(utc=now(),completed=n,total=len(jobs),key=j['key']))
  s=split['people'][j['person']];assert s['partition']=='dev','CLOSED_HOLDOUT';d=lookup[j['recording']];m=j['metadata'];src=read(j['source_info'])
  tests={'person_fold':s['fold']==j['fold']==d['fold'],'person_identity':j['person']==d['person']==re.match(r'^(S_(?:AB|N)\d+)',j['recording']).group(1),'recording':j['recording'] in s['recordings'],'annotation_row':m['sheet_row']==d['sheet_row'] and m['group']==d['group'],'label':m['pr_bpm']==d['pr_bpm'] and all(w['truth']==d['pr_bpm'] for w in j['windows']),'source_binding':j['source_sha256']==d['sha256']==src['source_sha256'],'file_hash':sha(j['source'])==j['source_sha256'],'split_binding':src['split_sha256']==sha(O/'btc_split.json')}
  for k,v in tests.items():
   if not v:issues.append(dict(recording=j['recording'],check=k))
  preds=read(P/'original'/j['key']/'native/PREDICTIONS.json')
  for r in preds:
   if (r['person'],r['fold'],r['recording'],r['truth'])!=(j['person'],j['fold'],j['recording'],d['pr_bpm']):issues.append(dict(recording=j['recording'],check='prediction_join',model=r['model'],start=r['start']))
   if r['model'].startswith('padding') and r['model']!=f"padding_f{j['fold']}":issues.append(dict(recording=j['recording'],check='padding_OOF_checkpoint'))
  lineage.append(dict(recording=j['recording'],group=m['group'],sheet_row=m['sheet_row'],person=j['person'],fold=j['fold'],source_sha256=j['source_sha256'],PR=d['pr_bpm'],PR_measurement_time=None,checks=tests))
  # Independently re-read display PTS from bitstream; do not infer from FPS tags.
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',j['source']]))
  pts=np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']]);pts-=pts[0]
  native=np.load(Path(j['crop'])/'meta.npz');clock_exact=np.array_equal(pts,native['t'])
  if not clock_exact:issues.append(dict(recording=j['recording'],check='cached_native_PTS_not_bitstream_exact'))
  for condition in ['native','c5','c5prime_default','c5prime_bf0']:
   try:mp=crop_path(j,condition);a=np.load(mp/'meta.npz')
   except (ValueError,FileNotFoundError):continue
   t=a['t'];valid=a['valid']
   for w in j['windows']:
    q=w['start']+np.arange(301)/30;ix=np.searchsorted(t,q,side='right').clip(1,len(t)-1);weight=((q-t[ix-1])/(t[ix]-t[ix-1])).astype(np.float32);effective=t[ix-1]*(1-weight.astype(float))+t[ix]*weight.astype(float)
    sup=bool(supported(t,valid,q[0],q[-1]));span=float(effective[-1]-effective[0]);nearest=np.where(weight<=.5,ix-1,ix);nearestspan=float(t[nearest[-1]]-t[nearest[0]])
    ptrows.append(dict(recording=j['recording'],person=j['person'],source_fps=float(m['media']['avg_fps']),condition=condition,start=w['start'],points=301,grid_span_s=float(q[-1]-q[0]),effective_source_PTS_span_s=span,nearest_frame_span_s=nearestspan,bracket_span_s=float(t[ix[-1]]-t[ix[0]-1]),unique_bracketing_source_frames=len(np.unique(np.r_[ix,ix-1])),max_time_residual_s=float(abs(effective-q).max()),extrapolation=bool(np.any((weight<0)|(weight>1))),supported=sup,span_violation=bool(abs(span-10)>.05),nearest_span_violation=bool(abs(nearestspan-10)>.05),native_source_PTS_exact=clock_exact))
 save(C/'LINEAGE.json',lineage);save(C/'LINEAGE_INCONSISTENCIES.json',issues);csvout(C/'PTS_WINDOWS.csv',ptrows)
 groups=defaultdict(list)
 for r in ptrows:groups[(r['condition'],round(r['source_fps'],2))].append(r)
 summary=[dict(condition=k[0],source_fps=k[1],windows=len(rr),supported=sum(r['supported'] for r in rr),violations=sum(r['span_violation'] for r in rr),nearest_frame_violations=sum(r['nearest_span_violation'] for r in rr),extrapolation=sum(r['extrapolation'] for r in rr),max_effective_time_residual_s=max(r['max_time_residual_s'] for r in rr)) for k,rr in sorted(groups.items())]
 csvout(C/'PTS_BY_FPS.csv',summary)
 associations();save(C/'METADATA_COMPLETE.json',dict(utc=now(),videos=len(jobs),people=158,issues=len(issues),PTS_windows=len(ptrows),actual_source_PTS_verified=True,original_workbook_reopen='pending: original XLSX not found; frozen dev annotation chain checked only',PR_measurement_timing='unknown',holdout_evaluations=0,span_violations=sum(r['span_violation'] for r in ptrows),extrapolation=sum(r['extrapolation'] for r in ptrows)))
if __name__=='__main__':
 try:main()
 except Exception as e:save(C/'METADATA_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
