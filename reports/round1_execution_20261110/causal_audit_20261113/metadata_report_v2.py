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
    out.append(dict(zip(['condition','input','candidate','readout'],key),group=group,row=name,**{k:v for k,v in stats(rs).items() if k!='Spearman_person_means'},Pearson_person_means=float(pearsonr(y,v).statistic) if ok else None,Spearman_person_means=float(spearmanr(y,v).statistic) if ok else None,baseline_condition='native saved, matched videos' if name!='candidate' else key[0]))
 csvout(C/'ASSOCIATIONS.csv',out);save(C/'ASSOCIATIONS_COMPLETE.json',dict(utc=now(),rows=len(out),source_sha256=sha(P/'report_original/BTC_VIDEOS.csv'),no_fitting=True))


def main():
 lease=lock('metadata.lock');check_deadline();b=budget(True)
 assert not (C/'METADATA_COMPLETE.json').exists()
 assert not (C/'METADATA_REPORT_V2_STARTED.json').exists()
 lineage=read(C/'LINEAGE.json');issues=read(C/'LINEAGE_INCONSISTENCIES.json')
 rows=list(csv.DictReader((C/'PTS_WINDOWS.csv').open()))
 assert len(lineage)==218 and len({r['person'] for r in lineage})==158
 assert len({r['recording'] for r in rows})==218
 assert all(r['checks']['file_hash'] and r['checks']['split_binding'] for r in lineage)
 evidence={n:sha(C/n) for n in ['LINEAGE.json','LINEAGE_INCONSISTENCIES.json','PTS_WINDOWS.csv','PTS_BY_FPS.csv','METADATA_ERROR.json','audit_metadata.py']}
 save(C/'METADATA_REPORT_V2_STARTED.json',dict(utc=now(),pid=os.getpid(),budget=b,reused_inputs=evidence,no_rescan=True,no_inference=True,code_sha256=sha(__file__)))
 associations()
 assert all(sha(C/n)==h for n,h in evidence.items())
 save(C/'METADATA_COMPLETE.json',dict(utc=now(),videos=218,people=158,issues=len(issues),PTS_windows=len(rows),actual_source_PTS_verified=True,original_workbook_reopen='Frozen source unavailable; current selected dev A/N agrees, export SHA differs. See CURRENT_ANNOTATION_AUDIT.',PR_measurement_timing='unknown',holdout_evaluations=0,span_violations=sum(r['span_violation']=='True' for r in rows),extrapolation=sum(r['extrapolation']=='True' for r in rows),report_recovery='METADATA_REPORT_V2_STARTED.json',old_report_error_preserved=True))
if __name__=='__main__':
 try:main()
 except Exception as e:save(C/'METADATA_REPORT_V2_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
