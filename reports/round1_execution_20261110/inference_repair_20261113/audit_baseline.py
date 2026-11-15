"""Compare original-input .7 outputs to saved native baselines before interpretation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common_repair import *
import numpy as np,csv,time,traceback
from collections import defaultdict

def main():
 lease=lock('baseline_audit.lock');assert not (P/'BASELINE_AUDIT_START.json').exists();save(P/'BASELINE_AUDIT_START.json',dict(utc=now(),pid=os.getpid(),code=sha(__file__)));began=time.time()
 while not (P/'ORIGINAL_COMPLETE.json').exists():
  assert time.time()-began<12*3600 and not (P/'ORIGINAL_ERROR.json').exists();time.sleep(15)
 baseline=list(csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open()));old={}
 for method,key in [('V3_median15','V3'),('POS_old_argmax','POS'),('CHROM_old','CHROM'),('calibrated_median_POS_CHROM_V3','calibrated')]:old[key]={r['recording']:float(r['prediction']) for r in baseline if r['condition']=='native' and r['method']==method}
 old['padding']={r['recording']:r['prediction'] for r in read(E/'final_priorities_20261111/padding_evaluation/BTC_VIDEO.json') if r['condition']=='native'}
 records=[]
 for j in manifest():
  if j['dataset']!='BTC':continue
  groups=defaultdict(list);root=P/'original'/j['key']/'native';receipt=read(root/'COMPLETE.json');assert sha(root/'PREDICTIONS.json')==receipt['rows_sha256']
  for r in read(root/'PREDICTIONS.json'):groups[('padding' if r['model'].startswith('padding') else 'V3',r['start'])].append(r['readouts']['power_0.7'])
  win=defaultdict(list)
  for (method,start),values in groups.items():
   assert len(values)==(15 if method=='V3' else 1)
   if all(v is not None for v in values):win[method].append(float(np.median(values)))
  for r in read(P/'classical_original'/j['key']/'native/CLASSICAL.json'):
   if r['readouts']['0.7'] is not None:win[r['model']].append(r['readouts']['0.7'])
  record=dict(recording=j['recording'],person=j['person'],fold=j['fold'],truth=j['windows'][0]['truth'],predictions={k:float(np.median(v)) for k,v in win.items() if v});records.append(record)
 # Identical old cohort, person-balanced OOF calibration.
 for fold in range(5):
  tr=sorted([r for r in records if r['fold']!=fold],key=lambda r:r['recording']);va=[r for r in records if r['fold']==fold];counts={p:sum(r['person']==p for r in tr) for p in {r['person'] for r in tr}}
  def med(r):return float(np.median([r['predictions'][m] for m in ['POS','CHROM','V3']]))
  w=np.sqrt([1/counts[r['person']] for r in tr]);x=np.array([[med(r),1] for r in tr]);y=np.array([r['truth'] for r in tr]);coef=np.linalg.lstsq(x*w[:,None],y*w,rcond=None)[0]
  for r in va:r['predictions']['calibrated']=float(coef[0]*med(r)+coef[1])
 comparisons=[]
 for method,saved in old.items():
  delta=[dict(recording=r['recording'],new=r['predictions'].get(method),saved=saved.get(r['recording']),difference=r['predictions'][method]-saved[r['recording']]) for r in records if method in r['predictions'] and r['recording'] in saved]
  differences=np.array([r['difference'] for r in delta]);comparisons.append(dict(method=method,compared=len(delta),saved=len(saved),max_absolute_difference=float(abs(differences).max()),changed_exact=int((differences!=0).sum()),changed_over_1e8=int((abs(differences)>1e-8).sum()),rows=delta))
 save(P/'BASELINE_AUDIT.json',dict(utc=now(),videos=len(records),people=len({r['person'] for r in records}),comparisons=comparisons,interpretation='Numerical differences must be reviewed; never waive source/reference hash mismatches',all_within_1e8=all(a['changed_over_1e8']==0 and a['compared']==218 for a in comparisons)))
 save(P/'BASELINE_AUDIT_COMPLETE.json',dict(utc=now(),receipt_sha256=sha(P/'BASELINE_AUDIT.json')))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'BASELINE_AUDIT_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
