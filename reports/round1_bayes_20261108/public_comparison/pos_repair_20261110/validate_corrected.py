"""Regression to frozen POS predictions on all public native/legacy-C5 windows."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,csv,hashlib,sys,fcntl
from pathlib import Path
import numpy as np
from pos_corrected import estimate_window, FRONTEND
P=Path(__file__).resolve().parent;PUBLIC=P.parent;ROOT=Path('/mnt/samsung_t5/khangle/facevitals')
sys.path.insert(0,str(PUBLIC));from metrics_frozen import summarize
def save(n,x):(P/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
 lock=(P/'validation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (P/'VALIDATION_COMPLETE.json').exists()
 refs=json.loads((PUBLIC/'REFERENCE_COMPACT.json').read_text());assert hashlib.sha256((ROOT/'reports/round1_final_20261105/revision_02/frozen_data/FROZEN_WINDOWS.json').read_bytes()).hexdigest()==refs['source_sha256']
 rows=[];failures=[];hashes={};parity=[]
 for item in refs['records']:
  ds,rid,cond=item['dataset'],item['recording_id'],item['variant']
  f=ROOT/'reports/round1_final_20261105/revision_02/frozen_data/spatial_rgb'/ds/rid/(cond+'.npz');hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  with np.load(f) as z:t,x,v=z['t'],z['rgb'],z['valid']
  for b in item['windows']:
   if not b['reference_eligible']:continue
   baseline=b['baseline']['POS']
   common=dict(dataset=ds,condition=cond,person=b['subject_id'],video=rid,recording=rid+'@'+str(b['window_start_s']),start=b['window_start_s'],truth=b['reference_hr_bpm'])
   try:
    out=estimate_window(t,x,v,b['window_start_s'],frontend=FRONTEND)
    if out['hr'] is None:raise ValueError(out['reason'])
    if baseline['accepted']:parity.append(abs(out['hr']-baseline['prediction_hr_bpm']))
    # Strict historical support for comparison; broader valid support is not silently included.
    if baseline['accepted']:rows.append(dict(**common,prediction=out['hr'],snr_db=out['snr_db']))
   except ValueError as e:
    failures.append(dict(**common,reason=str(e),historical_accepted=baseline['accepted']))
    assert not baseline['accepted'],('regression_support',common,e)
  print(ds,rid,cond,flush=True)
 assert parity and max(parity)<1e-8
 metrics=[]
 for ds in ['MCD','UBFC2']:
  for cond in ['native','f10_b500']:
   rr=[r for r in rows if r['dataset']==ds and r['condition']==cond];m=summarize(rr);m['windows']=m.pop('videos');m['recordings']=len({r['video'] for r in rr});m['snr_db_median']=float(np.median([r['snr_db'] for r in rr]));metrics.append(dict(dataset=ds,condition=cond,**m))
 with (P/'CORRECTED_PREDICTIONS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save('CORRECTED_METRICS.json',metrics);save('CORRECTED_FAILURES.json',failures);save('CORRECTED_INPUT_HASHES.json',hashes)
 m=next(r for r in metrics if r['dataset']=='MCD' and r['condition']=='native')
 save('CORRECTED_GATE.json',dict(passed=m['MAE']<=6 and m['snr_db_median']>0,MAE=m['MAE'],blind_SNR_median=m['snr_db_median'],rule_unchanged=True))
 save('VALIDATION_COMPLETE.json',dict(parity_windows=len(parity),max_hr_difference=max(parity),holdout_evaluations=0,frozen_originals_unchanged=True))
 print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
