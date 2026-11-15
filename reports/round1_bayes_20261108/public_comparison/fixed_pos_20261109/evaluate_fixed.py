"""Read saved spectra and outputs only; fixed POS comparison, no Bayes fits."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,gzip,csv,fcntl,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;PUBLIC=P.parent;ROOT=Path('/mnt/samsung_t5/khangle/facevitals');DOMAIN=PUBLIC.parent/'domain_sweep_20261109'
sys.path.insert(0,str(PUBLIC));from metrics_frozen import summarize
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,a):p.write_text(json.dumps(a,indent=2,allow_nan=False))
def main():
 lock=(P/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (P/'COMPLETE.json').exists()
 for f,h in read(P/'FROZEN.json').items():assert sha(Path(f))==h
 sys.path.insert(0,str(DOMAIN));from sweep import budget
 save(P/'PREFLIGHT.json',budget(read(DOMAIN/'PLAN.json'),True))
 for server,n in [('P100',119),('A100',35)]:
  for prefix in ['COMPLETE_','FEATURES_COMPLETE_']:assert read(PUBLIC/(prefix+server+'.json'))['records']==n
 relay=read(PUBLIC/'A100_RELAY_RECEIPT.json');assert relay['verified_all_files'] and relay['feature_files']==70
 for f,h in relay['files'].items():assert sha(PUBLIC/f)==h
 refs=read(PUBLIC/'REFERENCE_COMPACT.json');assert sha(ROOT/'reports/round1_final_20261105/revision_02/frozen_data/FROZEN_WINDOWS.json')==refs['source_sha256']
 inputs={};v4={}
 for fold in range(5):
  run=ROOT/f'reports/round1_v4_20261105/v4_repaired/fold{fold}/seed2027/joint';s=read(run/'status.json');assert s['state']=='complete' and s['completed_epochs']==30 and sha(run/'final.pt')==s['checkpoint_sha256'];inputs[str(run/'final.pt')]=s['checkpoint_sha256']
  for ds in ['MCD','UBFC2']:
   for cond in ['native','f10_b500']:
    ev=run/'evaluation'/(ds+'_'+cond);reuse=read(ev/'REFERENCE_REUSE.json');base=ROOT/f't8_polarity_v3_20261102/fold{fold}/seed2027/joint/evaluation/{ds}_{cond}'
    assert Path(reuse['path'])==base/'reference_signal.csv' and sha(Path(reuse['path']))==reuse['sha256']
    assert Path(reuse['raw_path'])==base/'raw_reference_signal.csv' and sha(Path(reuse['raw_path']))==reuse['raw_sha256']
    assert read(ev/'metrics.json')['reference_sha256']==reuse['sha256']
    f=ev/'processed_output.csv';inputs[str(f)]=sha(f)
    for r in csv.DictReader(f.open()):
     key=ds,cond,r['subject_id'],r['recording_id'],float(r['window_start_s']);assert key not in v4
     v4[key]=float(r['hr_bpm']) if r['hr_bpm'] and r['abstain'].lower()=='false' else None
 rows=[];coverage=[];snrs=[]
 for item in refs['records']:
  ds,cond,video=item['dataset'],item['variant'],item['recording_id'];f=PUBLIC/'features'/ds/(video+'_'+cond+'.json.gz');inputs[str(f)]=sha(f)
  with gzip.open(f,'rt') as z:a=json.load(z)
  assert a['reference_sha256']==sha(PUBLIC/'REFERENCE_COMPACT.json') and a['protocol_sha256']==sha(PUBLIC/'PROTOCOL.md')
  receipt=read(PUBLIC/'traces'/ds/video/cond/'PUBLIC_COMPLETE.json') if ds=='MCD' else None
  if receipt:
   assert a['trace_sha256']==receipt['trace_sha256'] and a['clock_sha256']==receipt['clock_sha256']
   for name,field in [('trace.npz','trace_sha256'),('clock.npz','clock_sha256')]:assert sha(PUBLIC/'traces'/ds/video/cond/name)==receipt[field]
  by={float(r['start']):r for r in a['rows']};expected={float(r['window_start_s']) for r in item['windows'] if r['reference_eligible']};assert set(by)==expected
  coverage.append(dict(dataset=ds,condition=cond,video=video,all_windows=len(item['windows']),reference_eligible=len(expected),new_median3_accepted=0))
  for b in item['windows']:
   if not b['reference_eligible']:continue
   r=by[float(b['window_start_s'])];assert r['truth']==b['reference_hr_bpm'] and r['person']==b['subject_id'] and r['fold']==b['fold']
   common={k:r[k] for k in ['recording','video','person','dataset','fold','condition','truth','start']};new={}
   for c in r['candidates']:
    if c['method']=='POS':
     assert c['roi'] not in new;peak=c['peaks'][0];new[c['roi']]=peak
     rows.append(dict(**common,method='new_POS_'+c['roi'],prediction=peak['hr']))
   if len(new)==3:
    rows.append(dict(**common,method='new_POS_median3',prediction=float(np.median([p['hr'] for p in new.values()]))));coverage[-1]['new_median3_accepted']+=1
    snrs.append(dict(**common,snr_db=float(np.median([p['snr_db'] for p in new.values()]))))
   for method,pred in b['baseline'].items():
    if pred['accepted']:rows.append(dict(**common,method=method+'_seed2027' if method in ['v3_joint','ubfc_only'] else 'old_'+method,prediction=pred['prediction_hr_bpm']))
   value=v4[ds,cond,r['person'],video,float(r['start'])]
   if value is not None:rows.append(dict(**common,method='V4_original_padding_seed2027',prediction=value))
 stats=[];sqi=[]
 for ds in ['MCD','UBFC2']:
  for cond in ['native','f10_b500']:
   rr=[r for r in rows if r['dataset']==ds and r['condition']==cond];methods=sorted({r['method'] for r in rr});keys=set.intersection(*({r['recording'] for r in rr if r['method']==m} for m in methods))
   for scope in ['available','matched_all_methods']:
    ss=[r for r in snrs if r['dataset']==ds and r['condition']==cond and (scope=='available' or r['recording'] in keys)];values=[r['snr_db'] for r in ss]
    sqi.append(dict(dataset=ds,condition=cond,scope=scope,windows=len(values),people=len({r['person'] for r in ss}),quantile_levels=[0,.25,.5,.75,1],snr_db_quantiles=np.quantile(values,[0,.25,.5,.75,1]).tolist() if values else None))
    for m in methods:
     selected=[r for r in rr if r['method']==m and (scope=='available' or r['recording'] in keys)]
     z=summarize(selected);z['windows']=z.pop('videos');z['recordings']=len({r['video'] for r in selected});z['reference_eligible_windows']=sum(c['reference_eligible'] for c in coverage if c['dataset']==ds and c['condition']==cond)
     if scope=='matched_all_methods':
      old=[r for r in rr if r['method']=='old_POS' and r['recording'] in keys];delta=summarize(selected,old);delta['windows']=delta.pop('videos');z['delta_old_POS']=delta
     stats.append(dict(dataset=ds,condition=cond,scope=scope,method=m,**z))
 tests=[r for r in stats if r['dataset']=='MCD' and r['condition']=='native' and r['method']=='new_POS_median3'];snr=next(r for r in sqi if r['dataset']=='MCD' and r['condition']=='native' and r['scope']=='available')
 gate=dict(new_MCD_MAE_by_scope={r['scope']:r['MAE'] for r in tests},blind_SNR_median=snr['snr_db_quantiles'][2],passed=all(r['MAE']<=6 for r in tests) and snr['snr_db_quantiles'][2]>0,rule='suspend if MAE>6 or blind median SNR<=0; no outcome retuning',automatic_promotion=False)
 for name,a in [('METRICS.json',stats),('SNR.json',sqi),('COVERAGE.json',coverage),('GATE.json',gate),('INPUT_HASHES.json',inputs)]:save(P/name,a)
 for name,rr in [('PREDICTIONS.csv',rows),('BLIND_SNR.csv',snrs)]:
  with (P/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 save(P/'COMPLETE.json',dict(fits=0,features=308,reference_eligible_windows=sum(x['reference_eligible'] for x in coverage),holdout_evaluations=0,no_new_inference=True));print(json.dumps(gate),flush=True)
if __name__=='__main__':main()
