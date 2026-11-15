"""Crossed frozen operators and saved traces; no parameter fitting or label selection."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,ast,math,hashlib,csv,fcntl
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;PUBLIC=P.parent;ROOT=Path('/mnt/samsung_t5/khangle/facevitals')
sys.path.insert(0,str(PUBLIC));import pipeline_frozen as new
from clock_contract import align,window
sys.path.insert(0,'/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102')
from score_round1 import reference_hr
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,a):(P/n).write_text(json.dumps(a,indent=2,allow_nan=False))
src=ROOT/'reports/round1_final_20261105/revision_02/code/vendor/hr_baseline.py'
tree=ast.parse(src.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pos');ns=dict(np=np,math=math)
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(src),'exec'),ns);oldpos=ns['pos']
def oldhr(w):return reference_hr(dict(window_start_s=0.,ppg_timestamps_json=json.dumps((np.arange(301)/30).tolist()),ppg_values_json=json.dumps(np.r_[w,w[-1]].tolist())))
def newhr(w):return new.peak_info(*new.psd(w,30))[0]['hr']
def main():
 lock=(P/'audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (P/'COMPLETE.json').exists()
 rows=[];errors=[];parity=[];inputs={str(src):sha(src)}
 refs=read(PUBLIC/'REFERENCE_COMPACT.json')
 for item in refs['records']:
  if item['dataset']!='MCD' or item['variant']!='native':continue
  rid=item['recording_id'];d=PUBLIC/'traces/MCD'/rid/'native';receipt=read(d/'PUBLIC_COMPLETE.json')
  for n,k in [('trace.npz','trace_sha256'),('clock.npz','clock_sha256')]:assert sha(d/n)==receipt[k];inputs[str(d/n)]=receipt[k]
  with np.load(d/'trace.npz') as z:rgb=z['rgb'].copy();pixel=z['pixels'].copy();area=z['areas'].copy()
  with np.load(d/'clock.npz') as z:t,rgb=align(z['original_t'],z['canonical_t'],rgb,receipt['allowed_tail_drop'])
  spatial=ROOT/'reports/round1_final_20261105/revision_02/frozen_data/spatial_rgb/MCD'/rid/'native.npz';inputs[str(spatial)]=sha(spatial)
  with np.load(spatial) as z:ot=z['t'].copy();ox=z['rgb'].copy()
  for b in item['windows']:
   if not b['reference_eligible'] or not b['baseline']['POS']['accepted']:continue
   st=float(b['window_start_s']);q=st+np.arange(300)/30
   common=dict(video=rid,person=b['subject_id'],start=st,truth=b['reference_hr_bpm'])
   traces={'oval':np.stack([np.interp(q,ot,ox[:,c]) for c in range(3)],1)}
   try:
    tw,xw=window(t,rgb,st)
    for ri,name in enumerate(['forehead','left_cheek','right_cheek']):
     try:
      _,xx,fs=new.native_grid(tw,xw[:,ri]);good=np.isfinite(xw[:,ri]).all(1)
      traces[name]=np.stack([np.interp(q,tw[good],xw[good,ri,c]) for c in range(3)],1)
     except ValueError as e:errors.append(dict(**common,roi=name,error=str(e)))
   except (ValueError,AssertionError) as e:errors.append(dict(**common,error=str(e)))
   for roi,x in traces.items():
    waves={'old':oldpos(x,30),'new':new.pulse(x,30,'POS')}
    for operator,w in waves.items():
     for decoder,func in [('old',oldhr),('new',newhr)]:
      try:
       pred=func(w);rows.append(dict(**common,roi=roi,operator=operator,decoder=decoder,prediction=pred))
       if roi=='oval' and operator==decoder=='old':parity.append(abs(pred-b['baseline']['POS']['prediction_hr_bpm']))
      except (ValueError,IndexError) as e:errors.append(dict(**common,roi=roi,operator=operator,decoder=decoder,error=str(e)))
   save('STATUS.json',dict(recording=rid,start=st,rows=len(rows),pid=os.getpid()))
  print(rid,len(rows),flush=True)
 assert max(parity)<1e-8,('old parity',max(parity))
 keys=[(r['roi'],r['operator'],r['decoder']) for r in rows];methods=sorted(set(keys))
 sets=[{(r['video'],r['start']) for r in rows if (r['roi'],r['operator'],r['decoder'])==k} for k in methods];shared=set.intersection(*sets)
 metrics=[]
 for scope in ['available','matched']:
  for k in methods:
   rr=[r for r in rows if (r['roi'],r['operator'],r['decoder'])==k and (scope=='available' or (r['video'],r['start']) in shared)]
   people=sorted({r['person'] for r in rr});mae=np.mean([np.mean([abs(r['prediction']-r['truth']) for r in rr if r['person']==p]) for p in people])
   metrics.append(dict(scope=scope,roi=k[0],operator=k[1],decoder=k[2],windows=len(rr),people=len(people),MAE=float(mae)))
 with (P/'CROSSED.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save('METRICS.json',metrics);save('ERRORS.json',errors);save('INPUT_HASHES.json',inputs);save('COMPLETE.json',dict(old_parity_max=max(parity),old_parity_windows=len(parity),matched_windows=len(shared),holdout_evaluations=0))
 print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
