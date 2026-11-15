"""Diagnose jitter rate failure; validate every planned C5 selection before launch."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,math,copy
from pathlib import Path
import numpy as np
from codec_provider import Provider,c5_selection,ROOT,read,sha
P=Path(__file__).resolve().parent;OLD=P.with_name('domain_sweep_20261109')
def save(n,x):(P/n).write_text(json.dumps(x,indent=2,allow_nan=False))

def main():
 assert not (P/'PLAN.json').exists(), 'Never overwrite inspected plan'
 old=read(OLD/'PLAN.json');index=read(ROOT/'t8_oval_20261031/training_index.json');p=Provider(index,old['runs'][0]['btc_bpp_quartiles'])
 # Pure clock regression, without a label or model output.
 t=np.r_[np.arange(301)/30*.999,10.05];initial=max(1,math.ceil((len(t)-1)/(t[-1]-t[0])/10));tt=t[::initial]
 assert (len(tt)-1)/(tt[-1]-tt[0])>10
 sel=c5_selection(t);tt=t[sel];assert (len(tt)-1)/(tt[-1]-tt[0])<=10
 assert np.array_equal(c5_selection(np.arange(301)/30),np.arange(0,301,3))
 # Reproduce exact first five batch draws of failed fold0, without decoding.
 rng=np.random.default_rng(np.random.SeedSequence([2027,0,1]));failed=[]
 for step in range(5):
  for ds in ['UBFC2','MCD']*4:
   options=old['runs'][0]['public_jobs'][ds];j=options[int(rng.integers(len(options)))];rate=float(rng.uniform(*j['rates']))
   if j['recipe']!='c5_proxy':continue
   raw,_=p.metadata(ds,j['recording']);t=raw['original_t'];st=j['start'];lo=max(0,np.searchsorted(t,st,side='right')-1);hi=min(len(t)-1,np.searchsorted(t,min(st+12.01,t[-1])));ts=t[lo:hi+1];fs=(len(ts)-1)/(ts[-1]-ts[0]);stride=max(1,math.ceil(fs/10));tt=ts[::stride];actual=float((len(tt)-1)/(tt[-1]-tt[0]))
   if actual>10+1e-8:failed.append(dict(step=step+1,job=j,rate=rate,before_fps=float(fs),old_stride=stride,old_after_fps=actual,new_stride=int(c5_selection(ts)[1])))
 assert failed and failed[0]['step']==5
 save('FAILURE_REPRO.json',failed);print('REPRO',json.dumps(failed),flush=True)
 plan=copy.deepcopy(old);changes=[];checks=[]
 for run in plan['runs']:
  provider=Provider(index,run['btc_bpp_quartiles'])
  for ds,jobs in list(run['public_jobs'].items()):
   retained=[]
   for j in jobs:
    if j['recipe']!='c5_proxy':retained.append(j);continue
    rates=provider.feasible_rate(ds,j['recording'],j['start'],j['recipe'])
    ix,t,v,_=provider.source_selection(ds,j['recording'],j['start'],j['recipe']);fps=float((len(t)-1)/(t[-1]-t[0]));assert fps<=10+1e-8
    checks.append(dict(fold=run['fold'],dataset=ds,recording=j['recording'],start=j['start'],fps=fps,selected_frame_indices=ix.tolist(),rates=rates))
    if rates!=j['rates']:changes.append(dict(fold=run['fold'],job=j,old_rates=j['rates'],new_rates=rates))
    if rates is None:run['public_exclusions'].append(dict(**j,reason='Corrected post-selection <=10fps clock has no valid minimum8s support; unchanged200ms gap guard'))
    else:retained.append(dict(j,rates=rates))
   run['public_jobs'][ds]=retained;assert retained
  # Keep original number of updates, seed, batch and all other training settings.
  assert run['steps_per_epoch']==old['runs'][run['fold']]['steps_per_epoch']
  del provider
 plan['operational_amendment']=dict(reason='Verify post-skip rate on selected original timestamps',original_plan_sha256=sha(OLD/'PLAN.json'),original_attempt='failed at step5, no checkpoint; restart same initial weights',no_metric_selection=True)
 save('PLAN.json',plan);save('C5_SELECTION_CHECKS.json',checks);save('PLAN_CHANGES.json',changes)
 # Decode/encode the exact previously failing clip once; exercise all feasible query rates.
 j=failed[0]['job'];run=plan['runs'][0];j2=next(x for x in run['public_jobs'][j['dataset']] if x['recording']==j['recording'] and x['start']==j['start'])
 rates=j2['rates'];cases=[]
 for rate in sorted(set([rates[0],float(np.mean(rates)),rates[1]])):
  x,y,receipt=p.clip(j['dataset'],j['recording'],j['start'],j['recipe'],j['lowres'],0,rate)
  assert x.shape==(301,3,36,36) and y.shape==(300,) and np.isfinite(x).all() and np.isfinite(y).all()
  assert receipt['fs']<=10+1e-8 and receipt['whole_file_kbps']<=500
  cases.append(dict(rate=rate,receipt=receipt))
 # Source/codec validation for unchanged paths is inherited explicitly, not claimed rerun.
 save('CODEC_SMOKE.json',dict(passed=True,cases=cases,all_planned_C5_checks=len(checks),max_C5_source_fps=max(z['fps'] for z in checks),inherited_unchanged_h264_hevc_smoke_sha256=sha(OLD/'CODEC_SMOKE.json'),changed_paths_tested='C5 selection on every retained/planned window plus real failed clip',MATLAB_parity=False))
 prior=read(OLD/'CPU_VALIDATION.json');assert prior['passed'] and not prior['CUDA_initialized']
 prior.update(plan_sha256=sha(P/'PLAN.json'),validation_basis=dict(inherited_unchanged_model_loss_and_h264_B8=sha(OLD/'CPU_VALIDATION.json'),changed_C5_path_real_cache_sha256=sha(P/'CODEC_SMOKE.json'),training_updates_unchanged=True,initial_checkpoints_unchanged=True))
 save('CPU_VALIDATION.json',prior)
 print('REPAIR_PASS',json.dumps(dict(C5_windows=len(checks),changes=len(changes),cases=len(cases),max_fps=max(z['fps'] for z in checks))),flush=True)
if __name__=='__main__':main()
