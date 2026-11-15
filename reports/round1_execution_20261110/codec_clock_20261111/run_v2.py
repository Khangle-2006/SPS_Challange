"""Operational fix: unsupported low-rate CHROM is a recorded diagnostic failure.
Frozen 30Hz primary readout and all encoding/selection parameters unchanged.
"""
from audit_clock import *
import audit_clock as base

def metrics_v2(traces,clock,fs,d):
 rows=[];arrays={};qall=np.arange(len(clock))/fs
 for roi,rgb in traces.items():
  for wi,start in enumerate([0,10,20]):
   use=(qall>=start)&(qall<start+10)
   for mode,t in [('sampled_clock',qall),('source_clock',np.asarray(clock))]:
    sl=(t>=start-.2)&(t<=start+10+.2);q=start+np.arange(300)/30;reason=None
    if t[0]>q[0]+1e-8 or t[-1]<q[-1]-1e-8 or np.any(~np.isfinite(rgb[sl])) or np.max(np.diff(t[sl]))>.2+1e-8:reason='invalid_support'
    x=None if reason else np.stack([np.interp(q,t,rgb[:,c]) for c in range(3)],axis=1)
    for method in ['G','POS','CHROM']:
     row=dict(roi=roi,window=wi,start=start,clock=mode,method=method,prediction=None,reason=reason)
     if x is not None:
      w=-x[:,1] if method=='G' else NS[method.lower()](x,30)
      try:row['prediction']=decode_wave(w)
      except ValueError as e:row['reason']=str(e)
      f,p,stats=spectrum(w,30);row.update(stats);key=f'{roi}_{wi}_{mode}_{method}';arrays[key+'_f']=f;arrays[key+'_p']=p;arrays[key+'_w']=w
     rows.append(row)
   for method in ['G','POS','CHROM']:
    row=dict(roi=roi,window=wi,start=start,clock='native_rate',method=method,prediction=None,reason=None)
    try:
     if not np.isfinite(rgb[use]).all():raise ValueError('invalid_support')
     w=-rgb[use,1] if method=='G' else NS[method.lower()](rgb[use],fs)
     f,p,stats=spectrum(w,fs);row.update(prediction=stats['peak_bpm'],**stats);key=f'{roi}_{wi}_native_rate_{method}';arrays[key+'_f']=f;arrays[key+'_p']=p;arrays[key+'_w']=w
    except ValueError as e:row['reason']='native_rate_unsupported: '+str(e)
    rows.append(row)
 np.savez_compressed(d/'spectra.npz',**arrays)
 return rows

def test():
 import tempfile
 checks=[]
 for fs in [8,10]:
  t=np.arange(fs*30+1)/fs;rgb=np.array([180.,130.,110.])[None,:]+np.sin(2*np.pi*1.2*t)[:,None]*np.array([.2,1,.1])
  with tempfile.TemporaryDirectory(prefix='clock_cpu_',dir='/dev/shm') as d:
   rows=metrics_v2(dict(skin=rgb),t,fs,Path(d))
   primary=[r for r in rows if r['clock']=='source_clock'];assert len(primary)==9 and all(r['prediction'] is not None for r in primary)
   failed=[r for r in rows if r['clock']=='native_rate' and r['method']=='CHROM'];assert len(failed)==3 and all('padlen' in r['reason'] for r in failed)
   for r in primary:
    if r['method']=='POS':assert abs(r['prediction']-72)<1
   checks.append(dict(fs=fs,primary_finite=len(primary),native_CHROM_unsupported=len(failed)))
 save(P/'TEST_V2.json',dict(passed=True,checks=checks,code_sha=sha(__file__),primary_math_unchanged=True))

def launch():
 lease=(P/'launch_v2.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (P/'WORKERS_V2.json').exists()
 for w in read(P/'WORKERS.json')['workers']:assert not Path(f"/proc/{w['pid']}").exists(),'Old worker alive'
 assert not list((P/'real').glob('*/*/f*_b*/video.mp4')) and not list((P/'synthetic').glob('*/b*/video.mp4'))
 test();guard(True);archive=P/'ATTEMPT1';archive.mkdir()
 for name in ['real','synthetic']:
  source=P/name;target=archive/name
  assert source.resolve().parent==P.resolve() and target.resolve().parent==archive.resolve()
  source.rename(target)
 amendment=dict(utc=now(),reason='Native-rate CHROM uses 1.6s windows shorter than scipy default padlen21 at8/10fps. Preserve failure as unsupported; primary30Hz unchanged.',original_plan_sha=sha(P/'PLAN.json'),v2_sha=sha(__file__),original_script_sha=sha(P/'audit_clock.py'),no_encode_before_fix=True,workers=[])
 save(P/'AMENDMENT_V2.json',amendment)
 for mode,cpu in [('real',7),('synthetic',8)]:
  with (P/(mode+'_v2.log')).open('xb') as log:
   cmd=['nice','-n','15','taskset','-c',str(cpu),sys.executable,str(Path(__file__)),mode]
   q=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=P);amendment['workers'].append(dict(mode=mode,pid=q.pid,args=cmd))
 save(P/'WORKERS_V2.json',amendment);print(json.dumps(amendment))
if __name__=='__main__':
 try:
  mode=sys.argv[1]
  if mode=='launch':launch()
  elif mode=='test':test()
  else:
   lease=(P/(mode+'.lock')).open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
   a=read(P/'AMENDMENT_V2.json');assert sha(__file__)==a['v2_sha'] and sha(P/'PLAN.json')==a['original_plan_sha'] and base.binding()==read(P/'PLAN.json')['bindings']
   guard(True);base.metrics=metrics_v2
   if mode=='real':base.real(read(P/'PLAN.json'))
   elif mode=='synthetic':base.synth(read(P/'PLAN.json'))
   else:raise ValueError(mode)
 except Exception as e:save(P/(sys.argv[1]+'_V2_ERROR.json'),dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
