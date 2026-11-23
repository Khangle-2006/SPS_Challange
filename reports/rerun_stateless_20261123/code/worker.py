"""One frozen model/condition task; compare output support against old run, never silently drop rows."""
from expert_spectrum import *
import traceback
taskid=sys.argv[1];plan=json.loads((OUT/'PLAN.json').read_text());task=next(t for t in plan['tasks'] if t['id']==taskid)
started=time.monotonic();name=task['model'];cond=task['condition'];receipt=OUT/'receipts'/(taskid+'.json')
try:
 assert not receipt.exists(),'Task already finished; do not rerun'
 mode=selected_mode();verify_weights()
 assert all((OUT/'gates'/(n+'_PASS.json')).exists() for n in MODELS),'Parity gate missing'
 keys=json.loads((OUT/'BTC_KEYS.json').read_text());public={j['key']:j for j in json.loads((OUT/'PUBLIC_DEV_MANIFEST.json').read_text())}
 def progress(n,total):save(OUT/'status'/(taskid+'.json'),dict(task=taskid,completed=n,total=total,seconds=time.monotonic()-started))
 def guard():
  if (OUT/'STOP.json').exists():raise RuntimeError('Global STOP receipt exists')
  rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
  if rss>8*1024**3:raise RuntimeError('Own RSS exceeds 8 GiB')
  while int(dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())['MemAvailable'].split()[0])*1024<15*1024**3:
   if (OUT/'STOP.json').exists():raise RuntimeError('Global STOP while waiting for RAM')
   time.sleep(10)
 if task['kind']=='btc':
  old=Path(task['old']);assert sha(old)==task['old_sha256'];S=np.full((218,277),np.nan);secs=np.zeros(218);done=np.zeros(218,bool)
  dest=OUT/'btc_conditions'/name/(cond+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
  if task['reuse_native']:
   with np.load(OUT/'native'/(name+'.npz')) as z:S[task['ids']]=z['S'][task['ids']]
   done[task['ids']]=True;np.savez_compressed(dest,S=S,keys=np.array(keys),seconds=secs,done=done)
  else:
   expert=Expert(name)
   for n,i in enumerate(task['ids']):
    guard();p=W/('out/x10' if cond=='split10' else 'out/x15/'+cond)/(keys[i]+'.npz')
    with np.load(p) as z:s,sec,nwin=btc_spectrum(expert,z['crop72'],z['t'],z['valid'].astype(bool),mode,split=cond=='split10',legacy_operator=True)
    done[i]=True;secs[i]=sec
    if s is not None:S[i]=s
    with np.load(old) as z:expected=bool(np.isfinite(z['S'][i]).all())
    actual=bool(np.isfinite(S[i]).all())
    tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,S=S,keys=np.array(keys),seconds=secs,done=done);tmp.replace(dest)
    progress(n+1,len(task['ids']));print(taskid,n+1,len(task['ids']),keys[i],actual,round(sec,2),flush=True)
    assert expected==actual,dict(stop='BTC output coverage changed',key=keys[i],old=expected,new=actual)
  with np.load(old) as z:assert np.array_equal(np.isfinite(S).all(1),np.isfinite(z['S']).all(1))
  data=dict(videos_with_output=int(np.isfinite(S).all(1).sum()),videos_processed=int(done.sum()),reuse_native=task['reuse_native'],sha256=sha(dest))
 else:
  expert=Expert(name);base=OUT/('public_native' if cond=='native' else 'public_cond/'+cond)/name;base.mkdir(parents=True,exist_ok=True);counts={};files={}
  for n,key in enumerate(task['keys']):
   guard();j=public[key];old=Path(task['old_base'])/(key+'.npz');assert sha(old)==task['old_receipts'][key]['sha256']
   p=W/('out/x10' if cond in ('native','c5_split10') else 'out/x16/'+cond)/(key+'.npz');dest=base/(key+'.npz');assert not dest.exists()
   with np.load(p) as z:d=public_spectrum(expert,z['crop72'],z['t'],z['valid'].astype(bool),j,mode,cond)
   tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,**d);tmp.replace(dest)
   with np.load(old) as z:
    assert np.array_equal(d['start'],z['start']) and np.array_equal(d['truth'],z['truth']),(key,'window clock/truth changed')
    a=np.isfinite(d['spec']).all(1);b=np.isfinite(z['spec']).all(1)
    assert np.array_equal(a,b),dict(stop='public window coverage changed',key=key,old=int(b.sum()),new=int(a.sum()),differing_starts=d['start'][a!=b].tolist())
   counts[j['dataset']]=counts.get(j['dataset'],0)+int(a.sum());files[key]=sha(dest)
   progress(n+1,len(task['keys']));print(taskid,n+1,len(task['keys']),key,int(a.sum()),round(d['seconds'],2),flush=True)
  assert counts==task['expected_windows'],(counts,task['expected_windows'])
  data=dict(recordings=len(task['keys']),windows=counts,file_sha256=files)
 save(receipt,dict(task=taskid,mode=mode,seconds=time.monotonic()-started,coverage_exact=True,**data))
 print(taskid,'COMPLETE',flush=True)
except BaseException as e:
 save(OUT/'status'/(taskid+'_ERROR.json'),dict(error=repr(e),traceback=traceback.format_exc()))
 if not (OUT/'STOP.json').exists():save(OUT/'STOP.json',dict(stage='inference',task=taskid,error=repr(e)))
 raise
