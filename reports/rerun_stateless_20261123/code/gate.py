"""Mandatory exact tests for both modes; STOP receipt on any mismatch."""
from expert_spectrum import *
import traceback
name=sys.argv[1];started=time.monotonic()
try:
 mode=selected_mode();verify_weights();keys=json.loads((OUT/'BTC_KEYS.json').read_text())
 # Only read z17 after the authoritative decision has been locked.
 refs={}
 c=json.loads((NC/'output/validation/reset_state'/(name+'_COMPLETE.json')).read_text())
 p=NC/'output/validation/reset_state'/c.get('result_file',name+'.npz')
 with np.load(p) as z:assert list(z['keys'])==keys;refs['reset']=z['S'].copy()
 warm=np.full((218,277),np.nan);coverage=np.zeros(218,int);source_hashes={str(p):sha(p)}
 for p in sorted((W/'out/z17').glob(name+'_shard*.npz')):
  with np.load(p) as z:
   assert list(z['keys'])==keys;m=np.isfinite(z['S']).all(1);coverage+=m;warm[m]=z['S'][m]
  source_hashes[str(p)]=sha(p)
 assert np.all(coverage==1) and np.isfinite(warm).all();refs['selfwarm']=warm
 refs_path=OUT/'native'/(name+'.npz')
 np.savez_compressed(refs_path,S=refs[mode],keys=np.array(keys))
 expert=Expert(name);rows=[]
 pre=OUT/'gates'/(name+'_RESET_PRE_PASS.json')
 modes=['reset','selfwarm']
 if pre.exists():
  d=json.loads(pre.read_text());assert d['common_code_sha256']==sha(OUT/'code/expert_spectrum.py')
  assert all(sha(Path(p))==h for p,h in d['reference_sha256'].items())
  rows+=d['checks'];modes=['selfwarm']
 ids=[keys.index('BTC_S_AB3_Resting'),keys.index('BTC_S_AB39_Resting')]
 for md in modes:
  spectra=[]
  for i in [ids[0],ids[1],ids[0]]:
   with np.load(W/'out/x10'/(keys[i]+'.npz')) as z:s,sec,n=btc_spectrum(expert,z['crop72'],z['t'],z['valid'].astype(bool),md)
   assert s is not None and np.isfinite(s).all()
   delta=float(np.max(np.abs(s-refs[md][i])));assert delta==0.0,(name,md,keys[i],delta)
   spectra.append(s);rows.append(dict(mode=md,key=keys[i],max_reference_delta=delta,windows=n,seconds=sec))
  delta=float(np.max(np.abs(spectra[0]-spectra[2])));assert delta==0.0,(name,md,'A-B-A',delta)
 save(OUT/'gates'/(name+'_PASS.json'),dict(model=name,selected_mode=mode,checks=rows,order_delta_both_modes=0.0,seconds=time.monotonic()-started,reference_sha256=source_hashes,native_sha256=sha(refs_path)))
 print(name,'all exact parity/order gates PASS',flush=True)
except BaseException as e:
 save(OUT/'gates'/(name+'_ERROR.json'),dict(error=repr(e),traceback=traceback.format_exc()));save(OUT/'STOP.json',dict(stage='gate',model=name,error=repr(e)));raise
