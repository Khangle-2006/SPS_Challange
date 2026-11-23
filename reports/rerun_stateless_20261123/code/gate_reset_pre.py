"""Independent step 1(a)/reset-order control while authoritative z18 is pending."""
from expert_spectrum import *
import traceback
name=sys.argv[1];started=time.monotonic()
try:
 verify_weights();keys=json.loads((OUT/'BTC_KEYS.json').read_text());root=NC/'output/validation/reset_state'
 c=json.loads((root/(name+'_COMPLETE.json')).read_text());p=root/c.get('result_file',name+'.npz')
 with np.load(p) as z:assert list(z['keys'])==keys;ref=z['S'].copy()
 expert=Expert(name);ids=[keys.index('BTC_S_AB3_Resting'),keys.index('BTC_S_AB39_Resting')];spectra=[];rows=[]
 for i in [ids[0],ids[1],ids[0]]:
  with np.load(W/'out/x10'/(keys[i]+'.npz')) as z:s,sec,n=btc_spectrum(expert,z['crop72'],z['t'],z['valid'].astype(bool),'reset')
  assert s is not None and np.isfinite(s).all();delta=float(np.max(abs(s-ref[i])));assert delta==0.0,(name,keys[i],delta)
  rows.append(dict(mode='reset',key=keys[i],max_reference_delta=delta,windows=n,seconds=sec));spectra.append(s)
 assert np.array_equal(spectra[0],spectra[2]);save(OUT/'gates'/(name+'_RESET_PRE_PASS.json'),dict(model=name,checks=rows,order_delta=0.0,reference_sha256={str(p):sha(p)},seconds=time.monotonic()-started,common_code_sha256=sha(OUT/'code/expert_spectrum.py')))
 print(name,'RESET EXACT + ORDER PASS',flush=True)
except BaseException as e:
 save(OUT/'gates'/(name+'_RESET_PRE_ERROR.json'),dict(error=repr(e),traceback=traceback.format_exc()));save(OUT/'STOP.json',dict(stage='reset_gate',model=name,error=repr(e)));raise
