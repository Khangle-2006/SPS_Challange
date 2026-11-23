"""Run path-only z13, then serialize predictions/fits without changing any fitting logic."""
from expert_spectrum import *
import traceback
started=time.monotonic()
try:
 selected_mode();assert (OUT/'inference_COMPLETE.json').exists();assert not (OUT/'STOP.json').exists()
 original_out=OUT
 # exec in the actual __main__ namespace keeps unchanged multiprocessing fit picklable.
 src=OUT/'code/z13_public_p4_prefilter.py';exec(compile(src.read_text(),str(src),'exec'),globals())
 def serial(x):
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
  if isinstance(x,(tuple,list)):return [serial(v) for v in x]
  return x
 save(original_out/'PUBLIC_FITS.json',serial(FITS));payload={}
 for c,d in D_.items():
  for k in FIELDS:payload[c+'__'+k]=d[k]
  for nm in READERS:
   pr,cst=apply(d,nm);payload[c+'__'+nm]=pr;payload[c+'__constant_'+nm]=cst
 np.savez_compressed(original_out/'PUBLIC_PREDICTIONS_PRIVATE.npz',**payload)
 save(original_out/'PUBLIC_SCORE_COMPLETE.json',dict(seconds=time.monotonic()-started,conditions=list(D_),readers=list(READERS)))
except BaseException as e:
 save(OUT/'PUBLIC_SCORE_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));save(OUT/'STOP.json',dict(stage='public_scoring',error=repr(e)));raise
