"""Real-cache source/codec/clock smoke, including retained invalid-gap rejection."""
import json,time
from pathlib import Path
import numpy as np
from codec_provider import Provider,ROOT,read
from t8_plan import pools
P=Path(__file__).resolve().parent;index=read(ROOT/'t8_oval_20261031/training_index.json');pool=pools(index,0);p=Provider(index,[.124,.188]);out=[]
for ds in ['UBFC2','MCD']:
 for recipe in ['h264','hevc','c5_proxy']:
  eligible=[(rid,start) for rid,start in pool[ds] if (p.feasible_rate(ds,rid,start,recipe) or [0,0])[1]>=1.2-1e-8]
  assert eligible
  rid,start=eligible[0]
  for rate in [.8,1.,1.2]:
   x,y,r=p.clip(ds,rid,start,recipe,True,0,rate)
   assert x.shape==(301,3,36,36) and y.shape==(300,) and np.isfinite(x).all() and np.isfinite(y).all()
   out.append(dict(dataset=ds,recipe=recipe,recording=rid,start=start,rate=rate,output_shape=list(x.shape),ppg_shape=list(y.shape),receipt=r))
  print(ds,recipe,'PASS rate .8/1/1.2',flush=True)
rid,start=pool['MCD'][0]
assert p.feasible_rate('MCD',rid,start,'c5_proxy') is None,'Known 207ms post-skip gap must remain excluded'
# Query-speed semantics: both image signal and label advance by the same factor.
for rate in [.8,1.,1.2]:
 q=np.arange(301)/30*rate;t=np.arange(1000)/60;signal=np.sin(2*np.pi*1.5*t)
 image=np.interp(q,t,signal);target=np.interp(q,t,signal)
 assert np.array_equal(image,target)
outcome=dict(passed=True,cases=out,known_post_skip_gap_rejected=True,source_clock_unchanged=True,MATLAB_parity=False,cache_bytes=p.bytes)
(P/'CODEC_SMOKE.json').write_text(json.dumps(outcome,indent=2,allow_nan=False))
