from vis_common import *
import numpy as np
assert read(P/'MODEL_COMPLETE.json')['errors']==[]
rows=manifest();features=[];labels=[];mass=[];areas=[];checks=[]
for r in rows:
 d=P/'model'/r['key'];receipt=read(d/'COMPLETE.json');assert sha(d/'outputs.npz')==receipt['outputs_sha256'];a=np.load(d/'outputs.npz');checks.append(receipt['finite_difference'])
 for i,x in enumerate(a['features_common']):features.append(x);labels.append(dict(recording=r['recording'],person=r['person'],domain=r['domain'],selection_group=r['group'],window_start=float(a['starts'][i])))
 for method,v in receipt['regional'].items():mass.append(dict(group=r['group'],recording=r['recording'],method=method,**v,non_skin=v['oval_edge']+v['eyes_mouth']+v['exterior']))
 if receipt['region_area'] is not None:areas.append(dict(group=r['group'],recording=r['recording'],area=receipt['region_area']))
np.savez_compressed(P/'COMMON_FEATURES.npz',features=features);save(P/'FEATURE_LABELS.json',labels)
summary=[];triggers=[]
for group in sorted({r['group'] for r in rows}):
 for method in ['gradient','occlusion']:
  rr=[r for r in mass if r['group']==group and r['method']==method];ar=[r['area'] for r in areas if r['group']==group]
  summary.append(dict(group=group,method=method,n=len(rr),means={k:float(np.mean([r[k] for r in rr])) for k in ['skin_proxy','oval_edge','eyes_mouth','exterior','non_skin']},mean_region_area=np.mean(ar,axis=0).tolist()))
for method in ['gradient','occlusion']:
 ub=[r['non_skin'] for r in mass if r['group']=='UBFC' and r['method']==method];btc=[r['non_skin'] for r in mass if r['group'].startswith('BTC') and r['method']==method];triggers.append(dict(method=method,UBFC_mean=float(np.mean(ub)),BTC_mean=float(np.mean(btc)),UBFC_n=len(ub),BTC_n=len(btc),trigger=bool(np.mean(btc)>.4 and np.mean(ub)<.2)))
save(P/'ATTRIBUTION_SUMMARY.json',dict(utc=now(),groups=summary,trigger=triggers,static_passed=len(rows),finite_difference_max_relative_error=max(x['relative_error'] for x in checks if x),n_feature_windows=len(labels),feature_sha256=sha(P/'COMMON_FEATURES.npz'),interpretation='Landmark skin proxy, not segmentation; diagnostic association only'))
print(json.dumps(dict(trigger=triggers,n=len(rows),features=len(labels))))
