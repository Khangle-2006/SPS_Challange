import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common_repair import *
from report_repair import stats
from collections import defaultdict
assert (P/'ORIGINAL_COMPLETE.json').exists()
groups=defaultdict(list)
for j in manifest():
 if j['dataset']=='BTC':continue
 for condition in ['native','c5','c5prime_default','c5prime_bf0']:
  root=P/'original'/j['key']/condition
  if (root/'EXCLUDED.json').exists():continue
  cp=read(root/'COMPLETE.json');assert sha(root/'PREDICTIONS.json')==cp['rows_sha256']
  for r in read(root/'PREDICTIONS.json'):
   if r['truth'] is None:continue
   method='padding' if r['model'].startswith('padding') else 'V3'
   for decoder,value in r['readouts'].items():
    if value is not None:groups[j['dataset'],condition,method,decoder].append(dict(recording=r['recording']+'@'+str(r['start']),person=r['person'],truth=r['truth'],prediction=value))
metrics=[]
for (ds,condition,method,decoder),rows in sorted(groups.items()):
 baseline=groups[ds,condition,method,'power_0.7'];paired={r['recording'] for r in baseline};rows=[r for r in rows if r['recording'] in paired];a=stats(rows);a['matched_windows']=a.pop('videos');delta=stats(rows,baseline);delta['matched_windows']=delta.pop('videos');metrics.append(dict(dataset=ds,condition=condition,method=method,readout=decoder,**a,delta_original=delta,reference='frozen_0.7'))
save(P/'READOUT_PUBLIC_COMPLETE.json',dict(utc=now(),metrics=metrics,spatial_complete=False,public_reference_unchanged=True,no_configuration_selection=True))
print([{k:r[k] for k in ['dataset','condition','readout','people','matched_windows','MAE','bias']} for r in metrics if r['method']=='padding' and r['readout'] in ['power_0.7','power_0.6','prominence_0.6','tilt_0.6']])
