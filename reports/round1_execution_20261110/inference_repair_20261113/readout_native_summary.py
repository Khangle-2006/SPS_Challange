import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common_repair import *
from report_repair import stats
import numpy as np,csv
from collections import defaultdict

assert (P/'ORIGINAL_COMPLETE.json').exists()
saved=list(csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open()));cal=[dict(r,prediction=float(r['prediction']),truth=float(r['truth'])) for r in saved if r['condition']=='native' and r['method']=='calibrated_median_POS_CHROM_V3'];const=[dict(r,prediction=float(r['prediction']),truth=float(r['truth'])) for r in saved if r['condition']=='native' and r['method']=='constant'];videos=[]
for j in manifest():
 if j['dataset']!='BTC':continue
 root=P/'original'/j['key']/'native';receipt=read(root/'COMPLETE.json');assert sha(root/'PREDICTIONS.json')==receipt['rows_sha256'];buckets=defaultdict(list)
 for r in read(root/'PREDICTIONS.json'):buckets[('padding' if r['model'].startswith('padding') else 'V3',r['start'])].append(r)
 window=defaultdict(list)
 for (method,start),rows in buckets.items():
  for decoder in rows[0]['readouts']:
   values=[r['readouts'][decoder] for r in rows]
   if all(v is not None for v in values):window[method,decoder].append(float(np.median(values)))
 for (method,decoder),values in window.items():videos.append(dict(recording=j['recording'],person=j['person'],truth=j['windows'][0]['truth'],prediction=float(np.median(values)),method=method,readout=decoder))
metrics=[]
for method,decoder in sorted({(r['method'],r['readout']) for r in videos}):
 rr=[r for r in videos if (r['method'],r['readout'])==(method,decoder)];metrics.append(dict(method=method,readout=decoder,**stats(rr),delta_saved_calibrated=stats(rr,cal),delta_constant=stats(rr,const)))
save(P/'READOUT_NATIVE_COMPLETE.json',dict(utc=now(),spatial_complete=False,metrics=metrics,constant=stats(const),saved_calibrated=stats(cal),no_configuration_selection=True,static_failure_V3_preserved=True))
print([{k:r[k] for k in ['method','readout','videos','people','MAE','bias']} for r in metrics])
