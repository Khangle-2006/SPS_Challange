import json,csv
from pathlib import Path
import numpy as np
from mcd_errata_audit import timed,nearest
ROOT=Path('/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG')
AUDIT=ROOT.parent.parent/'mapping_audit_20261030'
db={r['video']:r for r in csv.DictReader((ROOT/'db.csv').open())}
out=[]
for d in json.loads((AUDIT/'combined_diagnostics.json').read_text()):
    r=db[d['video']];_,mt=timed(ROOT/r['meta']);pv,pt=timed(ROOT/r['ppg'])
    sy=np.loadtxt(ROOT/r['ppg_sync'],ndmin=2)
    values,delta=nearest(mt[:len(sy)],pt,pv)
    out.append({'video':r['video'],'sync_rows':len(sy),'metadata_rows':len(mt),'decoded_frames':d['frames_default'],
                'nearest_prefix_value_exact_fraction':float(np.mean(values==sy[:,0])),
                'nearest_prefix_distance_match_1microsecond_fraction':float(np.mean(abs(delta-sy[:,1])<=1.000001e-6)),
                'max_distance_discrepancy_s':float(np.max(abs(delta-sy[:,1]))),
                'raw_ppg_timestamp_nonpositive_intervals':int(np.sum(np.diff(pt)<=0))})
(AUDIT/'curated_v3/sync_prefix_values36.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'n':len(out),'length_equals_decode':sum(x['sync_rows']==x['decoded_frames'] for x in out),
 'all_values_exact':sum(x['nearest_prefix_value_exact_fraction']==1 for x in out),
 'min_value_match_fraction':min(x['nearest_prefix_value_exact_fraction'] for x in out),
 'min_distance_match_fraction':min(x['nearest_prefix_distance_match_1microsecond_fraction'] for x in out)},indent=2))
