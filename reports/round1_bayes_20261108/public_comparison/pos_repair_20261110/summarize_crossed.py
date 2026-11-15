import csv,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent));from metrics_frozen import summarize
raw=list(csv.DictReader((P/'CROSSED.csv').open()))
for r in raw:
 for k in ['start','truth','prediction']:r[k]=float(r[k])
 r['recording']=r['video']+'@'+str(r['start'])
by=defaultdict(list)
for r in raw:by[r['roi'],r['operator'],r['decoder']].append(r)
shared=set.intersection(*({r['recording'] for r in rr} for rr in by.values()))
methods={}
for (roi,op,de),rr in by.items():
 if roi=='oval':methods[f'oval_POS{op}_decoder{de}']=[r for r in rr if r['recording'] in shared]
for op in ['new','old']:
 for de in ['new','old']:
  rois=[{r['recording']:r for r in by[roi,op,de]} for roi in ['forehead','left_cheek','right_cheek']]
  methods[f'3ROI_POS{op}_decoder{de}']=[dict(rois[0][k],prediction=float(np.median([z[k]['prediction'] for z in rois]))) for k in sorted(shared)]
base=methods['oval_POSold_decoderold'];out=[]
for name,rr in methods.items():out.append(dict(method=name,**summarize(rr),delta_old_oval=summarize(rr,base)))
(P/'CROSSED_SUMMARY.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps([dict(method=r['method'],MAE=r['MAE'],people=r['people'],windows=r['videos']) for r in out]))
