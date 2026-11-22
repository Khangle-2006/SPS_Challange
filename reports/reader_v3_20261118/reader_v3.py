"""Frozen reader-v3 inference. Resolution is always that of the ORIGINAL source."""
import json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'source'))
from lib_btc import avg_spectra,zlog,GRID
from reader_models import pmed

def source_class(source_width,source_height):
    if int(source_width)<=0 or int(source_height)<=0:raise ValueError('Original source dimensions required')
    return 'low' if int(source_width)*int(source_height)<1800000 else 'high'

def predict(t,valid,rgb_mean,model,*,source_width,source_height,activity):
    if activity not in ('Resting','Walking','Task1','Task2','Task3'):raise ValueError('Unknown activity')
    cls=source_class(source_width,source_height);task=activity.startswith('Task')
    with np.errstate(divide='ignore',invalid='ignore'):a=avg_spectra(np.asarray(t),np.asarray(valid,bool),np.asarray(rgb_mean))
    z=None if a is None else (zlog(a[0])+zlog(a[1]))/2
    if z is not None and not np.isfinite(z).all():z=None
    prior=np.asarray(model['prior']);assert np.array_equal(model['grid_bpm'],GRID)
    offset=model['offset_by_cell'][cls+('_task' if task else '_not_task')]
    value=pmed(np.zeros(len(GRID)) if z is None else z,prior,model['beta_by_class'][cls])+offset
    return dict(prediction=float(value),abstain=z is None,source_class=cls,method='reader_v3',pulse_observed=z is not None)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--meta',required=True);p.add_argument('--model',required=True)
    p.add_argument('--source-width',required=True,type=int);p.add_argument('--source-height',required=True,type=int)
    p.add_argument('--activity',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    with np.load(a.meta,allow_pickle=False) as n:r=predict(n['t'],n['valid'],n['rgb_mean'],json.loads(Path(a.model).read_text()),source_width=a.source_width,source_height=a.source_height,activity=a.activity)
    Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps(r))
