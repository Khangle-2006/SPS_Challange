"""Inference-only reader_v2. Requires a supplied Task/non-Task metadata flag."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'reader_v1_20261116'))
import reader_v1

def predict(t,valid,rgb,model,is_task):
    if not isinstance(is_task,(bool,np.bool_)):
        raise ValueError('An explicit boolean Task flag is required')
    result=reader_v1.predict(t,valid,rgb,model)
    offset=float(model['offset_task' if is_task else 'offset_rest'])
    return dict(prediction=result['prediction']+offset,abstain=result['abstain'],reader_v1_bpm=result['prediction'],offset_bpm=offset,is_task=bool(is_task),method='reader_v2')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--meta',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--task',choices=['task','non-task'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with np.load(a.meta,allow_pickle=False) as data:r=predict(data['t'],data['valid'],data['rgb_mean'],json.loads(a.model.read_text()),a.task=='task')
    a.out.write_text(json.dumps(r,indent=2));print(json.dumps(r))
