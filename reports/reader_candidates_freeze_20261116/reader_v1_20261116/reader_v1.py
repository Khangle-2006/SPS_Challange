"""Versioned pipeline API around the hash-verified reference reader. No CNN."""
import numpy as np
import btc_posterior_reader as ref
def scores(t,valid,rgb):
 with np.errstate(divide='ignore',invalid='ignore'):
  z=ref.video_scores(np.asarray(t),np.asarray(valid,bool),np.asarray(rgb))
 return z if z is not None and np.isfinite(z).all() else None
def predict(t,valid,rgb,model):
 z=scores(t,valid,rgb)
 return dict(prediction=ref.posterior_median(z,np.asarray(model['prior']),model['beta']),abstain=z is None,method='reader_v1')
if __name__=='__main__':
 import argparse,json
 from pathlib import Path
 parser=argparse.ArgumentParser();parser.add_argument('--meta',required=True,type=Path);parser.add_argument('--model',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
 with np.load(args.meta,allow_pickle=False) as a:result=predict(a['t'],a['valid'],a['rgb_mean'],json.loads(args.model.read_text()))
 args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
