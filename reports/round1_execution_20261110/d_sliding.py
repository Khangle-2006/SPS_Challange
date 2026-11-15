"""CPU inference for one selected outer-fold model; train features are not OOF scores."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,traceback
import numpy as np
from common import *
from sweep import make_model
from t8_evaluate import decode_difference
def main(fold):
 import torch
 torch.set_num_threads(1);lease=lock(f'd_sliding_{fold}.lock');guard(True);selection=read(P/'D_SELECTION.json');entry=selection['models'][fold];assert entry['fold']==fold and sha(entry['path'])==entry['sha256']
 dest=P/'D'/f'fold{fold}';assert not dest.exists(),'Inspect partial; no restart';dest.mkdir(parents=True)
 model=make_model(entry['initial_run'],'cpu');model.load_state_dict(torch.load(entry['path'],map_location='cpu',weights_only=False)['model']);static=model_static(model);model.eval();data=BTCData(fold);out=[]
 for v in data.rows:
  source=read(V/'btc_zeroshot/dev'/v['drive_id']/'SOURCE.json');assert source['source_sha256']==v['sha256'];span=source['source_span_s'];starts=np.arange(0,max(0,span-10+1e-8),1)
  candidates=sorted({r['start'] for r in data.candidates(v,'native') if r['scheme']=='10s'})
  for start in starts:
   prediction=None;reason=None
   try:
    x=data.clip(v,float(start))
    with torch.inference_mode():w=model(torch.from_numpy(x[None]))[0].numpy()
    prediction=decode_difference(w)
   except ValueError as e:reason=str(e)
   out.append(dict(recording=v['recording'],person=v['person'],fold=v['fold'],model_fold=fold,partition='validation' if v['fold']==fold else 'train_features',group=v['group'],task=v['task'],truth=v['pr_bpm'],start=float(start),prediction=prediction,reason=reason,also_disjoint=any(abs(start-s)<1e-8 for s in candidates)))
  save(dest/'STATUS.json',dict(state='running',last=v['recording'],windows=len(out),utc=now(),pid=os.getpid()))
 save(dest/'WINDOWS.json',out);save(dest/'COMPLETE.json',dict(utc=now(),model_sha256=entry['sha256'],selection_sha256=sha(P/'D_SELECTION.json'),static=static,input_hashes=data.hashes,excluded_videos=data.exclusions,holdout_evaluations=0))
if __name__=='__main__':
 fold=int(sys.argv[1])
 try:main(fold)
 except Exception as e:save(P/f'D/fold{fold}/ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
