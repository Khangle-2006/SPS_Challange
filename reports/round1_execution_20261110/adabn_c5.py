"""Finish predeclared AdaBN evaluation on C5-prime after its preparation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import traceback
from common import *
from sweep import make_model
from model_eval import c5_rows
def main():
 import torch
 torch.set_num_threads(1);lease=lock('adabn_c5.lock');guard(True)
 assert (P/'adabn/NATIVE_COMPLETE.json').exists() and (P/'C5_COMPLETE.json').exists()
 assert not (P/'adabn/C5_STARTED.json').exists(),'No duplicate/restart'
 save(P/'adabn/C5_STARTED.json',dict(pid=os.getpid(),utc=now(),code=hashes()))
 for run in read(S/'PLAN.json')['runs']:
  fold=run['fold'];d=P/'adabn'/f'fold{fold}';a=read(d/'COMPLETE.json');assert sha(d/'final.pt')==a['checkpoint_sha256']
  model=make_model(run,'cpu');model.load_state_dict(torch.load(d/'final.pt',map_location='cpu',weights_only=False)['model'])
  save(d/'C5PRIME.json',c5_rows(model,fold,'cpu','AdaBN',btc_only=True))
 save(P/'adabn/C5_COMPLETE.json',dict(utc=now(),folds=5,holdout_evaluations=0))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'adabn/C5_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
