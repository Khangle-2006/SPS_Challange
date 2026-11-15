"""Read completed C5-prime receipts only; public frozen references and BTC dev."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import time,sys,json,traceback,argparse
import numpy as np
from common import *
sys.path.insert(0,str(V/'btc_zeroshot'))
from infer_dev import classical_functions,image_clip,decode_wave,decode_model
from EfficientPhys_fixed import EfficientPhys
def loadcrop(d):
 a=read(d/'crop/COMPLETE.json');assert sha(d/'crop/crop.mkv')==a['crop_sha256'] and sha(d/'crop/meta.npz')==a['meta_sha256']
 with np.load(d/'crop/meta.npz') as z:t=z['t'].copy();valid=z['valid'].copy();mean=z['rgb_mean'].copy()
 raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(d/'crop/crop.mkv'),'-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
 assert len(raw)==len(t)*36*36*3
 return np.frombuffer(raw,np.uint8).reshape(-1,36,36,3),t,valid,mean
def main():
 import torch
 torch.set_num_threads(1);lease=lock('c5evaluation.lock');guard(True);ns,fnsha=classical_functions();refs=read(B/'public_comparison/REFERENCE_COMPACT.json');dev=read(V/'btc_zeroshot/dev_manifest.json')['videos'];data=BTCData(0);models={}
 source=R/'reports/round1_final_20261105/revision_02/frozen_data/FROZEN_WINDOWS.json';assert sha(source)==refs['source_sha256']
 def model(method,fold):
  key=method,fold
  if key not in models:
   root=R/'t8_polarity_v3_20261102' if method=='V3' else V/'v4_repaired';p=root/f'fold{fold}/seed2027/joint';st=read(p/'status.json');assert st['state']=='complete' and sha(p/'final.pt')==st['checkpoint_sha256']
   m=EfficientPhys(frame_depth=10,img_size=36);m.load_state_dict(torch.load(p/'final.pt',map_location='cpu',weights_only=False)['model']);models[key]=m.eval()
  return models[key]
 while True:
  progress=False
  for cp in sorted((P/'c5prime').glob('*/*/*/COMPLETE.json')):
   a=read(cp);ds,rid,variant=a['dataset'],a['recording'],a['variant'];out=P/'c5_predictions'/ds/rid/(variant+'.json')
   if out.exists():continue
   if a['state']!='complete':save(out,dict(source_receipt_sha256=sha(cp),rows=[],failure=a['reason']));continue
   d=cp.parent;rgb,t,valid,means=loadcrop(d);rows=[]
   if ds=='BTC':
    v=next(v for v in dev if v['recording']==rid);windows=[dict(start=w['start'],truth=v['pr_bpm'],eligible=True) for w in data.candidates(v,'native') if w['scheme']=='10s'];windows=list({w['start']:w for w in windows}.values())
   else:
    r=next(r for r in refs['records'] if r['dataset']==ds and r['recording_id']==rid and r['variant']=='native');windows=[dict(start=w['window_start_s'],truth=w['reference_hr_bpm'],eligible=w['reference_eligible']) for w in r['windows']]
   for w in windows:
    for method in ['POS','CHROM','V3','V4']:
     pred=None;reason=None
     try:
      if not w['eligible']:raise ValueError('reference_ineligible')
      if not a['bitrate_pass']:raise ValueError('C5prime_whole_file_bitrate_above_500_no_retuning')
      # image_clip enforces source timestamps/support with a real terminal frame.
      x=image_clip(rgb,t,valid,w['start'],10)
      if method in ['POS','CHROM']:
       q=w['start']+np.arange(300)/30;trace=np.stack([np.interp(q,t,means[:,k]) for k in range(3)],axis=1);pred=decode_wave(ns[method.lower()](trace,30))
      else:
       with torch.inference_mode():wave=model(method,a['fold'])(torch.from_numpy(x[None]))[0].numpy()
       pred=decode_model(wave)
     except ValueError as e:reason=str(e)
     rows.append(dict(dataset=ds,recording=rid,person=a['person'],fold=a['fold'],condition='c5prime_'+variant,method=method,**w,prediction=pred,reason=reason))
   save(out,dict(rows=rows,source_receipt_sha256=sha(cp),functions=fnsha,code=hashes(),model_policy='same-public-fold seed2027 only; V4 original padding comparator',holdout_evaluations=0));progress=True
   save(P/'C5_EVALUATION_STATUS.json',dict(state='running',pid=os.getpid(),last=str(out),completed_files=len(list((P/'c5_predictions').glob('*/*/*.json'))),utc=now()));guard()
  if (P/'C5_COMPLETE.json').exists():
   expected=len(read(P/'C5_COMPLETE.json')['rows']);done=len(list((P/'c5_predictions').glob('*/*/*.json')))
   if done==expected:save(P/'C5_EVALUATION_COMPLETE.json',dict(files=done,utc=now(),holdout_evaluations=0));break
  if not progress:time.sleep(20)
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'C5_EVALUATION_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
