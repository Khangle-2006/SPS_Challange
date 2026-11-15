"""Versioned .6-Hz CHROM correction from saved RGB; no video/codec rerun."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common_repair import *
from classical_repair_v2 import functions,waveform_decode
import numpy as np,time,traceback

def main():
 lease=lock('codec_band_v2.lock');assert not (P/'CODEC_BAND_V2_START.json').exists();budget(True);save(P/'CODEC_BAND_V2_START.json',dict(utc=now(),pid=os.getpid(),code=sha(__file__)));began=time.time();done=0
 rows=read(P/'CODEC_SELECTION.json');ns=functions(.6)
 for j in rows:
  root=P/'codec_four'/j['key']
  while not (root/'COMPLETE.json').exists():
   assert time.time()-began<36*3600 and not (P/'CODEC_FOUR_ERROR.json').exists();time.sleep(15)
  a=np.load(root/'COLOR_TIME.npz');results=[]
  for condition in ['native','lossless10','default','bf0']:
   t=a['native_t'] if condition=='native' else a['source_t'];rgb=a['native_all_rgb'] if condition=='native' else a['rgb_'+condition]
   for start in [0.,10.,20.]:
    if start<t[0] or start+10>t[-1]:continue
    q=start+np.arange(300)/30;x=np.stack([np.interp(q,t,rgb[:,k]) for k in range(3)],axis=1)
    for method in ['POS','CHROM','G']:
     try:y=x[:,1] if method=='G' else ns[method.lower()](x,30);hr=waveform_decode(y,.6);reason=None
     except ValueError as e:hr=None;reason=str(e)
     results.append(dict(dataset=j['dataset'],recording=j['recording'],condition=condition,start=start,method=method,prediction=hr,near150=hr is not None and 148<=hr<=152,reason=reason,band_hz=[.6,3]))
  old=read(root/'WINDOWS.json');lookup={(r['condition'],r['start'],r['method']):r for r in old}
  for r in results:
   if r['method']!='CHROM':assert r['prediction']==lookup[r['condition'],r['start'],r['method']]['prediction']
  save(root/'WINDOWS_BAND_V2.json',results);save(root/'BAND_V2_COMPLETE.json',dict(utc=now(),RGB_sha256=sha(root/'COLOR_TIME.npz'),windows_sha256=sha(root/'WINDOWS_BAND_V2.json'),old_windows_preserved=True,POS_G_parity=True,CHROM_inner_and_outer_band_hz=[.6,3]));done+=1
 save(P/'CODEC_BAND_V2_COMPLETE.json',dict(utc=now(),completed=done,no_video_or_codec_rerun=True))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'CODEC_BAND_V2_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
