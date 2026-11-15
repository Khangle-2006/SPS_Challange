"""Predeclared follow-up on ALL ten selected clips, regardless of HR/track outcome.
Same encoded stage4 bitstream; inverse known injected90deg before detector.
Primary ladder remains unchanged. First completed one-case control is reused.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LIBGL_ALWAYS_SOFTWARE='1',EGL_PLATFORM='surfaceless',MESA_LOADER_DRIVER_OVERRIDE='swrast',GALLIUM_DRIVER='llvmpipe',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
from common_causal import *
from ladder_v2 import MemoryVideo,Video,clip,functions,waveform_decode,OvalCrop,np,cv2,HRPY
import traceback
def one(case,a,path):
 j=case['job'];w=case['window'];data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==a['encoded_sha256'];mem=MemoryVideo('localization_all');os.write(mem.fd,data);del data
 out=C/'localization_all'/j['key'];out.mkdir(parents=True,exist_ok=False);save(out/'STARTED.json',dict(utc=now(),encoded_sha256=a['encoded_sha256'],angle_ccw=-90,angle_source='inverse experiment injection, no PR/HR search'))
 v=Video(mem.path,out/'decode.log');det=OvalCrop();rgb=[];valid=[];boxes=[];points=[]
 try:
  for im in v:
   check_deadline();ok,crop,box,pt=det.step(np.rot90(im,-1).copy());rgb.append(crop);valid.append(ok);boxes.append(box);points.append(pt)
 finally:det.close();mem.close()
 rgb=np.asarray(rgb);valid=np.asarray(valid);np.savez_compressed(out/'FRONTEND.npz',t=v.t,valid=valid,boxes=boxes,points=points,rgb=rgb)
 result=dict(utc=now(),key=j['key'],recording=j['recording'],dataset=j['dataset'],person=j['person'],source_stage_sha256=a['encoded_sha256'],truth=w['truth'],before_valid_fraction=a['valid_fraction'],after_valid_fraction=float(valid.mean()),before_POS=a.get('POS'),before_padding=a.get('padding'),angle_ccw=-90,no_PR_angle_selection=True)
 try:_,_,_,_,x=clip(rgb,v.t,valid,w['start'])
 except ValueError as e:save(out/'COMPLETE.json',dict(result,status='unscorable',reason=str(e),POS=None,padding=None));return
 ck=read(P/'FROZEN.json')['padding_checkpoints'][str(j['fold'])];np.save(out/'tensor_storage_env.npy',x);save(out/'INPUT.json',dict(checkpoint=ck['path'],checkpoint_sha256=ck['sha256'],start=w['start']))
 subprocess.run([HRPY,str(C/'ladder_infer_v2.py'),str(out)],check=True,stdout=open(out/'model.log','wb'),stderr=subprocess.STDOUT)
 mean=rgb.astype(float).mean((1,2));grid=w['start']+np.arange(300)/30;trace=np.stack([np.interp(grid,v.t,mean[:,k]) for k in range(3)],axis=1);hr=waveform_decode(functions()['pos'](trace,30),.7);model=read(out/'INFERENCE.json')
 save(out/'COMPLETE.json',dict(result,status='scorable',POS=hr,padding=model['HR'],static_pass=model['static_HR'] is None))

def main():
 lease=lock('localization_all.lock');assert not (C/'LOCALIZATION_ALL_STARTED.json').exists();budget(True);cases=read(C/'LADDER_SELECTION.json')
 save(C/'LOCALIZATION_ALL_STARTED.json',dict(utc=now(),pid=os.getpid(),cases=[c['job']['key'] for c in cases],selection_sha256=sha(C/'LADDER_SELECTION.json'),intervention='inverse known injected+90 before detector on ALL ten stage4 bitstreams; no HR-dependent selection',code_sha256=sha(__file__),label='follow-up after first synthetic tracking failure, not pre-original-ladder hypothesis'))
 results=[]
 for index,case in enumerate(cases):
  j=case['job'];p=C/'ladder'/j['key']/'s4_848x480/COMPLETE.json'
  while not p.exists():
   check_deadline();assert not (C/'LADDER_ERROR.json').exists(),'PRIMARY_LADDER_ERROR';save(C/'LOCALIZATION_ALL_STATUS.json',dict(utc=now(),completed=index,total=10,waiting=j['key']));time.sleep(10)
  a=read(p);out=C/'localization_all'/j['key']/'COMPLETE.json'
  if index==0:
   old=read(C/'LOCALIZATION_COMPLETE.json');assert old['same_bitstream_sha256']==a['encoded_sha256'];save(out,dict(old,key=j['key'],dataset=j['dataset'],person=j['person'],truth=old['PPG_truth'],before_POS=a.get('POS'),before_padding=a.get('padding'),reused_first_control=True))
  else:budget(True);one(case,a,a['commands'][-1][-1])
  results.append(read(out))
 save(C/'LOCALIZATION_ALL_COMPLETE.json',dict(utc=now(),cases=results,scope='Controlled synthetic stage4 orientation before detector only; BTC cohort effect not measured',no_training=True,holdout_evaluations=0))
if __name__=='__main__':
 try:main()
 except Exception as e:save(C/'LOCALIZATION_ALL_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
