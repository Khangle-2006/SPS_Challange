"""One-factor repair control after the first synthetic stage4 tracking failure.
Inverse of known injected90deg, fixed before new prediction, not PR-fitted angle.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LIBGL_ALWAYS_SOFTWARE='1',EGL_PLATFORM='surfaceless',MESA_LOADER_DRIVER_OVERRIDE='swrast',GALLIUM_DRIVER='llvmpipe',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
from common_causal import *
from ladder_v2 import MemoryVideo,Video,clip,functions,waveform_decode,OvalCrop,np,cv2,HRPY
import traceback
def main():
 lease=lock('localization.lock');assert not (C/'LOCALIZATION_STARTED.json').exists();check_deadline();budget(True)
 case=read(C/'LADDER_SELECTION.json')[0];j=case['job'];w=case['window'];base=C/'ladder'/j['key']/'s4_848x480';a=read(base/'COMPLETE.json');assert a['valid_fraction']==0 and a['status']=='unscorable'
 path=a['commands'][-1][-1];data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==a['encoded_sha256'];mem=MemoryVideo('localization');os.write(mem.fd,data);del data
 out=C/'orientation_localization';out.mkdir(exist_ok=False);save(C/'LOCALIZATION_STARTED.json',dict(utc=now(),pid=os.getpid(),source_stage=str(base),encoded_sha256=a['encoded_sha256'],intervention='np.rot90(frame,-1) before original detector; inverse known injected+90 display matrix; no angle search/no PR',code_sha256=sha(__file__)))
 v=Video(mem.path,out/'decode.log');det=OvalCrop();rgb=[];valid=[];boxes=[];points=[]
 try:
  for im in v:
   frame=np.rot90(im,-1).copy();ok,crop,box,pt=det.step(frame);rgb.append(crop);valid.append(ok);boxes.append(box);points.append(pt)
 finally:det.close();mem.close()
 rgb=np.asarray(rgb);valid=np.asarray(valid);np.savez_compressed(out/'FRONTEND.npz',t=v.t,valid=valid,boxes=boxes,points=points,rgb=rgb)
 receipt=dict(utc=now(),recording=j['recording'],before_valid_fraction=0.,after_valid_fraction=float(valid.mean()),same_bitstream_sha256=a['encoded_sha256'],PPG_truth=w['truth'],intervention='inverse injected90deg before landmark, pure pixel permutation',BTC_effect_not_measured=True)
 try:q,ix,weight,raw,x=clip(rgb,v.t,valid,w['start'])
 except ValueError as e:save(C/'LOCALIZATION_COMPLETE.json',dict(receipt,status='still_unscorable',reason=str(e)));return
 ck=read(P/'FROZEN.json')['padding_checkpoints'][str(j['fold'])];np.save(out/'tensor_storage_env.npy',x);save(out/'INPUT.json',dict(checkpoint=ck['path'],checkpoint_sha256=ck['sha256'],start=w['start']))
 subprocess.run([HRPY,str(C/'ladder_infer_v2.py'),str(out)],check=True,stdout=open(out/'model.log','wb'),stderr=subprocess.STDOUT)
 mean=rgb.astype(float).mean((1,2));grid=w['start']+np.arange(300)/30;trace=np.stack([np.interp(grid,v.t,mean[:,k]) for k in range(3)],axis=1);hr=waveform_decode(functions()['pos'](trace,30),.7);model=read(out/'INFERENCE.json');cv2.imwrite(str(out/'roi.png'),cv2.cvtColor(cv2.resize(rgb[0],(288,288),interpolation=cv2.INTER_NEAREST),cv2.COLOR_RGB2BGR))
 save(C/'LOCALIZATION_COMPLETE.json',dict(receipt,status='scorable',POS=hr,padding=model['HR'],POS_error=abs(hr-w['truth']),padding_error=None if model['HR'] is None else abs(model['HR']-w['truth']),static_pass=model['static_HR'] is None,interpretation='If recovered, orientation before detection contributes to this synthetic low-resolution failure; not proof of BTC population MAE cause or phone acquisition loss.'))
if __name__=='__main__':
 try:main()
 except Exception as e:save(C/'LOCALIZATION_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
