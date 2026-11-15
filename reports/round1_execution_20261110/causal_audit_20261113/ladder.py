"""Ten fixed public positive controls, frozen inference, RAM-only encoded streams."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common_causal import *
from common_repair import manifest,O
import numpy as np,cv2,traceback,io
from collections import deque
from scipy import signal
sys.path.insert(0,str(O));from preprocess import Video
from oval_crop import OvalCrop
from t8_t1_loader import supported
from classical_repair import functions,waveform_decode
from readout import TREND
from compare_interventions import compare
cv2.setNumThreads(1)
HRPY='/data_pool/khangle/facevitals/envs/hr/bin/python'

class MemoryVideo:
 def __init__(self,name):
  self.fd=os.memfd_create('facevitals_causal_'+name);self.path=f'/proc/{os.getpid()}/fd/{self.fd}';self.commands=[]
 def close(self):
  if self.fd is not None:os.close(self.fd);self.fd=None
 def __enter__(self):return self
 def __exit__(self,*args):self.close()

def encode(src,out,name,vf=None,codec='libx265',extra=None,duration=None,metadata_rotation=None):
 mem=MemoryVideo(name);cmd=['ffmpeg','-v','error','-y','-threads','1']
 if metadata_rotation is not None:cmd+=['-display_rotation:v:0',str(metadata_rotation)]
 cmd+=['-i',str(src),'-map','0:v:0','-an','-sn','-dn','-filter_threads','1']
 if duration is not None:cmd+=['-t',str(duration)]
 if vf:cmd+=['-vf',vf]
 cmd+=['-fps_mode','passthrough','-threads','1','-c:v',codec]
 if codec=='libx265':cmd+=['-preset','fast','-b:v','8M','-maxrate','8M','-bufsize','16M','-x265-params','pools=none:frame-threads=1:log-level=error','-tag:v','hvc1']
 if codec=='ffv1':cmd+=['-level','3','-pix_fmt','bgr0','-f','matroska']
 else:cmd+=['-f','mp4']
 if extra:cmd+=extra
 cmd+=[mem.path];mem.commands.append(cmd)
 try:
  with (out/(name+'_encode.log')).open('wb') as f:subprocess.run(cmd,stderr=f,check=True)
  return mem
 except Exception:mem.close();raise

def raw_mem(frames,w,h,fps,out,name):
 mem=MemoryVideo(name);cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',f'{fps:.12f}','-i','pipe:0','-an','-threads','1','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-f','matroska',mem.path];mem.commands=[cmd]
 with (out/(name+'_encode.log')).open('wb') as log:
  p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
  try:
   for frame in frames:p.stdin.write(np.ascontiguousarray(frame).tobytes())
   p.stdin.close();assert p.wait()==0
  except Exception:
   p.stdin.close();p.wait();mem.close();raise
 return mem

def temporal_nlm(video):
 # True temporal NLM: five adjacent RGB frames. FFmpeg nlmeans is spatial,
 # deliberately NOT used as a temporal substitute. Boundary replication fixed.
 iterator=iter(video);first=cv2.cvtColor(next(iterator),cv2.COLOR_RGB2BGR);ring=deque([first,first,first]);last=first
 for frame in iterator:
  frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR);ring.append(frame);last=frame
  if len(ring)==5:
   yield cv2.cvtColor(cv2.fastNlMeansDenoisingColoredMulti(list(ring),2,5,None,10,10,3,7),cv2.COLOR_BGR2RGB);ring.popleft()
 for _ in range(2):
  ring.append(last)
  if len(ring)==5:
   yield cv2.cvtColor(cv2.fastNlMeansDenoisingColoredMulti(list(ring),2,5,None,10,10,3,7),cv2.COLOR_BGR2RGB);ring.popleft()

def select():
 rows=[]
 for ds in ['UBFC2','MCD']:
  people=set()
  for j in sorted([j for j in manifest() if j['dataset']==ds],key=lambda x:x['key']):
   if j['person'] in people:continue
   if ds=='MCD' and 'FullHD' not in j['recording']:continue
   pr=read(P/'original'/j['key']/'native/PREDICTIONS.json');cr=read(P/'classical_original_v2'/j['key']/'native/CLASSICAL.json')
   for w in j['windows']:
    pp=next((r['readouts']['power_0.7'] for r in pr if r['model'].startswith('padding') and r['start']==w['start']),None);pos=next((r['readouts']['0.7'] for r in cr if r['model']=='POS' and r['start']==w['start']),None)
    if w['eligible'] and pp is not None and pos is not None and max(abs(pp-w['truth']),abs(pos-w['truth']))<=5:
     rows.append(dict(job=j,window=w,saved_native_padding=pp,saved_native_POS=pos));people.add(j['person']);break
   if len(people)==5:break
  assert len(people)==5,f'INSUFFICIENT_ACCURATE_POSITIVE_CONTROLS_{ds}'
 return rows

def clip(rgb,t,valid,start):
 q=start+np.arange(301)/30
 if not supported(t,valid,q[0],q[-1]):raise ValueError('image_support_gap_or_tracking_failure')
 ix=np.searchsorted(t,q,side='right').clip(1,len(t)-1);weight=((q-t[ix-1])/(t[ix]-t[ix-1])).astype(np.float32)
 x=np.asarray(rgb[ix-1],np.float32)*(1-weight[:,None,None,None])+np.asarray(rgb[ix],np.float32)*weight[:,None,None,None]
 if x.std()<1e-6:raise ValueError('constant_input')
 return q,ix,weight,x,((x-x.mean())/x.std()).transpose(0,3,1,2).copy()

def stage(case,name,path,parent,commands=None,authoritative_t=None,until=None):
 j=case['job'];w=case['window'];out=C/'ladder'/j['key']/name;out.mkdir(parents=True,exist_ok=False);save(C/'LADDER_STATUS.json',dict(utc=now(),key=j['key'],stage=name,state='production_BTC_decode_and_oval'))
 v=Video(path,out/'decode.log');t=v.t
 if authoritative_t is not None:
  assert len(authoritative_t)==len(t);t=np.asarray(authoritative_t)
 take=len(t) if until is None else int(np.searchsorted(t,until,side='right'));det=OvalCrop();crops=[];valid=[];boxes=[];points=[];framehash=hashlib.sha256();last=None
 try:
  for i,im in enumerate(v):
   if i>=take:break
   framehash.update(im.tobytes());ok,small,box,pt=det.step(im);crops.append(small);valid.append(ok);boxes.append(box);points.append(pt)
   if abs(t[i]-w['start'])<.04:last=small
 finally:det.close()
 rgb=np.asarray(crops);t=t[:len(rgb)];valid=np.asarray(valid);np.savez_compressed(out/'FRONTEND.npz',t=t,valid=valid,boxes=boxes,points=points,rgb=rgb)
 if last is not None:cv2.imwrite(str(out/'roi.png'),cv2.cvtColor(cv2.resize(last,(288,288),interpolation=cv2.INTER_NEAREST),cv2.COLOR_RGB2BGR))
 ck=read(P/'FROZEN.json')['padding_checkpoints'][str(j['fold'])];receipt=dict(utc=now(),key=j['key'],stage=name,parent=parent,start=w['start'],truth=w['truth'],source_sha256=j['source_sha256'],encoded_sha256=sha(path),decoded_RGB_sha256=framehash.hexdigest(),frames=len(t),valid_fraction=float(valid.mean()),width=v.width,height=v.height,stream=v.stream,source_clock=t.tolist(),commands=commands or [],decode_command=v.command,authoritative_source_time=authoritative_t is not None,checkpoint_sha256=ck['sha256'],selected_positive_control_not_representative=True)
 try:q,ix,weight,raw,x=clip(rgb,t,valid,w['start'])
 except ValueError as e:save(out/'COMPLETE.json',dict(receipt,status='unscorable',reason=str(e),POS=None,padding=None));return
 arrays={'PTS':t[ix-1]*(1-weight.astype(float))+t[ix]*weight.astype(float),'ROI':raw.astype(np.float32),'RGB_before_normalization':raw.mean((1,2)).astype(np.float32),'tensor':x}
 for k,a in arrays.items():np.save(out/(k+'.npy'),a)
 save(out/'INPUT.json',dict(checkpoint=ck['path'],checkpoint_sha256=ck['sha256'],start=w['start']))
 subprocess.run([HRPY,str(C/'ladder_infer.py'),str(out)],check=True,stdout=open(out/'model.log','wb'),stderr=subprocess.STDOUT)
 model=read(out/'INFERENCE.json');ns=functions();mean=rgb.astype(float).mean((1,2));grid=w['start']+np.arange(300)/30;tr=np.stack([np.interp(grid,t,mean[:,k]) for k in range(3)],axis=1);y=ns['pos'](tr,30)
 try:
  hr=waveform_decode(y,.7);z=signal.sosfiltfilt(signal.butter(1,[.7,3],fs=30,btype='bandpass',output='sos'),y-TREND(y));f,p=signal.periodogram(z,fs=30,window='hann',nfft=4096,detrend=False);b=(f>=.7)&(f<=3);peak=f[b][np.argmax(p[b])];near=b&(abs(f-peak)<=.1);snr=float(10*np.log10((p[near].sum()+1e-30)/(p[b&~near].sum()+1e-30)))
 except ValueError:hr=None;snr=None;f=np.array([]);p=np.array([])
 np.savez_compressed(out/'POS.npz',waveform=y,f=f,power=p)
 specs=[]
 for k,axis,units,tol in [('PTS','T','seconds',1e-6),('ROI','THWC','RGB8 intensity',0.),('RGB_before_normalization','TC','RGB8 intensity',0.),('tensor','TCHW','normalized input',1e-6),('waveform','T','predicted difference',2e-6)]:specs.append(dict(name=k,path=k+'.npy',axis_order=axis,units=units,atol=tol,rtol=0))
 save(out/'STAGES.json',dict(schema_version=1,comparison_kind='controlled_degradation',case_id=j['key'],common_input_sha256=j['source_sha256'],checkpoint_sha256=ck['sha256'],pipeline_contract_id='frozen_BTC_route_plus_declared_intervention_ladder_v1',model_mode='eval_frozen',stages=specs))
 base=out.parent/'s0_native/STAGES.json'
 if name!='s0_native' and base.exists():save(out/'COMPARE_TO_S0.json',compare(base,out/'STAGES.json'))
 save(out/'COMPLETE.json',dict(receipt,status='complete',POS=hr,padding=model['HR'],POS_blind_SNR_dB=snr,padding_blind_SNR_dB=model['blind_peak_SNR_dB'],static_pass=model['static_HR'] is None,POS_bias=None if hr is None else hr-w['truth'],padding_bias=None if model['HR'] is None else model['HR']-w['truth']))

def case_run(case):
 j=case['job'];w=case['window'];out=C/'ladder'/j['key'];out.mkdir(parents=True,exist_ok=False);assert sha(j['source'])==j['source_sha256'];end=w['start']+10.25;mems=[]
 try:
  stage(case,'s0_native',j['source'],None,until=end)
  source=Video(j['source'],out/'source.log')
  # Source SDR frame decode is explicit before resize/matrix/range conversion.
  vf=source.vf+',scale=1920:1080:force_original_aspect_ratio=decrease:out_color_matrix=bt709:out_range=tv,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p'
  a=encode(j['source'],out,'s1_hevc',vf,extra=['-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv'],duration=end);mems.append(a);stage(case,'s1_hevc',a.path,'s0_native',a.commands)
  b=encode(a.path,out,'s2_display90',codec='copy',metadata_rotation=90);mems.append(b)
  bp=Video(b.path,out/'rotate_probe.log');rotation=next((float(s['rotation']) for s in bp.stream.get('side_data_list',[]) if 'rotation' in s),0);assert abs(rotation)==90,'DISPLAY_MATRIX_NOT_WRITTEN'
  stage(case,'s2_display90',b.path,'s1_hevc',b.commands)
  vf='zscale=pin=bt709:tin=bt709:min=bt709:rin=limited:t=linear:npl=100,format=gbrpf32le,zscale=p=bt2020,zscale=t=arib-std-b67:m=bt2020nc:r=limited,format=yuv420p10le'
  c=encode(b.path,out,'s3_HLG10',vf,extra=['-color_primaries','bt2020','-color_trc','arib-std-b67','-colorspace','bt2020nc','-color_range','tv']);mems.append(c);stage(case,'s3_HLG10',c.path,'s2_display90',c.commands)
  d=encode(c.path,out,'s4_848x480','scale=848:480:force_original_aspect_ratio=decrease,pad=848:480:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p10le',extra=['-color_primaries','bt2020','-color_trc','arib-std-b67','-colorspace','bt2020nc','-color_range','tv']);mems.append(d);stage(case,'s4_848x480',d.path,'s3_HLG10',d.commands)
  # Both denoisers receive the same production-tone-mapped SDR frames.
  v=Video(d.path,out/'s4_sdr_decode.log');fps=(len(v.t)-1)/(v.t[-1]-v.t[0]);sd=raw_mem(iter(v),v.width,v.height,fps,out,'s4_SDR_FFV1');mems.append(sd);times=v.t.copy()
  e=encode(sd.path,out,'s5_hqdn3d','format=yuv444p,hqdn3d=4:3:12:9,format=rgb24',codec='ffv1');mems.append(e);stage(case,'s5_hqdn3d',e.path,'s4_848x480',sd.commands+e.commands,authoritative_t=times)
  v2=Video(sd.path,out/'nlm_input.log');e2=raw_mem(temporal_nlm(v2),v2.width,v2.height,fps,out,'s5_temporalNLM');mems.append(e2);stage(case,'s5_temporalNLM',e2.path,'s4_848x480',sd.commands+e2.commands,authoritative_t=times)
  # Fixed original-time frame selection, no interpolation before encoding.
  targets=np.arange(0,times[-1]+1e-9,.1);idx=np.unique(np.searchsorted(times,targets).clip(0,len(times)-1));expr='select='+ '+'.join(f'eq(n\\,{int(i)})' for i in idx)
  for name,parent in [('s6_hqdn3d_10fps',e),('s6_temporalNLM_10fps',e2)]:
   final=encode(parent.path,out,name,expr+',setpts=N/(10*TB)',codec='ffv1');mems.append(final);stage(case,name,final.path,'s5_hqdn3d' if parent is e else 's5_temporalNLM',final.commands,authoritative_t=times[idx])
  save(out/'CASE_COMPLETE.json',dict(utc=now(),source_sha256=j['source_sha256'],all_encoded_streams_RAM_only=True))
 finally:
  for m in mems:m.close()

def main():
 lease=lock('ladder.lock');assert not (C/'LADDER_STARTED.json').exists();check_deadline();b=budget(True);cases=select();save(C/'LADDER_SELECTION.json',cases);save(C/'LADDER_STARTED.json',dict(utc=now(),pid=os.getpid(),budget=b,selection_sha256=sha(C/'LADDER_SELECTION.json'),code_sha256=sha(__file__),no_training=True,device='CPU'))
 for case in cases:check_deadline();budget(True);case_run(case)
 save(C/'LADDER_COMPLETE.json',dict(utc=now(),cases=len(cases),no_training=True))
if __name__=='__main__':
 try:main()
 except Exception as e:save(C/'LADDER_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
