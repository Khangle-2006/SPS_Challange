"""Correct the audit's public reference clock, without recropping/reencoding.

The actual BTC route used AVI container time. UBFC's public reference uses the
recorded acquisition clock. Map by preserved source frame identity, never fit a
lag/scale against HR. Preserve the original BTC-route results as a diagnostic.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from common_causal import *
import numpy as np,torch,traceback
from scipy import signal
from infer_original import clip
os.environ['CUDA_VISIBLE_DEVICES']=''
from EfficientPhys_fixed import EfficientPhys
from replicate_tsm import use_replicate_padding
from t8_evaluate import decode_difference
from readout import spectra,TREND
from classical_repair import functions,waveform_decode
from compare_interventions import compare

MODELS={}
def model(j):
 f=str(j['fold']);a=read(P/'FROZEN.json')['padding_checkpoints'][f]
 if f not in MODELS:
  assert sha(a['path'])==a['sha256']
  m=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36));m.load_state_dict(torch.load(a['path'],map_location='cpu',weights_only=True)['model'],strict=True);MODELS[f]=m.eval()
 return MODELS[f],a['sha256']

def reference_clock(case):
 j=case['job'];d=Path(j['crop']);receipt=read(d.parent/'complete.json');assert sha(d/'meta.npz')==receipt['cache_files']['meta.npz']
 with np.load(d/'meta.npz') as a:
  t=a['original_t'].copy();assert np.array_equal(np.unique(t),a['t']),'PUBLIC_DUPLICATE_MERGE_MAPPING_NOT_EXACT'
 proof=dict(meta_sha256=sha(d/'meta.npz'),source_sha256=j['source_sha256'],recording=j['recording'])
 if j['dataset']=='UBFC2':
  gt=Path(j['source']).parent/'ground_truth.txt';a=np.loadtxt(gt,ndmin=2);assert a.shape[0]==3;tt=a[2]-a[2,0];assert np.array_equal(tt,t),'UBFC_ACQUISITION_CLOCK_NOT_EXACT'
  proof.update(clock='ground_truth acquisition time, exactly matches frozen public cache',ground_truth_sha256=sha(gt))
 else:proof.update(clock='frozen MCD video source clock; independent contact PPG reference unchanged')
 return t,proof

def analyze(case,source,out,reference_t,proof,stage,base):
 j=case['job'];w=case['window'];a=read(source/'COMPLETE.json') if (source/'COMPLETE.json').exists() else read(C/'LOCALIZATION_COMPLETE.json')
 with np.load(source/'FRONTEND.npz',allow_pickle=False) as z:rgb=z['rgb'];valid=z['valid'];container_t=z['t']
 with np.load(base/'s0_native/FRONTEND.npz',allow_pickle=False) as z:native_t=z['t']
 if stage.startswith('s6_'):
  with np.load(base/'s4_848x480/FRONTEND.npz',allow_pickle=False) as z:parent_t=z['t']
  ids=np.searchsorted(parent_t,container_t);assert np.all(ids<len(parent_t)) and np.all(np.abs(parent_t[ids]-container_t)<1e-9),'FRAME_SKIP_IDENTITY_UNPROVEN'
 else:
  ids=np.arange(len(container_t));assert len(ids)<=len(native_t) and np.max(abs(container_t-native_t[:len(ids)]))<2e-5,'FRAME_ORDER_OR_CLOCK_CHANGED_UNEXPECTEDLY'
 assert len(ids)==len(rgb) and len(ids)<=len(reference_t) and np.all(np.diff(ids)>0)
 t=reference_t[ids];out.mkdir(parents=True,exist_ok=False)
 receipt=dict(utc=now(),key=j['key'],dataset=j['dataset'],person=j['person'],stage=stage,truth=w['truth'],start=w['start'],valid_fraction=float(valid.mean()),source_frontend_sha256=sha(source/'FRONTEND.npz'),frame_identity_rule='same ordered source frames; stage6 uses exact saved selected source timestamps',reference_clock=proof,container_vs_reference_max_delta_s=float(np.max(abs(t-container_t))),reference_clock_verified=True)
 np.savez_compressed(out/'CLOCK.npz',source_indices=ids,container_PTS=container_t,reference_time=t,valid=valid)
 # Exact frozen public adapter policy: average frames at identical acquisition
 # times, AND valid support, store float16. No fitted time or signal alignment.
 unique,inverse,counts=np.unique(t,return_inverse=True,return_counts=True)
 merged=np.zeros((len(unique),36,36,3),np.float32);np.add.at(merged,inverse,rgb);merged/=counts[:,None,None,None]
 usable=np.ones(len(unique),bool);np.logical_and.at(usable,inverse,valid)
 receipt.update(duplicate_frames_merged=int(len(t)-len(unique)),duplicate_policy='frozen prepare_recording: mean RGB float32 -> float16, logical AND support')
 rgb=merged.astype(np.float16);t=unique;valid=usable
 try:x,raw=clip(rgb,t,valid,w['start'])
 except ValueError as e:save(out/'COMPLETE.json',dict(receipt,status='unscorable',reason=str(e),POS=None,padding=None));return
 m,ck=model(j)
 with torch.inference_mode():y=m(torch.from_numpy(x[None]))[0].numpy();sy=m(torch.from_numpy(np.repeat(x[:1],301,axis=0)[None]))[0].numpy()
 try:hr=decode_difference(y)
 except ValueError:hr=None
 try:static=decode_difference(sy)
 except ValueError:static=None
 assert static is None,'STATIC_FAILED'
 mean=rgb.astype(float).mean((1,2));grid=w['start']+np.arange(300)/30;tr=np.stack([np.interp(grid,t,mean[:,k]) for k in range(3)],axis=1);pos=functions()['pos'](tr,30)
 try:ph=waveform_decode(pos,.7)
 except ValueError:ph=None
 def snr(f,p):
  b=(f>=.7)&(f<=3);peak=f[b][np.argmax(p[b])];near=b&(abs(f-peak)<=.1);return float(10*np.log10((p[near].sum()+1e-30)/(p[b&~near].sum()+1e-30)))
 z=signal.sosfiltfilt(signal.butter(1,[.7,3],fs=30,btype='bandpass',output='sos'),pos-TREND(pos));pf,pp=signal.periodogram(z,fs=30,window='hann',nfft=4096,detrend=False)
 try:mf,_,_,mp=spectra(y,.7);ms=snr(mf,mp)
 except ValueError:mf=np.array([]);mp=np.array([]);ms=None
 np.savez_compressed(out/'STAGES.npz',PTS=w['start']+np.arange(301)/30,ROI=raw,RGB_before_normalization=raw.mean((1,2)),tensor=x,waveform=y,POS_waveform=pos,POS_f=pf,POS_power=pp,model_f=mf,model_power=mp)
 specs=[dict(name=k,path='STAGES.npz',key=k,axis_order=ax,units=u,atol=tol,rtol=0.) for k,ax,u,tol in [('PTS','T','seconds',1e-6),('ROI','THWC','RGB8',0.),('RGB_before_normalization','TC','RGB8',0.),('tensor','TCHW','zscore',1e-6),('waveform','T','difference',2e-6)]]
 save(out/'STAGES.json',dict(schema_version=1,comparison_kind='controlled_degradation',case_id=j['key'],common_input_sha256=j['source_sha256'],checkpoint_sha256=ck,pipeline_contract_id='source_frame_identity_public_reference_clock_fixed',model_mode='eval_frozen',stages=specs))
 base=out.parent/'s0_native/STAGES.json'
 if stage!='s0_native' and base.exists():save(out/'COMPARE_TO_S0.json',compare(base,out/'STAGES.json'))
 save(out/'COMPLETE.json',dict(receipt,status='complete',POS=ph,padding=hr,POS_blind_SNR_dB=snr(pf,pp),padding_blind_SNR_dB=ms,static_pass=True,checkpoint_sha256=ck,original_container_clock_POS=a.get('POS'),original_container_clock_padding=a.get('padding'),no_training=True))

def main():
 try:lease=lock('public_clock.lock')
 except BlockingIOError:return
 check_deadline();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 cases=read(C/'LADDER_SELECTION.json');rows=[];local=[]
 repaired={r['key'] for r in read(C/'PREFIX_CLOCK_AUDIT.json') if r['insufficient']} if (C/'PREFIX_CLOCK_AUDIT.json').exists() else set()
 for case in cases:
  j=case['job'];branch=C/'prefix_repair' if j['key'] in repaired else C;root=branch/'ladder'/j['key'];t,proof=reference_clock(case);result_root=C/('reference_clock_prefix' if j['key'] in repaired else 'reference_clock')
  for source in sorted(root.glob('s*')):
   if not source.is_dir() or not (source/'COMPLETE.json').exists():continue
   out=result_root/j['key']/source.name
   if not (out/'COMPLETE.json').exists():
    assert not out.exists(),'PRESERVE_PARTIAL_NO_BLIND_RESTART';check_deadline();budget(True);analyze(case,source,out,t,proof,source.name,root)
   rows.append(read(out/'COMPLETE.json'))
  lc=branch/'localization_all'/j['key']
  if (lc/'COMPLETE.json').exists():
   if read(lc/'COMPLETE.json').get('reused_first_control'):lc=C/'orientation_localization'
   out=result_root/j['key']/'inverse90_stage4'
   if not (out/'COMPLETE.json').exists():
    assert not out.exists(),'PRESERVE_PARTIAL_NO_BLIND_RESTART';check_deadline();budget(True);analyze(case,lc,out,t,proof,'inverse90_stage4',root)
   local.append(read(out/'COMPLETE.json'))
 save(C/'REFERENCE_CLOCK_ROWS.json',rows);save(C/'REFERENCE_CLOCK_LOCALIZATION.json',local)
 save(C/'REFERENCE_CLOCK_STATUS.json',dict(utc=now(),stages=len(rows),target_stages=90,localization=len(local),target_localization=10,no_video_decode_or_training=True))
 if len(rows)==90 and len(local)==10:save(C/'REFERENCE_CLOCK_COMPLETE.json',dict(utc=now(),stages=90,localization=10,old_outputs_preserved=True))

if __name__=='__main__':
 try:main()
 except Exception as e:save(C/'REFERENCE_CLOCK_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
