"""Frozen padding inference on CPU only, original production tensor and decoder."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from common_causal import *
import torch,numpy as np
from EfficientPhys_fixed import EfficientPhys
from replicate_tsm import use_replicate_padding
from t8_evaluate import decode_difference
from readout import spectra
if __name__=='__main__':
 d=Path(sys.argv[1]);a=read(d/'INPUT.json');torch.set_num_threads(2);torch.set_num_interop_threads(1)
 assert sha(a['checkpoint'])==a['checkpoint_sha256'];m=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36));m.load_state_dict(torch.load(a['checkpoint'],map_location='cpu',weights_only=True)['model'],strict=True);m.eval();x=np.load(d/'tensor_storage_env.npy')
 from infer_original import clip as production_clip
 os.environ['CUDA_VISIBLE_DEVICES']=''
 cached=np.load(d/'FRONTEND.npz');xx,_=production_clip(cached['rgb'],cached['t'],cached['valid'],a['start']);save(d/'NUMERIC_ENVIRONMENT.json',dict(max_abs_difference=float(abs(x-xx).max()),storage_tensor_sha256=sha(d/'tensor_storage_env.npy'),numpy_version=np.__version__,authority='actual production_clip in hr environment; no manually duplicated normalization'));x=xx;np.save(d/'tensor.npy',x)
 with torch.inference_mode():
  y=m(torch.from_numpy(x[None]))[0].numpy();sy=m(torch.from_numpy(np.repeat(x[:1],301,axis=0)[None]))[0].numpy()
 np.save(d/'waveform.npy',y)
 try:hr=decode_difference(y)
 except ValueError:hr=None
 try:static=decode_difference(sy)
 except ValueError:static=None
 assert static is None,'STATIC_FAILED'
 try:
  f,_,_,p=spectra(y,.7);b=(f>=.7)&(f<=3);peak=f[b][np.argmax(p[b])];near=b&(abs(f-peak)<=.1);snr=float(10*np.log10((p[near].sum()+1e-30)/(p[b&~near].sum()+1e-30)))
 except ValueError:f=np.array([]);p=np.array([]);snr=None
 np.savez_compressed(d/'model_spectrum.npz',f=f,power=p)
 save(d/'INFERENCE.json',dict(utc=now(),HR=hr,blind_peak_SNR_dB=snr,static_HR=static,checkpoint_sha256=a['checkpoint_sha256'],waveform_sha256=sha(d/'waveform.npy'),training=False,device='CPU',readout='production decode_difference: cumsum+detrend+0.7..3Hz'))
