from common_repair import *
import numpy as np,ast,math
from scipy import signal
from readout import TREND,decode
from t8_t1_loader import supported

def functions():
 ns=dict(np=np,json=json,math=math,signal=signal)
 for name,file in [('pos','hr_baseline.py'),('chrom','t7_pipeline.py')]:
  p=O/'vendor'/file;node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
 return ns

def waveform_decode(y,low):
 y=np.asarray(y,float)
 if y.shape!=(300,) or not np.isfinite(y).all() or y.std()<1e-10:raise ValueError('flat_or_invalid_classical')
 z=signal.sosfiltfilt(signal.butter(1,[low,3],fs=30,btype='bandpass',output='sos'),y-TREND(y));f,p=signal.periodogram(z,fs=30,window='hann',nfft=4096,detrend=False);s=(f>=low)&(f<=3)
 if not np.isfinite(p).all() or p[s].sum()<=1e-30:raise ValueError('invalid_classical_spectrum')
 return float(60*f[s][np.argmax(p[s])])

def run(j,c,variant,rgb,t,valid,out):
 ns=functions();mean=np.asarray(rgb,float).mean((1,2));rows=[];waves=[];keys=[]
 for w in j['windows']:
  for method in ['POS','CHROM']:
   ans={str(low):None for low in [.7,.6]};reason=None
   try:
    if not w['eligible'] or not supported(t,valid,w['start'],w['start']+10):raise ValueError('ineligible_reference_or_image_support')
    q=w['start']+np.arange(300)/30;x=np.stack([np.interp(q,t,mean[:,k]) for k in range(3)],axis=1);y=ns[method.lower()](x,30);waves.append(y);keys.append([method,str(w['start'])])
    for low in [.7,.6]:
     try:ans[str(low)]=waveform_decode(y,low)
     except ValueError:pass
   except ValueError as e:reason=str(e)
   rows.append(dict(dataset=j['dataset'],recording=j['recording'],person=j['person'],fold=j['fold'],condition=c,input=variant,model=method,start=w['start'],truth=w['truth'],readouts=ans,reason=reason))
 save(out/'CLASSICAL.json',rows);np.savez_compressed(out/'classical_waves.npz',waves=waves,keys=keys)
