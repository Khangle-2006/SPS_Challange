import os,sys,json,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;OLD=P.parent/'codec_clock_20261111'
sys.path.insert(0,str(OLD))
import audit_clock as a
from nuisance import regress
def amplitude(y,t,f):
    X=np.stack([np.ones(len(t)),np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t)],1)
    b=np.linalg.lstsq(X,y,rcond=None)[0];return float(np.hypot(b[1],b[2]))
def main():
    out=[]
    for root in sorted((OLD/'synthetic').glob('scene*')):
        parts=root.name.split('_');pulse=int(parts[2][5:]);fs=int(parts[3][1:])
        if pulse!=150:continue
        for bf in [0,3,4]:
            d=root/f'b{bf}';g=a.read(d/'GOP.json')['frames'];types=[x['pict_type'] for x in g]
            with np.load(d/'traces.npz') as z:rgb=z['skin'];t=z['t']
            boxes=np.stack([.8*np.sin(2*np.pi*.73*t),.8*np.cos(2*np.pi*.41*t),np.full(len(t),110),np.full(len(t),90)],1)
            cleaned,fit=regress(rgb,types,boxes);before=amplitude(rgb[:,1],t,2.5);after=amplitude(cleaned[:,1],t,2.5)
            retained=after/before if before>1e-10 else None
            out.append(dict(case=root.name,bf=bf,fs=fs,true_pulse_bpm=pulse,G_amplitude_before=before,G_amplitude_after=after,retained_fraction=retained,preserves_at_least90percent=retained is not None and retained>=.9,fit=fit,source_sha=a.sha(d/'traces.npz'),schedule_sha=a.sha(d/'GOP.json')))
    # Mathematical identifiability control: true pulse exactly in encoder-phase subspace.
    fs=10;t=np.arange(301)/fs;types=['P' if i%4==0 else 'B' for i in range(len(t))]
    rgb=np.array([180,130,110])[None,:]+np.sin(2*np.pi*2.5*t)[:,None]*np.array([.2,1,.1])
    cleaned,_=regress(rgb,types)
    exact=amplitude(cleaned[:,1],t,2.5)/amplitude(rgb[:,1],t,2.5)
    assert exact<1e-10,'Phase-regression identifiability test unexpectedly failed'
    result=dict(utc=a.now(),cases=out,ideal_phase_locked_true150_retained_fraction=exact,all_preserve90percent=all(r['preserves_at_least90percent'] for r in out),deployment_allowed=False,interpretation='A true 150 bpm pulse is collinear with a 10fps four-frame schedule; unprotected projection can erase real pulse. Full dataset evaluation is diagnostic only, no automatic adoption.',code_sha256=a.sha(__file__),regressor_sha256=a.sha(P/'nuisance.py'))
    a.save(P/'REGRESSION_PULSE_PROBE.json',result);print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
if __name__=='__main__':main()
