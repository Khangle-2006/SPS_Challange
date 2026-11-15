"""Predeclared crop-level robustness audit; opens train/validation only."""
import argparse, hashlib, json, time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_efficientphys import Clips, evaluate, sha
from EfficientPhys_fixed import EfficientPhys

ROOT=Path('/data_pool/khangle/facevitals')
DATA=ROOT/'datasets/UBFC-rPPG/efficientphys36_trainval_v1'
CONDITIONS=['clean','resolution24','resolution18','resolution12','fps15','fps10',
            'jpeg90','jpeg50','jpeg20','occlude_top','occlude_bottom','occlude_left','occlude_right',
            'dim_half','dim_quarter','noise2','noise5','frozen_frame']
RUNS={2027:'efficientphys36_ubfc_v1',2028:'efficientphys36_ubfc_seed2028',2029:'efficientphys36_ubfc_seed2029'}

def degrade(x,condition,rng):
    x=x.copy()
    if condition.startswith('resolution'):
        size=int(condition.removeprefix('resolution'))
        x=np.stack([cv2.resize(cv2.resize(f,(size,size),interpolation=cv2.INTER_AREA),(36,36),interpolation=cv2.INTER_LINEAR) for f in x])
    elif condition.startswith('fps'):
        stride=30//int(condition.removeprefix('fps'))
        indices=np.arange(0,301,stride); q=np.arange(301)
        j=np.clip(np.searchsorted(indices,q,side='right'),1,len(indices)-1)
        w=((q-indices[j-1])/(indices[j]-indices[j-1])).astype(np.float32)[:,None,None,None]
        x=x[indices[j-1]]*(1-w)+x[indices[j]]*w
    elif condition.startswith('jpeg'):
        quality=int(condition.removeprefix('jpeg')); frames=[]
        for f in x:
            bgr=cv2.cvtColor(np.clip(np.rint(f),0,255).astype(np.uint8),cv2.COLOR_RGB2BGR)
            ok,buf=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,quality]); assert ok
            frames.append(cv2.cvtColor(cv2.imdecode(buf,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB))
        x=np.asarray(frames,dtype=np.float32)
    elif condition.startswith('occlude'):
        # Static temporal-mean texture; geometry refers to crop coordinates, not landmarks.
        mean=x.mean(axis=0,keepdims=True)
        if condition=='occlude_top':x[:,:12]=mean[:,:12]
        if condition=='occlude_bottom':x[:,24:]=mean[:,24:]
        if condition=='occlude_left':x[:,:,:12]=mean[:,:,:12]
        if condition=='occlude_right':x[:,:,24:]=mean[:,:,24:]
    elif condition.startswith('dim'):
        x=np.rint(x*(.5 if condition=='dim_half' else .25))
    elif condition.startswith('noise'):
        sigma=float(condition.removeprefix('noise'))
        x=np.clip(x+rng.normal(0,sigma,x.shape).astype(np.float32),0,255)
    elif condition=='frozen_frame':x[:]=x[0]
    else:assert condition=='clean'
    assert x.shape==(301,36,36,3) and np.isfinite(x).all()
    return x.astype(np.float32)

class StressClips(Clips):
    def __init__(self,tag,condition):
        super().__init__(DATA,tag);self.condition=condition
    def __getitem__(self,i):
        r=self.rows[i];s=r['subject']
        if s not in self.cache:
            meta=np.load(self.root/(s+'_meta.npz'))
            self.cache[s]=(np.load(self.root/(s+'_rgb.npy'),mmap_mode='r'),meta['t'],meta['ppg'])
        raw,t,ppg=self.cache[s];q=r['start_s']+np.arange(301)/30
        assert q[0]>=t[0] and q[-1]<=t[-1]
        j=np.clip(np.searchsorted(t,q,side='right'),1,len(t)-1)
        w=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)[:,None,None,None]
        x=raw[j-1].astype(np.float32)*(1-w)+raw[j].astype(np.float32)*w
        seed=int.from_bytes(hashlib.sha256(f"{s}:{r['start_s']}:{self.condition}".encode()).digest()[:8],'little')
        x=degrade(x,self.condition,np.random.default_rng(seed))
        std=float(x.std());assert std>1e-6
        x=(x-float(x.mean()))/std
        y=np.diff(np.interp(q,t,ppg).astype(np.float32)); y/=float(y.std())
        return torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y),i

def selfcheck():
    base=Clips(DATA,'val'); clean=StressClips('val','clean')
    for i in (0,20,41):
        x,y,_=base[i];a,b,_=clean[i]
        torch.testing.assert_close(x,a,rtol=0,atol=0);torch.testing.assert_close(y,b,rtol=0,atol=0)
    t=np.arange(301)/30
    signal=np.sin(2*np.pi*1.2*t).astype(np.float32)
    x=np.broadcast_to(signal[:,None,None,None]+100,(301,36,36,3)).copy()
    from hr_baseline import estimate_hr
    for cond in ('fps15','fps10'):
        out=degrade(x,cond,np.random.default_rng(1))
        assert abs(estimate_hr(out[:-1,0,0,0],30)-72)<.5
    assert np.max(np.abs(np.diff(degrade(x,'frozen_frame',np.random.default_rng(1)),axis=0)))==0

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(1);selfcheck()
    for name,m in json.loads((DATA/'inventory.json').read_text()).items():
        assert (DATA/name).stat().st_size==m['bytes'] and sha(DATA/name)==m['sha256']
    lock=json.loads((ROOT/'results/efficientphys36_three_seeds/frozen_protocol.json').read_text())
    protocol={'conditions':CONDITIONS,'validation_seeds':[2027,2028,2029],'train_seeds':[2027],
      'script_sha256':sha(Path(__file__)),'inventory_sha256':sha(DATA/'inventory.json'),
      'checkpoint_hashes':{e['seed']:e['checkpoint_sha256'] for e in lock['checkpoints']},
      'scope':'36px cached crop after successful detection; not original-camera or detector robustness',
      'fps':'Decimate 30Hz grid then linearly interpolate same 10s interval back to 30Hz; no time-axis relabeling.',
      'jpeg':'Independent-frame JPEG on 36px crop; not H264/video-codec benchmark.',
      'dim_noise':'Brightness quantization and additive digital RGB noise are proxies, not calibrated lux/sensor physics.',
      'frozen':'Negative control with no temporal physiological information; numerical HR is not a valid measurement.',
      'selection':'No tuning in audit; no test input opened.'}
    pp=a.out/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==json.loads(json.dumps(protocol))
    else:pp.write_text(json.dumps(protocol,indent=2))
    summary=[]
    for split,seeds in [('val',[2027,2028,2029]),('train',[2027])]:
        for seed in seeds:
            ckpath=ROOT/'results'/RUNS[seed]/'best.pt'
            assert sha(ckpath)==protocol['checkpoint_hashes'][seed]
            model=EfficientPhys(frame_depth=10,img_size=36).cuda()
            model.load_state_dict(torch.load(ckpath,map_location='cuda',weights_only=False)['model'])
            for condition in CONDITIONS:
                target=a.out/f'{split}_{seed}_{condition}.json'
                if target.exists():result=json.loads(target.read_text())
                else:
                    start=time.time();ds=StressClips(split,condition)
                    result=evaluate(model,DataLoader(ds,batch_size=2,num_workers=2),'cuda')
                    result.update(split=split,seed=seed,condition=condition,elapsed_s=time.time()-start)
                    result['subjects']={s:float(np.mean([r['abs_error'] for r in result['records'] if r['subject']==s])) for s in sorted({r['subject'] for r in result['records']})}
                    result['over5bpm']=sum(r['abs_error']>5 for r in result['records'])
                    if split=='val' and condition=='clean':
                        expected=next(e['validation_MAE'] for e in lock['checkpoints'] if e['seed']==seed)
                        assert abs(result['subject_macro_MAE']-expected)<1e-8
                    tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(target)
                compact={k:v for k,v in result.items() if k not in ('records','subjects')}
                summary.append(compact);print(json.dumps(compact),flush=True)
            del model
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',len(summary),flush=True)

if __name__=='__main__':main()
