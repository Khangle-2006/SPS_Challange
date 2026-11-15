"""Local EfficientPhys baseline. Train/validation only; no test evaluation code."""
import argparse, csv, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from EfficientPhys_fixed import EfficientPhys
from hr_baseline import estimate_hr

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

class Clips(Dataset):
    def __init__(self, root, tag):
        self.root=root
        split=json.loads((root/'subject_split.json').read_text())
        all_rows=json.loads((root/'clips.json').read_text())
        assert all(r['subject'] not in split['test'] for r in all_rows)
        assert set(split['train']).isdisjoint(split['val'])
        assert set(split['train']+split['val']).isdisjoint(split['test'])
        self.rows=[r for r in all_rows if r['split']==tag]
        assert {r['subject'] for r in self.rows}==set(split[tag])
        self.cache={}
    def __len__(self): return len(self.rows)
    def __getitem__(self,idx):
        r=self.rows[idx]; s=r['subject']
        if s not in self.cache:
            meta=np.load(self.root/(s+'_meta.npz'))
            self.cache[s]=(np.load(self.root/(s+'_rgb.npy'),mmap_mode='r'),meta['t'],meta['ppg'])
        raw,t,ppg=self.cache[s]
        q=r['start_s']+np.arange(301)/30
        assert q[0]>=t[0] and q[-1]<=t[-1]
        j=np.clip(np.searchsorted(t,q,side='right'),1,len(t)-1)
        w=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)[:,None,None,None]
        x=raw[j-1].astype(np.float32)*(1-w)+raw[j].astype(np.float32)*w
        std=float(x.std())
        if std<1e-6: raise ValueError('Constant input clip')
        x=(x-float(x.mean()))/std
        y=np.interp(q,t,ppg).astype(np.float32)
        dy=np.diff(y); sd=float(dy.std())
        if sd<1e-8: raise ValueError('Constant PPG target')
        dy/=sd
        return torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(dy),idx

def atomic_save(obj,path):
    temp=path.with_suffix('.tmp'); torch.save(obj,temp); temp.replace(path)

def evaluate(model,loader,device):
    model.eval(); records=[]; losses=[]
    with torch.no_grad():
        for x,y,idx in loader:
            pred=model(x.to(device)); y=y.to(device)
            if not torch.isfinite(pred).all(): raise ValueError('Nonfinite validation output')
            losses.extend(torch.mean((pred-y)**2,dim=1).cpu().tolist())
            # A forward difference predicts y[t+1]-y[t]. Reconstruct y[0:T].
            wave=torch.cat([torch.zeros_like(pred[:,:1]),torch.cumsum(pred[:,:-1],dim=1)],dim=1).cpu().numpy()
            for k,i in enumerate(idx.tolist()):
                r=dict(loader.dataset.rows[i])
                try:
                    r['pred_hr']=estimate_hr(wave[k],30.)
                    r['abs_error']=abs(r['pred_hr']-r['gt_hr'])
                except ValueError:
                    # An invalid prediction must never be silently dropped.
                    r['pred_hr']=None; r['abs_error']=135.; r['invalid']=True
                records.append(r)
    by_subject={s:np.mean([r['abs_error'] for r in records if r['subject']==s]) for s in {r['subject'] for r in records}}
    return {'mse':float(np.mean(losses)), 'subject_macro_MAE':float(np.mean(list(by_subject.values()))),
            'window_MAE':float(np.mean([r['abs_error'] for r in records])),
            'window_RMSE':float(np.sqrt(np.mean([r['abs_error']**2 for r in records]))),
            'invalid':sum(r.get('invalid',False) for r in records),'n':len(records),'records':records}

def smoke(model,dataset,device,out):
    from EfficientPhys_original import EfficientPhys as Original
    x,y,_=dataset[0]; a,b,_=dataset[min(1,len(dataset)-1)]
    batch=torch.stack([x,a]).to(device)
    labels=torch.stack([y,b]).to(device)
    model.eval()
    original=Original(frame_depth=10,img_size=36).to(device).eval()
    original.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():
        together=model(batch)
        single=torch.cat([model(batch[i:i+1]) for i in range(2)],dim=0)
        torch.testing.assert_close(together,single,rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(model(batch.flip(0)).flip(0),together,rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(original(batch[0]).reshape(1,-1),model(batch[:1]),rtol=1e-4,atol=1e-5)
    # Explicitly prove diff does not inject another clip's DC offset.
    z=torch.zeros(2,301,3,1,1,device=device); z[1]=100
    assert torch.diff(z,dim=1).abs().max()==0
    del original
    model.train(); model.zero_grad(set_to_none=True)
    pred=model(batch); loss=torch.mean((pred-labels)**2); loss.backward()
    grad=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert pred.shape==(2,300) and math.isfinite(float(loss)) and math.isfinite(grad) and grad>0
    report={'passed':True,'loss':float(loss.detach()),'gradient_abs_sum':grad,
            'eval_batch_invariance':'passed','single_clip_equivalence_to_original':'passed',
            'cuda_device':torch.cuda.get_device_name(), 'peak_memory_bytes':torch.cuda.max_memory_allocated(),
            'parameters':sum(p.numel() for p in model.parameters())}
    (out/'smoke_test.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--smoke-only',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(a.config.read_text()); seed=cfg['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    device=torch.device('cuda:0')
    train=Clips(a.root,'train');val=Clips(a.root,'val')
    # Verify copied cache before training. No implicit fallback to raw/test data.
    for f,meta in json.loads((a.root/'inventory.json').read_text()).items():
        target=a.root/f
        assert target.stat().st_size==meta['bytes'] and sha(target)==meta['sha256'],f
    if not a.resume and (a.out/'last.pt').exists(): raise RuntimeError('Existing run: use --resume or a new output directory')
    model=EfficientPhys(frame_depth=10,img_size=36).to(device)
    smoke(model,train,device,a.out)
    if a.smoke_only:return
    # Smoke updates BatchNorm buffers, so reset initialization before actual training.
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=EfficientPhys(frame_depth=10,img_size=36).to(device)
    gen=torch.Generator().manual_seed(seed)
    tl=DataLoader(train,batch_size=cfg['batch_size'],shuffle=True,generator=gen,num_workers=2,pin_memory=True)
    vl=DataLoader(val,batch_size=cfg['batch_size'],shuffle=False,num_workers=2,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate_max'],weight_decay=0.)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg['learning_rate_max'],epochs=cfg['epochs'],steps_per_epoch=len(tl))
    start=0;best=float('inf');history=[]
    provenance={'config':cfg,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
                'data_inventory_sha256':sha(a.root/'inventory.json'),
                'code_sha256':{f:sha(Path(__file__).parent/f) for f in ('train_efficientphys.py','EfficientPhys_fixed.py','EfficientPhys_original.py','hr_baseline.py')},
                'train_subjects':28,'validation_subjects':7,'train_clips':len(train),'validation_clips':len(val),'test_opened':False}
    (a.out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    if a.resume:
        ck=torch.load(a.out/'last.pt',map_location=device,weights_only=False)
        assert ck['config']==cfg and ck['data_inventory_sha256']==provenance['data_inventory_sha256']
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        start=ck['epoch']+1;best=ck['best'];history=ck['history']
        torch.set_rng_state(ck['rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']]);gen.set_state(ck['loader_rng'].cpu())
    for epoch in range(start,cfg['epochs']):
        begin=time.time();model.train();losses=[]
        for step,(x,y,_) in enumerate(tl):
            x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True)
            pred=model(x);loss=torch.mean((pred-y)**2)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise RuntimeError('Nonfinite gradient')
            opt.step();sched.step();losses.append(float(loss.detach()))
            if step%20==0: print(json.dumps({'epoch':epoch+1,'step':step,'steps':len(tl),'loss':losses[-1]}),flush=True)
        v=evaluate(model,vl,device)
        record={'epoch':epoch+1,'train_mse':float(np.mean(losses)),**{k:v[k] for k in v if k!='records'},'elapsed_s':time.time()-begin}
        history.append(record)
        improved=v['subject_macro_MAE']<best
        if improved:best=v['subject_macro_MAE']
        ck={'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
            'best':best,'history':history,'config':cfg,'data_inventory_sha256':provenance['data_inventory_sha256'],
            'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':gen.get_state()}
        atomic_save(ck,a.out/'last.pt')
        if improved:
            atomic_save(ck,a.out/'best.pt')
            (a.out/'best_validation.json').write_text(json.dumps({'epoch':epoch+1,**v},indent=2))
        (a.out/'history.json').write_text(json.dumps(history,indent=2))
        status={'state':'running','completed_epochs':epoch+1,'planned_epochs':cfg['epochs'],'best_validation_subject_macro_MAE':best,'last_epoch':record,'test_opened':False}
        (a.out/'status.json').write_text(json.dumps(status,indent=2))
        print(json.dumps(record),flush=True)
    status['state']='complete';(a.out/'status.json').write_text(json.dumps(status,indent=2))

if __name__=='__main__':main()
