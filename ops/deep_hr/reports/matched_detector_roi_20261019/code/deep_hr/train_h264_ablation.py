"""Matched 3-seed clean-vs-H264 training; train/validation only."""
import argparse,json,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from train_efficientphys import Clips,evaluate,atomic_save,sha
from EfficientPhys_fixed import EfficientPhys
ROOT=Path('/data_pool/khangle/facevitals')
DATA=ROOT/'datasets/UBFC-rPPG'
CLEAN=DATA/'efficientphys36_trainval_v1'
COMP_TRAIN=DATA/'efficientphys36_train_h264_500k_v1'
COMP_VAL=DATA/'efficientphys36_val_h264_500k_v1'
OUT=ROOT/'results/h264_ablation_v1'
SEEDS=[2027,2028,2029]

def verify(root):
    for name,m in json.loads((root/'inventory.json').read_text()).items():
        assert (root/name).stat().st_size==m['bytes'] and sha(root/name)==m['sha256'],name

def same_pair(clean,encoded):
    assert clean.rows==encoded.rows
    for s in {r['subject'] for r in clean.rows}:
        a=np.load(clean.root/(s+'_meta.npz'));b=np.load(encoded.root/(s+'_meta.npz'))
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])

class Mixed(Dataset):
    def __init__(self,seed,mode):
        self.clean=Clips(CLEAN,'train');self.rows=self.clean.rows
        self.comp=Clips(COMP_TRAIN,'train') if mode=='augmentation' else None
        if self.comp is not None:same_pair(self.clean,self.comp)
        self.seed=seed;self.epoch=0
    def __len__(self):return len(self.rows)
    def use_compressed(self,i):
        # Deterministic Bernoulli mix, independent of Torch/dropout/shuffle RNG.
        return bool(self.comp is not None and np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,i])).random()<.5)
    def __getitem__(self,i):return (self.comp if self.use_compressed(i) else self.clean)[i]

def model_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['control','augmentation'],required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args()
    out=OUT/f'{a.mode}_{a.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists():raise RuntimeError('Refusing overwrite of an existing training run')
    config={'seed':a.seed,'mode':a.mode,'epochs':30,'batch_size':2,'max_lr':.009,'weight_decay':0.,
            'optimizer':'AdamW','scheduler':'OneCycleLR','loss':'MSE normalized PPG forward differences',
            'training_mix':{'clean':.5,'h264_500k':.5} if a.mode=='augmentation' else {'clean':1.},
            'validation_conditions':['clean','h264_500k'],
            'selection':'minimum equal-weight mean of clean and H264 validation subject-macro MAE; earliest tie',
            'test_opened':False}
    protocol={'seeds':SEEDS,'modes':['control','augmentation'],'epochs':30,
              'keep_rule':'Across paired seeds: >=10% improvement in mean clean/H264 selection score, average clean MAE increase <=0.25 bpm, H264 MAE improves in at least 2 of 3 seeds. Report all outcomes; no test evaluation.',
              'scope':'Development comparison on existing 28/7 subject split; H264 at same recipe as validation, no claim of unseen codec/domain generalization.',
              'train_clean_inventory':sha(CLEAN/'inventory.json'),'val_h264_inventory':sha(COMP_VAL/'inventory.json')}
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2))
    for root in ([CLEAN,COMP_VAL,COMP_TRAIN] if a.mode=='augmentation' else [CLEAN,COMP_VAL]):verify(root)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    model=EfficientPhys(frame_depth=10,img_size=36).cuda()
    train=Mixed(a.seed,a.mode)
    vc=Clips(CLEAN,'val');vh=Clips(COMP_VAL,'val');same_pair(vc,vh)
    generator=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(train,batch_size=2,shuffle=True,generator=generator,num_workers=2,pin_memory=True)
    val={'clean':DataLoader(vc,batch_size=2,num_workers=2),'h264_500k':DataLoader(vh,batch_size=2,num_workers=2)}
    prov={'config':config,'initial_model_sha256':model_hash(model),'protocol_sha256':sha(pp),
          'train_h264_inventory':sha(COMP_TRAIN/'inventory.json') if a.mode=='augmentation' else None,
          'code_sha256':{f:sha(Path(__file__).parent/f) for f in ('train_h264_ablation.py','train_efficientphys.py','EfficientPhys_fixed.py','hr_baseline.py')},
          'torch':torch.__version__,'gpu':torch.cuda.get_device_name()}
    (out/'provenance.json').write_text(json.dumps(prov,indent=2))
    opt=torch.optim.AdamW(model.parameters(),lr=.009,weight_decay=0.)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.009,epochs=30,steps_per_epoch=len(loader))
    best=float('inf');history=[]
    for epoch in range(30):
        start=time.time();train.epoch=epoch;model.train();losses=[];order=[]
        for x,y,ids in loader:
            order.extend(ids.tolist());opt.zero_grad(set_to_none=True)
            pred=model(x.cuda());loss=((pred-y.cuda())**2).mean();assert torch.isfinite(loss)
            loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            opt.step();sched.step();losses.append(float(loss.detach()))
        reports={c:evaluate(model,dl,'cuda') for c,dl in val.items()}
        score=float(np.mean([r['subject_macro_MAE'] for r in reports.values()]))
        item={'epoch':epoch+1,'train_mse':float(np.mean(losses)),'score':score,
              'conditions':{c:{k:r[k] for k in ('subject_macro_MAE','window_MAE','window_RMSE','invalid')} for c,r in reports.items()},
              'compressed_train_clips':sum(train.use_compressed(i) for i in range(len(train))),
              'sample_order_sha256':hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest(),
              'elapsed_s':time.time()-start}
        history.append(item);improved=score<best
        if improved:best=score
        ck={'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
            'config':config,'best':best,'history':history,'rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':generator.get_state()}
        atomic_save(ck,out/'last.pt')
        if improved:
            atomic_save(ck,out/'best.pt');(out/'best_validation.json').write_text(json.dumps({'epoch':epoch+1,'score':score,'conditions':reports},indent=2))
        (out/'history.json').write_text(json.dumps(history,indent=2))
        (out/'status.json').write_text(json.dumps({'state':'running','completed_epochs':epoch+1,'best':best}))
        print(json.dumps({'seed':a.seed,'mode':a.mode,**item}),flush=True)
    saved=json.loads((out/'best_validation.json').read_text())
    model.load_state_dict(torch.load(out/'best.pt',map_location='cuda',weights_only=False)['model'])
    for c,dl in val.items():assert abs(evaluate(model,dl,'cuda')['subject_macro_MAE']-saved['conditions'][c]['subject_macro_MAE'])<1e-8
    (out/'status.json').write_text(json.dumps({'state':'complete','completed_epochs':30,'best_epoch':saved['epoch'],
        'best':best,'reload_verified':True,'checkpoint_sha256':sha(out/'best.pt')}))
    print('COMPLETE',a.mode,a.seed,best,flush=True)
if __name__=='__main__':main()
