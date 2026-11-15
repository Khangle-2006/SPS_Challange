"""Paired clean/augmentation pilot; fixed architecture and development-only selection."""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from stress_ubfc import ROOT, DATA, StressClips
from train_efficientphys import evaluate, atomic_save, sha
from EfficientPhys_fixed import EfficientPhys
VAL_CONDITIONS=['clean','jpeg90','jpeg50']

class TrainClips(StressClips):
    def __init__(self,seed,aug):
        super().__init__('train','clean');self.seed=seed;self.aug=aug;self.epoch=0
    def __getitem__(self,i):
        if self.aug:
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,i]))
            self.condition=rng.choice(['clean','jpeg90','jpeg50'],p=[.5,.25,.25])
        else:self.condition='clean'
        return super().__getitem__(i)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--mode',choices=['control','augmentation'],required=True);a=p.parse_args()
    out=ROOT/'results/compression_ablation_v1'/f'{a.mode}_{a.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists():raise RuntimeError('Refusing overwrite; no automatic rerun')
    config={'seed':a.seed,'mode':a.mode,'epochs':30,'batch_size':2,'optimizer':'AdamW','weight_decay':0.,
            'max_lr':.009,'scheduler':'OneCycleLR','validation_conditions':VAL_CONDITIONS,
            'selection':'mean of subject-macro MAE over clean/jpeg90/jpeg50; earliest minimum',
            'training_augmentation':{'clean':.5,'jpeg90':.25,'jpeg50':.25} if a.mode=='augmentation' else {'clean':1.},
            'pilot_gate':'At seed2027: >=10% reduction in mean clean/jpeg90/jpeg50 MAE versus matched control, with clean MAE increase <=0.25 bpm. Otherwise do not expand to more seeds.',
            'test_opened':False,'note':'Development experiment motivated by validation stress audit; no independent generalization claim.'}
    prov={'config':config,'data_inventory_sha256':sha(DATA/'inventory.json'),
          'code_sha256':{f:sha(Path(__file__).parent/f) for f in ('train_compression_ablation.py','stress_ubfc.py','train_efficientphys.py','EfficientPhys_fixed.py','hr_baseline.py')}}
    (out/'provenance.json').write_text(json.dumps(prov,indent=2))
    for name,m in json.loads((DATA/'inventory.json').read_text()).items():
        assert (DATA/name).stat().st_size==m['bytes'] and sha(DATA/name)==m['sha256']
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    model=EfficientPhys(frame_depth=10,img_size=36).cuda()
    train=TrainClips(a.seed,a.mode=='augmentation')
    generator=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(train,batch_size=2,shuffle=True,generator=generator,num_workers=2,pin_memory=True)
    val={c:DataLoader(StressClips('val',c),batch_size=2,num_workers=2) for c in VAL_CONDITIONS}
    opt=torch.optim.AdamW(model.parameters(),lr=.009,weight_decay=0.)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.009,epochs=30,steps_per_epoch=len(loader))
    history=[];best=float('inf')
    for epoch in range(30):
        start=time.time();train.epoch=epoch;model.train();losses=[]
        for x,y,_ in loader:
            opt.zero_grad(set_to_none=True);pred=model(x.cuda());loss=((pred-y.cuda())**2).mean()
            assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            opt.step();sched.step();losses.append(float(loss.detach()))
        reports={c:evaluate(model,dl,'cuda') for c,dl in val.items()}
        score=float(np.mean([r['subject_macro_MAE'] for r in reports.values()]))
        item={'epoch':epoch+1,'train_mse':float(np.mean(losses)),'selection_score':score,
              'conditions':{c:{k:r[k] for k in ('subject_macro_MAE','window_MAE','window_RMSE','invalid')} for c,r in reports.items()},'elapsed_s':time.time()-start}
        history.append(item);improved=score<best
        if improved:best=score
        ck={'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
            'config':config,'best':best,'history':history,'rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':generator.get_state()}
        atomic_save(ck,out/'last.pt')
        if improved:
            atomic_save(ck,out/'best.pt')
            (out/'best_validation.json').write_text(json.dumps({'epoch':epoch+1,'selection_score':score,'conditions':reports},indent=2))
        (out/'history.json').write_text(json.dumps(history,indent=2))
        (out/'status.json').write_text(json.dumps({'state':'running','completed_epochs':epoch+1,'best':best}))
        print(json.dumps(item),flush=True)
    model.load_state_dict(torch.load(out/'best.pt',map_location='cuda',weights_only=False)['model'])
    saved=json.loads((out/'best_validation.json').read_text())
    for c,dl in val.items():
        report=evaluate(model,dl,'cuda')
        assert abs(report['subject_macro_MAE']-saved['conditions'][c]['subject_macro_MAE'])<1e-8
    (out/'status.json').write_text(json.dumps({'state':'complete','completed_epochs':30,'best_epoch':saved['epoch'],
        'best':best,'checkpoint_sha256':sha(out/'best.pt'),'reload_verified':True}))
    print('COMPLETE',a.mode,a.seed,best,flush=True)

if __name__=='__main__':main()
