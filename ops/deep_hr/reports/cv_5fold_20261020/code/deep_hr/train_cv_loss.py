"""Final-epoch-only paired five-fold loss comparison; no heldout inference here."""
import argparse,random,time,hashlib,json
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from cv_common import ROOT,WORK,OUT,SEED,read,write,sha,CVClips,rows_for
from train_efficientphys import atomic_save
from EfficientPhys_fixed import EfficientPhys
from spectral_objective import SpectralObjective,WEIGHT
def model_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
class Mixed(Dataset):
    def __init__(self,fold):
        self.rows=rows_for(fold,'train');self.clean=CVClips(WORK/'cache/clean',self.rows);self.comp=CVClips(WORK/'cache/native_500k',self.rows);self.epoch=0
        assert not self.clean.invalid and not self.comp.invalid
    def __len__(self):return len(self.rows)
    def use_compressed(self,i):return bool(np.random.default_rng(np.random.SeedSequence([SEED,self.epoch,i])).random()<.5)
    def __getitem__(self,i):return (self.comp if self.use_compressed(i) else self.clean)[i]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=range(5),required=True);ap.add_argument('--loss',choices=['mse','spectral'],required=True);a=ap.parse_args()
    p=OUT/f'fold{a.fold}'/a.loss;p.mkdir(parents=True,exist_ok=True)
    if (p/'last.pt').exists():raise RuntimeError('Refusing overwrite of an existing CV run')
    protocol=read(OUT/'protocol.json');assert sha(WORK/'folds.json')==protocol['folds_sha256'] and sha(WORK/'primary_manifest.json')==protocol['primary_manifest_sha256']
    for name,h in protocol['code_sha256'].items():assert sha(ROOT/'code/deep_hr'/name)==h
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=EfficientPhys(frame_depth=10,img_size=36).cuda();initial=model_hash(model);train=Mixed(a.fold)
    gen=torch.Generator().manual_seed(SEED);loader=DataLoader(train,batch_size=2,shuffle=True,generator=gen,num_workers=2,pin_memory=True)
    objective=SpectralObjective().cuda() if a.loss=='spectral' else None
    cfg={'seed':SEED,'fold':a.fold,'loss':a.loss,'spectral_weight':WEIGHT if objective is not None else 0.,'epochs':30,'batch_size':2,'max_lr':.009,'weight_decay':0.,'optimizer':'AdamW','scheduler':'OneCycleLR','train_windows':len(train),'selection':'final epoch30, no validation during training','test_opened':False}
    write(p/'provenance.json',{'config':cfg,'initial_model_sha256':initial,'protocol_sha256':sha(OUT/'protocol.json'),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'train_rows_sha256':hashlib.sha256(json.dumps(train.rows,sort_keys=True).encode()).hexdigest(),'code_sha256':protocol['code_sha256']})
    opt=torch.optim.AdamW(model.parameters(),lr=.009,weight_decay=0.);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.009,epochs=30,steps_per_epoch=len(loader));history=[]
    for epoch in range(30):
        start=time.time();train.epoch=epoch;model.train();losses=[];mse_losses=[];kl_losses=[];order=[]
        for x,y,ids in loader:
            order.extend(ids.tolist());opt.zero_grad(set_to_none=True);pred=model(x.cuda());target=y.cuda()
            if objective is None:mse=(pred-target).square().mean();kl=torch.zeros((),device='cuda');loss=mse
            else:loss,mse,kl=objective(pred,target)
            assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
            opt.step();sched.step();losses.append(float(loss.detach()));mse_losses.append(float(mse.detach()));kl_losses.append(float(kl.detach()))
        item={'epoch':epoch+1,'train_total_loss':float(np.mean(losses)),'train_mse':float(np.mean(mse_losses)),'train_KL':float(np.mean(kl_losses)),
            'compressed_train_clips':sum(train.use_compressed(i) for i in range(len(train))),'sample_order_sha256':hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest(),'elapsed_s':time.time()-start}
        history.append(item);ck={'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'config':cfg,'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':gen.get_state()}
        atomic_save(ck,p/'last.pt');write(p/'history.json',history);write(p/'status.json',{'state':'running','completed_epochs':epoch+1})
        print(json.dumps({'fold':a.fold,'loss':a.loss,**item}),flush=True)
    atomic_save({'model':model.state_dict(),'config':cfg,'epoch':29},p/'final.pt');expected=model_hash(model)
    model.load_state_dict(torch.load(p/'final.pt',map_location='cuda',weights_only=False)['model']);assert model_hash(model)==expected
    write(p/'status.json',{'state':'complete','completed_epochs':30,'final_epoch':30,'reload_verified':True,'checkpoint_sha256':sha(p/'final.pt'),'model_sha256':expected,'test_opened':False})
    print('COMPLETE',a.fold,a.loss,flush=True)
if __name__=='__main__':main()
