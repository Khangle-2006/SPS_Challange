import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
import sys,json,csv,time,hashlib,fcntl,traceback,gc
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;E=R/'round1_execution_20261110';S=R/'round1_bayes_20261108/domain_sweep_v2_20261110'
sys.path[:0]=[str(E),str(S)]
import common
from sweep import make_model,deadline
from repair_loss import phase_loss
from domain_loss import weak_ce
from btc_data import BTCData
from training_provider import DiskProvider
from test_static_tsm import static_result
from t8_t1_loader import supported
from t8_evaluate import decode_difference

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def check_budget():
 used=sum(f.stat().st_size for f in P.rglob('*') if f.is_file());assert used<1_000_000_000,'STOP_NEW_SCOPE_1GB'
 deadline(json.load(open(E/'PLAN_ELIGIBLE.json')))
def cohorts():
 rows=list(csv.DictReader(open(E/'inference_repair_20261113/report_original/BTC_VIDEOS.csv')))
 ids={r['recording'] for r in rows if r['condition']=='native'};assert len(ids)==218;return ids

class RateBTC(BTCData):
 def __init__(self,fold,shuffle=False):
  super().__init__(fold,max_bytes=450_000_000);ids=cohorts();self.rows=[v for v in self.rows if v['recording'] in ids]
  self.train=[(v,s) for v,s in self.train if v['recording'] in ids]
  self.people=sorted({v['person'] for v,s in self.train});self.by_person={p:[z for z in self.train if z[0]['person']==p] for p in self.people}
  self.targets={v['recording']:float(v['pr_bpm']) for v in self.rows if v['fold']!=fold};self.permutation={}
  if shuffle:
   groups={p:sorted({v['recording'] for v,s in self.by_person[p]}) for p in self.people};strata={}
   for p,vs in groups.items():strata.setdefault(len(vs),[]).append(p)
   rng=np.random.default_rng(0);original=self.targets.copy()
   for n,ps in sorted(strata.items()):
    donors=rng.permutation(ps)
    for person,donor in zip(ps,donors):
     self.permutation[person]=str(donor)
     for target,source in zip(groups[person],groups[donor]):self.targets[target]=original[source]
 def batch(self,rng):
  xs=[];labels=[];draws=[]
  for person in rng.choice(self.people,8,replace=False):
   choices=self.by_person[person]
   for attempt in range(1000):
    v,_=choices[int(rng.integers(len(choices)))];rate=float(np.exp(rng.uniform(np.log(.67),np.log(1.5))));label=self.targets[v['recording']]*rate
    if not 45<=label<=170:continue
    rgb,t,valid=self.load(v,'native');latest=t[-1]-10*rate
    if latest<t[0]:continue
    start=float(rng.uniform(t[0],latest)) if latest>t[0] else float(t[0]);q=start+rate*np.arange(301)/30
    if not supported(t,valid,q[0],q[-1]):continue
    j=np.searchsorted(t,q,side='right').clip(1,len(t)-1);w=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)
    x=rgb[j-1].astype(np.float32)*(1-w[:,None,None,None])+rgb[j].astype(np.float32)*w[:,None,None,None]
    if not np.isfinite(x).all() or x.std()<1e-6:continue
    xs.append(((x-x.mean())/x.std()).transpose(0,3,1,2).copy());labels.append(label);draws.append(dict(person=str(person),recording=v['recording'],start=start,rate=rate,label=label));break
   else:raise RuntimeError('No valid speed-augmented interval for selected person')
  assert len({x['person'] for x in draws})==8
  return np.stack(xs),np.asarray(labels,np.float32),draws

def evaluate(model,data,videos,device):
 import torch
 model.eval();out=[]
 for v in videos:
  preds=[];windows=[]
  # Same frozen10s candidates, no cherry-picking successful windows.
  starts=sorted({r['start'] for r in data.candidates(v,'native') if r['scheme']=='10s'})
  for start in starts:
   pred=None;reason=None
   try:
    with torch.inference_mode():wave=model(torch.from_numpy(data.clip(v,start)[None]).to(device))[0].cpu().numpy()
    pred=float(decode_difference(wave));assert np.isfinite(pred);preds.append(pred)
   except ValueError as exc:reason=str(exc)
   windows.append(dict(start=start,prediction=pred,reason=reason))
  out.append(dict(recording=v['recording'],person=v['person'],fold=v['fold'],group=v['group'],task=v['task'],truth=float(v['pr_bpm']),prediction=float(np.median(preds)) if preds else None,windows=windows))
 return out
def metric(rows):
 people=sorted({r['person'] for r in rows});valid=[r for r in rows if r['prediction'] is not None]
 mae=float(np.mean([np.mean([abs(r['prediction']-r['truth']) for r in valid if r['person']==p]) for p in people])) if len(valid)==len(rows) else None
 return dict(videos=len(rows),accepted=len(valid),people=len(people),MAE=mae)

def one(plan,fold,shuffle,device):
 import torch,t8_run as base
 dest=P/'runs'/('fold2_shuffle' if shuffle else f'fold{fold}');assert not dest.exists(),'No automatic restart'
 dest.mkdir(parents=True);base.seed_all(2027,fold);run=next(x for x in plan['runs'] if x['fold']==fold)
 data=RateBTC(fold,shuffle);model=make_model(run,device)
 training=[v for v in data.rows if v['fold']!=fold and v['person'] in data.people and data.starts(v,'native')]
 rng=np.random.default_rng(2027+fold);fixed=[training[i] for i in sorted(rng.choice(len(training),40,replace=False))]
 save(dest/'TRAIN40.json',[v['recording'] for v in fixed]);save(dest/'SHUFFLE.json',data.permutation)
 save(dest/'BEFORE_TRAIN40.json',evaluate(model,data,fixed,device))
 provider=DiskProvider(common.read(common.R/'t8_oval_20261031/training_index.json'),run['btc_bpp_quartiles'],300_000_000)
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=plan['weight_decay']);history=[]
 save(dest/'PROVENANCE.json',dict(initial=run,epochs=10,lr=1e-4,weak_weight=1.,sigma=5.,AiPEX=False,plan_sha256=sha(E/'PLAN_ELIGIBLE.json'),code_sha256=sha(__file__),shuffle=shuffle,shuffle_seed=0,shuffle_policy='whole-person blocks within identical training-video-count strata; singleton/fixed permutations reported'))
 for epoch in range(1,11):
  check_budget();model.train();rng=np.random.default_rng(np.random.SeedSequence([2027,fold,epoch]));losses=[];drawhash=hashlib.sha256();started=time.monotonic()
  for step in range(run['steps_per_epoch']):
   xs=[];ys=[]
   for ds in ['UBFC2','MCD']*4:
    choices=run['public_jobs'][ds];j=choices[int(rng.integers(len(choices)))];rate=float(rng.uniform(*j['rates']))
    x,y,_=provider.clip(ds,j['recording'],j['start'],j['recipe'],j['lowres'],fold,rate);xs.append(x);ys.append(y)
   optimizer.zero_grad(set_to_none=True)
   output=model(torch.from_numpy(np.stack(xs)).to(device));sup=phase_loss(output,torch.from_numpy(np.stack(ys)).to(device)).mean();assert torch.isfinite(sup);sup.backward();del output,xs,ys
   xx,labels,draws=data.batch(rng);output=model(torch.from_numpy(xx).to(device));aux=weak_ce(output,torch.from_numpy(labels).to(device),sigma_bpm=5.);assert torch.isfinite(aux);aux.backward()
   assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   optimizer.step();losses.append([float(sup.detach()),float(aux.detach())]);drawhash.update(json.dumps(draws,sort_keys=True).encode());del output,xx,aux,sup
   save(dest/'STATUS.json',dict(state='training',epoch=epoch,step=step+1,steps=run['steps_per_epoch'],pid=os.getpid()))
  model.cpu();static=static_result(model);save(dest/f'STATIC_{epoch}.json',static)
  entry=dict(epoch=epoch,phase_loss=float(np.mean(losses,axis=0)[0]),weak_ce=float(np.mean(losses,axis=0)[1]),seconds=time.monotonic()-started,static_pass=static['hr'] is None,draw_sha256=drawhash.hexdigest())
  model.to(device);tr=evaluate(model,data,fixed,device);save(dest/f'TRAIN40_EPOCH{epoch}.json',tr);entry['train40']=metric(tr);history.append(entry);save(dest/'history.json',history)
  torch.save(dict(model=model.cpu().state_dict(),epoch=epoch,fold=fold,seed=2027),dest/'last.pt')
  if not entry['static_pass']:
   save(dest/'REJECTED_STATIC.json',entry);return
  model.to(device)
 torch.save(dict(model=model.cpu().state_dict(),epoch=10,fold=fold,seed=2027),dest/'final.pt');model.to(device)
 rows=evaluate(model,data,[v for v in data.rows if v['fold']==fold],device);save(dest/'NATIVE.json',rows);save(dest/'COMPLETE.json',dict(**metric(rows),checkpoint_sha256=sha(dest/'final.pt'),static_pass=True))
 del model,optimizer,provider,data;gc.collect();torch.cuda.empty_cache()

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--validate',action='store_true');a=parser.parse_args()
 plan=json.load(open(E/'PLAN_ELIGIBLE.json'));check_budget()
 if a.validate:
  import torch
  torch.set_num_threads(2)
  data=RateBTC(2);x,y,draws=data.batch(np.random.default_rng(0));assert x.shape==(8,301,3,36,36) and np.isfinite(x).all();assert np.all((y>=45)&(y<=170))
  control=RateBTC(2,True);assert set(control.targets)==set(data.targets);assert all(v['fold']!=2 for v,s in data.train)
  model=make_model(next(r for r in plan['runs'] if r['fold']==2),'cpu');model.train();z=model(torch.from_numpy(x));loss=weak_ce(z,torch.from_numpy(y));assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  save(P/'CPU_VALIDATION.json',dict(passed=True,shape=list(x.shape),finite_backward=True,weak_ce=float(loss.detach()),distinct_people=len({d['person'] for d in draws}),shuffle_changed_people=sum(p!=d for p,d in control.permutation.items())));return
 assert json.load(open(P/'CPU_VALIDATION.json'))['passed']
 guards=json.load(open(P/'LAUNCH_GUARDS.json'));assert guards['passed'] and time.time()-guards['measured_epoch']<3600
 assert guards['train_sha256']==sha(__file__)
 lease=(P/'train.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 shared=(E/'gpu_execution.lock').open('a');fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
 from runtime_shared_v4 import gpu
 device=gpu();save(P/'TRAIN_STARTED.json',dict(pid=os.getpid(),device=device,source_sha256=sha(__file__)))
 for fold,shuf in [(2,False),(2,True),(0,False),(1,False),(3,False),(4,False)]:
  one(plan,fold,shuf,device)
 save(P/'TRAINING_TERMINAL.json',dict(runs=6,no_score_based_early_stop=True,holdout_evaluations=0))
if __name__=='__main__':
 try:main()
 except Exception:save(P/'TRAIN_ERROR.json',dict(traceback=traceback.format_exc()));raise
