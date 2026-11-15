"""Five fold-safe fine-tunes; frozen plan, budgets and locks precede CUDA allocation."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,time,math,fcntl,subprocess,traceback,gc
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from codec_provider import ROOT,V3,Provider,read,sha
P=Path(__file__).resolve().parent;OLD=P.parent.parent/'round1_v4_20261105';OUT=P/'runs';INDEX=ROOT/'t8_oval_20261031/training_index.json'
sys.path.insert(0,str(OLD/'phase_repair'));sys.path.insert(0,str(OLD/'tsm_padding'));sys.path.insert(0,str(OLD/'code'))
from repair_loss import phase_loss
from domain_loss import weak_ce,sinc_loss
from replicate_tsm import use_replicate_padding
from test_static_tsm import static_result
from btc_data import BTCData
def save(p,a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(a,indent=2,allow_nan=False));tmp.replace(p)
def now():return datetime.now(timezone.utc).isoformat()
def deadline(plan):assert datetime.now(timezone.utc)<datetime.fromisoformat(plan['deadline_utc']),'Hard stop 22 November ICT'
def size(p):return sum(x.stat().st_size for x in Path(p).rglob('*') if x.is_file()) if Path(p).exists() else 0
def budget(plan,fresh):
 approval=read(P/'SWEEP_BUDGET_APPROVAL.json');assert approval['status']=='AUTHORIZED_BY_USER' and approval['scope']=='five_domain_runs_and_CPU_outputs'
 r=read(P/'INVENTORY.json');age=(datetime.now(timezone.utc)-datetime.fromisoformat(r['measured_utc'])).total_seconds();assert age>=-60 and (not fresh or age<=3600),'fresh cross-server inventory required'
 from polarity_guard import storage_guard
 b=storage_guard(read(INDEX),ROOT/'t8_oval_20261031',ROOT/'t8_polarity_v3_20261102')
 whole=b['total_bytes']+size(OLD)+size('/data_pool/khangle/facevitals/reports/round1_v4_20261105')+r['a100_v4_bytes']+size(P.parent)
 project=sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in (ROOT,Path('/data_pool/khangle/facevitals')))+r['a100_bytes']
 # Pending outputs reserve decreases only as this new run root grows. Other workers retain their own reserves.
 pending=max(0,plan['output_reserve_bytes']-size(OUT));reserve=250_000_000+70_000_000+pending+100_000_000
 assert whole+reserve<=approval['T8_cap_bytes'],'T8 whole including CPU/public/sweep/headroom'
 assert size(OLD/'btc_zeroshot')+size(P.parent)+70_000_000+pending<=approval['BTC_cap_bytes'],'BTC/Bayes/sweep derivatives cap'
 assert project+reserve<=500_000_000_000,'project cap'
 return dict(utc=now(),whole_bytes=whole,project_bytes=project,reserved_bytes=reserve,receipt_sha256=sha(P/'INVENTORY.json'))
def make_model(run,device):
 import torch
 from EfficientPhys_fixed import EfficientPhys
 assert sha(run['checkpoint'])==run['checkpoint_sha256']
 assert sha(Path(run['checkpoint']).parent/'provenance.json')==run['checkpoint_provenance_sha256']
 ck=torch.load(run['checkpoint'],map_location='cpu',weights_only=False)
 model=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36));model.load_state_dict(ck['model'],strict=True)
 assert static_result(model)['hr'] is None,'Initial static contract failed'
 return model.to(device)
def evaluate_btc(model,fold,dest,device):
 import torch
 from t8_evaluate import decode_difference
 data=BTCData(fold);out=[];model.eval()
 for v in data.rows:
  if v['fold']!=fold:continue
  for condition in ['native','c5']:
   # Candidate list includes failures; never obtain a smaller denominator from successful predictions.
   candidates=data.candidates(v,condition)
   starts=sorted({r['start'] for r in candidates if r['scheme']=='10s'})
   for start in starts:
    pred=None;reason=None
    try:
     x=data.clip(v,start,condition)
     with torch.no_grad():wave=model(torch.from_numpy(x[None]).to(device))[0].cpu().numpy()
     pred=decode_difference(wave)
    except ValueError as e:reason=str(e)
    out.append(dict(person=v['person'],recording=v['recording'],fold=fold,group=v['group'],task=v['task'],skin=v['skin'],condition=condition,start=start,truth=v['pr_bpm'],prediction=pred,reason=reason))
 save(dest/'BTC_DEV_OOF.json',dict(rows=out,input_hashes=data.hashes,excluded_videos=data.exclusions,split_sha256=sha(OLD/'btc_zeroshot/btc_split.json'),C5_recipe='historical_x264_comparator_not_new_C5proxy',holdout_evaluations=0))
 # Person means first; report group signed bias beside error and explicit coverage.
 metrics=[]
 for condition in ['native','c5']:
  for group in ['all','Normal','Abnormal']:
   rr=[r for r in out if r['condition']==condition and (group=='all' or r['group']==group)];vv=[]
   for rec in sorted({r['recording'] for r in rr}):
    clips=[r for r in rr if r['recording']==rec and r['prediction'] is not None]
    if clips:vv.append(dict(person=clips[0]['person'],truth=clips[0]['truth'],prediction=float(np.median([r['prediction'] for r in clips]))))
   errors=[];bias=[];constant_errors=[]
   constant=float(np.median([v['pr_bpm'] for v in data.rows if v['fold']!=fold]))
   for person in sorted({r['person'] for r in vv}):
    a=[r for r in vv if r['person']==person];errors.append(np.mean([abs(r['prediction']-r['truth']) for r in a]));bias.append(np.mean([r['prediction']-r['truth'] for r in a]));constant_errors.append(np.mean([abs(constant-r['truth']) for r in a]))
   metrics.append(dict(condition=condition,group=group,videos=len(vv),candidate_videos=sum(v['fold']==fold and (group=='all' or v['group']==group) for v in data.rows),people=len(errors),valid_windows=sum(r['prediction'] is not None for r in rr),candidate_windows=len(rr),MAE=float(np.mean(errors)) if errors else None,bias=float(np.mean(bias)) if bias else None,constant_trainfold_median=constant,constant_same_videos_MAE=float(np.mean(constant_errors)) if constant_errors else None))
 save(dest/'BTC_SUMMARY.json',metrics)
def run_one(plan,run,device):
 import torch
 import t8_run as baseline
 fold=run['fold'];dest=OUT/f'fold{fold}'
 if dest.exists():
  st=read(dest/'status.json');assert st['state']=='complete' and st['plan_sha256']==sha(P/'PLAN.json') and sha(dest/'final.pt')==st['checkpoint_sha256'],'Partial or altered run: manual inspection required'
  return
 deadline(plan);budget(plan,False);baseline.resource_guard();baseline.seed_all(2027,fold)
 assert int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])*1024>=16_000_000_000,'RAM guard'
 model=make_model(run,device);provider=Provider(read(INDEX),run['btc_bpp_quartiles'],plan['codec_cache_bytes']);btc=BTCData(fold) if fold>=2 else None
 if btc:assert set(btc.people)<=set(run['btc_train_people']) and set(btc.people).isdisjoint(run['btc_val_people'])
 opt=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay']);dest.mkdir(parents=True)
 save(dest/'provenance.json',dict(plan_sha256=sha(P/'PLAN.json'),frozen_sha256=sha(P/'FROZEN.json'),initial=run,selection=plan['selection'],official_MATLAB_parity=False));history=[]
 for epoch in range(1,plan['epochs']+1):
  deadline(plan);budget(plan,False);model.train();started=time.monotonic();rng=np.random.default_rng(np.random.SeedSequence([2027,fold,epoch]));sums=[];auxs=[];parts=[];draws=[]
  for step in range(run['steps_per_epoch']):
   deadline(plan);xs=[];ys=[]
   for ds in ['UBFC2','MCD']*4:
    options=run['public_jobs'][ds];j=options[int(rng.integers(len(options)))];rate=float(rng.uniform(*j['rates']))
    x,y,receipt=provider.clip(ds,j['recording'],j['start'],j['recipe'],j['lowres'],fold,rate);xs.append(x);ys.append(y);draws.append(dict(**j,rate=rate))
   x=torch.from_numpy(np.stack(xs)).to(device);y=torch.from_numpy(np.stack(ys)).to(device);opt.zero_grad(set_to_none=True)
   pred=model(x);sup=phase_loss(pred,y).mean();assert torch.isfinite(sup);sup.backward();auxvalue=0.
   del x,y,pred
   if btc:
    xx,pr,freq,ids=btc.aux(rng,fold==2);pp=model(torch.from_numpy(xx).to(device))
    if fold==2:aux=weak_ce(pp,torch.tensor(pr,device=device,dtype=pp.dtype));components={}
    else:aux,components=sinc_loss(pp,freq)
    weighted=plan['btc_aux_weight'][str(fold)]*aux;assert torch.isfinite(weighted);weighted.backward();auxvalue=float(weighted.detach());parts.append({k:float(v.detach()) for k,v in components.items()});draws.append(dict(btc=ids,loss=run['btc_training']))
    del pp,xx,aux,weighted
   assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'Nonfinite gradient'
   opt.step();sums.append(float(sup.detach()));auxs.append(auxvalue)
   save(dest/'status.json',dict(state='training',epoch=epoch,step=step+1,steps=run['steps_per_epoch'],utc=now(),pid=os.getpid(),plan_sha256=sha(P/'PLAN.json')))
  entry=dict(epoch=epoch,supervised_loss=float(np.mean(sums)),weighted_BTC_loss=float(np.mean(auxs)),total_loss=float(np.mean(sums)+np.mean(auxs)),seconds=time.monotonic()-started,updates=len(sums),codec_cache_bytes=provider.bytes,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),SiNC_components={k:float(np.mean([c[k] for c in parts])) for k in parts[0]} if parts and parts[0] else {})
  history.append(entry);save(dest/'history.json',history);save(dest/f'epoch{epoch}_draws.json',draws);save(dest/'CODEC_RECEIPTS.json',provider.receipts)
  if btc:save(dest/'BTC_TRAIN_INPUT_HASHES.json',btc.hashes)
  baseline.save_checkpoint(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,plan_sha256=sha(P/'PLAN.json')),dest/'last.pt')
  print(json.dumps(dict(fold=fold,**entry)),flush=True)
  # Existing pilot gate is not reused for pretrained fine-tuning; retain measured epoch5 trend.
  if epoch==5:save(dest/'EPOCH5_TREND.json',dict(epoch1=history[0],epoch5=entry,relative_total_reduction=1-entry['total_loss']/history[0]['total_loss'],action='report_only_no_checkpoint_selection'))
 baseline.save_checkpoint(dict(model=model.state_dict(),epoch=plan['epochs'],fold=fold,seed=2027,plan_sha256=sha(P/'PLAN.json')),dest/'final.pt')
 model.cpu();static=static_result(model);save(dest/'STATIC.json',static);assert static['hr'] is None,'Static test failed; stop queue and report'
 del provider,btc;gc.collect();torch.cuda.empty_cache();model.to(device)
 deadline(plan);evaluate_btc(model,fold,dest,device)
 from phase_evaluate import evaluate
 deadline(plan);evaluate(model,read(INDEX),fold,2027,dest/'public_evaluation',ROOT/'t8_polarity_v3_20261102',device)
 save(dest/'status.json',dict(state='complete',completed_epochs=plan['epochs'],checkpoint_sha256=sha(dest/'final.pt'),plan_sha256=sha(P/'PLAN.json'),static_pass=True,utc=now()))
 del model,opt;gc.collect();torch.cuda.empty_cache()
def main():
 os.environ['CUDA_VISIBLE_DEVICES']='0'
 plan=read(P/'PLAN.json');deadline(plan);assert sha(INDEX)==plan['index_sha256']
 frozen=read(P/'FROZEN.json')
 for file,h in frozen.items():assert sha(Path(file) if Path(file).is_absolute() else P/file)==h,('frozen file changed',file)
 assert read(P/'CPU_VALIDATION.json')['passed'] and read(P/'CODEC_SMOKE.json')['passed']
 locks=[]
 for path in [ROOT/'t8_oval_20261031/runs/queue.lock',ROOT/'t8_polarity_v3_20261102/queue.lock',OLD/'v4_repaired/queue.lock',OLD/'v4_padding_pilot/queue.lock',P.parent/'domain_sweep_20261109/queue.lock',P/'queue.lock']:
  h=path.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 assert not (P/'ERROR.json').exists(),'Inspect preserved error before any restart'
 b=budget(plan,True)
 import t8_run as baseline
 resources=baseline.resource_guard();save(P/'PREFLIGHT.json',dict(storage=b,resources=resources,utc=now(),plan_sha256=sha(P/'PLAN.json')))
 import torch
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35,0)
 save(P/'QUEUE_STATUS.json',dict(state='running',pid=os.getpid(),utc=now()))
 try:
  for run in plan['runs']:run_one(plan,run,'cuda:0')
  save(P/'QUEUE_STATUS.json',dict(state='complete',runs=5,utc=now()))
 except Exception:
  save(P/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));save(P/'QUEUE_STATUS.json',dict(state='failed',utc=now(),error='See ERROR.json'));raise
if __name__=='__main__':main()
