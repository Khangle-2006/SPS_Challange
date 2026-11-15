"""Dependency controller; one launch per stage, no automatic partial restart."""
import time,traceback
from common import *
PY='/data_pool/khangle/facevitals/envs/hr/bin/python'
def exists(name):return (P/name).exists()
def fresh():
 return max((datetime.now(timezone.utc)-datetime.fromisoformat(t)).total_seconds() for t in [read(P/'BUDGET_ESTIMATE.json')['utc'],read(P/'A100_INVENTORY.json')['measured_utc']])<3600
def available_gpu():
 row=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0];idx,uuid,free,util=[v.strip() for v in row.split(',')]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines();others=[x for x in apps if x.split(',')[0].strip()==uuid and int(x.split(',')[1])!=3368998]
 return int(idx)==0 and int(free)>=6000 and int(util)==0 and not others
def launch(name,script,cpu,args=(),gpu=False):
 folder=P/'execution_v2';folder.mkdir(exist_ok=True);receipt=folder/(name+'_LAUNCH.json');assert not receipt.exists(),'Inspect prior stage; no restart'
 guard(True)
 for file,h in read(P/'EXECUTION_FROZEN_V2.json')['files'].items():assert sha(file)==h,('Execution dependency changed',file)
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
  # D uses multiple explicitly disjoint folds; each has its own lease/output root.
  if script!='d_sliding.py':assert str(P/script) not in cmd,'Duplicate worker'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='2' if gpu else '1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2' if gpu else '1')
 cmd=['nice','-n','10' if gpu else '15','taskset','-c',cpu,PY,str(P/script),*map(str,args)]
 with (folder/(name+'.log')).open('ab') as log:q=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(receipt,dict(utc=now(),pid=q.pid,command=cmd,code_sha256=sha(P/script),dependencies=hashes()));return q
def main():
 lease=lock('continue_af_v2.lock');assert not exists('execution_v2/CONTROLLER_STARTED.json')
 for file,h in read(P/'EXECUTION_FROZEN_V2.json')['files'].items():assert sha(file)==h,('Execution dependency changed',file)
 save(P/'execution_v2/CONTROLLER_STARTED.json',dict(pid=os.getpid(),utc=now(),code=hashes()));children={};reported=set();progress_signature=None
 while True:
  for name,q in children.items():
   rc=q.poll()
   if rc is not None and name not in reported:
    save(P/f'execution_v2/{name}_EXIT.json',dict(utc=now(),pid=q.pid,returncode=rc));reported.add(name);assert rc==0,('Child failed; inspect preserved partial',name,rc)
  errors=[*P.glob('btc_c5ensemble/*ERROR.json'),*P.glob('constraint_preparation/*ERROR.json')]
  prep=list(P.glob('parallel/*ERROR.json'));known={}
  for f in prep:
   a=read(f)
   if f.name.startswith('warm_') and a['error'].startswith('C5 proxy bitrate failed:') and 'no automatic rate retuning' in a['error']:known[int(f.name.split('_')[1])]=sha(f)
  parent=read(P/'parallel/ERROR.json') if exists('parallel/ERROR.json') else {}
  codes=parent.get('exit_codes',[]);known_only=bool(codes) and all(code in [None,0] or (i>=4 and i-4 in known) for i,code in enumerate(codes))
  for f in prep:
   if f.name.startswith('warm_') and int(f.name.split('_')[1]) in known:continue
   if f.name in ['ERROR.json','controller_ERROR.json'] and known_only:continue
   errors.append(f)
  # Constraint failures remain explicit exclusions; C cannot start until every job has a verified eligibility receipt.

  for name in ['ADABN_CONTROLLER_ERROR.json','adabn/ERROR.json','adabn/C5_ERROR.json','C5_EVALUATION_ERROR.json','GOP_ERROR.json','SWEEP_ERROR.json']:
   if exists(name):errors.append(P/name)
  assert not errors,('Existing active-stage errors',list(map(str,errors)))
  progress_files=[*P.glob('adabn/fold*/COMPLETE.json'),*P.glob('adabn/fold*/C5PRIME.json'),*P.glob('sweep/fold*/NATIVE.json'),*P.glob('sweep/fold*/C5PRIME.json')]
  signature={str(f):sha(f) for f in progress_files}
  if signature and signature!=progress_signature:
   subprocess.run(['nice','-n','15','taskset','-c','5',PY,str(P/'report_progress.py')],check=True)
   save(P/'execution_v2/RESULT_UPDATE_READY.json',dict(utc=now(),files=signature,report='BTC_PROGRESS.md'));progress_signature=signature
  a3=exists('C5_EVALUATION_COMPLETE.json') and all(exists(f'btc_c5ensemble/SHARD{s}_COMPLETE.json') for s in range(4))
  ready=fresh();state='waiting_dependencies' if ready else 'waiting_fresh_inventory'
  if ready and exists('adabn/NATIVE_COMPLETE.json') and exists('C5_COMPLETE.json') and 'B_c5' not in children:
   children['B_c5']=launch('B_c5','adabn_c5.py','4')
  if ready and a3 and 'A3_report' not in children:children['A3_report']=launch('A3_report','report_c5.py','5')
  c_ready=a3 and exists('adabn/C5_COMPLETE.json') and exists('TRAIN_CACHE_ELIGIBLE_COMPLETE.json') and exists('AIPEX_COMPLETE.json') and exists('GOP_COMPLETE.json') and exists('FROZEN_ELIGIBLE.json')
  if ready and c_ready and 'C_sweep' not in children:
   state='waiting_GPU0_for_sweep'
   if available_gpu():children['C_sweep']=launch('C_sweep','start_sweep_eligible_v2.py','0,1',gpu=True)
  c_done=exists('SWEEP_STATUS.json') and read(P/'SWEEP_STATUS.json')['state']=='complete'
  if ready and c_done and 'D_select' not in children:children['D_select']=launch('D_select','select_d.py','4')
  if ready and exists('D_SELECTION.json'):
   for fold in range(5):
    name=f'D_fold{fold}'
    if name not in children:children[name]=launch(name,'d_sliding.py',str(34+fold),[fold])
  if ready and all(exists(f'D/fold{f}/COMPLETE.json') for f in range(5)) and 'D_stack' not in children:children['D_stack']=launch('D_stack','d_stack.py','4')
  save(P/'execution_v2/STATUS.json',dict(state=state,utc=now(),pid=os.getpid(),A3_complete=a3,C_prerequisites=c_ready,children={n:dict(pid=q.pid,returncode=q.poll()) for n,q in children.items()},holdout_evaluations=0))
  if exists('D_STACK_COMPLETE.json'):
   if ready and 'E_report' not in children:children['E_report']=launch('E_report','report_final.py','4')
   if exists('FINAL_REPORT_RECEIPT.json'):
    save(P/'execution_v2/COMPUTE_COMPLETE.json',dict(utc=now(),report_review_pending=True,holdout_evaluations=0));break
  time.sleep(20)
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'execution_v2/ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
