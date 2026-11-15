"""One authorized AdaBN launch at first eligible resource window; fail closed."""
import os,time,traceback
from common import *
def main():
 lease=lock('adabn_controller.lock')
 assert not (P/'ADABN_CONTROLLER_LAUNCH.json').exists(),'Existing controller must be inspected'
 assert not (P/'adabn_LAUNCH.json').exists() and not (P/'adabn/ERROR.json').exists()
 save(P/'ADABN_CONTROLLER_LAUNCH.json',dict(pid=os.getpid(),utc=now(),code=hashes()))
 while True:
  q=read(P/'BUDGET_ESTIMATE.json');a=read(P/'A100_INVENTORY.json')
  ages=[(datetime.now(timezone.utc)-datetime.fromisoformat(x)).total_seconds() for x in [q['utc'],a['measured_utc']]]
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines();idx,uuid,free,util=[x.strip() for x in rows[0].split(',')]
  other=[]
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
  for row in apps:
   device,pid,memory=[x.strip() for x in row.split(',')]
   if device==uuid and int(pid)!=3368998:other.append(dict(pid=int(pid),memory_MiB=memory))
  state='waiting_fresh_inventory' if max(ages)>3600 else 'waiting_GPU0' if int(free)<6000 or int(util)!=0 or other else 'eligible'
  save(P/'ADABN_CONTROLLER_STATUS.json',dict(state=state,utc=now(),pid=os.getpid(),free_MiB=int(free),utilization=int(util),other_GPU0_processes=other,inventory_age_s=ages))
  if state=='eligible':
   guard(True)
   subprocess.run(['/data_pool/khangle/facevitals/envs/hr/bin/python',str(P/'launch.py'),'adabn'],check=True)
   save(P/'ADABN_CONTROLLER_STATUS.json',dict(state='launch_requested_check_actual_worker_and_GPU',utc=now(),worker=read(P/'adabn_LAUNCH.json')));break
  time.sleep(20)
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'ADABN_CONTROLLER_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
