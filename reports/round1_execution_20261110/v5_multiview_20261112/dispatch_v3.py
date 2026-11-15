"""One bounded dependency controller; failures are preserved, never restarted."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import time,subprocess,traceback
from runtime import *
os.environ['CUDA_VISIBLE_DEVICES']=''
def main():
    lease=lock(P/'dispatch_v3.lock');assert not (P/'DISPATCH_V3_STARTED.json').exists(),'Existing controller; inspect instead'
    save(P/'DISPATCH_V3_STARTED.json',dict(utc=now(),pid=os.getpid(),code_sha256=sha(__file__),action='wait current augmentation then validate then final V5 queue'))
    while not (P/'FOLD0_READY.json').exists():
        hard_stop()
        for name in ['PREP_ERROR.json','AUG_ERROR.json','READY_ERROR.json']:
            if (P/name).exists():raise RuntimeError('Dependency failed: '+name)
        save(P/'DISPATCH_V3_STATUS.json',dict(state='waiting_augmentation',utc=now(),pid=os.getpid()));time.sleep(20)
    for script,cpus,env,label in [('train_v2.py','0,1','0','GPU_training')]:
        while True:
            try:budget(fresh=True);break
            except AssertionError as e:
                if str(e)!='FRESH_CROSS_SERVER_INVENTORY_REQUIRED':raise
                save(P/'DISPATCH_V3_STATUS.json',dict(state='waiting_fresh_inventory',stage=label,utc=now()));hard_stop();time.sleep(20)
        with (P/(script+'.log')).open('x') as log:
            cmd=['taskset','-c',cpus,'nice','-n','10','/data_pool/khangle/facevitals/envs/hr/bin/python','-u',str(P/script)]
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env={**os.environ,'CUDA_VISIBLE_DEVICES':env,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
            save(P/'DISPATCH_V3_STATUS.json',dict(state=label,utc=now(),pid=os.getpid(),child_pid=child.pid,command=cmd));rc=child.wait()
            save(P/(script+'.EXIT.json'),dict(utc=now(),exit_code=rc,child_pid=child.pid));assert rc==0,script+' failed, no restart'
    save(P/'DISPATCH_V3_COMPLETE.json',dict(utc=now(),training_complete=True,report_review_pending=True))
if __name__=='__main__':
    try:main()
    except Exception as e:save(P/'DISPATCH_V3_ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
