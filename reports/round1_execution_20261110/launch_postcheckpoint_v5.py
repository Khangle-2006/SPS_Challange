"""One-time CPU controller launch for already-running fasttrack sweep."""
import os, subprocess
from runtime_shared_v4 import install
install()
from common import *

def main():
    lease=lock('postcheckpoint_launch_v5.lock')
    assert not (P/'execution_v5/LAUNCH.json').exists()
    assert not (P/'DEFERRED_EVAL_V5_STARTED.json').exists()
    assert not (P/'execution_v5/CONTROLLER_STARTED.json').exists()
    files=dict(read(P/'FROZEN_FASTTRACK_V5.json')['files'])
    for file,digest in files.items():assert sha(file)==digest, file
    for name in ['FROZEN_FASTTRACK_V5.json','deferred_eval_v5.py','cpu_entry_v5.py','continue_af_v5.py','launch_postcheckpoint_v5.py']:
        files[str(P/name)]=sha(P/name)
    save(P/'POSTCHECKPOINT_FROZEN_V5.json',dict(utc=now(),files=files,training_unchanged=True))
    guard(True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    folder=P/'execution_v5';folder.mkdir(exist_ok=True)
    launched={}
    for script,cpu in [('deferred_eval_v5.py','7'),('continue_af_v5.py','6')]:
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            except (PermissionError,FileNotFoundError,ProcessLookupError):continue
            assert str(P/script) not in cmd, 'Duplicate '+script
        cmd=['nice','-n','15','taskset','-c',cpu,'/data_pool/khangle/facevitals/envs/hr/bin/python',str(P/script)]
        with (folder/(script+'.log')).open('ab') as log:
            q=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launched[script]=dict(pid=q.pid,command=cmd)
        save(folder/'LAUNCH.json',dict(utc=now(),workers=launched,holdout_evaluations=0))
    print(launched)

if __name__=='__main__':main()
