"""Single bounded launch after current guards. Does not wait or retry."""
import os,sys,subprocess,json
from common import *
task=sys.argv[1]
assert task in ['adabn','c5smoke','c5smoke2','c5full']
guard(True)
if task=='adabn':
 t=read(P/'ADABN_CPU_TEST.json');assert t['passed'] and t['adabn_sha256']==sha(P/'adabn.py') and t['common_sha256']==sha(P/'common.py')
 a=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]
 _,free,util=map(lambda s:int(s.strip()),a.split(','));assert free>=6000 and util==0,('WAIT_GPU0',a)
 script='adabn.py';args=[];cpu='0,1';env='/data_pool/khangle/facevitals/envs/hr/bin/python'
else:
 script='c5prime_v2.py';args=['--smoke'] if task.startswith('c5smoke') else [];cpu='7';env='/data_pool/khangle/facevitals/envs/storage_pilot/bin/python'
 if task=='c5full':assert (P/'C5_SMOKE.json').exists() and not (P/'C5_ERROR_V2.json').exists()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except (PermissionError,FileNotFoundError,ProcessLookupError):continue
 assert str(P/script) not in cmd,'Duplicate worker'
assert not (P/(task+'_LAUNCH.json')).exists(),'Inspect prior launch'
cmd=['nice','-n','10' if task=='adabn' else '15','taskset','-c',cpu,env,str(P/script)]+args
worker_env=dict(os.environ)
if task!='adabn':worker_env.update(LIBGL_ALWAYS_SOFTWARE='1',EGL_PLATFORM='surfaceless',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
with (P/(task+'.log')).open('ab') as log:q=subprocess.Popen(cmd,cwd=P,env=worker_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
save(P/(task+'_LAUNCH.json'),dict(pid=q.pid,utc=now(),command=cmd,code=hashes()))
print(q.pid)
