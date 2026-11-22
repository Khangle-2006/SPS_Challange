import os,json,time,subprocess,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
r=json.loads((P/'PREFLIGHT_C_RECOVERY_V3.json').read_text())
assert r['passed'] and time.time()-r['measured_epoch']<3600
assert not (P/'C_RECOVERY_V3_LAUNCH.json').exists()
assert not list(P.glob('C_ERROR_SHARD*.json'))
v=os.statvfs('/dev/shm');assert v.f_bavail*v.f_frsize>2_100_000_000
m=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
assert int(m['MemAvailable'].split()[0])*1024>3_000_000_000
old=(P/'constraints_reader_v2.py').read_text()
for a,b in [('PREFLIGHT.json','PREFLIGHT_C_RECOVERY_V3.json'),('380_000_000','1_900_000_000'),('400_000_000','2_000_000_000'),('reader_C_parallel_20261116','reader_C_recovery_20261117'),('C_PROTOCOL_V2.json','C_PROTOCOL_V3.json'),('ephemeral400MB','ephemeral2000MB'),('STOP_RAM400MB','STOP_RAM2000MB')]:old=old.replace(a,b)
assert old==(P/'constraints_reader_v3.py').read_text()
for f in ['constraints_reader_v3.py','report_followup.py']:compile((P/f).read_text(),str(P/f),'exec')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
jobs=[]
for i in range(4):
 with (P/f'C_SHARD{i}.log').open('a') as log:
  proc=subprocess.Popen(['nice','-n','15','taskset','-c',str(12+i),'/data_pool/khangle/facevitals/envs/storage_pilot/bin/python','-u',str(P/'constraints_reader_v3.py'),str(i)],stdout=log,stderr=log,env=env,start_new_session=True)
 jobs.append(dict(shard=i,pid=proc.pid))
with (P/'report_followup_recovery_v3.log').open('a') as log:
 proc=subprocess.Popen(['nice','-n','15','taskset','-c','11','/data_pool/khangle/facevitals/envs/hr/bin/python','-u',str(P/'report_followup.py')],stdout=log,stderr=log,env=env,start_new_session=True)
receipt=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),workers=jobs,reporter=proc.pid,code_sha256=hashlib.sha256((P/'constraints_reader_v3.py').read_bytes()).hexdigest(),completed_preserved=483,recipe_unchanged=True)
(P/'C_RECOVERY_V3_LAUNCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
