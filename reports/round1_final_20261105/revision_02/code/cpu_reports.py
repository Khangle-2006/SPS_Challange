import json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);code=root/'code';started=time.time()
while not (root/'frozen_data/FROZEN_WINDOWS.json').exists():
    if time.time()-started>7200:raise TimeoutError('Frozen preparation did not finish within2hours')
    time.sleep(5)
steps=[['freeze_report.py','--root',str(root)],['lag_check.py','--root',str(root),'--pos-source','/data_pool/khangle/facevitals/code/hr_baseline.py'],['mcd_c5_report.py','--root',str(root)],['hybrid.py','--root',str(root),'--code','/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102','--controls','/mnt/samsung_t5/khangle/facevitals/t8_oval_20261031/runs','--joint','/mnt/samsung_t5/khangle/facevitals/t8_polarity_v3_20261102']]
for step in steps:
    print('START',step[0],flush=True)
    subprocess.run([sys.executable,str(code/step[0]),*step[1:]],check=True)
    print('COMPLETE',step[0],flush=True)
(root/'CPU_REPORTS_COMPLETE.json').write_text(json.dumps(dict(state='complete',scripts=[s[0] for s in steps],elapsed_s=time.time()-started)))
