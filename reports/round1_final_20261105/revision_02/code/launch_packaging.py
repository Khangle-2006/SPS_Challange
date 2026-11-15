import os,json,subprocess,sys,datetime
from pathlib import Path
r=Path(sys.argv[1]);ds=sys.argv[2];base=Path('/mnt/samsung_t5/khangle/facevitals') if ds=='MCD' else Path('/data/khangle/facevitals')
job=r/('packaging_'+ds)/'attempt2';venv=job/'venv';out=job/'artifact';assert venv.is_dir() and not out.exists()
assert not (job/'LAUNCH.json').exists()
rec='1115_FullHDwebcam_after' if ds=='MCD' else 'subject1'
data=base/'datasets'/('MCD-rPPG' if ds=='MCD' else 'UBFC-rPPG/DATASET_2')
trace=base/'reports/t4_submission_20261030'/('mcd_e2e_v1' if ds=='MCD' else 'ubfc_e2e_v1')/'artifact/roi_trace.npz'
cmd=['nice','-n','10','taskset','-c','5' if ds=='MCD' else '3',str(venv/'bin/python'),str(r/'code/process_round1.py'),'--out',str(out),'--index',str(r/'packaging_inputs/training_index.json'),'--data-root',str(data),'--checkpoint',str(r/'packaging_inputs'/ds/'final.pt'),'--checkpoint-manifest',str(r/'packaging_inputs'/ds/'checkpoint_manifest.json'),'--roi-trace',str(trace),'--oof-windows',str(r/'packaging_inputs'/ds/'oof_windows.json'),'--dataset',ds,'--recording',rec,'--branch','v3_joint','--frozen-cache',str(base/'t8_oval_20261031'/ds/rec/'native/crop')]
if ds=='MCD':cmd+=['--audit',str(base/'mapping_audit_20261030')]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
log=open(job/'process.log','x');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(job/'LAUNCH.json').write_text(json.dumps(dict(pid=p.pid,command=cmd,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True),indent=2))
print(p.pid)
