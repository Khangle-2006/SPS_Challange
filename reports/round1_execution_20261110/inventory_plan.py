import os,sys,json,math,subprocess
from pathlib import Path
from datetime import datetime,timezone
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent;B=P.parent/'round1_bayes_20261108';V=P.parent/'round1_v4_20261105';S=B/'domain_sweep_v2_20261110'
sys.path.insert(0,str(S))
from codec_provider import Provider,read,sha
from polarity_guard import storage_guard
index=read(R/'t8_oval_20261031/training_index.json');plan=read(S/'PLAN.json');dev=read(V/'btc_zeroshot/dev_manifest.json')['videos']
provider=Provider(index,plan['runs'][0]['btc_bpp_quartiles'])
pub=read(B/'public_comparison/MANIFEST.json')['records'];dur={};crops=0
for r in pub:
 raw,meta=provider.metadata(r['dataset'],r['recording']);t=raw['original_t'];seconds=float(t[-1]-t[0]);dur[r['dataset']]=dur.get(r['dataset'],0)+seconds;crops+=int(math.ceil(seconds*10)+2)*36*36*3*2
dur['BTC']=sum(float(v['media']['seconds']) for v in dev)
crops+=sum(int(math.ceil(float(v['media']['seconds'])*10)+2)*36*36*3*2 for v in dev)
# Raw uint8 bound; actual compressed arrays cannot be assumed smaller for permission.
trainbytes=0
for run in plan['runs']:
 for ds,jobs in run['public_jobs'].items():
  for j in jobs:
   _,t,valid,m=provider.source_selection(ds,j['recording'],j['start'],j['recipe'])
   trainbytes+=len(t)*(36*36*3+16)+8192
c5bytes=math.ceil(sum(dur.values())*2*500000/8*1.03)+375*2*250000
size=lambda p:sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file()) if Path(p).exists() else 0
a=read(P/'A100_INVENTORY.json')
base=storage_guard(index,R/'t8_oval_20261031',R/'t8_polarity_v3_20261102')['total_bytes']+size(V)+size('/data_pool/khangle/facevitals/reports/round1_v4_20261105')+a['a100_v4_bytes']+size(B)
project=sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in (R,Path('/data_pool/khangle/facevitals')))+a['a100_bytes']
reserve=dict(C5_fullframe_two_recipes=c5bytes,C5_oval_uint8_upper=crops,training_crop_uint8_upper=trainbytes,AiPEX_crops_and_receipts=600000000,checkpoints_B_and_C=400000000,evaluations_reports=400000000,CPU_public_headroom=420000000)
out=dict(utc=datetime.now(timezone.utc).isoformat(),source_duration_seconds=dur,existing_T8_whole_bytes=base,existing_project_bytes=project,reserves=reserve,estimated_T8_whole_bytes=base+sum(reserve.values()),estimated_project_bytes=project+sum(reserve.values()),maximum_authorized_T8_bytes=80000000000,project_cap_bytes=500000000000,source_plan_sha256=sha(S/'PLAN.json'),no_deletion=True)
out['within_authorized_limits']=out['estimated_T8_whole_bytes']<=80000000000 and out['estimated_project_bytes']<=500000000000
(P/'BUDGET_ESTIMATE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
