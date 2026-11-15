"""Refresh operational inventory; preserve every prior receipt before replacement."""
from common import *
from polarity_guard import storage_guard
a=read(P/'A100_INVENTORY.json')
assert (datetime.now(timezone.utc)-datetime.fromisoformat(a['measured_utc'])).total_seconds()<3600
q=read(P/'BUDGET_ESTIMATE.json');stamp=now().replace(':','').replace('+','_')
history=P/'inventory_history';history.mkdir(exist_ok=True)
(history/(stamp+'_budget_previous.json')).write_bytes((P/'BUDGET_ESTIMATE.json').read_bytes())
size=lambda p:sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file()) if Path(p).exists() else 0
index=read(R/'t8_oval_20261031/training_index.json')
base=storage_guard(index,R/'t8_oval_20261031',R/'t8_polarity_v3_20261102')['total_bytes']+size(V)+size('/data_pool/khangle/facevitals/reports/round1_v4_20261105')+a['a100_v4_bytes']+size(B)
project=sum(int(subprocess.check_output(['du','-sb',str(x)],text=True).split()[0]) for x in (R,Path('/data_pool/khangle/facevitals')))+a['a100_bytes']
q['reserves']['AiPEX_crops_and_receipts']=read(P/'BUDGET_AIPEX_ADDENDUM.json')['raw_uint8_upper_bytes']
q['reserves']['private_Octave_archives_and_prefix']=300000000
q.update(utc=now(),existing_T8_whole_bytes=base,existing_project_bytes=project,current_root_bytes=size(P),estimated_T8_whole_bytes=base+sum(q['reserves'].values()),maximum_authorized_T8_bytes=84000000000,a100_receipt_sha256=sha(P/'A100_INVENTORY.json'),scope='user-approved A-F',measurement_complete=True)
q['estimated_project_bytes']=project+max(0,sum(q['reserves'].values())-q['current_root_bytes'])
q['within_authorized_limits']=q['estimated_T8_whole_bytes']<=84000000000 and q['estimated_project_bytes']<=500000000000
save(history/(stamp+'_fresh.json'),q);assert q['within_authorized_limits'],'Storage estimate exceeds approved cap; ask user, no waiver'
save(P/'BUDGET_ESTIMATE.json',q)
print(json.dumps({k:q[k] for k in ['utc','existing_T8_whole_bytes','current_root_bytes','estimated_T8_whole_bytes','existing_project_bytes','estimated_project_bytes']}))
