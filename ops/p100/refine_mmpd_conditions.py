import json
from pathlib import Path
base=Path('reports/storage_pilot_20261030')
old=json.loads((base/'mmpd_subset_plan.json').read_text())
out=Path('reports/mcd_mapping_20261030');out.mkdir(exist_ok=True)
people=[]
for p in old['subjects']:
    files=[f for f in p['files'] if f['light'] in ['LED-low','LED-high','Nature'] and f['motion'] in ['Stationary','Rotation'] and f['exercise']=='False']
    assert len(files)==6
    people.append({**p,'files':files})
batches=[]
for i in range(0,len(people),2):
    ps=people[i:i+2];batches.append({'subjects':[p['subject'] for p in ps],'bytes':sum(f['bytes'] for p in ps for f in p['files'])})
plan={**old,'subjects':people,'files':sum(len(p['files']) for p in people),
    'total_bytes':sum(f['bytes'] for p in people for f in p['files']),
    'condition_rule':'LED-low/LED-high/Nature x Stationary/Rotation, exercise=False; same ten preselected people',
    'two_person_batches':batches,'raw_prune_allowed':False,
    'notes':'Access pending. Natural-light variability remains a confound: report light separately or interaction; never claim LED-low <=50lux without measurement. Raw deletion after frontend freeze and verified caches only.'}
(out/'mmpd_60_files_plan.json').write_text(json.dumps(plan,indent=2))
print(json.dumps({'files':plan['files'],'bytes':plan['total_bytes'],'batches':batches}))
