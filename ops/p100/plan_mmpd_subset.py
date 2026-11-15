"""Public metadata only; no restricted subject data downloaded."""
import csv, json, urllib.request
from pathlib import Path
out = Path('reports/storage_pilot_20261030'); out.mkdir(parents=True, exist_ok=True)
base = 'https://raw.githubusercontent.com/THU-CS-PI/MMPD_rPPG_dataset/main/'
for name in ['meta_label.csv', 'size_MMPD.csv']:
    (out/name).write_bytes(urllib.request.urlopen(base+name, timeout=60).read())
rows = list(csv.DictReader((out/'meta_label.csv').open()))
sizes = {Path(p).name:int(n) for p,n in csv.reader((out/'size_MMPD.csv').open())}
people = {}
for r in rows:
    sid = int(r['p'])
    p = people.setdefault(sid, {'subject': sid, 'skin_color':int(r['skin_color']), 'gender':r['gender'], 'files':[]})
    p['files'].append({'path':r['file'].lstrip('/'), 'bytes':sizes[Path(r['file']).name], 'light':r['light'], 'motion':r['motion'], 'exercise':r['exercise']})
# Deterministic exploratory plan, before inspecting any video/model performance.
# Alternate available genders, then subject ID. Counts are not a power calculation.
selected = []
for tone, count in [(3,2),(4,2),(5,3),(6,3)]:
    pools = {g:sorted([p for p in people.values() if p['skin_color']==tone and p['gender']==g],key=lambda p:p['subject']) for g in ('female','male')}
    for i in range(count):
        g = ('female','male')[i%2]
        if not pools[g]: g = 'male' if g=='female' else 'female'
        selected.append(pools[g].pop(0))
result = {'state':'proposed_access_pending', 'selection_rule':'2/2/3/3 across labels 3/4/5/6; alternate genders where available, lowest subject ID; no model outcomes used',
    'scope':'exploratory external evaluation only; not all six Fitzpatrick types, not full C2 or vitiligo validation',
    'subjects':selected, 'total_bytes':sum(f['bytes'] for p in selected for f in p['files']),
    'source_total_bytes':sum(sizes.values()), 'files':sum(len(p['files']) for p in selected)}
(out/'mmpd_subset_plan.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='subjects'}))
print([(p['subject'],p['skin_color'],p['gender']) for p in selected])
