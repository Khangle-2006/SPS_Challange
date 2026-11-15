import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from summarize_dev import boot,save
HERE=Path(__file__).resolve().parent
def main():
    manifest=json.loads((HERE/'dev_manifest.json').read_text());sample=json.loads((HERE/'BTC_CONTROL_SAMPLE.json').read_text())
    people=sample['people'];videos=manifest['videos'];byrec={v['recording']:v for v in videos}
    groups=defaultdict(set)
    for v in videos:
        if v['person'] in people:groups[v['group']].add(v['person'])
    donor={}
    for ids in groups.values():
        ids=sorted(ids)
        for i,p in enumerate(ids):donor[p]=ids[(i+1)%len(ids)]
    reference={p:float(np.median([v['pr_bpm'] for v in videos if v['person']==p])) for p in people}
    stats=defaultdict(list)
    for f in (HERE/'btc_controls').glob('*/*/*.json'):
        r=json.loads(f.read_text());v=byrec[r['recording']]
        stats[r['method'],r['condition'],'static'].append(dict(person=v['person'],recording=v['recording'],truth=v['pr_bpm'],prediction=r['static_hr'],pass_static=r['static_pass']))
    if (HERE/'VIDEO_PREDICTIONS.csv').exists():
        for r in csv.DictReader((HERE/'VIDEO_PREDICTIONS.csv').open()):
            if r['person'] not in people or r['scheme']!='10s':continue
            p=float(r['prediction']) if r['prediction'] else None
            a=dict(person=r['person'],recording=r['recording'],truth=float(r['truth']),prediction=p)
            stats[r['method'],r['condition'],'main'].append(a)
            stats[r['method'],r['condition'],'wrong_person_label'].append(dict(a,truth=reference[donor[r['person']]]))
    result=[]
    for (method,condition,test),rr in sorted(stats.items()):
        numeric=[r for r in rr if r['prediction'] is not None]
        result.append(dict(method=method,condition=condition,test=test,**boot(rr),videos=len(rr),numeric=len(numeric),ceilings=sum(r['prediction']>=179 for r in numeric),
            ceiling_fraction=sum(r['prediction']>=179 for r in numeric)/len(numeric) if numeric else None,
            all_static_pass=all(r['pass_static'] for r in rr) if test=='static' else None))
    save(HERE/'BTC_CONTROL_RESULTS.json',dict(sample_people=people,donor_people=donor,results=result,
        note='Static MAE only describes invalid physiological input, not clinical accuracy. Wrong-person PR is a deterministic evaluation negative control, never a training label. Missing controls remain pending, not pass.'))
if __name__=='__main__':main()
