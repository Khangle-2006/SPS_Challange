"""Strict matched-window loading shared by controls and final summaries."""
import csv, hashlib, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np

REPORT=Path('/mnt/samsung_t5/khangle/facevitals/reports/round1_v4_20261105')
ROOT=REPORT.parents[1]
V3CODE=Path('/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102')
FROZEN=REPORT.parent/'round1_final_20261105/revision_02/frozen_data/FROZEN_WINDOWS.json'
CONDITIONS={'MCD native':('MCD','native'),'MCD-C5':('MCD','f10_b500'),'UBFC clean':('UBFC2','native'),'UBFC C5':('UBFC2','f10_b500')}
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(p)
def key(r): return r['subject_id'],r['recording_id'],float(r['window_start_s'])
def completed(base=None):
    base=base or REPORT/'v4_repaired'
    return [p.parent for p in sorted(base.glob('fold*/seed*/joint/status.json')) if read(p)['state']=='complete']
def load_run(run, frozen):
    fold=int(run.parent.parent.name[4:]);seed=int(run.parent.name[4:])
    status=read(run/'status.json');assert status['state']=='complete' and status['completed_epochs']==30
    assert sha(run/'final.pt')==status['checkpoint_sha256']
    rows=[]
    for cond,(ds,var) in CONDITIONS.items():
        d=run/'evaluation'/f'{ds}_{var}'
        reuse=read(d/'REFERENCE_REUSE.json');m=read(d/'metrics.json')
        ref=Path(reuse['path']);raw=Path(reuse['raw_path'])
        assert sha(ref)==reuse['sha256']==m['reference_sha256']
        assert sha(raw)==reuse['raw_sha256']
        pol=read(ref.parent/'REFERENCE_POLARITY.json')
        assert sha(ref.parent/'REFERENCE_POLARITY.json')==reuse['polarity_receipt_sha256']
        assert pol['canonical_reference_sha256']==reuse['sha256'] and pol['raw_reference_sha256']==reuse['raw_sha256'] and pol['all_hr_and_failure_parity']
        assert sha(V3CODE/'score_round1.py')==m['scorer_sha256']
        assert sha(d/'processed_output.csv')==m['output_sha256']
        old={key(r):r for r in frozen if r['branch']=='v3_joint' and r['condition']==cond and r['fold']==fold and r['seed']==seed}
        pred=list(csv.DictReader((d/'processed_output.csv').open()))
        assert len(pred)==len(old) and {key(r) for r in pred}==set(old)
        for p in pred:
            b=old[key(p)];r=dict(b);r['branch']='v4_repaired'
            r['accepted']=b['reference_eligible'] and p['abstain'].lower()=='false'
            assert r['accepted']==b['accepted'],'Support/failure mismatch vs frozen V3'
            r['prediction_hr_bpm']=float(p['hr_bpm']) if p['abstain'].lower()=='false' else None
            if r['accepted']:assert math.isfinite(r['prediction_hr_bpm'])
            rows.append(r)
    return rows
def metrics(rows, pred='prediction_hr_bpm', ref='reference_hr_bpm'):
    people=defaultdict(list);numeric=0;ceilings=0
    for r in rows:
        h=r[pred]
        if h is not None:
            assert math.isfinite(h)
            numeric+=1;ceilings+=h>=179
            people[r['subject_id']].append(h-r[ref])
    return dict(windows=len(rows),numeric=numeric,abstentions=len(rows)-numeric,people_with_numeric=len(people),
                MAE=float(np.mean([np.abs(v).mean() for v in people.values()])) if people else None,
                P95=float(np.mean([np.quantile(np.abs(v),.95) for v in people.values()])) if people else None,
                bias=float(np.mean([np.mean(v) for v in people.values()])) if people else None,
                ceilings=ceilings,ceiling_fraction_numeric=ceilings/numeric if numeric else None,
                ceiling_fraction_all=ceilings/len(rows) if rows else None,coverage=numeric/len(rows) if rows else None)
