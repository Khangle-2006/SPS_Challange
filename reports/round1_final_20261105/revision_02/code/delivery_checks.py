import json,csv,collections,hashlib,sys
from pathlib import Path
r=Path(sys.argv[1]);d={}
for name in ('per_person_metrics.csv','baselines_per_person.csv','mcd_c5_per_person.csv','hybrid_per_person.csv'):
    for row in csv.DictReader((r/name).open()):
        key=tuple(row[k] for k in ('person','dataset','condition','branch','fold','seed'))
        if key in d:assert row==d[key],key
        d[key]=row
groups=collections.defaultdict(list)
for k,row in d.items():groups[k[:-1]].append(row)
out=[]
for key,rr in sorted(groups.items()):
    assert {x['seed'] for x in rr}=={'2027','2028','2029'}
    row=dict(zip(('person','dataset','condition','branch','fold'),key));row['seeds']='2027|2028|2029'
    for m in ('MAE','signed_bias','P95_abs_error','coverage'):row[m]=sum(float(x[m]) for x in rr)/3
    out.append(row)
with (r/'per_person_seed_averaged.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
s=json.loads((r/'ALL_BRANCH_BOOTSTRAP.json').read_text())
for b,cc in s['branches'].items():
    for c in ('UBFC clean','UBFC C5','MCD native','MCD-C5'):
        rows=[x for x in out if x['branch']==b and x['condition']==c]
        for m in ('MAE','signed_bias','P95_abs_error','coverage'):
            assert abs(sum(x[m] for x in rows)/len(rows)-cc[c][m]['value'])<1e-10,(b,c,m)
assert s['recommended_branch']=='v3_joint' and not any(g['pass_all'] for g in s['gates'].values())
for host,ds in [('P100','MCD'),('A100','UBFC2')]:
    a=r/f'receipts_{host}/packaging_{ds}/attempt2/artifact'
    assert json.loads((a/'status.json').read_text())['state']=='complete'
    assert json.loads((a/'OOF_PARITY.json').read_text())['max_HR_delta_bpm']==0
    assert json.loads((a/'FRONTEND_PARITY.json').read_text())['all_pixels_exact']
paths=['v3_frozen_report.md','lag_after_polarity.md','mcd_c5_report.md','hybrid_report.md','branch72_report.md','schema_check.md','branch_choice.md','REPRODUCE.md']
for p in paths:assert (r/p).is_file() and (r/p).stat().st_size>0
assert (r.parent/'SUMMARY.md').exists()
receipt=dict(state='PASS',person_seed_rows=len(d),seed_averaged_rows=len(out),independent_metric_rechecks=8*4*4,both_packaging_HR_delta_zero=True,all_required_reports_present=True,branch72_state='not_run_storage_guard',optional_state='skipped',no_submission=True)
(r/'DELIVERY_VALIDATION.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
