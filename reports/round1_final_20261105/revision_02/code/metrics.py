"""Person-first paired bootstrap with the newly requested fixed seed0."""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

CONDITIONS=('UBFC clean','UBFC C5','MCD native','MCD-C5')

def write_json(p,x):
    p=Path(p)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def csv_write(p,rows):
    p=Path(p)
    if p.exists():raise FileExistsError(p)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def person_metrics(windows):
    groups=defaultdict(list)
    for r in windows:groups[r['subject_id'],r['dataset'],r['condition'],r['branch'],r['fold'],r['seed']].append(r)
    rows=[]
    for (sid,ds,c,b,f,s),rr in sorted(groups.items()):
        valid=[r for r in rr if r['reference_eligible']]
        good=[r for r in rr if r['accepted']]
        err=np.array([r['prediction_hr_bpm']-r['reference_hr_bpm'] for r in good])
        rows.append(dict(person=sid,dataset=ds,condition=c,branch=b,fold=f,seed=s,MAE=float(np.abs(err).mean()) if len(err) else None,signed_bias=float(err.mean()) if len(err) else None,P95_abs_error=float(np.quantile(np.abs(err),.95)) if len(err) else None,coverage=len(good)/len(valid) if valid else None,candidates=len(rr),eligible=len(valid),accepted=len(good)))
    return rows

def aggregate(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r['person'],r['condition'],r['branch']].append(r)
    persons={}
    for k,rr in grouped.items():
        assert len(rr)==3 and {r['seed'] for r in rr}=={2027,2028,2029},'Require complete3seeds'
        assert len({r['fold'] for r in rr})==1
        assert all(r['MAE'] is not None and r['coverage'] is not None for r in rr),'Undefined person metric; do not silently remove person'
        persons[k]={m:float(np.mean([r[m] for r in rr])) for m in ('MAE','signed_bias','P95_abs_error','coverage')}
    branches=sorted({r['branch'] for r in rows});result={};boots={};paired={}
    rng=np.random.default_rng(0)
    samples={35:rng.integers(35,size=(10000,35)),60:rng.integers(60,size=(10000,60))}
    for b in branches:
        result[b]={}
        for c in CONDITIONS:
            ids=sorted(k[0] for k in persons if k[1:]==(c,b))
            assert len(ids)==(60 if c.startswith('MCD') else 35)
            a=np.array([[persons[sid,c,b][m] for m in ('MAE','signed_bias','P95_abs_error','coverage')] for sid in ids])
            sample=samples[len(ids)]
            boot=a[sample].mean(axis=1);boots[b,c]=boot[:,0]
            result[b][c]={m:dict(value=float(a[:,j].mean()),CI95=np.quantile(boot[:,j],[.025,.975]).tolist()) for j,m in enumerate(('MAE','signed_bias','P95_abs_error','coverage'))}
            result[b][c].update(persons=len(ids),median_MAE=float(np.median(a[:,0])),P95_person_MAE=float(np.quantile(a[:,0],.95)))
        # Same UBFC resampling indices are used across both conditions; MCD has its own person population.
        sb=np.mean([boots[b,c] for c in CONDITIONS[:3]],axis=0)
        result[b]['S']=dict(value=float(np.mean([result[b][c]['MAE']['value'] for c in CONDITIONS[:3]])),CI95=np.quantile(sb,[.025,.975]).tolist())
    for b in branches:
        for control in ('ubfc_only','v3_joint'):
            if b==control or control not in branches:continue
            name=b+' minus '+control;paired[name]={}
            for c in CONDITIONS:
                delta=boots[b,c]-boots[control,c]
                ci=np.quantile(delta,[.025,.975]).tolist()
                paired[name][c]=dict(delta=result[b][c]['MAE']['value']-result[control][c]['MAE']['value'],CI95=ci,reading='better' if ci[1]<0 else 'worse' if ci[0]>0 else 'no detectable difference')
            d=np.mean([boots[b,c]-boots[control,c] for c in CONDITIONS[:3]],axis=0)
            paired[name]['S']=dict(delta=result[b]['S']['value']-result[control]['S']['value'],CI95=np.quantile(d,[.025,.975]).tolist())
    gates={}
    ref=result['v3_joint']
    for b in branches:
        if b.startswith(('constant','POS','CHROM')) or b=='v3_joint':continue
        v=result[b];red=1-v['S']['value']/ref['S']['value'];clean=v['UBFC clean']['MAE']['value']/ref['UBFC clean']['MAE']['value']-1;p95=v['UBFC C5']['P95_abs_error']['value']-ref['UBFC C5']['P95_abs_error']['value']
        gates[b]=dict(S_relative_reduction=red,UBFC_clean_relative_increase=clean,UBFC_C5_P95_delta=p95,pass_S=red>=.1,pass_clean=clean<=.05,pass_P95=p95<=0,pass_all=red>=.1 and clean<=.05 and p95<=0)
    passing=[b for b,g in gates.items() if g['pass_all']]
    choice=min(passing,key=lambda b:result[b]['S']['value']) if passing else 'v3_joint'
    return dict(branches=result,paired=paired,gates=gates,recommended_branch=choice,bootstrap=dict(resamples=10000,seed=0,unit='person',seeds_averaged_first=True,P95_definition='mean of seed-averaged within-person window-error P95s'))

def table(summary,branches=None):
    lines=['| Branch | Condition | MAE [95% CI] | P95 [95% CI] | Bias [95% CI] | Coverage [95% CI] |','|---|---|---:|---:|---:|---:|']
    for b in branches or summary['branches']:
        for c in CONDITIONS:
            r=summary['branches'][b][c]
            def fmt(k):
                x=r[k];return f"{x['value']:.3f} [{x['CI95'][0]:.3f}, {x['CI95'][1]:.3f}]"
            lines.append('| '+' | '.join([b,c,fmt('MAE'),fmt('P95_abs_error'),fmt('signed_bias'),fmt('coverage')])+' |')
    return '\n'.join(lines)
