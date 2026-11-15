"""Versioned person-paired V4 final summary; no partial full-OOF scores."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv,argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'controls_20261107'))
from common import *
HERE=Path(__file__).resolve().parent
METRICS=('MAE','P95_abs_error','bias','ceiling_fraction','coverage')
BRANCHES=('v4_repaired','v3_joint','POS','CHROM')

def person_seed_rows(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['subject_id'],r['condition'],r['branch'],r['fold'],r['seed']].append(r)
    result=[]
    for (sid,c,b,f,s),rr in sorted(groups.items()):
        accepted=[r for r in rr if r['accepted']]
        eligible=sum(r['reference_eligible'] for r in rr)
        assert accepted and eligible,'Undefined person metrics: fail closed'
        h=np.array([r['prediction_hr_bpm'] for r in accepted]);y=np.array([r['reference_hr_bpm'] for r in accepted]);e=h-y
        result.append(dict(person=sid,condition=c,branch=b,fold=f,seed=s,MAE=float(np.abs(e).mean()),P95_abs_error=float(np.quantile(np.abs(e),.95)),
            bias=float(e.mean()),ceiling_fraction=float((h>=179).mean()),coverage=len(e)/eligible,candidates=len(rr),eligible=eligible,accepted=len(e)))
    return result
def person_means(rows,required_seeds=(2027,2028,2029)):
    grouped=defaultdict(list)
    for r in rows:grouped[r['person'],r['condition'],r['branch']].append(r)
    result={}
    for key,rr in grouped.items():
        assert len(rr)==len(required_seeds) and {r['seed'] for r in rr}==set(required_seeds)
        assert len({r['fold'] for r in rr})==1
        result[key]={m:float(np.mean([r[m] for r in rr])) for m in METRICS}
    return result
def summary(persons,branches=BRANCHES,expected_counts=True):
    ids={c:sorted(s for s,cc,b in persons if cc==c and b==branches[0]) for c in CONDITIONS}
    for c in CONDITIONS:
        if expected_counts:assert len(ids[c])==(60 if c.startswith('MCD') else 35)
        for b in branches:assert sorted(s for s,cc,bb in persons if cc==c and bb==b)==ids[c]
    assert ids['UBFC clean']==ids['UBFC C5'] and ids['MCD native']==ids['MCD-C5']
    rng=np.random.default_rng(0)
    samples={ds:rng.integers(len(ids[c]),size=(10000,len(ids[c]))) for ds,c in [('UBFC','UBFC clean'),('MCD','MCD native')]}
    report={};boots={}
    def stat(a,b):return dict(value=float(a),CI95=np.quantile(b,[.025,.975]).tolist())
    for b in branches:
        report[b]={}
        for c in CONDITIONS:
            a=np.array([[persons[s,c,b][m] for m in METRICS] for s in ids[c]])
            boot=a[samples['MCD' if c.startswith('MCD') else 'UBFC']].mean(1)
            boots[b,c]=boot
            report[b][c]={m:stat(a[:,j].mean(),boot[:,j]) for j,m in enumerate(METRICS)}
            report[b][c]['people']=len(ids[c])
        for score,cc in [('S',('UBFC clean','UBFC C5','MCD native')),('S_compressed',('UBFC C5','MCD-C5'))]:
            boot=np.mean([boots[b,c][:,0] for c in cc],axis=0);boots[b,score]=boot
            report[b][score]=stat(np.mean([report[b][c]['MAE']['value'] for c in cc]),boot)
    paired={}
    for control in branches[1:]:
        paired[control]={}
        for c in CONDITIONS:
            delta=boots[branches[0],c]-boots[control,c]
            paired[control][c]={m:stat(report[branches[0]][c][m]['value']-report[control][c][m]['value'],delta[:,j]) for j,m in enumerate(METRICS)}
        for score in ('S','S_compressed'):
            paired[control][score]=stat(report[branches[0]][score]['value']-report[control][score]['value'],boots[branches[0],score]-boots[control,score])
    return dict(branches=report,paired_V4_minus=paired,bootstrap=dict(unit='person',resamples=10000,seed=0,seeds_averaged_first=True,
        P95='mean of seed-averaged within-person window absolute-error P95s',ceiling='person-macro fraction among accepted numeric HR; raw counts reported separately',independent_dataset_resampling=True))
def table_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def matched_all(runs):
    frozen=read(FROZEN)
    freeze=HERE/'INPUTS.json'
    hashes=dict(frozen_windows_sha256=sha(FROZEN),protocol_sha256=sha(REPORT/'controls_20261107/PROTOCOL.md'))
    if freeze.exists():assert read(freeze)==hashes,'Frozen inputs changed'
    else:save(freeze,hashes)
    new=[r for run in runs for r in load_run(run,frozen)]
    wanted={(r['condition'],r['fold'],r['seed'],key(r)):r for r in new}
    rows=list(new)
    for branch in BRANCHES[1:]:
        old={(r['condition'],r['fold'],r['seed'],key(r)):r for r in frozen if r['branch']==branch and (r['condition'],r['fold'],r['seed'],key(r)) in wanted}
        assert set(old)==set(wanted),'Comparator window identity mismatch'
        for k,r in old.items():
            v=wanted[k]
            assert r['reference_eligible']==v['reference_eligible'] and r['accepted']==v['accepted'],'Comparator failure/support mismatch; do not silently intersect'
            if r['reference_eligible']:assert r['reference_hr_bpm']==v['reference_hr_bpm']
            rows.append(r)
    return rows
def fold3(rows):
    rr=[r for r in rows if r['fold']==3 and r['condition'] in ('UBFC clean','MCD-C5') and r['branch'] in ('v4_repaired','v3_joint')]
    if not rr:return
    seeds=sorted({r['seed'] for r in rr});pm=person_means(person_seed_rows(rr),seeds)
    output=[];concentration={}
    for cond in ('UBFC clean','MCD-C5'):
        persons=sorted(s for s,c,b in pm if c==cond and b=='v4_repaired')
        for sid in persons:
            a=pm[sid,cond,'v3_joint']['MAE'];b=pm[sid,cond,'v4_repaired']['MAE']
            output.append(dict(condition=cond,person=sid,seeds=len(seeds),V3_MAE=a,V4_MAE=b,delta=b-a))
        group=[r for r in output if r['condition']==cond];positive=sorted([r['delta'] for r in group if r['delta']>0],reverse=True)
        d=np.array([r['delta'] for r in group]);rng=np.random.default_rng(0);boot=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
        concentration[cond]=dict(people=len(group),worse=sum(r['delta']>0 for r in group),better=sum(r['delta']<0 for r in group),
            top2_share_of_positive_increases=sum(positive[:2])/sum(positive) if positive else 0,net_MAE_delta=float(d.mean()),CI95=np.quantile(boot,[.025,.975]).tolist())
    save(HERE/'FOLD3.json',dict(partial=len(seeds)<3,seeds=seeds,people=output,concentration=concentration))
    table_csv(HERE/'FOLD3_people.csv',output)
    lines=['# Fold3 — đối chiếu theo người',f'\nSeed đã xong: {seeds}. '+('CÒN THIẾU SEED; chưa phải kết luận cuối.' if len(seeds)<3 else 'Đủ3seed.'),
        '\nMAE là sai số tuyệt đối trung bình, bpm; delta=V4−V3, số dương là xấu hơn. Mỗi người được trung bình qua cùng các seed.',
        '\n| Điều kiện | Người | V3 | V4 | Delta |','|---|---|---:|---:|---:|']
    for r in output:lines.append(f"|{r['condition']}|{r['person']}|{r['V3_MAE']:.3f}|{r['V4_MAE']:.3f}|{r['delta']:+.3f}|")
    for c,v in concentration.items():lines.append(f"\n{c}: {v['worse']}/{v['people']} người xấu hơn; hai người có mức tăng lớn nhất chiếm {100*v['top2_share_of_positive_increases']:.1f}% tổng phần tăng dương. Tỷ trọng này không phải tỷ trọng sai số tổng, và bỏ qua phần giảm để mô tả tập trung. Delta trung bình {v['net_MAE_delta']:+.3f}, CI95 {v['CI95']}.")
    (HERE/'FOLD3.md').write_text('\n'.join(lines),encoding='utf-8')
def main(partial):
    runs=completed();rows=matched_all(runs);fold3(rows)
    if partial:return
    pairs={(int(p.parent.parent.name[4:]),int(p.parent.name[4:])) for p in runs}
    assert pairs=={(f,s) for f in range(5) for s in (2027,2028,2029)},'Full S requires all15 final runs'
    ps=person_seed_rows(rows);persons=person_means(ps);out=summary(persons)
    checks={}
    for run in runs:
        name=run.parent.parent.name+'_'+run.parent.name;p=REPORT/'controls_20261107'/name/'STATIC.json'
        if p.exists():
            r=read(p);assert r['checkpoint_sha256']==sha(run/'final.pt')
            checks[name]=r['pass_static']
        else:checks[name]=None
    delta=out['paired_V4_minus']['v3_joint']
    gate=dict(S_CI_improves=delta['S']['CI95'][1]<0,UBFC_clean_within_half_bpm=delta['UBFC clean']['MAE']['value']<=.5,
        static_pass=all(v is True for v in checks.values()),static_checks=checks,missing_static=sum(v is None for v in checks.values()))
    gate['all_pass']=all(gate[k] for k in ('S_CI_improves','UBFC_clean_within_half_bpm','static_pass'))
    out.update(utc=datetime.now(timezone.utc).isoformat(),partial=False,completed_runs=15,gates=gate,recommended_branch='v4_repaired' if gate['all_pass'] else 'v3_joint',
        counts={b:{c:dict(candidates=sum(r['branch']==b and r['condition']==c for r in rows),eligible=sum(r['branch']==b and r['condition']==c and r['reference_eligible'] for r in rows),accepted=sum(r['branch']==b and r['condition']==c and r['accepted'] for r in rows),ceilings=sum(r['branch']==b and r['condition']==c and r['accepted'] and r['prediction_hr_bpm']>=179 for r in rows)) for c in CONDITIONS} for b in BRANCHES})
    save(HERE/'FINAL.json',out);table_csv(HERE/'person_seed_metrics.csv',ps)
    lines=['# V4 repaired — đủ15lượt, đối chiếu ghép cặp theo người',
        '\nMAE/P95/bias đơn vị bpm. Bias=dự đoán−nhãn. P95 là trung bình P95 cửa sổ trong từng người/seed; không phải P95 gộp. Các ô là điểm [CI95]. Ceiling là tỷ lệ>=179 trong dự đoán numeric; bảng dùng trọng số bằng nhau giữa người. Các số đếm cửa sổ có3seed nên không phải số video/người độc lập.',
        '\n| Nhánh | Điều kiện | MAE | P95 | Bias | >=179 (%) | Coverage (%) |','|---|---|---:|---:|---:|---:|---:|']
    def fmt(v,scale=1):return f"{v['value']*scale:.3f} [{v['CI95'][0]*scale:.3f}, {v['CI95'][1]*scale:.3f}]"
    for b in BRANCHES:
        for c in CONDITIONS:
            a=out['branches'][b][c];lines.append('| '+' | '.join([b,c]+[fmt(a[m],100 if m in ('ceiling_fraction','coverage') else 1) for m in METRICS])+' |')
    lines+=['\nS=mean(UBFC sạch,UBFC C5,MCD native) MAE. S nhóm nén=mean(UBFC C5,MCD-C5), chỉ mô tả.','\n| Nhánh | S | S nhóm nén |','|---|---:|---:|']
    for b in BRANCHES:lines.append(f"|{b}|{fmt(out['branches'][b]['S'])}|{fmt(out['branches'][b]['S_compressed'])}|")
    for control in BRANCHES[1:]:
        lines+=['\n## V4 trừ '+control,'\nSố âm nghĩa MAE/P95 thấp hơn; bias có dấu nên không diễn giải âm luôn tốt. Tỷ lệ chênh lệch dùng điểm phần trăm.','\n| Điều kiện | Delta MAE | Delta P95 | Delta bias | Delta >=179 (pp) | Delta coverage (pp) |','|---|---:|---:|---:|---:|---:|']
        for c in CONDITIONS:lines.append('| '+' | '.join([c]+[fmt(out['paired_V4_minus'][control][c][m],100 if m in ('ceiling_fraction','coverage') else 1) for m in METRICS])+' |')
        lines.append('\nDelta S '+fmt(out['paired_V4_minus'][control]['S'])+'; delta S nhóm nén '+fmt(out['paired_V4_minus'][control]['S_compressed']))
    lines+=['\n## Cổng đã ghi trước',json.dumps(gate,indent=2),'\nỨng viên theo cổng: '+out['recommended_branch']+'. Không tự nộp bài.']
    (HERE/'FINAL.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(completed=15,gates=gate,recommendation=out['recommended_branch'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial-fold3',action='store_true');a=p.parse_args();main(a.partial_fold3)
