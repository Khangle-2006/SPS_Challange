"""One-shot crop diagnostic report; thresholds frozen before predictions."""
import argparse,json,csv
from pathlib import Path
from collections import defaultdict
import numpy as np

def metrics(rows,key):
    valid=[r for r in rows if r.get(key) is not None and r.get('reference_hr') is not None]
    e=np.array([r[key]-r['reference_hr'] for r in valid]); people=defaultdict(list)
    for r,x in zip(valid,e):people[r['subject']].append(abs(x))
    return dict(expected=len(rows),accepted=len(valid),coverage=len(valid)/len(rows) if rows else None,
        subject_macro_MAE=float(np.mean([np.mean(v) for v in people.values()])) if people else None,
        window_MAE=float(abs(e).mean()) if len(e) else None,P95=float(np.quantile(abs(e),.95)) if len(e) else None,
        bias=float(e.mean()) if len(e) else None,errors_gt20=int((abs(e)>20).sum()),
        positive_tail=int((e>20).sum()),negative_tail=int((e< -20).sum()),
        per_subject={s:float(np.mean(v)) for s,v in people.items()})

def main(a):
    raw=json.loads((a.input/'results.json').read_text());status=json.loads((a.input/'status.json').read_text())
    rows=raw['rows'];assert status['state']=='complete' and status['completed']==119 and len(rows)==1860 and not raw['errors']
    ids={(r['video'],r['index']) for r in rows};assert len(ids)==1860
    result=dict(records=119,subjects=len({r['subject'] for r in rows}),metrics={k:metrics(rows,k) for k in ('crop_POS','large_oval','control_small')},
        before_after={c:{k:metrics([r for r in rows if r['step']==c],k) for k in ('crop_POS','large_oval')} for c in ('before','after')})
    d=defaultdict(list)
    for r in rows:
        if r['crop_POS'] is not None and r['large_oval'] is not None:
            d[r['subject']].append(abs(r['crop_POS']-r['reference_hr'])-abs(r['large_oval']-r['reference_hr']))
    v=np.array([np.mean(d[s]) for s in sorted(d)])
    boot=np.random.default_rng(20261031).choice(v,(10000,len(v)),replace=True).mean(1) if len(v) else []
    result['paired_crop_minus_oval']=dict(windows=sum(map(len,d.values())),subjects=len(v),delta=float(v.mean()) if len(v) else None,CI95=np.quantile(boot,[.025,.975]).tolist() if len(v) else None)
    paired_rows=[r for r in rows if r['crop_POS'] is not None and r['large_oval'] is not None]
    result['paired_metrics']={k:metrics(paired_rows,k) for k in ('crop_POS','large_oval')}
    m=result['metrics']['crop_POS']; complete=m['accepted']==1860 and result['subjects']==60
    result['decision']='KEEP_T8_CROP' if complete and m['subject_macro_MAE']<=6 else ('REVISE_MCD_TO_OVAL' if complete and m['subject_macro_MAE']>=8 else 'UNRESOLVED_RANGE_OR_COVERAGE')
    result['limitations']='POS proxy on dev60; not a learned-model result or untouched test. Full-resolution mean of actual saved crop; no 36x36 resize in this diagnostic.'
    if (a.input/'roi_areas.json').exists():
        areas=json.loads((a.input/'roi_areas.json').read_text())
        vv=[r['bbox_area_median'] for r in areas if r['bbox_area_median'] is not None]
        result['crop_area_pixels']=dict(records=len(areas),median_of_video_medians=float(np.median(vv)),min=float(min(vv)),max=float(max(vv)),by_recording=areas)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'summary.json').write_text(json.dumps(result,indent=2))
    with (a.output/'windows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    keys=['control_small','crop_POS','large_oval'];labels=[f"{name}\n{result['metrics'][k]['accepted']}/1860" for k,name in zip(keys,['Three small ROIs','Exact T8 crop','Large oval'])];axs[0].bar(labels,[result['metrics'][k]['subject_macro_MAE'] or 0 for k in keys]);axs[0].set(ylabel='Subject-macro MAE (bpm)',title='Fixed 1,860 expected windows');axs[0].tick_params(axis='x',labelrotation=15)
    axs[1].hist(v,bins=18);axs[1].axvline(0,color='k',ls='--');axs[1].set(xlabel='Crop minus oval MAE (bpm)',ylabel='Subjects',title='Paired subject differences')
    for k,label in [('crop_POS','T8 crop'),('large_oval','Oval')]:
        vv=[r for r in rows if r[k] is not None];axs[2].scatter([r['reference_hr'] for r in vv],[r[k]-r['reference_hr'] for r in vv],s=5,alpha=.3,label=label)
    axs[2].axhline(0,color='k',ls='--');axs[2].set(xlabel='Reference HR (bpm)',ylabel='Signed error (bpm)',title='Errors versus HR');axs[2].legend()
    fig.suptitle('T8 frontend gate — POS diagnostic, MCD dev60');fig.savefig(a.output/'comparison.png',dpi=160);plt.close(fig)
    lines=['T8 exact-crop POS diagnostic',f"Decision: {result['decision']}",json.dumps(result['metrics'],indent=2),json.dumps(result['paired_crop_minus_oval'],indent=2),result['limitations']]
    (a.output/'REPORT.txt').write_text('\n\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','before_after','crop_area_pixels','paired_metrics')},indent=2))
    print('MAEs:',{k:v['subject_macro_MAE'] for k,v in result['metrics'].items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
