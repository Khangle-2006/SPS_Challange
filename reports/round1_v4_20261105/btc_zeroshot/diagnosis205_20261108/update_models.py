"""Read finished JSONs only. No model or cache mutation; full15 V4 membership."""
import json,csv,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
O=Path(__file__).resolve().parent;B=O.parent
sys.path.insert(0,str(O))
from summarize import strata,metric,write,readcsv
def update():
    co=json.loads((O/'COHORT.json').read_text());keys=co['recordings'];man=json.loads((B/'dev_manifest.json').read_text());by={v['recording']:v for v in man['videos']}
    meta={r['recording']:r for r in readcsv(O/'VIDEO_METADATA.csv')};names=[f'v4_repaired_f{f}_s{s}' for f in range(5) for s in [2027,2028,2029]]
    rows=[];ready=True
    for rec in keys:
        v=by[rec]
        for cond in ['native','c5']:
            pp=O/'padding205'/(rec+'_'+cond+'.json')
            if pp.exists():
                data=json.loads(pp.read_text());pred=[w['prediction'] for w in data['rows'] if w['prediction'] is not None]
                rows.append(dict(recording=rec,person=v['person'],fold=v['fold'],truth=v['pr_bpm'],method='padding_fixed_single_f0_s2027',condition=cond,prediction=float(np.median(pred)) if pred else None))
            files=[B/'predictions'/v['drive_id']/cond/(name+'.json') for name in names]
            if not all(f.exists() for f in files):ready=False;continue
            windows=defaultdict(dict)
            for name,p in zip(names,files):
                d=json.loads(p.read_text());assert d['split_sha256']==co['split_sha256'] and d['checkpoint']['name']==name
                f=int(name.split('_f')[1].split('_')[0]);seed=int(name.split('_s')[1]);status=json.loads((B.parent/f'v4_repaired/fold{f}/seed{seed}/joint/status.json').read_text())
                assert status['state']=='complete' and d['checkpoint']['sha256']==status['checkpoint_sha256']
                for w in d['rows']:
                    if w['scheme']=='10s':windows[w['start']][name]=w
            pred=[]
            for start,ww in windows.items():
                assert set(ww)==set(names),'Incomplete window membership'
                if all(w['prediction'] is not None for w in ww.values()):pred.append(float(np.median([w['prediction'] for w in ww.values()])))
            rows.append(dict(recording=rec,person=v['person'],fold=v['fold'],truth=v['pr_bpm'],method='v4_repaired_median15',condition=cond,prediction=float(np.median(pred)) if pred else None))
    # A branch enters the fixed205 tables only when all205 native+C5 records are present.
    complete=[m for m in ['padding_fixed_single_f0_s2027','v4_repaired_median15'] if sum(r['method']==m for r in rows)==410]
    write('MODEL_EXTENSION_VIDEO.csv',rows)
    base=readcsv(O/'BASELINE_VIDEO.csv');metrics=[]
    for m in complete:
        for cond in ['native','c5']:
            rr=[r for r in rows if r['method']==m and r['condition']==cond and r['prediction'] is not None];accepted={r['recording'] for r in rr}
            for dim in strata(next(iter(meta.values()))):
                values=sorted({strata(meta[r])[dim] for r in accepted})
                for val in values:
                    kk={r for r in accepted if strata(meta[r])[dim]==val};use=[r for r in rr if r['recording'] in kk]
                    metrics.append(dict(method=m,condition=cond,dimension=dim,group=val,**metric(use)))
                    for bm in ['constant','POS','CHROM','ubfc_only_median15','v3_joint_median15']:
                        bb=[dict(recording=r['recording'],person=r['person'],truth=float(r['truth']),prediction=float(r['prediction'])) for r in base if r['scheme']=='10s' and r['method']==bm and r['condition']==cond and r['recording'] in kk]
                        assert len(bb)==len(use)
                        metrics.append(dict(method=bm+'_matched_to_'+m,condition=cond,dimension=dim,group=val,**metric(bb)))
    write('MODEL_EXTENSION_STRATA.csv',metrics)
    (O/'MODEL_EXTENSION_STATUS.json').write_text(json.dumps(dict(complete_branches=complete,V4_all205_complete=ready,holdout_evaluations=0,cohort_videos=205),indent=2))
    return ready
if __name__=='__main__':print(update())
