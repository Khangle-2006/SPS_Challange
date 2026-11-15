"""Preregistered native-only stopping rule; no holdout or C5 dependency."""
import csv
import numpy as np
from common import *
from report_metrics import metric, belongs

ORDER=[3,2,0,1,4]  # User run numbers 4,3,1,2,5; original zero-based folds.

def stop_after_priority(results):
    return all(f in results for f in [3,2]) and not any(results[f]['passed_native_gate'] for f in [3,2])

def below_twelve(mae, valid=True):
    return bool(valid and mae is not None and np.isfinite(mae) and mae < 12.0)

def report_fold(fold):
    d=P/'sweep'/f'fold{fold}';status=read(d/'status.json')
    if not (d/'NATIVE.json').exists():
        assert status['state']=='rejected_static', 'Missing native results require investigation'
        out=dict(fold=fold,user_run=fold+1,MAE=None,passed_native_gate=False,reason='rejected_static',rows=[])
    else:
        raw=read(d/'NATIVE.json');before=read(d/'BEFORE_NATIVE.json')
        dev=read(V/'btc_zeroshot/dev_manifest.json')['videos'];meta={x['recording']:x for x in dev}
        cal={r['recording']:float(r['prediction']) for r in csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open()) if r['condition']=='native' and r['method']=='calibrated_median_POS_CHROM_V3'}
        constant=float(np.median([x['pr_bpm'] for x in dev if x['fold']!=fold]))
        videos=[]
        for rid in sorted({r['recording'] for r in raw}):
            ww=[r for r in raw if r['recording']==rid];vv=[r['prediction'] for r in ww if r['prediction'] is not None]
            if not vv:continue
            a=ww[0];m=meta[rid];assert m['fold']==fold
            videos.append(dict(a,group=m['group'],task=m['task'],method='fasttrack',prediction=float(np.median(vv)),candidate_windows=len(ww),valid_windows=len(vv)))
        oldkeys={r['recording'] for r in before if r['prediction'] is not None}
        coverage_preserved=oldkeys.issubset({r['recording'] for r in videos})
        matched=[r for r in videos if r['recording'] in cal];rows=[]
        for r in matched:rows += [r,dict(r,method='constant',prediction=constant),dict(r,method='calibrated',prediction=cal[r['recording']])]
        tables=[]
        for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
            rr=[r for r in rows if belongs(r,group)]
            for x in metric(rr,['constant','calibrated','fasttrack']):
                pp=[r['prediction'] for r in rr if r['method']==x['method']]
                tables.append(dict(group=group,SD_video_bpm=float(np.std(pp,ddof=1)) if len(pp)>1 else None,**x))
        own=next((x for x in tables if x['group']=='all' and x['method']=='fasttrack'),None)
        mae=own['MAE'] if own else None
        valid=status['state']=='complete' and status['static_pass'] and status['weak_gate']['passed'] and coverage_preserved
        out=dict(utc=now(),fold=fold,user_run=fold+1,MAE=mae,passed_native_gate=below_twelve(mae,valid),valid=valid,
                 coverage_preserved=coverage_preserved,candidate_videos=len({r['recording'] for r in raw}),valid_videos=len(videos),matched_videos=len(matched),
                 candidate_windows=len(raw),valid_windows=sum(r['prediction'] is not None for r in raw),
                 rows=tables,static_pass=status['static_pass'],weak_gate=status['weak_gate'],
                 scope='This fold validation people only; not full BTC dev; threshold is strict MAE<12',
                 inference_only_no_holdout=True)
    save(d/'NATIVE_STOP_GATE.json',out)
    lines=[f'# Lượt {fold+1} — fold {fold}: BTC native validation','',
           'Kết quả riêng fold, không phải toàn BTC dev. MAE theo người; CI ghép cặp10.000 bootstrap seed0. Hai baseline dùng đúng giao video.', '',
           '| Nhóm | Phương pháp | Video/người | MAE | CI95% | Bias | SD dự đoán video | Spearman theo người | Δ hiệu chỉnh [CI95%] |',
           '|---|---|---:|---:|---|---:|---:|---:|---|']
    for x in out['rows']:
        lines.append(f"| {x['group']} | {x['method']} | {x['videos']}/{x['people']} | {x['MAE']:.3f} | {x['CI95']} | {x['bias']:+.3f} | {x['SD_video_bpm']} | {x['Spearman']} | {x['delta_calibrated']:+.3f} {x['delta_calibrated_CI']} |")
    lines += ['',f"Đạt cổng native <12: {out['passed_native_gate']}. Phép thử ảnh tĩnh/weak-collapse và độ phủ phải hợp lệ."]
    (d/'NATIVE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(P/f'FASTTRACK_RESULT_RUN{fold+1}.json',dict(utc=now(),fold=fold,report=str(d/'NATIVE_REPORT.md'),receipt_sha256=sha(d/'NATIVE_STOP_GATE.json'),passed_native_gate=out['passed_native_gate'],MAE=out['MAE']))
    return out
