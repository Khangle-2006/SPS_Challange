"""Paired BTC reports with both required comparators; no training or selection."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
from report_metrics import metric,belongs
P=Path(__file__).resolve().parent;B=P.parent/'round1_bayes_20261108';V=P.parent/'round1_v4_20261105'
dev=json.loads((V/'btc_zeroshot/dev_manifest.json').read_text())['videos'];meta={v['recording']:v for v in dev}
old=list(csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open()));oldcal={r['recording']:float(r['prediction']) for r in old if r['condition']=='native' and r['method']=='calibrated_median_POS_CHROM_V3'}
allrows=[];windowrows=[]
for file in (P/'adabn').glob('fold*/*NATIVE.json'):windowrows+=json.loads(file.read_text())
for file in (P/'sweep').glob('fold*/NATIVE.json'):windowrows+=json.loads(file.read_text())
for folder in ['adabn','sweep']:
 for file in (P/folder).glob('fold*/C5PRIME.json'):windowrows += [r for r in json.loads(file.read_text()) if r.get('dataset')=='BTC']
c5_baseline=json.loads((P/'C5_BTC_VIDEO.json').read_text()) if (P/'C5_BTC_VIDEO.json').exists() else []
calibrated={('native',rid):p for rid,p in oldcal.items()}
calibrated.update({(r['condition'],r['recording']):r['prediction'] for r in c5_baseline if r['method']=='calibrated'})
for condition in sorted({r['condition'] for r in windowrows}):
 for method in sorted({r['method'] for r in windowrows if r['condition']==condition}):
  for rid in sorted({r['recording'] for r in windowrows if r['method']==method and r['condition']==condition}):
   a=[r for r in windowrows if r['method']==method and r['recording']==rid and r['condition']==condition];valid=[r for r in a if r['prediction'] is not None]
   if valid:allrows.append(dict(valid[0],prediction=float(np.median([r['prediction'] for r in valid])),valid_windows=len(valid),candidate_windows=len(a),near175=float(np.mean([r['prediction']>=175 for r in valid]))))
native=[]
for condition in sorted({r['condition'] for r in allrows}):
 for method in sorted({r['method'] for r in allrows if r['condition']==condition}):
  for fold in sorted({r['fold'] for r in allrows if r['method']==method and r['condition']==condition})+['pooled_partial']:
   rr=[r for r in allrows if r['method']==method and r['condition']==condition and (fold=='pooled_partial' or r['fold']==fold)]
   for r in list(rr):
    train=[v['pr_bpm'] for v in dev if v['fold']!=r['fold']];rr.append(dict(r,method='constant',prediction=float(np.median(train))))
    if (condition,r['recording']) in calibrated:rr.append(dict(r,method='calibrated',prediction=calibrated[condition,r['recording']]))
   for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
    for result in metric([r for r in rr if belongs(r,group)],['constant','calibrated',method]):native.append(dict(fold=fold,group=group,condition=condition,target=method,**result))
out=dict(rows=native,windows=len(windowrows),holdout_evaluations=0,complete_folds=sorted({int(f.parent.name[4:]) for f in (P/'adabn').glob('fold*/COMPLETE.json')}))
(P/'BTC_PROGRESS.json').write_text(json.dumps(out,indent=2,allow_nan=False))
lines=['# BTC dev — tiến độ đo được','', 'Số partial không phải kết quả toàn bộ; chưa chọn ứng viên. MAE theo người; CI bootstrap10.000 seed0. Hằng số và trung vị hiệu chỉnh được tính trên đúng giao video.']
for target in sorted({r['target'] for r in native}):
 for fold in sorted({str(r['fold']) for r in native if r['target']==target}):
  lines+=['',f'## {target}, fold {fold}','', '| Điều kiện | Nhóm | Phương pháp | Video/người | MAE [CI95%] | Bias | Δ hiệu chỉnh [CI95%] |','|---|---|---|---:|---:|---:|---:|']
  for r in native:
   if r['target']!=target or str(r['fold'])!=fold:continue
   lo,hi=r['CI95'];dl,dh=r['delta_calibrated_CI'];lines.append(f"| {r['condition']} | {r['group']} | {r['method']} | {r['videos']}/{r['people']} | {r['MAE']:.3f} [{lo:.3f};{hi:.3f}] | {r['bias']:+.3f} | {r['delta_calibrated']:+.3f} [{dl:+.3f};{dh:+.3f}] |")
(P/'BTC_PROGRESS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps([r for r in native if r['group']=='all' and r['method'] not in ['constant','calibrated']]))
