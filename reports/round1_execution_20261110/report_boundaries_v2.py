"""Incremental requested boundary-rate tables; no model or threshold changes."""
import os,time,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import csv
import numpy as np
from common import *
from report_metrics import belongs

def once(files):
 dev=read(V/'btc_zeroshot/dev_manifest.json')['videos'];meta={v['recording']:v for v in dev}
 baseline={('native',r['recording']):float(r['prediction']) for r in csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open()) if r['condition']=='native' and r['method']=='calibrated_median_POS_CHROM_V3'}
 if (P/'C5_BTC_VIDEO.json').exists():baseline.update({(r['condition'],r['recording']):r['prediction'] for r in read(P/'C5_BTC_VIDEO.json') if r['method']=='calibrated'})
 constant={f:float(np.median([v['pr_bpm'] for v in dev if v['fold']!=f])) for f in range(5)};windows=[]
 for f in files:
  for r in read(f):
   if r.get('dataset','BTC')=='BTC':windows.append(dict(r,group=meta[r['recording']]['group'],task=meta[r['recording']]['task']))
 out=[]
 for c in sorted({r['condition'] for r in windows}):
  for target in sorted({r['method'] for r in windows if r['condition']==c}):
   for fold in sorted({r['fold'] for r in windows if r['condition']==c and r['method']==target})+['pooled_partial']:
    candidates=[r for r in windows if r['condition']==c and r['method']==target and (fold=='pooled_partial' or r['fold']==fold)]
    for g in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
     rr=[r for r in candidates if belongs(r,g)];matched=[r for r in rr if r['prediction'] is not None and (c,r['recording']) in baseline]
     if not matched:continue
     for method in ['constant','calibrated',target]:
      values=np.array([constant[r['fold']] if method=='constant' else baseline[c,r['recording']] if method=='calibrated' else r['prediction'] for r in matched]);errors=values-np.array([r['truth'] for r in matched]);people=sorted({r['person'] for r in matched});bias=float(np.mean([errors[[r['person']==p for r in matched]].mean() for p in people]))
      out.append(dict(condition=c,target=target,fold=fold,group=g,method=method,candidate_windows=len(rr),matched_windows=len(matched),videos=len({r['recording'] for r in matched}),people=len(people),at_least175=float(np.mean(values>=175)),within_150pm2=float(np.mean(abs(values-150)<=2)),near42=float(np.mean(abs(values-42)<=2)),near180=float(np.mean(abs(values-180)<=2)),bias_person=bias))
 save(P/'PROGRESS_WINDOWS.json',dict(utc=now(),rows=out,input_hashes={str(f):sha(f) for f in files},holdout_evaluations=0))
 lines=['# BTC — độ phủ và tỉ lệ dự đoán gần biên','', 'Partial đến khi đủ năm fold. Tỉ lệ là số cửa sổ, không phải MAE cấp người. Hằng số/hiệu chỉnh là dự đoán cấp video được lặp trên đúng cửa sổ của model để có cùng mẫu số; đây không phải ước lượng nhịp biến thiên của hai baseline. Bias được tính trong từng người rồi trung bình. Cửa sổ thất bại giữ trong mẫu số candidate.','', '| Điều kiện | Target | Fold | Nhóm | Phương pháp | Video/người | Cửa sổ ghép/candidate | ≥175 (%) | 150±2 (%) | Bias |','|---|---|---|---|---|---:|---:|---:|---:|---:|']
 for r in out:lines.append(f"| {r['condition']} | {r['target']} | {r['fold']} | {r['group']} | {r['method']} | {r['videos']}/{r['people']} | {r['matched_windows']}/{r['candidate_windows']} | {100*r['at_least175']:.2f} | {100*r['within_150pm2']:.2f} | {r['bias_person']:+.3f} |")
 (P/'PROGRESS_WINDOWS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def main():
 lease=lock('report_boundaries_v2.lock');assert not (P/'BOUNDARY_REPORTER_V2_STARTED.json').exists();guard(True);save(P/'BOUNDARY_REPORTER_V2_STARTED.json',dict(utc=now(),pid=os.getpid(),code_sha256=sha(__file__)));last=None
 while datetime.now(timezone.utc)<datetime(2026,11,23,tzinfo=timezone.utc):
  files=[*P.glob('adabn/fold*/NATIVE.json'),*P.glob('adabn/fold*/C5PRIME.json'),*P.glob('sweep/fold*/NATIVE.json'),*P.glob('sweep/fold*/C5PRIME.json')];signature={str(f):sha(f) for f in files}
  if (P/'C5_BTC_VIDEO.json').exists():signature['calibration']=sha(P/'C5_BTC_VIDEO.json')
  if files and signature!=last:once(files);last=signature
  if (P/'execution_v2/COMPUTE_COMPLETE.json').exists() or (P/'execution_v2/ERROR.json').exists():break
  time.sleep(30)
 save(P/'BOUNDARY_REPORTER_V2_FINISHED.json',dict(utc=now(),holdout_evaluations=0))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'BOUNDARY_REPORTER_V2_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
