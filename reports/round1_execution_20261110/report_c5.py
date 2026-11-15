"""Matched old/new codec comparison. No inference, training or holdout access."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv
from collections import defaultdict,Counter
import numpy as np
from common import *
from report_metrics import metric,belongs

def key(r):return r['recording'],round(float(r['start']),6)
def valid(r):return r['prediction'] is not None and np.isfinite(r['prediction']) and r['truth'] is not None
def tail(rows):
 a=np.array([r['prediction'] for r in rows if valid(r)])
 return dict(candidate_windows=len(rows),valid_windows=len(a),within_150pm2=float(np.mean(abs(a-150)<=2)) if len(a) else None,at_least175=float(np.mean(a>=175)) if len(a) else None,near42=float(np.mean(abs(a-42)<=2)) if len(a) else None,near180=float(np.mean(abs(a-180)<=2)) if len(a) else None,failures=dict(Counter(r.get('reason') or 'invalid' for r in rows if not valid(r))))

def public_score(rows):
 # Public truth varies by window: score windows within person, never median HR.
 people=sorted({r['person'] for r in rows});e=[];bias=[];ae=[]
 for person in people:
  x=np.array([r['prediction']-r['truth'] for r in rows if r['person']==person]);e.append(abs(x).mean());bias.append(x.mean());ae.extend(abs(x))
 if not people:return {}
 e=np.array(e);bias=np.array(bias);ix=np.random.default_rng(0).integers(len(people),size=(10000,len(people)))
 return dict(people=len(people),windows=len(rows),MAE=float(e.mean()),CI95=np.quantile(e[ix].mean(1),[.025,.975]).tolist(),bias=float(bias.mean()),P95_person_MAE=float(np.quantile(e,.95)),P95_window_abs_error=float(np.quantile(ae,.95)))

def main():
 lease=lock('report_c5.lock');refs=read(B/'public_comparison/REFERENCE_COMPACT.json');frozen=R/'reports/round1_final_20261105/revision_02/frozen_data/FROZEN_WINDOWS.json';assert sha(frozen)==refs['source_sha256']
 dev=read(V/'btc_zeroshot/dev_manifest.json')['videos'];meta={v['recording']:v for v in dev};public=[];btc_windows=[];evidence={};run_metrics={}
 for rec in refs['records']:
  ds,rid,variant=rec['dataset'],rec['recording_id'],rec['variant'];condition='native' if variant=='native' else 'c5'
  for w in rec['windows']:
   base=dict(dataset=ds,recording=rid,person=w['subject_id'],fold=w['fold'],condition=condition,start=w['window_start_s'],truth=w['reference_hr_bpm'])
   for method in ['POS','CHROM','V3','V4']:
    if method in ['POS','CHROM']:
     a=w['baseline'][method];pred=a['prediction_hr_bpm'] if a['accepted'] else None;reason=None if a['accepted'] else 'frozen_reference_or_output_invalid'
    else:
     cachekey=ds,variant,w['fold'],method
     if cachekey not in run_metrics:
      root=R/'t8_polarity_v3_20261102' if method=='V3' else V/'v4_repaired';run=root/f"fold{w['fold']}/seed2027/joint";d=run/'evaluation'/f'{ds}_{variant}';status=read(run/'status.json');assert status['state']=='complete' and sha(run/'final.pt')==status['checkpoint_sha256']
      if method=='V3':
       receipt=read(d/'REFERENCE_POLARITY.json');reference=d/'reference_signal.csv';assert sha(reference)==receipt['canonical_reference_sha256'];assert sha(d/'raw_reference_signal.csv')==receipt['raw_reference_sha256'];assert receipt['all_hr_and_failure_parity']
      else:
       receipt=read(d/'REFERENCE_REUSE.json');reference=Path(receipt['path']);assert sha(reference)==receipt['sha256'] and sha(receipt['raw_path'])==receipt['raw_sha256'];assert sha(reference.parent/'REFERENCE_POLARITY.json')==receipt['polarity_receipt_sha256']
      a=read(d/'metrics.json');assert a['reference_sha256']==sha(reference) and a['output_sha256']==sha(d/'processed_output.csv');evidence[str(d/'metrics.json')]=sha(d/'metrics.json');run_metrics[cachekey]={(x['recording_id'],round(x['window_start_s'],6)):x for x in a['windows']}
     a=run_metrics[cachekey][key(base)];assert a['reference_hr_bpm']==base['truth'] and a['reference_eligible']==w['reference_eligible'];pred=a['prediction_hr_bpm'] if a['accepted'] else None;reason=a['prediction_reason']
    public.append(dict(base,method=method,prediction=pred,reason=reason))
 for file in sorted((P/'c5_predictions').glob('*/*/*.json')):
  a=read(file);ds,rid,variant=file.parent.parent.name,file.parent.name,file.stem;cp=P/'c5prime'/ds/rid/variant/'COMPLETE.json';assert sha(cp)==a['source_receipt_sha256'];evidence[str(file)]=sha(file)
  if ds=='BTC':btc_windows.extend(r for r in a['rows'] if r['method'] in ['POS','CHROM'])
  else:public.extend(a['rows'])
 # Fifteen-checkpoint BTC median, then window median per video; partial ensembles invalid.
 for file in sorted((P/'btc_c5ensemble').glob('*/*.json')):
  a=read(file);rid,variant=file.parent.name,file.stem;assert rid in meta
  assert a['source_receipt_sha256']==sha(P/'c5prime/BTC'/rid/variant/'COMPLETE.json');evidence[str(file)]=sha(file)
  for branch,method in [('v3_joint','V3'),('v4_repaired','V4')]:
   for start in sorted({r['start'] for r in a['rows']}):
    rr=[r for r in a['rows'] if r['branch']==branch and r['start']==start];assert len(rr)==15 and len({r['method'] for r in rr})==15
    ok=all(valid(r) for r in rr);btc_windows.append(dict(rr[0],method=method,prediction=float(np.median([r['prediction'] for r in rr])) if ok else None,reason=None if ok else 'one_or_more_of_15_checkpoint_outputs_invalid'))
 # New sweep is only added after all five attempts finish; rejected folds remain absent.
 if (P/'SWEEP_STATUS.json').exists() and read(P/'SWEEP_STATUS.json')['state']=='complete':
  for fold in range(5):
   d=P/f'sweep/fold{fold}'
   if read(d/'status.json')['state']!='complete':continue
   for file in (d/'public_evaluation').glob('*/metrics.json'):
    a=read(file);receipt=read(file.parent/'REFERENCE_REUSE.json');assert sha(receipt['path'])==receipt['sha256']==a['reference_sha256'];assert sha(receipt['raw_path'])==receipt['raw_sha256'];assert sha(Path(receipt['path']).parent/'REFERENCE_POLARITY.json')==receipt['polarity_receipt_sha256'];assert sha(file.parent/'processed_output.csv')==a['output_sha256'];evidence[str(file)]=sha(file)
    ds,variant=file.parent.name.split('_',1)
    for w in a['windows']:public.append(dict(dataset=ds,recording=w['recording_id'],person=w['subject_id'],fold=fold,condition='native' if variant=='native' else 'c5',start=w['window_start_s'],truth=w['reference_hr_bpm'],method='new_sweep',prediction=w['prediction_hr_bpm'] if w['accepted'] else None,reason=w['prediction_reason']))
   file=d/'C5PRIME.json';evidence[str(file)]=sha(file);public.extend(dict(r,method='new_sweep') for r in read(file) if r.get('dataset') in ['UBFC2','MCD'])
 pubmetrics=[];pubcoverage=[]
 for ds in ['UBFC2','MCD']:
  all_ds=[r for r in public if r['dataset']==ds];conditions=['native','c5','c5prime_default','c5prime_bf0'];methods=['POS','CHROM','V3','V4']+(['new_sweep'] if any(r['method']=='new_sweep' for r in all_ds) else []);look={(c,m):{key(r):r for r in all_ds if r['condition']==c and r['method']==m and valid(r)} for c in conditions for m in methods}
  cross=set.intersection(*(set(x) for x in look.values()))
  for k in cross:assert len({look[c,m][k]['truth'] for c in conditions for m in methods})==1,'Cross-condition reference mismatch'
  for c in conditions:
   common=set.intersection(*(set(look[c,m]) for m in methods))
   for m in methods:
    rr=[r for r in all_ds if r['condition']==c and r['method']==m];pubcoverage.append(dict(dataset=ds,condition=c,method=m,**tail(rr)))
    for scope,keys in [('within_condition',common),('all_conditions_same_windows',cross)]:
     selected=[look[c,m][k] for k in sorted(keys)];ans=public_score(selected)
     if ans:pubmetrics.append(dict(dataset=ds,condition=c,method=m,scope=scope,**ans))
 btc=[];mapping={'constant':'constant','POS_old_argmax':'POS','CHROM_old':'CHROM','V3_median15':'V3','V4_original_padding_median15':'V4','calibrated_median_POS_CHROM_V3':'calibrated'}
 for file in [B/'VIDEO_PREDICTIONS.csv',B/'BTC_V4_SUPPLEMENT.csv']:
  evidence[str(file)]=sha(file)
  for r in csv.DictReader(file.open()):
   if r['method'] not in mapping:continue
   v=meta[r['recording']];btc.append(dict(v,truth=float(r['truth']),condition=r['condition'],method=mapping[r['method']],prediction=float(r['prediction'])))
 complete=(P/'C5_EVALUATION_COMPLETE.json').exists() and all((P/f'btc_c5ensemble/SHARD{s}_COMPLETE.json').exists() for s in range(4));fits=[]
 if complete:
  for c in ['c5prime_default','c5prime_bf0']:
   video=[]
   for m in ['POS','CHROM','V3','V4']:
    for rid,v in meta.items():
     rr=[r for r in btc_windows if r['recording']==rid and r['condition']==c and r['method']==m and valid(r)]
     if rr:video.append(dict(v,truth=v['pr_bpm'],condition=c,method=m,prediction=float(np.median([r['prediction'] for r in rr]))))
   look={m:{r['recording']:r for r in video if r['method']==m} for m in ['POS','CHROM','V3']};keys=set.intersection(*(set(x) for x in look.values()));med=[dict(look['POS'][k],prediction=float(np.median([look[m][k]['prediction'] for m in look]))) for k in sorted(keys)]
   for fold in range(5):
    train=[r for r in med if r['fold']!=fold];val=[r for r in med if r['fold']==fold];counts=Counter(r['person'] for r in train);assert not set(counts)&{r['person'] for r in val}
    assert len(counts)>=2,'Insufficient train support; no calibration fallback'
    w=np.sqrt([1/counts[r['person']] for r in train]);x=np.array([[r['prediction'],1] for r in train]);y=np.array([r['truth'] for r in train]);coef=np.linalg.lstsq(x*w[:,None],y*w,rcond=None)[0];fits.append(dict(condition=c,fold=fold,a=float(coef[0]),b=float(coef[1]),train_people=sorted(counts)))
    video.extend(dict(r,method='calibrated',prediction=float(coef[0]*r['prediction']+coef[1])) for r in val)
   for v in dev:video.append(dict(v,truth=v['pr_bpm'],condition=c,method='constant',prediction=float(np.median([x['pr_bpm'] for x in dev if x['fold']!=v['fold']]))))
   btc.extend(video)
 metrics=[];coverage=[];methods=['constant','calibrated','POS','CHROM','V3','V4']
 for c in sorted({r['condition'] for r in btc}):
  for g in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
   rr=[r for r in btc if r['condition']==c and belongs(r,g)];metrics.extend(dict(condition=c,group=g,**m) for m in metric(rr,methods))
  for m in methods:
   rr=[r for r in btc if r['condition']==c and r['method']==m];coverage.append(dict(condition=c,method=m,valid_videos=len(rr),candidate_videos=221,candidate_people=161,**tail([r for r in btc_windows if r['condition']==c and r['method']==m])))
 save(P/'C5_PUBLIC_METRICS.json',pubmetrics);save(P/'C5_PUBLIC_COVERAGE.json',pubcoverage);save(P/'C5_BTC_METRICS.json',metrics);save(P/'C5_BTC_VIDEO.json',btc);save(P/'C5_BTC_COVERAGE.json',coverage);save(P/'C5_BTC_CALIBRATION_FITS.json',fits)
 save(P/'C5_REPORT_RECEIPT.json',dict(utc=now(),full_C5_complete=complete,input_hashes=evidence,reference_sha256=refs['source_sha256'],public_policy='same-fold seed2027 V3/V4; paired windows; public truth varies by window',BTC_policy='15-checkpoint median per window then median windows; original native/c5 calibration preserved, new-condition calibration person-OOF fit only after complete',MATLAB_parity=False,holdout_evaluations=0))
 lines=['# Đối chiếu C5 cũ và C5′','',f"Trạng thái toàn bộ C5′: {'hoàn tất' if complete else 'PARTIAL — đang xử lý'}. Không chọn phương pháp từ bảng partial.",'','C5′ là mô phỏng ffmpeg, chưa xác minh tương đương MATLAB VideoWriter. `default` giữ GOP/B-frame mặc định; `bf0` tắt B-frame. Mỗi hàng tính lỗi trong từng người trước, rồi trung bình người; CI bootstrap người 10.000 lần, seed 0. Δ hiệu chỉnh âm là tốt hơn. P95_person_MAE là phân vị 95% của MAE từng người; khác P95 lỗi từng cửa sổ.','', '## UBFC/MCD — giao cùng cửa sổ giữa cả bốn điều kiện','', 'V3/V4 dùng cùng fold, seed2027; không trộn với trung vị 15 checkpoint của BTC. Khi còn partial, giao này chỉ gồm bản ghi đã hoàn tất ở cả hai C5′.','', '| Dataset | Điều kiện | Phương pháp | Người/cửa sổ | MAE [CI95%] | Bias |','|---|---|---|---:|---:|---:|']
 for r in pubmetrics:
  if r['scope']!='all_conditions_same_windows':continue
  lo,hi=r['CI95'];lines.append(f"| {r['dataset']} | {r['condition']} | {r['method']} | {r['people']}/{r['windows']} | {r['MAE']:.3f} [{lo:.3f}; {hi:.3f}] | {r['bias']:+.3f} |")
 lines+=['','## BTC — cùng giao video trong mỗi điều kiện','', 'Mốc hiệu chỉnh native giữ nguyên dự đoán đã lưu (11,37 trên cohort cũ). Hiệu chỉnh C5′ là cùng công thức hai hệ số, fit lại bằng train-fold của điều kiện đó; không được gọi là MAE 11,37 trên C5′. Mọi bảng có cả hằng số và hiệu chỉnh. Chỉ fit C5′ khi toàn bộ đầu ra sẵn sàng.','', '| Điều kiện | Nhóm | Phương pháp | Video/người | MAE [CI95%] | Δ hiệu chỉnh [CI95%] |','|---|---|---|---:|---:|---:|']
 for r in metrics:
  lo,hi=r['CI95'];dl,dh=r['delta_calibrated_CI'];lines.append(f"| {r['condition']} | {r['group']} | {r['method']} | {r['videos']}/{r['people']} | {r['MAE']:.3f} [{lo:.3f}; {hi:.3f}] | {r['delta_calibrated']:+.3f} [{dl:+.3f}; {dh:+.3f}] |")
 lines+=['','Độ phủ và lỗi từng cửa sổ nằm trong C5_PUBLIC_COVERAGE.json và C5_BTC_COVERAGE.json. Không có cửa sổ mới của phương pháp hằng/hiệu chỉnh vì đây là dự đoán cấp video; các số cửa sổ bằng 0 ở hai hàng này nghĩa là không áp dụng, không phải tỉ lệ lỗi 0%. Các bảng old/native ở đây không tải lại waveform và không tính lại boundary-window statistics.','', 'Không mở holdout; không thay ngưỡng, reference hoặc checkpoint theo điểm số.']
 (P/'C5_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print('C5_REPORT',dict(full=complete,public_metric_rows=len(pubmetrics),BTC_metric_rows=len(metrics)))

if __name__=='__main__':main()
