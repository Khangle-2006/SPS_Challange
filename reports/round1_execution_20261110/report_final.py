"""A-F conclusion from completed, immutable outputs. Never opens holdout."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from common import *
from report_metrics import metric,belongs

def main():
 lease=lock('final_report.lock');assert (P/'D_STACK_COMPLETE.json').exists()
 from report_c5 import main as refresh_c5
 refresh_c5();assert read(P/'C5_REPORT_RECEIPT.json')['full_C5_complete']
 baseline=read(P/'C5_BTC_VIDEO.json');rows=[r for r in baseline if r['method'] in ['constant','calibrated','POS','CHROM','V3','V4']];coverage=[];attempts=[];windowrows=[];hashes_used={str(P/'C5_BTC_VIDEO.json'):sha(P/'C5_BTC_VIDEO.json')}
 def collect(base,family,filename):
  combined=[];folds=[]
  for fold in range(5):
   d=base/f'fold{fold}';file=d/filename
   if not file.exists():continue
   if family=='new_sweep':
    st=read(d/'status.json')
    if st['state']!='complete':continue
   if family=='AdaBN':assert (d/'COMPLETE.json').exists()
   rr=read(file);combined.extend(dict(r,method=family) for r in rr if r.get('dataset','BTC')=='BTC');folds.append(fold);hashes_used[str(file)]=sha(file)
  for c in sorted({r['condition'] for r in combined}):
   for rid in sorted({r['recording'] for r in combined if r['condition']==c}):
    rr=[r for r in combined if r['condition']==c and r['recording']==rid];good=[r for r in rr if r['prediction'] is not None]
    coverage.append(dict(method=family,condition=c,recording=rid,candidate_windows=len(rr),valid_windows=len(good),folds=folds))
    if good:
     v=next(r for r in baseline if r['recording']==rid);rows.append(dict(v,method=family,condition=c,prediction=float(np.median([r['prediction'] for r in good])),complete_five_folds=folds==list(range(5))))
  windowrows.extend(combined)
 for family,folder in [('AdaBN','adabn'),('new_sweep','sweep')]:
  for filename in ['NATIVE.json','C5PRIME.json']:collect(P/folder,family,filename)
 for fold in range(5):attempts.append(dict(fold=fold,**read(P/f'sweep/fold{fold}/status.json')))
 drows=read(P/'D_VIDEO_PREDICTIONS.json');rows.extend(r for r in drows if r['method'] not in ['constant','calibrated']);hashes_used[str(P/'D_VIDEO_PREDICTIONS.json')]=sha(P/'D_VIDEO_PREDICTIONS.json')
 results=[];gates=[]
 for c in sorted({r['condition'] for r in rows}):
  for target in sorted({r['method'] for r in rows if r['condition']==c}-{'constant','calibrated'}):
   selected=[r for r in rows if r['condition']==c and r['method'] in ['constant','calibrated',target]]
   for g in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
    metrics=metric([r for r in selected if belongs(r,g)],['constant','calibrated',target]);results.extend(dict(condition=c,group=g,comparison_target=target,**m) for m in metrics)
    if c=='native' and g=='all' and metrics:
     m=next(x for x in metrics if x['method']==target);baseids={r['recording'] for r in selected if r['method']=='constant'}&{r['recording'] for r in selected if r['method']=='calibrated'};ids={r['recording'] for r in selected if r['method']==target and r['prediction'] is not None};full=baseids<=ids
     five=target!='new_sweep' or all(a['state']=='complete' for a in attempts);passed=m['MAE']<=10 and m['delta_calibrated']<=-1 and m['delta_calibrated_CI'][1]<0
     gates.append(dict(method=target,measured_criteria_pass=passed,full_comparator_cohort=full,valid_five_fold_family=five,status='PASS_DEV_EXPLORATORY' if passed and full and five else 'NOT_ASSESSABLE_FULL_COHORT' if not full or not five else 'FAIL',MAE=m['MAE'],delta_calibrated=m['delta_calibrated'],CI=m['delta_calibrated_CI'],people=m['people'],videos=m['videos']))
 save(P/'FINAL_BTC_METRICS.json',results);save(P/'FINAL_BTC_COVERAGE.json',coverage);save(P/'FINAL_GATES.json',gates)
 lines=['# BTC dev — kết quả chương trình A–F','', 'Kết quả dev có lựa chọn phương pháp trên dev, nên CI sau lựa chọn là thăm dò. Holdout 72 video/41 người vẫn khóa; chưa có kết luận test độc lập.','', '## Cổng đã khóa','', '| Phương pháp | MAE | Δ hiệu chỉnh [CI95%] | Trạng thái |','|---|---:|---:|---|']
 for r in gates:lines.append(f"| {r['method']} | {r['MAE']:.3f} | {r['delta_calibrated']:+.3f} [{r['CI'][0]:+.3f}; {r['CI'][1]:+.3f}] | {r['status']} |")
 lines+=['','Tiêu chí: MAE người ≤10 bpm, giảm ≥1 bpm so với trung vị hiệu chỉnh trên cùng video, CI95% Δ có cận trên <0. Thiếu dự đoán trên cohort baseline thì chỉ có kết quả chọn lọc, không khẳng định đạt cho toàn dev.','', '## Các lượt huấn luyện','', '| Fold | Trạng thái final |','|---|---|']
 for r in attempts:lines.append(f"| {r['fold']} | {r['state']} |")
 lines+=['','Năm fold dùng ba recipe đã ghi trước, không phải so sánh nhân quả giữa recipe. Lượt trượt ảnh tĩnh hoặc collapse bị loại; không thay bằng checkpoint khác.','', '## BTC — bảng ghép cặp','', 'Mỗi khối target có hằng số và trung vị hiệu chỉnh trên đúng giao video. Không so trực tiếp MAE giữa hai khối có số người/video khác nhau. P95 là phân vị MAE cấp người; bias có dấu = dự đoán − nhãn.','', '| Điều kiện | Nhóm | Target | Phương pháp | Video/người | MAE [CI95%] | P95 người | Bias |','|---|---|---|---|---:|---:|---:|---:|']
 for r in results:lines.append(f"| {r['condition']} | {r['group']} | {r['comparison_target']} | {r['method']} | {r['videos']}/{r['people']} | {r['MAE']:.3f} [{r['CI95'][0]:.3f}; {r['CI95'][1]:.3f}] | {r['P95_person_MAE']:.3f} | {r['bias']:+.3f} |")
 lines+=['','## Các giới hạn phải giữ','', '- C5′ là ffmpeg proxy; chưa xác minh MATLAB VideoWriter. C5 cũ và C5′ có cột riêng trong C5_REPORT.md.', '- AiPEX 15 fps là giả định chưa xác minh acquisition; không fit fps từ nhãn.', '- A1 khôi phục POS MCD 4.67015 bpm, nhưng blind SNR median −0.24956 dB vẫn trượt cổng; chưa dùng Bayes để kết luận không có tín hiệu.', '- D chọn nhánh trên dev. Biến neural dùng để fit stack là in-sample của model outer-fold; outer validation không được model/stack học. Không coi CI sau lựa chọn là kiểm chứng độc lập.', '- process_round1.m là POS packaging baseline, chưa phải cấu hình BTC cuối. Octave 8.4 --traditional khớp Python trên 5 video; chưa chạy MATLAB thật.', '', '## Tái lập','', 'Nguồn dữ liệu, checkpoint và cấu hình: PLAN.json, FROZEN.json, EXECUTION_FROZEN.json, GOP_COMPLETE.json, C5_REPORT_RECEIPT.json và từng COMPLETE.json. Lệnh đã chạy nằm ở execution/*_LAUNCH.json; không chạy lại vào cùng output. CPU đã hoàn tất ở CPU_REPORT.md; public/C5 ở C5_REPORT.md; D ở D_METRICS.json. Không thay seed, cửa sổ hay ngưỡng theo kết quả.']
 if not any(r['status']=='PASS_DEV_EXPLORATORY' for r in gates):lines+=['','Trong phạm vi chương trình thí nghiệm A–F đã khóa: **đã thử hết đòn bẩy có cơ sở văn liệu, dừng cải thiện BTC**. Câu này không có nghĩa đã thử mọi phương pháp trong văn liệu. Giữ phương án có bằng chứng dev tốt nhất và báo cả hằng số; không chỉnh ngưỡng để nhận thắng.']
 else:lines+=['','Có phương án đạt tiêu chí trên dev; cần review toàn bộ độ phủ/cổng và khóa cấu hình trước lần holdout duy nhất. Không tự mở holdout hoặc nộp bài.']
 (P/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf8');save(P/'FINAL_REPORT_RECEIPT.json',dict(utc=now(),input_hashes=hashes_used,holdout_evaluations=0,review_required=True))
 print('FINAL_REPORT_READY',gates)
if __name__=='__main__':main()
