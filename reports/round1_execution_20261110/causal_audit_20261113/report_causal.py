"""Evidence-driven review draft, no automatic promotion or acquisition attribution."""
from common_causal import *
import csv,traceback,zipfile
from collections import defaultdict
import numpy as np

def table(headers,rows):
 return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(v) for v in r)+'|' for r in rows])
def fmt(x):return '—' if x is None else f'{x:.5f}' if isinstance(x,float) else str(x)
def render():
 from compare_frontends import run as compare_recorded_frontends
 compare_recorded_frontends()
 ledger=[];lines=['# Truy nguyên BTC — báo cáo bằng chứng',f'Cập nhật UTC: {now()}. Không huấn luyện; holdout chưa mở. Đây là bản rà soát, không tự thay FREEZE hay chọn lại mô hình.','## 1. Phụ thuộc batch']
 if (C/'BATCH_COMPLETE.json').exists():
  a=read(C/'BATCH_COMPLETE.json');bad=[c['key'] for c in a['records'] if any(r['first_recorded_divergence'] is not None or r['permuted_first'] is not None for r in c['runs'])]
  lines.append(f"20 cửa sổ, 5 checkpoint cùng fold. Sai khác waveform lớn nhất chạy đơn so với batch: {a['max_waveform_abs_difference']:.9g}. Đổi clip đi kèm/hoán vị: {a['max_companion_or_permutation_difference']:.9g}. Repeated-run control: {a['max_repeated_run_difference']:.9g}. Ca vượt ngưỡng đã chốt: {len(bad)}. BN eval, track_running_stats=True; buffers không đổi. TSM chia mỗi clip300frame thành30 đoạn10frame, không cắt qua biên clip.")
  ledger.append(dict(hypothesis_id='H_batch_coupling',status='unsupported_in_test' if not bad else 'not_identifiable',evidence_files=['BATCH_COMPLETE.json'],first_recorded_divergence=bad or None,limitations=['20 fixed tensors/five checkpoints only; not a proof for all inputs'],person_level_effect=None))
 lines+=['## 2. Thang suy giảm', 'Mỗi hàng dùng nhãn PPG của cùng cửa sổ. Bias = dự đoán − nhãn, đơn vị bpm; số âm là đo thấp. SNR mù đo quanh đỉnh thuật toán tự chọn, không quanh nhãn. Mất tracking/không dự đoán được được giữ riêng, không điền bằng nhãn hoặc hằng số.']
 stages=['s0_native','s1_hevc','s2_display90','s3_HLG10','s4_848x480','s5_hqdn3d','s5_temporalNLM','s6_hqdn3d_10fps','s6_temporalNLM_10fps'];allrows=[];first=[]
 if (C/'LADDER_SELECTION.json').exists():
  selected=read(C/'LADDER_SELECTION.json')
  for case in selected:
   j=case['job'];reference_root=C/'physical_v3/ladder'/j['key'];basepath=reference_root/'s0_native/COMPLETE.json';base=read(basepath) if basepath.exists() else None
   for method in ['POS','padding']:
    baseline_ok=None if base is None else bool(base.get(method) is not None and abs(base[method]-base['truth'])<=5);firstbad=None
    for stage in stages:
     p=reference_root/stage/'COMPLETE.json'
     if not p.exists():continue
     a=read(p);prediction=a.get(method);error=None if prediction is None else abs(prediction-a['truth']);failure=baseline_ok and stage!='s0_native' and (prediction is None or (error>10 and error-abs(base[method]-base['truth'])>=5))
     if failure and firstbad is None:firstbad=stage
     allrows.append(dict(dataset=j['dataset'],person=j['person'],recording=j['recording'],method=method,stage=stage,truth=a['truth'],prediction=prediction,error=error,bias=None if prediction is None else prediction-a['truth'],blind_SNR_dB=a.get(method+'_blind_SNR_dB'),valid_fraction=a.get('valid_fraction'),baseline_positive_control_reproduced=baseline_ok,declared_HR_failure=bool(failure),status=a['status']))
    first.append(dict(recording=j['recording'],method=method,baseline_reproduced=baseline_ok,first_HR_failure=firstbad,all_stages_complete=all((reference_root/s/'COMPLETE.json').exists() for s in stages)))
  lines.append(table(['Ca','Phương pháp','Baseline đúng ≤5?','Bậc hỏng đầu','Đủ9bậc?'],[[r['recording'],r['method'],'chưa chạy' if r['baseline_reproduced'] is None else ('đạt' if r['baseline_reproduced'] else 'không đạt'),r['first_HR_failure'] or ('chưa đánh giá' if r['baseline_reproduced'] is None else 'không xác định: baseline không đạt' if not r['baseline_reproduced'] else 'chưa thấy / chưa đủ'),'đủ' if r['all_stages_complete'] else 'chưa đủ'] for r in first]))
  summary=[]
  for ds in ['UBFC2','MCD']:
   for method in ['POS','padding']:
    for stage in stages:
     rr=[r for r in allrows if (r['dataset'],r['method'],r['stage'])==(ds,method,stage)];ok=[r for r in rr if r['prediction'] is not None]
     if not rr:continue
     # One selected window per distinct person: direct means are person-first here.
     snr=[r['blind_SNR_dB'] for r in ok if r['blind_SNR_dB'] is not None]
     summary.append([ds,method,stage,f'{len(ok)}/{len(rr)}',fmt(float(np.mean([r['error'] for r in ok]))) if ok else '—',fmt(float(np.mean([r['bias'] for r in ok]))) if ok else '—',fmt(float(np.median(snr))) if snr else '—'])
  lines.append(table(['Bộ','Phương pháp','Bậc','Hợp lệ/đã chạy','MAE','Bias','SNR trung vị dB'],summary));save(C/'LADDER_ROWS.json',allrows);save(C/'LADDER_FIRST_FAILURE.json',first)
  lines.append('Khác mảng sau can thiệp là điều dự kiến, không tự động là bug. Stage comparison dùng phần số học của công cụ được cung cấp với hợp đồng controlled_degradation được công bố riêng. Không cô lập riêng codec khỏi upscale ở bậc1; HLG là round-trip tổng hợp; bậc xoay chủ ý thay hướng hiển thị. Chỉ gọi một yếu tố gây hại trong đúng phép can thiệp này sau khi baseline tái lập và kiểm đối chứng bổ sung.')
  lines.append('Cả bậc không chấm được vẫn xuất PTS, cờ dò mặt, ROI và RGB trước chuẩn hóa trong FRONTEND.npz, cùng FRONTEND_COMPARE_TO_S0.json. Không tạo tensor giả cho cửa sổ bị loại. Thư mục upstream_routes đối chiếu cache public cũ với đường đọc video BTC thật tại bậc gốc; đây là so sánh mô tả hai cách tạo ROI, không mặc định chúng phải giống từng pixel.')
  lines.append('MAE/bias trong bảng chính dùng thời gian public đã xác minh theo danh tính khung hình (physical_v3). UBFC dùng clock thu trong ground_truth, khác PTS container AVI. Kết quả đường BTC dùng PTS container ban đầu vẫn giữ nguyên trong ladder; không trộn hai clock hoặc gọi chênh lệch này là lỗi nhãn BTC. Xem PHYSICAL_CLOCK_V3.md.')
  ledger.append(dict(hypothesis_id='H_public_audit_clock_bridge',status='confirmed_defect',scope='Public-to-BTC audit harness reference clock only, not BTC production or frozen public cache',evidence_files=['PHYSICAL_CLOCK_V3.md','LADDER_ROWS.json','upstream_routes_physical'],repair='Use verified source frame identity and original public reference clock, unchanged pixels/checkpoint/readout; preserve container-clock experiment.',person_level_effect=None))
 else:lines.append('Đang chuẩn bị/chạy; chưa có bảng đầy đủ.')
 ladder_done=(C/'physical_v3/LADDER_COMPLETE.json').exists();ledger.append(dict(hypothesis_id='H_acquisition_loss',status='not_identifiable',evidence_files=['LADDER_ROWS.json'] if allrows else [],limitations=['Synthetic ladder does not recreate phone ISP, exposure, physiology or label timing. A finite passing test is not proof of acquisition loss.'],missing_evidence=['simultaneous team webcam/phone recordings with independent reference/time mapping'],person_level_effect=None))
 if (C/'LOCALIZATION_COMPLETE.json').exists():
  l=read(C/'LOCALIZATION_COMPLETE.json');lines.append(f"Đối chứng lịch sử trên cùng bitstream của ca đầu: xoay ngược90° đã chủ ý thêm, trước detector, khôi phục valid frames từ {l['before_valid_fraction']:.0%} lên {l['after_valid_fraction']:.0%}. Ca này dùng clock container và chỉ chứng minh thay đổi tracking trong can thiệp đó; không đưa HR của nó vào bảng PPG chính. Kết quả clock public hiện hành nằm trong PHYSICAL_LOCALIZATION_ROWS.json khi hoàn tất. Không train, không đổi decoder. Chưa đo đóng góp lên BTC.")
  ledger.append(dict(hypothesis_id='H_orientation_before_detection_synthetic_stage4',status='confirmed_contributor' if l.get('status')=='scorable' and l['after_valid_fraction']>l['before_valid_fraction'] else 'not_identifiable',evidence_files=['LOCALIZATION_STARTED.json','LOCALIZATION_COMPLETE.json','LOCALIZATION_FIRST_REPORT.md'],first_recorded_divergence='valid face support at rotated+padded low-resolution stage4',repair='inverse known injected90 before detector, pure pixel permutation',person_level_effect=None,limitations=['single synthetic positive control, not BTC cohort attribution; ten-case follow-up pending']))
 controls=[]
 for f in sorted((C/'physical_v3/ladder').glob('*/inverse90_stage4/COMPLETE.json')):
  a=read(f);before=read(f.parent.parent/'s4_848x480/COMPLETE.json');controls.append(dict(key=a['key'],truth=a['truth'],before_valid=before['valid_fraction'],after_valid=a['valid_fraction'],before_POS=before.get('POS'),after_POS=a.get('POS'),before_padding=before.get('padding'),after_padding=a.get('padding'),static_pass=a.get('static_pass'),same_bitstream=a['encoded_sha256']==before['encoded_sha256']))
 if controls:
  lines.append('Đối chứng hướng trên cùng bitstream, clock public đã sửa; đủ10ca mới tổng kết toàn bộ:')
  lines.append(table(['Ca','PPG','Valid trước','Valid sau','POS sau','Padding sau','Cùng bitstream'],[[r['key'],fmt(r['truth']),fmt(r['before_valid']),fmt(r['after_valid']),fmt(r['after_POS']),fmt(r['after_padding']),r['same_bitstream']] for r in controls]))
  save(C/'PHYSICAL_LOCALIZATION_ROWS.json',controls)
 lines+=['## 3. Đồng hồ PTS', '301 điểm mô hình gồm hai đầu của10giây; ở nguồn10fps chúng không phải301ảnh độc lập. Phân biệt PTS hiệu dụng sau nội suy, PTS ảnh gần nhất và khoảng ảnh dùng để kẹp nội suy.']
 if (C/'METADATA_COMPLETE.json').exists():
  a=read(C/'METADATA_COMPLETE.json');lines.append(f"Đã kiểm218video/158người; {a['PTS_windows']} cửa sổ–điều kiện. Vi phạm span hiệu dụng10±0,05s: {a['span_violations']}; có ngoại suy: {a['extrapolation']}. Xem PTS_BY_FPS.csv và PTS_WINDOWS.csv, có cả cửa sổ bị loại vì support.")
  pts=list(csv.DictReader((C/'PTS_WINDOWS.csv').open()));accepted=[r for r in pts if r['supported']=='True'];near=sum(r['nearest_span_violation']=='True' for r in accepted);extra=sum(r['extrapolation']=='True' for r in accepted)
  lines.append(f"Trong {len(accepted)} cửa sổ có support hợp lệ: {near} span của hai frame nguồn gần nhất vượt10±0,05s, {extra} cửa sổ cần ngoại suy. Span hiệu dụng đúng10s là hệ quả của nội suy lên lưới cố định, tự nó không chứng minh PTS nguồn đúng; đối chiếu bitstream, span frame thật và support mới là kiểm tra bổ sung. Không gọi mọi301điểm là301frame nguồn.")
  ledger.append(dict(hypothesis_id='H_time_span',status='unsupported_in_test' if a['span_violations']==0 and a['extrapolation']==0 else 'not_identifiable',evidence_files=['METADATA_COMPLETE.json','PTS_BY_FPS.csv'],limitations=['Interpolation span alone cannot prove physiological reference alignment.']))
 else:lines.append('Đang đối chiếu PTS trong bitstream gốc, không chỉ đọc fps từ metadata.')
 lines+=['## 4. Thu đối chứng thực tế','Hướng dẫn: HUONG_DAN_THU_DOI_CHUNG.md. Chưa nhận cặp webcam–điện thoại đồng thời của đội; chưa có kết quả mục4. Không suy luận rằng tổn thất nằm ở khâu thu chỉ vì chưa tìm thấy lỗi phần mềm.']
 lines+=['## 5. Annotation và tương quan','ASSOCIATIONS.csv dùng trung bình dự đoán/PR của từng người để tính Pearson và Spearman; MAE lấy |lỗi| theo video, trung bình trong người rồi trung bình người. Các bảng ghép cặp luôn có hằng số train-fold và trung vị hiệu chỉnh trên cùng video. CI bootstrap người10.000 lần, seed0. Không suy từ tương quan thấp thành “không có tín hiệu”.']
 if (C/'CURRENT_ANNOTATION_AUDIT.json').exists():
  a=read(C/'CURRENT_ANNOTATION_AUDIT.json');lines.append(f"Đối chiếu đúng cộtN (PR) và cộtA (ID) của218dòng dev trong bản bảng BTC tải lại: {len(a['inconsistencies'])} bất nhất. SHA bản export hiện tại khác SHA đóng băng; lưu riêng, không thay nhãn cũ và không tuyên bố đã kiểm lại byte của workbook gốc. Không mở dòng holdout. Thời điểm đo PR vẫn chưa biết.")
 if (C/'METADATA_COMPLETE.json').exists():
  a=read(C/'METADATA_COMPLETE.json');lines.append(f"Chuỗi frozen dev annotation → file hash → người → fold → prediction: {a['issues']} bất nhất trong218video. Chi tiết chỉ các bất nhất ở LINEAGE_INCONSISTENCIES.json; provenance tối thiểu ở LINEAGE.json.")
 if (C/'ROUTE_COMPLETE.json').exists():
  a=read(C/'ROUTE_COMPLETE.json');lines.append('Hai public positivecontrols và một BTC: '+', '.join(c['key']+': '+('không khác tại các boundary đã xuất' if c['first_recorded_divergence'] is None else 'khác đầu tại '+c['first_recorded_divergence']) for c in a['cases'])+'. Phạm vi từ decoded oval cache+PTS xuống tensor/mạng/readout; không đánh tráo với kiểm loader video đầy đủ. V5_TARGET_EVIDENCE.json dẫn đúng validation offset/sign/target V5 đã dùng, không dùng offset0 của hình minh họa.')
 lines+=['## 6. Bốn nhánh xoay/mask','Nhóm xoay được chốt bằng hình học landmark trước khi nhìn lỗi; readout power_0.7 là chính. Giữ original/rotate/mask/rotate_mask riêng, không chỉ đọc nhánh tốt nhất.']
 spatial=P/'report_spatial/BTC_METRICS.json'
 if spatial.exists():
  rows=[r for r in read(spatial) if r['condition']=='native' and r['readout']=='power_0.7' and r['method'] in ['padding','V3','calibrated_refit'] and r['group'] in ['all','rotated','upright','Abnormal']]
  matched=list(csv.DictReader((spatial.parent/'BTC_MATCHED_TABLES.csv').open()));shown=[]
  for r in rows:
   shown.append([r['input'],r['method'],r['group'],f"{r['people']}/{r['videos']}",fmt(r['MAE']),f"{r['delta_saved_calibrated']['MAE']:.3f} {r['delta_saved_calibrated']['CI95_MAE']}"])
   for b in matched:
    if (b['condition'],b['input'],b['candidate'],b['readout'],b['group'])==(r['condition'],r['input'],r['method'],r['readout'],r['group']) and b['row'] in ['constant_trainfold','calibrated_native_saved']:shown.append([r['input'],b['row']+' (cùng tập)',r['group'],f"{b['people']}/{b['videos']}",fmt(float(b['MAE'])),'đối chứng'])
  lines.append(table(['Nhánh','Mô hình','Nhóm','Người/video','MAE','Δ hiệu chỉnh [CI95]'],shown));lines.append('Không dùng11,37 làm giá trị chung cho nhóm con. Toàn bộ bảng và CI ở report_spatial/BTC_MATCHED_TABLES.csv.')
 else:lines.append('Chưa đủ A/B; không đưa số chưa hoàn tất vào kết luận.')
 lines+=['## 7. FREEZE','Chưa cập nhật FREEZE trong bước thu bằng chứng. Cần đọc thang đầy đủ, A/B, kiểm ảnh tĩnh và mọi bất nhất; phần đối chứng thu thực tế còn thiếu. Nếu không tìm được contributor, kết luận giới hạn ở các hợp đồng đã kiểm. Không ghi câu “tổn thất nằm ở khâu thu” như kết luận nhân quả khi chưa có bằng chứng mục4.']
 save(C/'CAUSAL_LEDGER.json',dict(utc=now(),hypotheses=ledger,review_required=True));(C/'REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
 save(C/'REPORT_STATUS.json',dict(utc=now(),ladder_complete=ladder_done,metadata_complete=(C/'METADATA_COMPLETE.json').exists(),batch_complete=(C/'BATCH_COMPLETE.json').exists(),spatial_report_complete=spatial.exists(),paired_capture_complete=False,freeze_changed=False))
 if (C/'export_causal.py').exists():
  from export_causal import export_if_ready
  export_if_ready()
if __name__=='__main__':
 try:render()
 except Exception as e:save(C/'REPORT_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
