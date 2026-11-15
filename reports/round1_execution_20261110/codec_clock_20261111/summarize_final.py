"""Readable tables from completed, verified audit. No fitting or selection."""
from report_clock import *

def csvrows(name):return list(csv.DictReader((P/name).open()))
def yes(x):return str(x).lower()=='true'
def main():
 assert read(P/'FINAL_VALIDATION.json')['status']=='PASS'
 assert read(P/'REPORT_STATUS.json')['state']=='COMPLETE_WITH_EXCLUSION'
 real,enc=loadrows('real');synth,senc=loadrows('synthetic');moves=csvrows('PAIRED_FPS_MOVEMENT.csv');ba=csvrows('BEFORE_AFTER.csv');roi=csvrows('ROI_COMPARISON.csv');cov=csvrows('COVERAGE.csv');btc=read(P/'BTC_METRICS.json')
 lines=['# Kết quả kiểm chứng codec và đường đọc nhịp', '', f'Hoàn tất {now()}. **Đã xử lý đủ 12 video được khóa trước: 11 video mã hóa được (66 bản mã hóa), 1 video không đủ điều kiện do khoảng mất timestamp; 120 bản tổng hợp; 186/186 SHA-256 kiểm lại khớp.** Với mỗi video mã hóa được, xét 3 cửa sổ 10 giây; cửa sổ thiếu hỗ trợ vẫn được ghi lỗi trong bảng độ phủ. Video MCD 1474_FullHDwebcam_after có khoảng mất timestamp 0,337102 giây, khiến lấy mẫu 8/10 fps lặp cùng frame; đã loại có lý do, không vá thời gian hoặc thay bằng video khác. Đây là kiểm tra cơ chế trên mẫu nhỏ, không phải đánh giá lại toàn bộ BTC.', '',
 '## Đọc các con số như thế nào?', '',
 '- **fps**: số khung hình mỗi giây. Cùng 30 giây nội dung được lấy mẫu ở 8 hoặc 10 fps; không tăng hay giảm tốc độ phát.',
 '- **B**: giới hạn số B-frame liên tiếp khi mã hóa. B-frame là một loại khung dự đoán của H.264. Lịch I/P/B thực tế được đọc lại từ file, không suy ra chỉ từ tham số lệnh.',
 '- **Chu kỳ codec**: tần số lặp của lịch khung. Ví dụ, nếu mỗi 4 khung có một khung không-B, ở 10 fps chu kỳ là 10/4 = 2,5 lần/giây, tương ứng 150 lần/phút. Đây không phải nhịp tim.',
 '- **Đỉnh phổ**: dao động mạnh nhất trong dải 0,7–3 Hz. Một đỉnh gần 150 bpm có thể là mạch thật hoặc nhiễu; riêng con số không phân biệt được.',
 '- Các số lẻ của đỉnh phổ đến từ lưới tính FFT; chúng không có nghĩa phép đo sinh lý chính xác tới0,01bpm. 1Hz tương ứng60 lần/phút.',
 '- **G / POS / CHROM**: tín hiệu kênh xanh lá / hai phép kết hợp các kênh màu nhằm lấy tín hiệu mạch. Bảng cơ chế dùng đỉnh phổ trước bộ lọc HR; bảng MAE dùng decoder HR đã đóng băng. Hai đại lượng không hoàn toàn giống nhau.',
 '- **Gần chu kỳ**: đỉnh phổ cách tần số codec quan sát không quá 2 bpm. Các cửa sổ cùng một người không được coi là người độc lập.', '',
 '## 1. Đường đọc đầu ra mô hình', '',
 '**Không tìm thấy lỗi thiếu tích phân trong các đường đánh giá đã kiểm.** Đầu ra sai phân đã được cumsum trước detrend, lọc và tìm đỉnh. Đối chứng 72 bpm đọc lại 72,07 bpm. Tín hiệu bằng 0 và hằng số bị từ chối.', '',
 'Có một điểm cần phân biệt: dao động cực nhỏ vẫn có thể được trả một HR nếu vượt ngưỡng biên độ 1e-10. Đầu ra ảnh tĩnh SiNC có SD khoảng 2,425e-9 nên vẫn trả 61,52 bpm. Nó trượt cổng ảnh tĩnh đã đặt; nhưng con số HR này tự nó không chứng minh có dao động giả biên độ lớn. Không sửa ngưỡng sau khi đã thấy kết quả.', '',
 '## 2. Video thật: đỉnh có bám chu kỳ codec không?', '',
 'Bảng dưới giữ nguyên video, ROI và ba khoảng thời gian. “Trước” là cùng các khung nguồn đã lấy mẫu, chưa qua lần encode thử nghiệm. Nguồn ban đầu vốn đã có nén. Bảng đếm đỉnh gần chu kỳ **của cấu hình đang thử** ở trước và sau encode.', '',
 '| Dataset | fps | B | Phép chiếu | Trước encode | Sau encode |', '|---|---:|---:|---|---:|---:|']
 for ds in ['UBFC2','MCD','BTC']:
  for fs in [8,10]:
   for bf in [3,4]:
    for method in ['POS','CHROM']:
     rr=[r for r in ba if r['dataset']==ds and int(r['fs'])==fs and int(r['bf'])==bf and r['method']==method]
     lines.append(f"| {ds} | {fs} | {bf} | {method} | {pct(sum(yes(r['before_near']) for r in rr),len(rr))} | {pct(sum(yes(r['after_near']) for r in rr),len(rr))} |")
 lines+=['', '### Thay fps: cùng cửa sổ có dịch đỉnh theo cả hai chu kỳ không?', '',
 'Ví dụ ở B=3: cần đỉnh gần 120 bpm tại 8 fps **và** gần 150 bpm tại 10 fps trên đúng cửa sổ đó. Đếm ghép cặp này mạnh hơn việc nhìn hai đỉnh trung vị ở hai nhóm khác nhau.', '', '| Dataset | B | Phép chiếu | Bám cả hai chu kỳ 8→10 fps |','|---|---:|---|---:|']
 for ds in ['UBFC2','MCD','BTC']:
  for bf in [3,4]:
   for method in ['POS','CHROM']:
    rr=[r for r in moves if r['dataset']==ds and int(r['bf'])==bf and r['method']==method and yes(r['eligible'])]
    lines.append(f"| {ds} | {bf} | {method} | {pct(sum(yes(r['follows_both']) for r in rr),len(rr))} |")
 lines+=['', 'Đổi B cũng đổi cách encoder phân bổ sai số và bitrate. Thí nghiệm này kiểm tác động của cấu hình mã hóa trong điều kiện đã giữ cố định, chưa tách riêng mọi cơ chế bên trong encoder. Không suy rộng thành mọi video BTC native đều có nhiễu này.', '',
 '## 3. Kiểm tra ảnh hưởng của ROI và nội suy', '',
 'Cùng một đa giác mặt được dùng cho mọi bản mã hóa. So sánh trung bình pixel bên trong oval ở ảnh đầy đủ với trung bình ảnh oval36 có nền mặt nạ. Sai khác ở đây cho biết hai cách lấy trace không thể được tự động coi là tương đương.', '', '| Dataset | Phép chiếu | Số cặp | Trung vị chênh đỉnh tuyệt đối (bpm) |','|---|---|---:|---:|']
 for ds in ['UBFC2','MCD','BTC']:
  for method in ['G','POS','CHROM']:
   rr=[float(r['abs_difference']) for r in roi if r['dataset']==ds and r['method']==method]
   lines.append(f"| {ds} | {method} | {len(rr)} | {np.median(rr):.2f} |")
 lines+=['', '### Đỉnh ở B=3 có tồn tại trước nội suy lên 30 Hz không?', '', '| Dataset | fps | Phép chiếu | Gần chu kỳ trước nội suy | Gần chu kỳ sau nội suy theo clock nguồn |','|---|---:|---|---:|---:|']
 for ds in ['UBFC2','MCD','BTC']:
  for fs in [8,10]:
   for method in ['G','POS']:
    values=[]
    for mode in ['native_rate','source_clock']:
     rr=[r for r in real if r['dataset']==ds and r['fs']==fs and r['bf']==3 and r['roi']=='oval36' and r['clock']==mode and r['method']==method and r.get('peak_bpm') is not None and r.get('cycle_bpm')]
     values.append(pct(sum(abs(r['peak_bpm']-r['cycle_bpm'])<=2 for r in rr),len(rr)))
    lines.append(f"| {ds} | {fs} | {method} | {' | '.join(values)} |")
 lines+=['', 'Phổ G/POS trước nội suy được giữ trong ALL_REAL_SPECTRA.csv. CHROM nguyên bản không chạy trực tiếp ở 8/10 Hz: cửa sổ của nó ngắn hơn yêu cầu padlen của filtfilt. Lần đầu dừng trước encode vì lỗi này; lần V2 ghi rõ nhánh đó không hỗ trợ và giữ nguyên nhánh 30 Hz dùng trong đánh giá trước đây. Không thay CHROM để làm kết quả đẹp hơn.', '',
 '## 4. Đối chứng tổng hợp: có nhiễu và có mất mạch không?', '',
 'Hai cảnh sáng/tối, hai seed nhiễu; mỗi cấu hình có 12 cửa sổ. Không có mạch được cài vào ở bảng sau. Thuật toán vẫn có thể trả một đỉnh vì đây là tìm cực đại phổ, không phải bộ phát hiện có/không có mạch.', '', '| fps | B | Chu kỳ bpm | G gần chu kỳ | POS gần chu kỳ | CHROM gần chu kỳ |','|---:|---:|---:|---:|---:|---:|']
 for fs in [8,10]:
  for bf in [3,4]:
   vals=[];cycle=None
   for method in ['G','POS','CHROM']:
    rr=[r for r in synth if r['pulse_bpm']==0 and r['fs']==fs and r['bf']==bf and r['clock']=='source_clock' and r['method']==method and r.get('cycle_bpm')]
    if rr:cycle=rr[0]['cycle_bpm']
    vals.append(pct(sum(abs(r['peak_bpm']-r['cycle_bpm'])<=2 for r in rr),len(rr)))
   lines.append(f"| {fs} | {bf} | {nfmt(cycle)} | {' | '.join(vals)} |")
 lines+=['', 'Với mạch được cài ở 72, 120, 150 và 180 bpm, bảng tiếp theo báo sai số đỉnh trung bình trên cả sáu cấu hình encode. Trường hợp xấu nhất là cấu hình có sai số trung bình lớn nhất, không phải cửa sổ xấu nhất.', '', '| Mạch cài bpm | Phép chiếu | n cửa sổ | MAE đỉnh toàn bộ | MAE cấu hình xấu nhất |','|---:|---|---:|---:|---:|']
 for pulse in [72,120,150,180]:
  for method in ['G','POS','CHROM']:
   rr=[r for r in synth if r['pulse_bpm']==pulse and r['bf'] is not None and r['clock']=='source_clock' and r['method']==method]
   groups=defaultdict(list)
   for r in rr:groups[(r['fs'],r['bf'])].append(abs(r['peak_bpm']-pulse))
   lines.append(f"| {pulse} | {method} | {len(rr)} | {np.mean([abs(r['peak_bpm']-pulse) for r in rr]):.2f} | {max(np.mean(v) for v in groups.values()):.2f} |")
 lines+=['', 'Tín hiệu dương tính này khá mạnh và đơn giản; không đại diện mọi màu da hay camera. Khi mạch cài trùng chu kỳ codec, dự đoán đúng chưa chứng minh thuật toán đang theo mạch. Chưa thử bộ khử nhiễu, nên chưa có kết luận “sửa artifact mà vẫn giữ mạch thật”.', '',
 '## 5. BTC: số trên đúng bốn video, chưa dùng để chọn phương pháp', '',
 'Mỗi nhóm Normal-Resting / Abnormal-Resting / Task / Walking có đúng một người. Lấy trung vị ba dự đoán cửa sổ cho mỗi video, rồi tính sai số theo người. Hằng số lấy trung vị PR của các fold train; hiệu chỉnh là dự đoán OOF native đã lưu trên chính video đó. Không fit lại trên bốn người này. CI bootstrap ghép cặp 10.000 lần, seed 0; với n=4 khoảng tin cậy rất hạn chế.', '',
 '| fps | B | Phương pháp | n người | MAE bpm | CI95% | Δ so hiệu chỉnh [CI95%] |', '|---:|---|---|---:|---:|---|---|']
 for x in btc:
  if x['group']!='all':continue
  lines.append(f"| {x['fs']} | {'Trước encode' if x['bf'] is None else x['bf']} | {x['method']} | {x['people']} | {x['MAE']:.2f} | {x['CI95'][0]:.2f}–{x['CI95'][1]:.2f} | {x['delta_calibrated']:+.2f} [{x['delta_calibrated_CI'][0]:+.2f}, {x['delta_calibrated_CI'][1]:+.2f}] |")
 valid=sum(int(r['valid']) for r in cov);expected=sum(int(r['expected_windows']) for r in cov)
 lines+=['', f'Coverage chính: {valid}/{expected} phép đo cửa sổ hợp lệ (đếm cả ba phương pháp và các cấu hình; không phải số cửa sổ độc lập). Chi tiết lỗi ở COVERAGE.csv. Bảng từng nhóm, P95, bias và CI nằm trong REPORT.md / BTC_METRICS.json.', '',
 '## 6. Phạm vi kết luận', '',
 '- Kiểm định cơ chế hoàn tất; không có lượt GPU mới và không mở holdout.',
 '- Một chu kỳ phổ bám cấu hình encoder là bằng chứng của thành phần do xử lý video trong thí nghiệm đó. Nó không đủ để quy mọi lỗi BTC cho codec hoặc nhãn.',
 '- Không có bằng chứng mới về MAE toàn bộ BTC native thấp hơn mốc hiệu chỉnh 11,37. Không thay ứng viên và không khởi động lại sweep đã dừng.',
 '- Chưa có MATLAB parity; nguồn đã nén, lịch B cố định và mẫu 4 UBFC / 3 MCD hợp lệ / 4 BTC đều hạn chế khả năng suy rộng.',
 '- Kết quả chi tiết, lệnh và hash được giữ để đưa vào báo cáo như một thí nghiệm kiểm soát. Chỉ cân nhắc sửa thuật toán khi có phương án không dùng nhãn để nhận diện artifact và vượt cả đối chứng giữ mạch thật.', '',
 '## File để đọc và tái lập', '',
 '- REPORT.md: toàn bộ bảng, các trường hợp thất bại và BTC theo nhóm.',
 '- PLAN.json / PROTOCOL.md: mẫu và cách làm được khóa trước kết quả.',
 '- FINAL_VALIDATION.json / VERIFIED_ARTIFACTS.csv: số lượng, hash và guard dung lượng.',
 '- PAIRED_FPS_MOVEMENT.csv / PAIRED_CLOCK_MOVEMENT.csv / BEFORE_AFTER.csv: ghép cặp thay fps, thay B và trước/sau encode.',
 '- ALL_REAL_SPECTRA.csv / ALL_SYNTHETIC_SPECTRA.csv: đỉnh phổ, biên độ, tỷ lệ năng lượng và trạng thái từng phép đo.',
 '- WINDOW_PREDICTIONS.csv / BTC_BASELINES_MATCHED.csv: dự đoán và nhãn tham chiếu cùng baseline.']
 (P/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print('SUMMARY_WRITTEN',sha(P/'SUMMARY.md'))

if __name__=='__main__':main()
