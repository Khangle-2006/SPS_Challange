"""Render measured results only; no inference, fitting, or model selection."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def n(x):return '—' if x is None else f'{x:.3f}'
def ci(x):return f'[{x[0]:+.3f}; {x[1]:+.3f}]'
def main():
 assert read(P/'REVIEW_20261120.json')['passed']
 rows=read(P/'conditions_20261120/METRICS.json');pub=read(P/'public_20261120/METRICS.json')
 labels={'c1_050':'C1 gain 0,5','c1_025':'C1 gain 0,25','c1_0125':'C1 gain 0,125','c4_QVGA':'C4 QVGA','c4_VGA':'C4 VGA','c5':'C5 cũ','c5prime_default':'C5′ mặc định','c5prime_bf0':'C5′ tắt B-frame'}
 methods={'reader_v3':'Reader v3 điều kiện','constant':'Hằng số train-fold','calibrated':'Hiệu chỉnh native lịch sử','reader_native':'Reader v3 native (cùng video)'}
 lines=['# Reader v3 và readout public — kết quả tích hợp 20/11/2026','',
 'Đã hoàn tất đánh giá CPU. Không huấn luyện, không điều chỉnh tham số reader, không mở holdout. Dữ liệu C1/C4 đã xử lý xong trước đó được tái sử dụng; không mã hóa lại. Các số dưới đây được tính lại độc lập từ dự đoán đã lưu.',
 '', '## Cách đọc', '',
 '- **MAE (bpm):** lấy sai số tuyệt đối trung bình trong từng người, rồi trung bình giữa người. Càng thấp càng tốt. Một người có nhiều video không được tính nặng hơn người khác.',
 '- **Bias:** dự đoán trừ nhãn, cũng tính trong từng người trước. Âm nghĩa là dự đoán thấp; dương là dự đoán cao.',
 '- **Spearman:** tương quan thứ hạng giữa dự đoán trung bình và nhãn trung bình của mỗi người; gần 1 là cùng tăng giảm, gần 0 là ít liên hệ.',
 '- **Δ và CI95:** MAE phương pháp trừ MAE đối chứng trên đúng cùng video/người. Số âm là tốt hơn. CI được bootstrap ghép cặp theo người 10.000 lần, seed 0; chứa 0 nghĩa là chưa có bằng chứng rõ về chênh lệch.',
 '', '## BTC dev — tổng thể theo điều kiện', '',
 'Năm model native train-fold đã đóng băng được giữ nguyên: prior, beta theo loại video và offset theo loại/hoạt động. Loại low/high lấy từ kích thước **video nguồn** (<1,8 triệu pixel), không lấy từ QVGA/VGA hay video đã nén. Không fit lại theo điều kiện. Đây là phép thử chuyển nguyên cấu hình native sang dữ liệu biến đổi.', '',
 'Native reader v3 trên đủ 218 video/158 người: **9,58706 bpm**; hằng số 12,07068; hiệu chỉnh cũ 11,37385. Những hàng có độ phủ khác chỉ được so với native/hằng số trên giao video tương ứng.', '',
 '| Điều kiện | Video/người | Reader v3 | Hằng số | Native v3 cùng video | Δ so hằng số [CI95] | Δ so native v3 [CI95] |',
 '|---|---:|---:|---:|---:|---|---|']
 for cond,label in labels.items():
  rr={r['method']:r for r in rows if r['condition']==cond and r['group']=='all'};r=rr['reader_v3']
  lines.append(f"| {label} | {r['videos']}/{r['people']} | {n(r['MAE'])} | {n(rr['constant']['MAE'])} | {n(rr['reader_native']['MAE'])} | {r['delta_constant']:+.3f} {ci(r['delta_constant_CI'])} | {r['delta_native']:+.3f} {ci(r['delta_native_CI'])} |")
 pred=read(P/'conditions_20261120/PREDICTIONS.json')
 lines+=['', '### Độ phủ và trường hợp lùi về prior', '',
 'Bảng MAE tính cả video được gắn cờ abstain: khi không có phổ hữu hạn/hợp lệ, reader dùng **prior + offset đã khóa**. Vì vậy “218 video có dự đoán” không đồng nghĩa 218 video đo được mạch.', '',
 '| Điều kiện | Có dự đoán / 218 | Có phổ dùng được | Abstain, dùng prior + offset |', '|---|---:|---:|---:|']
 for cond,label in labels.items():
  rr=[r for r in pred if r['condition']==cond and r['method']=='reader_v3'];ab=sum(r['abstain'] for r in rr)
  lines.append(f'| {label} | {len(rr)} / 218 | {len(rr)-ab} | {ab} |')
 lines+=['','C5′ mặc định loại 57 video; bf0 loại 60 video vì receipt không đạt giới hạn bitrate. Danh sách chi tiết trong EXCLUSIONS.json. Không thay ngưỡng hoặc dùng lại video không đạt.', '',
 'C1 gồm cả codec lưu CRF18 của simulate.py đã khóa, nên không tách được tác động của nhiễu khỏi codec bằng bảng này. C4 dùng FFV1. C5′ là mô phỏng ffmpeg, **chưa xác minh tương đương MATLAB VideoWriter**. C5 và C5′ có độ phủ khác nhau; không dùng chênh lệch giữa các tập khác nhau để quy nguyên nhân.', '',
 '“Hiệu chỉnh native lịch sử” dưới đây là dự đoán đã lưu của video gốc, không phải hiệu chỉnh được fit lại trên dữ liệu nén. Đối chứng hằng số luôn có mặt trên đúng cùng tập video. Reader v3 trong lần chấm này không áp lại phép chọn beta riêng cho mỗi điều kiện của reader v1.', '', '## BTC dev — bốn nhóm và tất cả đối chứng']
 for cond,label in labels.items():
  lines+=['',f'### {label}','', '| Nhóm | Phương pháp | Video/người | MAE [CI95] | Bias | Spearman | Δ hằng số [CI95] | Δ hiệu chỉnh [CI95] | Δ native v3 [CI95] |','|---|---|---:|---|---:|---:|---|---|---|']
  for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
   for method in ['constant','calibrated','reader_native','reader_v3']:
    r=next(r for r in rows if r['condition']==cond and r['group']==group and r['method']==method)
    lines.append(f"| {group} | {methods[method]} | {r['videos']}/{r['people']} | {n(r['MAE'])} {ci(r['CI95'])} | {n(r['bias'])} | {n(r['Spearman'])} | {r['delta_constant']:+.3f} {ci(r['delta_constant_CI'])} | {r['delta_calibrated']:+.3f} {ci(r['delta_calibrated_CI'])} | {r['delta_native']:+.3f} {ci(r['delta_native_CI'])} |")
 lines+=['','## Public — hợp nhất phổ và tracking b04','',
 'Tích hợp đúng mã b04 đã cung cấp. Trong mỗi fold, trọng số phổ và tham số tracking chỉ được fit trên bốn fold người còn lại. Đầu ra module triển khai được kiểm trùng hoàn toàn hàm b04 trên từng chuỗi cửa sổ. Không huấn luyện lại checkpoint. Đây là đánh giá phát triển có fit readout, không phải zero-shot thuần.', '',
 'Tracking xem cả chuỗi cửa sổ của bản ghi, vì vậy là **xử lý offline**, không phải hệ thống đo trực tiếp thời gian thực. Mỗi bảng dùng đúng giao cửa sổ public có phổ đã lưu; số người/cửa sổ được ghi để tránh nhầm với những bảng public trước có độ phủ khác.', '',
 '| Điều kiện | Bộ | Phương pháp | Người/cửa sổ | MAE [CI95] | Δ so padding [CI95] |', '|---|---|---|---:|---|---|']
 for r in pub:
  lines.append(f"| {r['condition']} | {r['dataset']} | {r['method']} | {r['people']}/{r['windows']} | {n(r['MAE'])} {ci(r['CI'])} | {r['delta_padding']:+.3f} {ci(r['delta_CI'])} |")
 lines+=['','## Kiểm soát ảnh tĩnh','',
 '**25/25 ROI thật của BTC dev, 5 checkpoint padding-fixed: PASS**, cả hợp nhất không tracking và có tracking. Lựa chọn 5 video mỗi fold theo ID và hỗ trợ thời gian, không theo lỗi/nhãn. Lặp một frame 301 lần qua checkpoint CPU, sau đó đưa sai phân đầu ra và POS/CHROM của RGB tĩnh vào readout mới. Cả 25 đầu ra CNN có độ lệch chuẩn bằng 0; readout trả `abstain`, không đưa ra HR.', '',
 '10/25 trace CHROM không hữu hạn vì chuẩn hóa tín hiệu tĩnh dẫn tới chia 0; nhánh kiểm không hữu hạn cũng trả abstain. Các ca còn lại bị nhận diện là DC/tĩnh. Không che giấu hoặc thay số NaN thành một HR. Đây là kiểm **crop thật → checkpoint → readout**; chưa chứng minh toàn bộ bộ dò mặt/video thô ở chế độ end-to-end. Kiểm trace tổng hợp và 60 ca phổ không suy biến trước đây vẫn được giữ.', '',
 '## Trạng thái và giới hạn','',
 'Reader v3 native đã đạt các ngưỡng số của dev (≤10 bpm, hơn bản hiệu chỉnh ≥1 bpm, CI chênh lệch loại 0), nhưng được phát triển sau nhiều thử nghiệm nên vẫn là kết quả khám phá. So v2, CI vẫn chứa 0. Các bảng mới không thay đổi cấu hình đã đóng băng hoặc tự động chọn cấu hình nộp. Holdout 72 video/41 người vẫn khóa; chỉ mở sau xác nhận riêng của người dùng.', '',
 'Cap chung đã được người dùng duyệt 120 GB; project 500 GB. Kiểm kê trước khởi chạy: 110,418 GB gồm dự phòng; project 354,942 GB. Không xóa dữ liệu hay dừng job khác.', '',
 '## Hiện vật và tái lập','',
 '- `conditions_20261120/`: PREDICTIONS.json, METRICS.json, SOURCE_CLASS_BINDING.json, EXCLUSIONS.json, COMPLETE.json.',
 '- `public_20261120/`: PREDICTIONS.npz, FITS.json, METRICS.json, COMPLETE.json.',
 '- `static_20261120/`: RESULTS.json ghi hash checkpoint/cache, COMPLETE.json.',
 '- `REVIEW_20261120.json`: đối chiếu độc lập MAE, bias, Spearman, CI và phân loại video nguồn.',
 '- `DOWNLOAD_HASHES_20261120.json`: hash hiện vật remote so với bản tải local.', '',
 'Lệnh chính (thay `R` bằng `/mnt/samsung_t5/khangle/facevitals/reports`, dùng thư mục output mới; không ghi đè kết quả cũ):', '',
 '```bash', 'P="$R/reader_v3_20261118"', 'PY=/data_pool/khangle/facevitals/envs/hr/bin/python',
 'CUDA_VISIBLE_DEVICES="" "$PY" "$P/evaluate_conditions.py" --reports-root "$R" --numeric-root "$R/w2_btc_20261116/numeric_dev" --source-metadata "$R/claude_research_20261118/out/dev_video_meta.csv" --out NEW_CONDITIONS',
 'CUDA_VISIBLE_DEVICES="" "$PY" "$P/evaluate_public.py" --spectra "$R/claude_research_20261118/out/public_spectra.npz" --out NEW_PUBLIC',
 'CUDA_VISIBLE_DEVICES="" "$PY" "$P/static_checkpoint_readout.py" --out NEW_STATIC', '```','']
 (P/'INTEGRATION_REPORT_20261120.md').write_text('\n'.join(lines),encoding='utf-8')
 print(P/'INTEGRATION_REPORT_20261120.md')
if __name__=='__main__':main()
