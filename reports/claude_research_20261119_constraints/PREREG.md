# Đăng ký trước — đợt chạy 7 constraint và thử hạ MAE/RMSE (19/11/2026)

Viết lúc 19/11/2026 ~18:40 (giờ VN), **trước khi có bất kỳ số nào của đợt chạy này trên server**.
Người viết: Claude. Các ngưỡng dưới đây là **đề xuất của Claude** (mục 11 của `claude/bay-constraint-2026-11-19.md`), nhóm chưa điều chỉnh.
Số đã biết trước khi viết: kết quả mô phỏng mức vết của đêm 18→19/11 (BTC dev) và báo cáo C1/C4 của Codex (`reports/reader_w2b_20261116`).

## 1. Pipeline được chấm (đóng băng, không huấn luyện lại)
- Front end: `t8_oval_20261031/oval_frontend.py` (FaceMesh, oval 36 điểm) → crop 36×36, chạy lại **trên khung đã bị làm xấu**.
- Bộ công khai (UBFC2 dev 35 người, MCD 119 bản ghi/60 người): CNN padding-fixed (checkpoint ngoài fold) + POS + CHROM → hợp nhất log-phổ chuẩn hoá
  (trọng số fit trên 4 fold còn lại) → gộp theo bản ghi (Viterbi; (σ,τ) fit trên 4 fold) — đúng `claude_research_20261118/code/b04_public_final.py`.
- BTC dev (218 video/158 người): reader_v3 (`a11_reader_v3.py`). Holdout BTC và 7 người UBFC test **không đụng tới**.
- Ước lượng chính: **fit lại theo điều kiện** (tham số readout fit trên 4 fold của chính điều kiện đó). Báo thêm: **đóng băng** (tham số fit trên native, áp vào điều kiện).
- Thống kê: MAE theo người; bootstrap ghép cặp theo người 10.000 lần, seed 0. Công khai: cửa sổ 10 s của `inference_repair_20261113/MANIFEST.json`; BTC: một dự đoán/video.
- "Hằng số không dùng video": trung vị nhãn của 4 fold huấn luyện (công khai: trung vị HR cửa sổ theo từng bộ; BTC: trung vị hậu nghiệm với β=0).

## 2. Điều kiện (tham số cố định trước)
| Mã | Phép làm xấu (ngữ nghĩa MATLAB trong tài liệu) |
|---|---|
| native | không đổi (kiểm tra tái lập so với cache cũ) |
| c1_g50p, **c1_g25p**, c1_g12p | uint8(gain·I), gain 0,5 / 0,25 / 0,125; rồi `imnoise(·,'poisson')` |
| c1_g25gv001, c1_g25gv01 | gain 0,25 rồi `imnoise(·,'gaussian',0,v)`, v = 0,001 / 0,01 |
| **c4_vga**, c4_qvga | `imresize` bicubic có khử răng cưa, vừa trong 640×480 / 320×240, giữ tỉ lệ (UBFC2/MCD gốc đã là 640×480 → chỉ có c4_qvga) |
| c6_us1 | `imfilter` nhân unsharp (alpha 0,2), 1 lượt, cả khung |
| **c7_blue**, c7_white, c7_black_high | đa giác đặc vùng mặt dưới theo landmark của khung sạch; 'standard' (xanh y tế, trắng), 'high' (đen) |
| c5_lit | 10 fps, ≤500 kbps, mã hoá mặc định (bản có sẵn của nhóm: BTC `c5`, công khai `c5prime_default`) |
| **c5_tuned** | 10 fps, x264 bf=0, không keyframe định kỳ, maxrate ≤500 kbps (công khai: `c5prime_bf0` có sẵn) |
| c5_vga | thu về ≤640×480 rồi mã hoá như c5_tuned |
| c5_face | "chế độ thiết bị": cắt vùng mặt (hộp mặt nới 30%) về ≤192×240 rồi mã hoá ≤500 kbps |
| c3_side, c3_big | ghép người thứ hai vào khung (cạnh nhau cùng cỡ; người ngoài to hơn 1,3×) — UBFC2 |
In đậm = điều kiện **chính** của constraint đó; các mức khác là đường liều–đáp ứng/stress.
Nhiễu C1: khung 640×480 → cả khung; khung HD (BTC) → gain cả khung, nhiễu chỉ trong hộp mặt (cache native) nới 50% mỗi phía (giới hạn tính toán; ghi rõ khi báo cáo).

## 3. Cổng đạt/không đạt
**Cổng suy giảm (C1, C4, C5, C7)** — xét trên điều kiện chính, ước lượng chính:
 (a) độ phủ: ≥ 90 % số cửa sổ (công khai) / video (BTC) có đầu ra ở native vẫn có đầu ra;
 (b) MAE không tệ hơn hằng số không dùng video: KTC95 % ghép cặp của (MAE điều kiện − MAE hằng số) **không nằm hoàn toàn trên 0**;
 (c) báo thêm (mức mạnh): tốt hơn hằng số với KTC không chứa 0. Luôn báo kèm chênh lệch ghép cặp so với native.
 C7: nhánh "biết che" (phát hiện hàng bị che trên lưới 6×6, bỏ các hàng đó) là ước lượng chính; nhánh không đổi báo làm đối chiếu; phải báo tỉ lệ khung/ video bị khai báo che sai trên mặt không che.
**Cổng nhóm (C2, C6)**: mọi nhóm có ≥ 20 người có MAE trong khoảng ±2 bpm so với MAE chung, và phép hoán vị nhãn nhóm giữa người (10.000 lần, thống kê = max−min MAE nhóm) không bác bỏ ở 5 %.
 C6: nhóm ≥ 60 tuổi báo riêng; nếu < 20 người thì chỉ báo số + KTC, không kết luận đạt/không đạt.
**Cổng C3**: 0 khung hợp lệ đọc từ người ngoài trên mọi video ghép; độ phủ ≥ 90 % cửa sổ một-mặt; KTC ghép cặp của (MAE ghép − MAE một mặt) chứa 0.
Một constraint được ghi "đạt" chỉ khi đạt trên **mọi bộ dữ liệu có chạy điều kiện đó**; nếu khác nhau giữa các bộ thì ghi từng bộ.

## 4. Thử hạ MAE/RMSE (danh sách cố định trước, chọn theo quy tắc)
Ứng viên: (1) điểm ước lượng = trung bình hậu nghiệm / pha trộn trung vị–trung bình (mục tiêu RMSE); (2) bằng chứng lược hài (cộng log-phổ tại 2f);
(3) thêm chuyên gia cổ điển vào hợp nhất (GREEN, PBV, LGI, OMIT) trên vết có sẵn; (4) nội suy đỉnh dưới bước lưới; (5) trọng số cửa sổ theo chất lượng (SNR phổ);
(6) gộp bản ghi bằng tích hậu nghiệm thay cho Viterbi. Mọi tham số mới fit trên 4 fold còn lại.
Quy tắc nhận: chỉ nhận thay đổi nếu KTC95 % ghép cặp (theo người) của chênh lệch so với bản hiện tại **không chứa 0** ở thước đo mục tiêu và thước đo còn lại không tệ đi có ý nghĩa.
Không đạt → ghi "đã thử, không nhận". Đối chứng bắt buộc cho thứ được nhận: xáo nhãn giữa người. Không mở holdout.
