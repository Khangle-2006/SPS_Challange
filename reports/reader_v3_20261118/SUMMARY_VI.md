# Reader v3 — cập nhật 20/11/2026

**Đã hoàn tất** reader_v3 trên C1/C4/C5, tích hợp readout public b04 và kiểm ảnh tĩnh qua checkpoint thật. Mọi việc chạy CPU; không huấn luyện hoặc mở holdout.

Báo cáo đầy đủ có bốn nhóm, tất cả đối chứng, bias, Spearman, CI, độ phủ và lệnh tái lập: [INTEGRATION_REPORT_20261120.md](INTEGRATION_REPORT_20261120.md).

## BTC dev

| Điều kiện | Video/người | Reader v3 MAE | Hằng số cùng video |
|---|---:|---:|---:|
| Native | 218/158 | 9,587 | 12,071 |
| C1 gain 0,5 | 218/158 | 12,390 | 12,071 |
| C1 gain 0,25 | 218/158 | 12,873 | 12,071 |
| C1 gain 0,125 | 218/158 | 12,415 | 12,071 |
| C4 QVGA | 218/158 | 9,927 | 12,071 |
| C4 VGA | 218/158 | 10,110 | 12,071 |
| C5 cũ | 218/158 | 12,488 | 12,071 |
| C5′ mặc định | 161/109 | 14,932 | 12,804 |
| C5′ bf0 | 158/106 | 12,298 | 13,072 |

MAE tính trong từng người trước rồi trung bình; đơn vị bpm. Giữ nguyên năm model native train-fold, không fit lại beta/offset theo điều kiện. Loại video lấy từ kích thước nguồn, không từ QVGA/VGA.

**C4 giữ được lợi thế so hằng số:** QVGA Δ −2,144 CI95 [−3,249; −1,073]; VGA Δ −1,960 [−3,057; −0,886]. Chênh lệch so native của cả hai còn chứa 0. QVGA có 23, VGA có 22 video abstain dùng prior + offset; không gọi toàn bộ 218 video là đo được mạch.

C1 và C5 cũ không hơn hằng số rõ ràng. C5′ mặc định tệ hơn hằng số: Δ +2,128 [0,252; 4,177]. C5′ bf0 có Δ −0,774 [−2,124; 0,547], chưa có bằng chứng hơn hằng số. Không so nhân quả giữa các điều kiện có độ phủ khác nhau. C1 còn gồm codec lưu CRF18; C5′ là proxy ffmpeg, chưa xác minh MATLAB VideoWriter.

## Public

Hợp nhất phổ + tracking b04, fit trên bốn fold người còn lại, xử lý offline:

| Điều kiện | UBFC MAE | MCD MAE |
|---|---:|---:|
| Native | 0,702 | 2,319 |
| C5 cũ | 7,490 | 5,820 |
| C5′ mặc định | 5,926 | 5,234 |
| C5′ bf0 | 1,105 | 2,334 |

35 người UBFC và 60 người MCD; số cửa sổ theo điều kiện ở báo cáo đầy đủ. Đây là kết quả trên phổ public đã lưu, không được so trực tiếp với bảng cũ khác mẫu số. MCD native cải thiện so padding trên cùng cửa sổ Δ −4,257 [−5,559; −3,018]; UBFC native Δ −0,326 [−0,776; +0,011], CI còn chứa 0. Không phải mọi ô đều cải thiện: UBFC C5′ mặc định Δ +0,271 [−0,853; +1,528].

## Kiểm chứng

- 160 dòng metric BTC được tính lại độc lập, gồm MAE, bias, Spearman và CI ghép cặp 10.000 lần seed 0.
- 8.522 dự đoán public, cả fusion và tracking, trùng hoàn toàn kết quả b04 nguồn. Module triển khai cũng trùng hàm b04 trên từng chuỗi.
- 25/25 ảnh tĩnh từ ROI dev thật, qua cả 5 checkpoint padding-fixed: abstain ở cả hai chế độ đọc. 10 ca CHROM không hữu hạn được chặn đúng; không biến NaN thành HR. Không phải kiểm bộ dò mặt từ video thô.
- 21 file kết quả tải local khớp SHA256 remote. Ba tiến trình hoàn tất exit 0; không còn worker của lần chạy này.

## Cấu hình và giới hạn

Tái lập native trước đó: v1 10,57015; v2 10,25048; v3 9,58706 trên 218 video/158 người, khớp nguồn <1e-9. V3 đạt cổng số dev nhưng là kết quả khám phá sau nhiều thử nghiệm; CI so v2 còn chứa 0. Không tự chọn cấu hình nộp và không mở holdout.

Bundle v1/v2/v3 ngày 18/11 giữ nguyên: `READER_CANDIDATES_FROZEN_20261118.zip`, SHA256 `3ca6f9db80903b2878874f48b6405266cf97ed9c95f7ad36bb8c62ce166cd080`. Model v3 gốc SHA256 `4570483ec06b44c9908fffeb82a19b9e05936fad305a5357d07359d54d1f55ae`. Chưa port v3 sang Octave/MATLAB; các kiểm Octave trước đó chỉ dành v1/v2.

Người dùng duyệt cap chung **120 GB**, project 500 GB. Trước chạy: 110,418 GB gồm dự phòng; project 354,942 GB. Thư mục remote mới dùng khoảng 2,52 MB sau tính toán. Không xóa dữ liệu, không dừng job khác. Holdout 72 video/41 người vẫn khóa, chỉ mở sau xác nhận riêng.
