# Kiểm CLI video thật — R1, 23/11/2026

## Code và model hiện hành

Ba expert chạy **self-warm**: reset về bản sao trạng thái ngay sau khởi tạo,
chạy cùng video hai lượt, lấy lượt thứ hai. Không mang trạng thái giữa video.
Reader deploy được fit lại một lần trên 218 video dev; 5 model fold khớp đủ
218 dự đoán OOF mới với sai khác 0. Model deploy không được dùng để chấm dev.

BTC native đạt **8,33327 bpm**, CI95 [7,08958; 9,65911], trên 218 video/158 người.
So reader_v3: Δ −1,25379 [−2,28246; −0,28346]. Đây là kết quả dev sau nhiều
thử nghiệm. Việc chọn 3/17 expert trên phổ lịch sử không được chạy lại.
[Bảng đầy đủ](RESULTS.md).

## Các kiểm đã chạy

| Kiểm | Kết quả |
|---|---|
| Reset so tham chiếu, 2 video × 3 expert | Sai khác phổ 0,0 |
| Self-warm so tham chiếu, 2 video × 3 expert | Sai khác phổ 0,0 |
| A → B → A, cả 3 expert | Sai khác phổ A bằng 0,0 |
| 63 lượt suy luận BTC/public | Đủ; số đầu ra và mask cửa sổ khớp bản cũ |
| 3 video dev gốc, CLI reference với fold đúng người | 3/3 PASS, sai khác PR so OOF **0 bpm** |
| Ảnh mặt lặp | `abstain=true`, `pr_bpm=null` |
| Video không mặt | `abstain=true`, không báo prior là số đo |
| Video quá ngắn | Exit 1; không đệm thành kết quả giả |
| Unit test runtime/tools | 16/16 PASS |
| Unit test scorer | 10/10 PASS |

Ba lượt raw mất lần lượt **206,79 / 184,77 / 173,26 giây**; cả bộ raw và kiểm âm
mất 589,13 giây trên máy thử. Đây không phải SLA cho mọi video hoặc máy.
[Receipt tổng hợp, không có ID/nhãn/dự đoán cá nhân](../reports/rerun_stateless_20261123/CLI_VALIDATION_AGGREGATE.json).

Lượt R1 dùng lại ba môi trường core/frontend/experts cô lập đã cài từ lock trên
clone NC trong lượt kiểm trước; **không tuyên bố đã cài lại môi trường mới cho R1**.
Source/config/model được kiểm bằng `tools/verify_source.py` trước khi push.
Lệnh và dependency: [RUN_VIDEO.md](RUN_VIDEO.md).

## Giới hạn được giữ nguyên

CLI mặc định là `reference` cho video một người. Frontend `guarded` từng thay
đổi một khung và làm lệch PR; cổng tương đương trước đó chưa đạt. R1 chỉ sửa
trạng thái expert, không thay detector, landmark hoặc nâng `guarded` lên mặc định.
Kiểm ảnh tĩnh ở đây kiểm toàn đường CLI có lớp từ chối đầu vào; không suy ra
mọi CNN riêng lẻ đều qua kiểm âm nếu bỏ lớp đó.

Holdout BTC và 7 người test UBFC2 không được mở. Không huấn luyện lại.
[Bản kiểm reset 9,130 và lịch sử lỗi được giữ ở commit cũ](history/R1_STATE_BUG_20261123.md).
