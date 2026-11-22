# Kiểm CLI video thật — 23/11/2026

## Kết quả sau sửa trạng thái expert

Đủ 218 video dev / 158 người, cùng 5 fold theo người. Ba CNN huấn luyện sẵn
không đổi trọng số. Chạy lại suy luận, đặt lại trạng thái ban đầu trước từng video;
fit lại reader bằng đúng hàm, grid beta và offset cũ, không tìm tham số mới.
Đây là kiểm OOF trên crop native lịch sử, không phải toàn bộ frontend guarded.

| Nhóm | Video / người | Reader reset | Hằng số | Hiệu chỉnh cũ |
|---|---:|---:|---:|---:|
| all | 218 / 158 | 9.130 | 12.071 | 11.374 |
| Normal-Resting | 62 / 62 | 8.224 | 8.661 | 9.182 |
| Abnormal-Resting | 76 / 76 | 9.807 | 14.197 | 12.708 |
| Task | 75 / 19 | 10.323 | 14.798 | 12.950 |
| Walking | 5 / 5 | 2.989 | 12.400 | 10.760 |

MAE tính theo người; bootstrap ghép cặp theo người 10.000 lần, seed 0.
Reader reset đạt **9,13014 bpm**, bias −0,03669, Spearman 0,57215.
So hằng số: Δ −2,94054 [−4,11441; −1,78768].
So hiệu chỉnh cũ: Δ −2,24371 [−3,23840; −1,25786].
So reader_v3: Δ −0,45692 [−1,27597; +0,31198], **chưa chứng minh tốt hơn v3**.
Đạt tiêu chí số học dev cũ (≤10 và giảm ≥1 so hiệu chỉnh với CI loại 0), nhưng
đã có nhiều thử nghiệm trên dev; không phải bằng chứng tổng quát hóa độc lập.
Holdout chưa mở. [Số đầy đủ](validation/video_cli_dev_20261123.json).

## Lỗi được tìm bằng kiểm từ file gốc

open-rppg 0.1.1 xóa buffer khi bắt đầu video nhưng giữ `model.state`. Script
nghiên cứu cũ tái sử dụng đối tượng Model. Thử A → B → A làm phổ của A thay đổi
0,0054915 ở phần tử lệch lớn nhất; reset state trước A cho sai khác 0.
Cùng video, môi trường expert cũ và môi trường mới cho phổ reset giống hệt.
Crop, PTS và phổ POS/CHROM cũng khớp, nên không quy lỗi này cho phiên bản môi trường.

Lượt cũ 8,35018 bpm được giữ dưới nhãn **lịch sử có trạng thái liên video**.
Không sử dụng nó để quảng bá sai số của CLI mới. P4m/public và các bảng constraint
chưa được chấm lại sau sửa này. ZIP Release cũ không bị ghi đè; `models/` trong
repo chứa fit mới và SHA256. Fit toàn-dev phục vụ triển khai; 5 fit fold dùng OOF.

## Kiểm raw video → kết quả

Ba video dev đã định trước, mỗi video chạy cả `reference` và `guarded`:

- 6/6 lượt kết thúc thành công. Thời gian khoảng 176–205 s/video trên máy thử,
  khi hai lượt cùng chạy CPU; không phải SLA trên mọi máy/video.
- Reference: crop72, PTS, valid trùng cache; phổ cả ba expert và PR theo fold
  trùng kết quả state-isolated ở cả 3/3 video, sai khác PR **0 bpm**.
- Guarded có thể phục hồi một frame mà reference bỏ. Không khẳng định hai frontend
  tương đương trên mọi video, không hạ ngưỡng cổng 22/11 đã thất bại.
- Video lặp một ảnh mặt thật: `abstain=true`, `pr_bpm=null`.
- Video không mặt: `abstain=true`, không báo prior là mạch đo được.
- Video 1 s: exit khác 0, báo lỗi, không đệm thành kết quả giả.

Ảnh/crop/ID/nhãn cá nhân chỉ nằm trong bằng chứng riêng trên server, không đưa vào repo.
Kiểm âm trên dữ liệu và unit test tách riêng: kiểm tĩnh đầu vào không phải chứng
minh tất cả CNN luôn qua kiểm âm nếu bỏ lớp kiểm đầu vào.

## Môi trường sạch và cách chạy

Linux x86_64, Python 3.12.3. Ba venv mới được cài từ lock trên clone riêng:
`core`, `frontend`, `experts`; không dùng môi trường nghiên cứu cũ.
`bootstrap_video.py` kiểm pins, pip check, SHA256 reader, hai weights và state.pkl.
Lệnh sử dụng và giới hạn: [RUN_VIDEO.md](RUN_VIDEO.md).

Không cần bảng nhãn, đường dẫn riêng của đội hoặc bundle private-assets khi
chạy video mới. Cần cung cấp video và cờ hoạt động đã biết; cần ≥15 GiB RAM
còn trống. P4m/public vẫn là pipeline nghiên cứu riêng, không phải nhánh CLI này.
