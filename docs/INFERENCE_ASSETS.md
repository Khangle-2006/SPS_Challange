# Checkpoint và cấu hình reader — bản xuất 23/11/2026

Gói này cung cấp trọng số và tham số đã có, không huấn luyện hay fit lại.
Đây là **tài sản để suy luận/tái lập nghiên cứu**, chưa phải gói raw video →
kết quả đã qua toàn bộ cổng tích hợp.

> Bản ZIP này giữ nguyên để tái lập lịch sử. Reader bên trong fit từ expert có
> thể mang trạng thái giữa video; không dùng thay cho `models/` của CLI mới.
> `predict.py` dùng tham số đã tính lại với trạng thái reset, có sẵn trong repo.

## Tải về

[GitHub Release inference-assets-20261123](https://github.com/Khangle-2006/SPS_Challange/releases/tag/inference-assets-20261123)

```bash
mkdir -p checkpoints/download
cd checkpoints/download
curl -fLO https://github.com/Khangle-2006/SPS_Challange/releases/download/inference-assets-20261123/facevitals-inference-assets-20261123.zip
curl -fLO https://github.com/Khangle-2006/SPS_Challange/releases/download/inference-assets-20261123/SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt
unzip facevitals-inference-assets-20261123.zip
cd ../..

python3 tools/check_inference_assets.py checkpoints/download/facevitals-inference-assets-20261123
.venv/torch-cpu/bin/python tools/check_inference_assets.py checkpoints/download/facevitals-inference-assets-20261123 --smoke
```

Trên Windows, có thể tải ZIP/SHA256SUMS từ trang Release và dùng `Get-FileHash`
để kiểm SHA256 trước khi giải nén. Môi trường CPU được cài theo
[GETTING_STARTED.md](https://github.com/Khangle-2006/SPS_Challange/blob/main/docs/GETTING_STARTED.md).

## Trong gói có gì?

| Thư mục/file | Nội dung |
|---|---|
| `checkpoints/padding_fixed_fold{0..4}_seed2027.pt` | 5 checkpoint epoch 30 của EfficientPhys + TSM replicate padding |
| `readers/btc_reader_v4_folds.json` | 5 bản train-fold của reader_v4: prior KDE, beta theo loại nguồn, offset theo loại nguồn × Task |
| `readers/public_p4m_parameters.json` | Tham số fusion/readout P4m của fold 0–4 và `deploy`, riêng UBFC2/MCD |
| `CONFIG.json` | Kiến trúc, kích thước đầu vào, lưới tần số, thứ tự kênh, quy tắc loại nguồn và giới hạn |
| `code/` | Định nghĩa mô hình/padding và toán readout cố định để nạp đúng trọng số |
| `MANIFEST.json` | SHA256 từng file; hash nguồn và phép chuyển đổi khi xuất |
| `VALIDATION.json` | Kiểm tensor trọng số, đầu ra tổng hợp và round-trip tham số |
| `TOOLBOX_LICENSE.txt` | Giấy phép upstream cho mã EfficientPhys/rPPG-Toolbox |

Checkpoint xuất chỉ giữ `model` và `epoch`; bỏ cấu hình huấn luyện. Mọi tensor
trọng số giữ nguyên và được so bằng `torch.equal`. Vì bỏ metadata, SHA256 file
xuất **khác** file `final.pt` gốc; cả hai hash có trong manifest, không thay hash
gốc trong các hồ sơ frozen.

Gói không chứa video, crop, nhãn PR/PPG, ID người, dự đoán từng video/cửa sổ,
split chứa ID, optimizer, mật khẩu hoặc dữ liệu holdout. P4m chỉ xuất 18 mảng
tham số `w_*`/`ro_*`, không chép nguyên file nghiên cứu `z14_p4m_frozen.npz`.
Prior KDE là tham số tổng hợp đã học, không phải danh sách nhãn từng người.

Ba expert không được đóng gói lại: cài `open-rppg==0.1.1` qua profile `experts`.
Hai weights upstream đã có hash kiểm trong `tools/smoke_environment.py`.

## Nạp checkpoint và tham số

Ví dụ từ repo sau khi giải nén:

```python
from pathlib import Path
import json, sys, torch

assets = Path("checkpoints/download/facevitals-inference-assets-20261123")
sys.path.insert(0, str(assets / "code"))
from EfficientPhys_fixed import EfficientPhys
from replicate_tsm import use_replicate_padding

fold = 0  # phải dùng đúng fold của mẫu khi chấm OOF
model = use_replicate_padding(EfficientPhys(frame_depth=10, img_size=36))
payload = torch.load(assets / "checkpoints" / f"padding_fixed_fold{fold}_seed2027.pt",
                     map_location="cpu", weights_only=True)
model.load_state_dict(payload["model"], strict=True)
model.eval()
btc_folds = json.loads((assets / "readers/btc_reader_v4_folds.json").read_text())
p4m = json.loads((assets / "readers/public_p4m_parameters.json").read_text())
```

CNN nhận clip đã chuẩn hóa `[batch, 301, 3, 36, 36]`, trả 300 sai phân; phải
`cumsum` trước toán tử readout. Crop public phải được lọc trung bình trượt theo
`CONFIG.json` trước nội suy và chuẩn hóa. Dùng mô hình với crop/clock khác không
đảm bảo tái lập số đã báo.

Reader BTC có lưới 42–180 bpm, bước 0,5. JSON dùng khóa chuỗi: trước khi gọi
`code/reader_v4.py:predG`, đổi `prior` thành mảng NumPy và khóa `beta`/`off` thành
số nguyên. Loại nguồn = bitrate stream của **video gốc** <4.000.000 bit/s; Task
là hoạt động bắt đầu bằng `Task`. Không phân loại lại bằng bitrate/kích thước
của video constraint. `CONFIG.json` ghi đúng thứ tự evidence và ô offset.

P4m dùng lưới FFT riêng, không phải lưới BTC 0,5 bpm. `w_*` có thứ tự CNN, POS,
CHROM, trung bình ba expert. `ro_*` có thứ tự gamma, sigma, tau, cờ median,
quality reference, constant baseline. `code/public_fixed_math.py` cung cấp
`qscale`, `marg`, `point`; chạy forward-backward riêng từng bản ghi. Không nối
cửa sổ của nhiều người thành một chuỗi. `deploy` chỉ dành cho dữ liệu mới;
chấm OOF phải dùng đúng `foldN`.

## Phần chưa có và cách dùng đúng

Đây là mô tả **Release lịch sử `inference-assets-20261123`**, không thay đổi
bytes/hash của Release đó. Lệnh video mới dùng cấu hình reader trong `models/`
của checkout, xem [RUN_VIDEO.md](RUN_VIDEO.md). Checkpoint CNN ở Release này
phục vụ nhánh public/nghiên cứu, không cần cho nhánh BTC reader_v4.

- **Release này không chứa model BTC `deploy` fit toàn dev.** Gói cung cấp đúng
  5 fit train-fold đã tồn tại; không lấy trung bình chúng rồi gọi là cấu hình
  triển khai đã kiểm. Việc xuất gói không fit hay mở holdout.
- Cổng tích hợp frontend mới ngày 22/11 còn sai khác đầu ra vượt ngưỡng. Vì vậy
  không coi gói này là bản nộp hoàn chỉnh. Các tham số ở đây tái hiện nhánh
  nghiên cứu đã có, không sửa hoặc bỏ qua cổng đang dừng.
- Để lặp bảng cần thêm dữ liệu, đúng split/mapping/cache/clock được cấp quyền.
  Các file đó không được công khai trong Release. Không dùng checkpoint của
  fold khác hoặc tham số all-dev để báo điểm validation.
- Không gán giấy phép chung mới cho trọng số/mã của đội. Giữ điều khoản upstream
  và thỏa thuận dataset; tham khảo `THIRD_PARTY_NOTICES.md` trong repo.

Vị trí dataset trên P100 được ghi tại
[SERVER_DATASETS.md](https://github.com/Khangle-2006/SPS_Challange/blob/main/docs/SERVER_DATASETS.md).
