# Từ clone mới đến PR và số dev R1

## 1. Điều kiện máy

Linux x86_64, Python 3.12, ffmpeg/ffprobe, ít nhất 15 GiB RAM available, khoảng 3 GiB cho ba môi trường và thêm chỗ cho output video. Cài gói hệ thống nếu chưa có:

```bash
sudo apt-get install git ffmpeg python3.12-venv libgl1 libegl1 libglib2.0-0t64
```

Lệnh apt ở trên dành cho Ubuntu 24.04. Nếu máy không có Python 3.12 trong kho hệ điều hành, nhờ quản trị viên cài đúng phiên bản trước; không thay lock bằng phiên bản ngẫu nhiên. Không cần GPU. Cài lần đầu cần Internet đến GitHub/PyPI.

## 2. Clone và lấy đủ tài sản

```bash
git clone https://github.com/Khangle-2006/SPS_Challange.git
cd SPS_Challange
python3.12 tools/bootstrap_video.py
```

Bootstrap tạo `core`, `frontend`, `experts`, kiểm pins/pip check, weights và reader; tự tải Release R1 gồm đủ 5 checkpoint public. Không tự quét dataset hoặc mở holdout.

## 3. Chọn video và ra PR

```bash
python3.12 predict.py --video /absolute/path/video.mp4 --activity Resting --output output/video_001
```

Hoạt động lấy từ lúc quay: Resting/Task/Walking. Đọc `output/video_001/result.json` hoặc CSV. `pr_bpm` là dự đoán; `abstain=true` nghĩa không báo số đo. Mỗi lượt cần một thư mục output mới. Nếu file không có stream bitrate, truyền `--source-bitrate` bằng bitrate **nguồn**; không dùng bitrate bản đã nén/resize để đổi loại nguồn. Ví dụ và giới hạn frontend: [RUN_VIDEO](RUN_VIDEO.md).

## 4. Lặp đúng các số dev trên P100

```bash
.venv/core/bin/python tools/reproduce_r1.py --output output/r1_replay
```

Mặc định server-root là `/mnt/samsung_t5/khangle/facevitals`. Máy khác có bản sao được cấp quyền dùng `--server-root /path/copy`. Private cache là `reports/repo_ready_20261124/dev_replay`; cấu hình công khai chỉ chứa hash/danh mục, không chứa nhãn. Cần quyền đọc cache. Nếu thiếu, chương trình dừng và chỉ rõ đường thiếu, không tải dataset hay quét sang holdout.

Replay dùng code reader trong clone và tham số đã khóa, từ phổ/tensor bằng chứng trước reader, không dùng mảng dự đoán lưu sẵn làm đầu ra. Dự đoán lưu sẵn chỉ để assert parity. Không fit lại. BTC: đủ 218 video/158 người native. Public: 24 hàng của 12 điều kiện × 2 bộ. `RESULTS.csv` và `REPLAY_COMPLETE.json` ghi MAE theo người và sai khác dự đoán. Đây là kiểm tái lập reader/readout; không phải tái mã hóa và suy luận CNN lại mọi video.

| Native | MAE phải tái lập, bpm |
|---|---:|
| BTC reader_v4 | 8,333268430 |
| UBFC2 P4m | 0,746442522 |
| MCD P4m | 1,510869770 |

Muốn chấm một video BTC dev từ file gốc, dùng `predict.py --fold N` đúng fold người; nếu không có `--fold`, CLI dùng deployment model nên **không được** so số đó với OOF. Dataset raw/mapping/clock có quyền truy cập nằm ở [SERVER_DATASETS](SERVER_DATASETS.md). Các script R1 và bảng 9 hàng BTC constraint được giữ trong `reports/rerun_stateless_20261123`; lệnh replay gọn này không tái chạy 8 điều kiện BTC hoặc toàn frontend public.

## 5. Phạm vi chứng minh

Kết quả là dev sau nhiều thử nghiệm. Số cũ 8,350 đã bị loại vì mang trạng thái giữa video; không dùng làm mốc hiện hành. Chọn 3/17 expert trên phổ cũ chưa được làm lại. Frontend guarded vẫn thử nghiệm. Chạy một video bất kỳ không bảo đảm sai số bằng trung bình dev. Holdout BTC và 7 người test UBFC2 vẫn khóa.
