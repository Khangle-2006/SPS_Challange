# Kiểm trên môi trường mới — 22/11/2026

Đã tạo **bốn venv mới**, không dùng `site-packages` hệ thống hay môi trường thí
nghiệm cũ, trên Linux x86_64/Python 3.12.3. Chạy CPU, nice 19; không đọc video,
nhãn người tham gia hoặc holdout. Đây là kiểm cài đặt và chức năng, **chưa phải
tái lập MAE BTC/public hoặc kiểm end-to-end pipeline đã chọn**.

Receipt đã bỏ đường dẫn riêng: [clean_env_20261122.json](validation/clean_env_20261122.json).
Lock dependency đầy đủ: [environments/locks](../environments/locks).

## Kết quả thực đo

| Môi trường / kiểm | Kết quả |
|---|---|
| Cài mới và `pip check` | PASS cả 4 profile; không lỗi dependency |
| core: POS + CHROM trên RGB tổng hợp 90 bpm | Hai đỉnh đều 90,0 bpm; tín hiệu tĩnh trả không có phổ hợp lệ |
| frontend: MediaPipe 0.10.21, OpenCV 4.11.0, PyAV 14.2.0 | Import và FaceMesh trên ảnh đen PASS; 0 mặt |
| torch-cpu: PyTorch 2.5.1+cpu | Phép tính tensor CPU PASS; CUDA runtime không có |
| experts: open-rppg 0.1.1 / JAX 0.12.1 / Keras 3.15.1 | FacePhys.rlap, ME-chunk.pure, ME-flow.pure đều trả 360 mẫu BVP hữu hạn từ tensor màu tổng hợp |
| Hai file weights expert | SHA256 khớp các giá trị trong [ASSETS.md](ASSETS.md) |
| Kiểm demo và bộ công cụ setup | 7/7 unit test PASS |
| `test_score_round1` | 10/10 unit test PASS |
| `verify_source.py` | 1.020 hash file archive khớp; mã Python parse được |
| Cài thêm một venv core từ lock cuối cùng | PASS; smoke POS/CHROM và static PASS |
| `doctor` khi chưa cấp bundle riêng | Exit 2 đúng thiết kế, báo thiếu asset manifest |

Ba expert chạy tuần tự trong cùng phép kiểm: tổng thời gian **1 phút 45,37 giây**,
RSS cực đại **2.713.808 KiB** theo `/usr/bin/time -v`. Đây không phải benchmark
thời gian trên video người thật. BVP hữu hạn không chứng minh mô hình đo đúng mạch.

Các test setup kiểm thay prefix không thay lặp, từ chối đường dẫn vượt workspace,
bundle sai scope/hash hoặc trùng đích; tạo workspace riêng, giữ nguyên nguồn,
từ chối ghi đè và phát hiện tài sản đã bị sửa sau prepare.

## Lệnh kiểm

Sau các lệnh cài ở [GETTING_STARTED.md](GETTING_STARTED.md):

```bash
.venv/core/bin/python -m unittest discover -s tools -p 'test_*.py'
.venv/core/bin/python -m unittest test_score_round1
.venv/core/bin/python tools/verify_source.py
.venv/core/bin/python tools/smoke_environment.py core
.venv/frontend/bin/python tools/smoke_environment.py frontend
.venv/torch-cpu/bin/python tools/smoke_environment.py torch-cpu
.venv/experts/bin/python tools/smoke_environment.py experts
```

## Giới hạn còn lại

- Chưa chạy lại các bảng nghiên cứu với bundle riêng trong các môi trường mới.
  Các phiên bản NumPy/SciPy và runtime có thể khác môi trường lịch sử; phải kiểm
  số thực trước khi tuyên bố tái lập các MAE trong README.
- Chưa nạp năm checkpoint CNN riêng của đội vào môi trường mới. Torch CPU
  smoke là kiểm runtime; expert smoke chỉ nạp weights upstream đã kiểm hash.
- FaceMesh ảnh đen chưa kiểm decoder, timestamp, ROI, target lock hay frontend
  trên video thật. Không thay thế các kiểm C3 hoặc parity 218 video.
- Launcher được kiểm với bundle tổng hợp nhỏ. Chưa kiểm đầy đủ mọi dependency
  của các script lịch sử sau relocation; hash assertion còn lại phải được xử
  lý bằng receipt đúng phiên bản, không bỏ assertion.
- Windows/WSL2 chưa được kiểm thực tế trong lượt này. Lock môi trường này dành
  cho Linux x86_64/Python 3.12; các nền tảng khác chưa được xác nhận.
- Không phát hành dữ liệu cá nhân, private checkpoint, nhãn holdout hoặc cache
  thí nghiệm. Cách xin đúng tài sản được ghi trong [ASSETS.md](ASSETS.md).

PR này bổ sung khả năng cài đặt, kiểm tài sản và chuẩn bị workspace; không thay
thuật toán, không huấn luyện, không mở holdout và không tự merge.
