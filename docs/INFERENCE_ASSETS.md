# Checkpoint và reader R1 — 24/11/2026

Gói hiện hành: [inference-assets-r1-20261124](https://github.com/Khangle-2006/SPS_Challange/releases/tag/inference-assets-r1-20261124).
`python3.12 tools/bootstrap_video.py` cài môi trường và tự tải gói này, kiểm SHA256 ZIP, mọi file, và đối chiếu reader với checkout. Không phải tự tìm checkpoint.

## Lệnh tải riêng

```bash
python3.12 tools/fetch_inference_assets.py
python3.12 tools/check_inference_assets.py checkpoints/r1/facevitals-inference-assets-r1-20261124 \
  --archive checkpoints/r1/facevitals-inference-assets-r1-20261124.zip
```

Có thể dùng `--destination /path/new-folder` cho lệnh tải riêng. Bootstrap và lệnh replay mặc định dùng `checkpoints/r1/`. Gói đã có và hash đúng sẽ được dùng lại; file sai hash hoặc bản tải dở được giữ để kiểm tra, không ghi đè. Khi tải dở, chọn thư mục mới để thử lại.

## Thành phần

| Thành phần | Vai trò |
|---|---|
| `checkpoints/padding_fixed_fold0..4_seed2027.pt` | Đủ 5 checkpoint public, epoch 30, giữ nguyên bytes từ gói đã kiểm tensor; không dùng fold0 cho mọi người |
| `readers/btc_reader_v4_deploy.json` | Cấu hình chạy video mới, đã fit một lần trên dev |
| `readers/btc_reader_v4_folds.json` | Đúng 5 cấu hình theo fold để chấm dev OOF |
| `readers/public_p4m_r1.json` | Trọng số fusion và readout public R1, từng fold và deploy |
| `code/` | Kiến trúc EfficientPhys/padding và toán reader cần để nạp đúng |
| `MANIFEST.json`, `CONFIG.json` | SHA256 từng file, nguồn gốc và quy tắc self-warm |

Ba expert FacePhys.rlap, ME-chunk.pure, ME-flow.pure được cài cùng `open-rppg==0.1.1`; package chứa weights. Bootstrap kiểm hai file weights và `state.pkl` theo hash đã khóa. Không cần tải chúng ở nơi khác. Nhánh BTC không dùng CNN padding-fixed; vẫn tải đủ checkpoint để dùng nhánh public/nghiên cứu.

CLI sử dụng reader trong `models/`, không chép đè từ Release. Downloader bắt buộc reader trong ZIP trùng từng byte với `models/`. Trước mỗi video, expert reset trạng thái rồi chạy chính video hai lượt, chỉ lấy lượt hai.

## Nạp và kiểm CNN public (tùy chọn)

```bash
python3.12 tools/setup_env.py torch-cpu
.venv/torch-cpu/bin/python tools/check_inference_assets.py \
  checkpoints/r1/facevitals-inference-assets-r1-20261124 --smoke
```

Đây là môi trường bổ sung, không cần cho lệnh BTC `predict.py` hoặc replay readout. Kiểm smoke nạp strict đủ 5 checkpoint, đầu ra hữu hạn và sóng phẳng trên đầu vào tĩnh; không thay cho kiểm video thật.

Checkpoint nhận `[batch,301,3,36,36]`, trả 300 sai phân; cần cumsum trước readout. Public P4m dùng crop lọc trước CNN và tham số theo đúng fold. Chạy BTC CLI trên webcam không tương đương chạy P4m. [Pipeline](PIPELINE.md) và [tái lập server](REPRODUCE_SERVER.md).

Gói không chứa ảnh mặt, video, nhãn cá nhân, split có ID, optimizer, thông tin đăng nhập hoặc holdout. Dữ liệu có quyền truy cập vẫn nằm trên server. Gói cũ `inference-assets-20261123` giữ nguyên để kiểm toán; **không dùng reader/P4m của gói cũ cho R1**. [Mô tả lịch sử](history/INFERENCE_ASSETS_20261123.md).
