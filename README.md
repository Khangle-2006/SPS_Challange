# FaceVitals — SP Cup 2027, Round 1

Ước lượng nhịp mạch (PR, bpm) từ video khuôn mặt. Trên BTC, `predict.py` mặc định dùng **BTC joint**. Trên UBFC2 và MCD, dùng **P4m** qua `predict_public.py`.

## Clone và chạy một video

Trên Linux x86_64, Python 3.12, có `ffmpeg`/`ffprobe` và ít nhất **15 GiB RAM còn trống**:

```bash
git clone https://github.com/Khangle-2006/SPS_Challange.git
cd SPS_Challange
python3.12 tools/bootstrap_video.py
python3.12 predict.py --video /path/video.mp4 --activity Resting --output output/my_video
```

Bootstrap tự cài ba môi trường đã khóa, tải đủ 5 checkpoint public và kiểm reader/weights bằng SHA256. Cần Internet khi cài lần đầu; không cần chép file từ thư mục riêng của tác giả.

Đọc `output/my_video/result.json` hoặc `result.csv`. `--activity` là `Resting`, `Task` hoặc `Walking`, lấy từ hoạt động lúc quay. Mặc định `reference` dành cho video một người. Nếu không đủ bằng chứng, `abstain=true`, `pr_bpm=null`; xem lý do trong trường `reason`. Thư mục output phải mới.

Cài gói hệ thống, xử lý bitrate nguồn, video HDR, tùy chọn fold và bộ gác nhiều mặt: **[RUN_VIDEO.md](docs/RUN_VIDEO.md)**. Lệnh này chạy CPU, không cần nhãn PR, dataset của đội hoặc checkpoint CNN của đội. Bootstrap tạo ba môi trường đã khóa phiên bản; ba expert nằm trong `open-rppg==0.1.1`, reader nhẹ có sẵn trong `models/`.

## Tái lập BTC joint trên dev

Với dữ liệu dự án được cấp quyền, lệnh sau chạy lại bộ đọc trên trace RGB và phổ ba expert theo đúng fold người, kiểm từng dự đoán của cả 218 video. Video mới dùng tham số triển khai trong `models/btc_joint/`.

```bash
.venv/core/bin/python tools/reproduce_btc_joint.py --reports /path/to/facevitals/reports
```

## Kết quả BTC joint

Mỗi video có một dự đoán nhịp mạch. MAE là sai số tuyệt đối trung bình, tính trong từng người rồi lấy trung bình giữa các người. Đơn vị là nhịp/phút (bpm), càng thấp càng tốt.

| Tập BTC | Video có dự đoán / tổng | Người được chấm / tổng | MAE (bpm) |
|---|---:|---:|---:|
| Dev, kiểm chéo theo người | 218/218 | 158/158 | **8,020** |
| Holdout đã đánh giá trước đây | 71/72 | 40/41 | **9,704** |

Khoảng tin cậy 95% của MAE trên dev là 6,790–9,350 bpm. Một video holdout thiếu cửa sổ hợp lệ để trả về PR.

## P4m trên UBFC2 và MCD

P4m ước lượng nhịp mạch theo cửa sổ 10 giây. Khi chấm một PR cho mỗi video như BTC, lấy trung vị các dự đoán và trung vị nhịp tham chiếu từ PPG đầu ngón tay trên cùng các cửa sổ hợp lệ.

| Tập dev, video gốc | Video / người | MAE theo cửa sổ (bpm) | MAE theo video (bpm) |
|---|---:|---:|---:|
| UBFC2 | 35 / 35 | 0,746 | **0,515** |
| MCD FullHD | 119 / 60 | 1,511 | **0,648** |

BTC joint đã được chấm trên video gốc. Kết quả C1–C7 của phiên bản R1 và hướng dẫn tái lập được lưu trong [bảng thí nghiệm R1](docs/RESULTS.md) và [hướng dẫn chạy trên server](docs/REPRODUCE_SERVER.md).

## Pipeline

| Bước | Xử lý |
|---|---|
| Frontend | PTS thật, FaceMesh một mặt, oval 36×36 cho tín hiệu màu và crop hộp 72×72 cho expert. `guarded` qua C3 và native reader_v3; chưa qua tái lập chính xác reader_v4/P4m R1. |
| Tín hiệu | POS + CHROM; FacePhys.rlap, ME-chunk.pure, ME-flow.pure, trọng số giữ nguyên và self-warm riêng từng video. Nhánh public thêm CNN padding-fixed được lọc đầu vào 0,1 s. |
| Phổ | Cửa sổ 10 s; BTC joint bước 2,5 s cho POS/CHROM, 5 s cho ba expert; BTC R1 bước 5 s, public R1 bước 10 s; nội suy 30 Hz, detrend λ=100, Butterworth 0,7–3 Hz, Hann periodogram nfft=4096. Giữ toán tử BTC/public tương ứng đã khóa. |
| BTC | BTC joint chọn giữa PR hợp nhất và năm PR riêng theo chất lượng phổ, mức đồng thuận, bitrate và cờ Task. Khi bitrate dưới 4 Mb/s và beta của nhánh thấp bằng 0, dùng bộ bù từ thông tin nguồn quay, chuyển động hộp mặt và tín hiệu màu/phổ. |
| Public | P4m gộp bốn kênh và readout theo thời gian; fit gộp native + 6 điều kiện trên fold train. Trọng số/readout theo fold, không dùng fit deploy để chấm dev. |

[Chi tiết kiến trúc và tham số](docs/PIPELINE.md). [Kiểm CLI từ video gốc](docs/VIDEO_CLI_VALIDATION.md).

## Dataset và checkpoint

Dataset đã có trên **P100**, cần tài khoản và quyền truy cập:

| Bộ | Đường dẫn |
|---|---|
| BTC | `/mnt/samsung_t5/khangle/facevitals/datasets/FaceVitals_BTC_20261107` |
| UBFC-rPPG | `/data_pool/khangle/facevitals/datasets/UBFC-rPPG` |
| MCD-rPPG | `/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG` |
| AiPEX | `/data_pool/khangle/facevitals/datasets/AiPEX` |

[Quyền, split và đường dữ liệu](docs/SERVER_DATASETS.md). Chỉ dùng manifest dev được cấp quyền; **holdout BTC 72 video/41 người và 7 người test UBFC2 vẫn khóa**. Không quét cả root để chấm.

[Release R1 hiện hành: checkpoint + reader](https://github.com/Khangle-2006/SPS_Challange/releases/tag/inference-assets-r1-20261124) được bootstrap tự tải và kiểm. **BTC joint** có sẵn trong `models/btc_joint/` cùng repo. Gói R1 cung cấp checkpoint public và các tham số R1. Release cũ được giữ riêng để tra cứu. [Chi tiết](docs/INFERENCE_ASSETS.md).

## Cấu trúc code và tái lập

| Thư mục / file | Vai trò |
|---|---|
| `predict.py`, `predict_public.py`, `runtime/` | Video đơn → BTC PR hoặc public P4m HR, không fit lúc suy luận |
| `models/` | Reader BTC deploy + 5 fold, tham số public P4m; không chứa nhãn hay phổ cá nhân |
| `environments/`, `tools/bootstrap_video.py` | Dependency đã khóa và cài môi trường |
| `configs/`, `tools/repro.py` | Khai báo đường dẫn riêng và chuẩn bị tái lập nghiên cứu |
| `reports/`, `ops/` | Mã thí nghiệm, bản gốc giữ hash; tài sản cá nhân không kèm repo |
| `docs/RESULTS.md`, `docs/SOURCE_MANIFEST.json` | Bảng mới và nguồn/hash |
| `submission/` | Mã đóng gói; không đồng nghĩa đã nộp bài |

[GETTING_STARTED](docs/GETTING_STARTED.md) hướng dẫn môi trường nghiên cứu và `tools/repro.py init/doctor/prepare/plan`; [ASSETS](docs/ASSETS.md) liệt kê cache/checkpoint riêng cần có để lặp bảng. Chạy video mới không đòi các tài sản riêng này. Public/P4m có lệnh riêng `predict_public.py`; đã kiểm từ video gốc trên 3 bản ghi, 26 cửa sổ khớp R1 0,000 bpm. [Gói Round 1 và giới hạn](docs/SUBMISSION_ROUND1.md).

Không huấn luyện lại, không mở rộng lưới và không mở holdout trong R1. Gói này không bảo đảm MAE dev cho video bất kỳ. [Nguồn và giấy phép bên thứ ba](THIRD_PARTY_NOTICES.md).
