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

### BTC dev dưới bảy ràng buộc C1–C7

Mỗi hàng so BTC joint với bản BTC R1 trên cùng video có dự đoán, trong tổng số 218 video dev. Δ là MAE của joint trừ MAE của R1 (bpm). Khoảng tin cậy 95% được tính bằng 10.000 lần bootstrap ghép cặp theo người, seed 0.

| Điều kiện | Video được chấm | BTC joint | BTC R1 | Δ joint − R1 [CI95] |
|---|---:|---:|---:|---:|
| C1 · Giảm sáng, nhiễu Poisson | 200 | 9,49 | 9,67 | −0,18 [−0,55; +0,16] |
| C1 · Giảm sáng, nhiễu Gaussian | 197 | 12,28 | 12,26 | +0,02 [−0,57; +0,57] |
| C2 · Mảng giảm sắc tố mô phỏng | 199 | 8,91 | 8,83 | +0,08 [−0,37; +0,61] |
| C3 · Hai mặt cạnh nhau | 216 | 9,03 | — | — |
| C3 · Người bên cạnh lớn hơn 1,3 lần | 212 | 9,05 | — | — |
| C4 · VGA | 200 | 8,73 | 9,10 | −0,37 [−0,73; −0,03] |
| C4 · QVGA | 200 | 9,12 | 9,44 | −0,32 [−0,88; +0,17] |
| C5 · Truyền crop `split10` | 218 | 9,30 | 9,48 | −0,18 [−0,50; +0,13] |
| C6 · Làm nét | 197 | 8,36 | 8,44 | −0,09 [−0,40; +0,23] |
| C7 · Khẩu trang xanh | 197 | 10,51 | 10,46 | +0,06 [−0,35; +0,46] |

BTC joint có MAE thấp hơn R1 ở 5/8 cấu hình có số ghép cặp. Chỉ C4 VGA có CI95 của chênh lệch nằm hoàn toàn dưới 0. C1 với nhiễu Gaussian cho sai số lớn nhất (12,28 bpm), gần mốc đoán hằng số 12,61 bpm trên cùng 197 video.

Hai bố cục C3 là phép thử mới trên BTC. Bộ gác danh tính chọn sang phía người bên cạnh ở **0/365.426 khung hợp lệ**. Khi bật cùng bộ gác trên cả video một mặt và hai mặt, sai số tăng 0,84 bpm ở bố cục cạnh nhau (215 video ghép cặp) và 1,02 bpm khi người bên cạnh lớn hơn (211 video). Cả hai đều vượt ngưỡng 0,05 bpm.

<details>
<summary>Mốc hằng số và bản hiệu chỉnh cũ trên cùng video BTC (MAE, bpm)</summary>

| Điều kiện | Hằng số theo fold train | Bản hiệu chỉnh native |
|---|---:|---:|
| C1 · Giảm sáng, nhiễu Poisson | 12,50 | 11,58 |
| C1 · Giảm sáng, nhiễu Gaussian | 12,61 | 11,73 |
| C2 · Mảng giảm sắc tố mô phỏng | 12,43 | 11,46 |
| C3 · Hai mặt cạnh nhau | 12,12 | 11,45 |
| C3 · Người bên cạnh lớn hơn 1,3 lần | 11,96 | 11,38 |
| C4 · VGA | 12,50 | 11,58 |
| C4 · QVGA | 12,50 | 11,58 |
| C5 · Truyền crop `split10` | 12,07 | 11,37 |
| C6 · Làm nét | 12,53 | 11,55 |
| C7 · Khẩu trang xanh | 12,69 | 11,74 |

</details>

## P4m trên UBFC2 và MCD

P4m ước lượng nhịp mạch theo cửa sổ 10 giây. Khi chấm một PR cho mỗi video như BTC, lấy trung vị các dự đoán và trung vị nhịp tham chiếu từ PPG đầu ngón tay trên cùng các cửa sổ hợp lệ.

| Tập dev, video gốc | Video / người | MAE theo cửa sổ (bpm) | MAE theo video (bpm) |
|---|---:|---:|---:|
| UBFC2 | 35 / 35 | 0,746 | **0,515** |
| MCD FullHD | 119 / 60 | 1,511 | **0,648** |

### UBFC2 và MCD dưới các ràng buộc

Bảng dưới dùng kết quả **P4m đã chấm ở R1**. Mỗi ô ghi **MAE (bpm) · số cửa sổ 10 giây được chấm**. P4m tính sai số trên từng cửa sổ rồi tổng hợp theo người. Bảng BTC phía trên chấm một dự đoán cho mỗi video.

| Điều kiện | UBFC2 | MCD FullHD |
|---|---:|---:|
| C1 · Giảm sáng, Poisson, gain 0,50 | 0,982 · 193 | 1,869 · 618 |
| C1 · Giảm sáng, Poisson, gain 0,25 | 1,532 · 193 | 2,443 · 618 |
| C1 · Giảm sáng, Poisson, gain 0,12 | 2,564 · 186 | 3,056 · 616 |
| C1 · Giảm sáng, Gaussian, gain 0,25, phương sai 0,001 | 3,082 · 192 | 4,139 · 616 |
| C1 · Giảm sáng, Gaussian, gain 0,25, phương sai 0,01 | 13,450 · 184 | 10,435 · 592 |
| C2 · Mảng giảm sắc tố mô phỏng | 0,750 · 193 | 1,734 · 618 |
| C4 · QVGA | 0,727 · 193 | 1,520 · 618 |
| C5 · Truyền crop `split10` | 0,646 · 179 | 1,704 · 1.929 |
| C6 · Làm nét | 0,732 · 193 | 1,641 · 618 |
| C7 · Khẩu trang xanh | 1,447 · 192 | 2,972 · 618 |
| C7 · Khẩu trang trắng | 1,154 · 176 | 3,602 · 614 |

**C3 trên UBFC2** được chấm trước khi sửa trạng thái mô hình ở R1. Trên 193 cửa sổ ghép cặp của 35 người, MAE một mặt là **0,755 bpm**. Với hai mặt cạnh nhau, MAE là **0,775** (Δ +0,020). Với người bên cạnh lớn hơn 1,3 lần, MAE là **0,771** (Δ +0,016). Cả hai đạt ngưỡng chênh lệch 0,05 bpm. Báo cáo cũ chỉ có C3 cho UBFC2.

[Khoảng tin cậy, định nghĩa các ràng buộc và giới hạn kiểm chứng C5](docs/RESULTS.md) · [Hướng dẫn tái lập trên server](docs/REPRODUCE_SERVER.md).

## Pipeline

| Bước | Xử lý |
|---|---|
| Frontend | PTS thật, FaceMesh một mặt, oval 36×36 cho tín hiệu màu và crop hộp 72×72 cho expert. Với nhiều mặt, chế độ `guarded` bám người mục tiêu. Kết quả C3 theo từng bộ ở các bảng trên. |
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
