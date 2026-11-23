# FaceVitals — SP Cup 2027, Round 1

Ước lượng **nhịp mạch PR (bpm) từ video RGB khuôn mặt**. Bản R1 ngày 23/11/2026 dùng ba expert **self-warm, tách trạng thái giữa video**. BTC dùng reader_v4; bộ công khai dùng P4m. Các số dưới đây là kiểm chéo theo người trên **dev**, chưa phải holdout.

## Clone và chạy một video

Trên Linux x86_64, Python 3.12, có `ffmpeg`/`ffprobe` và ít nhất **15 GiB RAM còn trống**:

```bash
git clone https://github.com/Khangle-2006/SPS_Challange.git
cd SPS_Challange
python3.12 tools/bootstrap_video.py
python3.12 predict.py --video /path/video.mp4 --activity Resting --output output/my_video
```

Bootstrap tự cài ba môi trường đã khóa, tải đủ 5 checkpoint public và kiểm reader/weights bằng SHA256. Cần Internet khi cài lần đầu; không cần chép file từ thư mục riêng của tác giả.

Đọc `output/my_video/result.json` hoặc `result.csv`. `--activity` là `Resting`, `Task` hoặc `Walking`, lấy từ hoạt động lúc quay. Mặc định `reference` dành cho video một người. Nếu không đủ bằng chứng, `abstain=true`, `pr_bpm=null`; prior dự phòng **không phải số đo mạch**. Thư mục output phải mới.

Cài gói hệ thống, xử lý bitrate nguồn, video HDR, tùy chọn fold và bộ gác nhiều mặt: **[RUN_VIDEO.md](docs/RUN_VIDEO.md)**. Lệnh này chạy CPU, không cần nhãn PR, dataset của đội hoặc checkpoint CNN của đội. Bootstrap tạo ba môi trường đã khóa phiên bản; ba expert nằm trong `open-rppg==0.1.1`, reader nhẹ có sẵn trong `models/`.

## Lặp số trên server P100

Sau bootstrap, chạy trong clone mới:

```bash
.venv/core/bin/python tools/reproduce_r1.py --output output/r1_replay
```

Lệnh kiểm hash cache dev được cấp quyền, chạy lại reader từ bằng chứng phổ với đúng model fold, rồi đối chiếu từng dự đoán; không fit lại, không đọc holdout. Kỳ vọng BTC native **8,333268**, UBFC2 native **0,746443**, MCD native **1,510870 bpm**. Có đủ 24 hàng public native/constraint. Đây là replay readout từ cache, **không phải chạy lại CNN/frontend từ video gốc**. Chạy một video mới dùng lệnh `predict.py` ở trên; chấm dev phải thêm `--fold` đúng người. [Hướng dẫn từng bước và phạm vi](docs/REPRODUCE_SERVER.md).

## Kết quả hiện hành

MAE là sai số tuyệt đối: tính trong từng người rồi trung bình giữa người; thấp hơn là tốt hơn. CI95 bootstrap theo người, 10.000 lần, seed 0.

| Nhánh / video gốc | Phạm vi dev | MAE mới [CI95], bpm | Hằng số train-fold |
|---|---|---:|---:|
| BTC reader_v4 self-warm | 218 video / 158 người | **8,333 [7,090; 9,659]** | 12,071 |
| Public P4m — UBFC2 | 193 cửa sổ / 35 người | **0,746 [0,516; 1,051]** | 14,592 |
| Public P4m — MCD | 1.959 cửa sổ / 60 người | **1,511 [1,247; 1,820]** | 11,684 |

BTC reader_v3 đạt 9,587. Delta reader_v4 − v3 **−1,254 [−2,282; −0,283] bpm**. Bản reset một lượt đạt 9,130; self-warm giảm **−0,797 [−1,632; −0,076] bpm**. Việc chọn 3/17 expert được thực hiện trên phổ cũ và **không làm lại** trong R1; không gọi các CI này là kiểm định holdout độc lập.

[Bảng đầy đủ: 9 hàng BTC, 24 hàng public, P4/P4m, cả số xấu, độ phủ và CI](docs/RESULTS.md). Trường hợp thiếu sáng kèm nhiễu mạnh `aidtrack_c1_g25gv01` không chứng minh thắng hằng số ở cả BTC, UBFC2 và MCD. C5 ở đây là **luồng crop tách ≤10 fps**, khác thử nghiệm nén video toàn khung trước đây.

## Lỗi trạng thái 23/11

`open-rppg 0.1.1` xóa bộ đệm khung nhưng giữ trạng thái hồi quy giữa các lời gọi. Các script cũ dùng chung model qua nhiều video, khiến kết quả phụ thuộc video chạy trước. **8,350 và public 0,75/1,59 cũ đã bị loại**, dù một số số mới gần chúng.

Bản hiện hành sao chép trạng thái khởi tạo ngay sau khi tạo model. Trước mỗi video, khôi phục trạng thái đó, xử lý **chính video này hai lượt** và lấy lượt hai. Kiểm reset/self-warm so tham chiếu và A→B→A đều lệch **0,0** cho cả ba expert. Đã chạy lại đủ 63 lượt mô hình × điều kiện; số đầu ra khớp bản cũ ở mọi lượt. [Lịch sử được giữ nguyên](docs/history/R1_STATE_BUG_20261123.md).

## Pipeline

| Bước | Xử lý |
|---|---|
| Frontend | PTS thật, FaceMesh một mặt, oval 36×36 cho tín hiệu màu và crop hộp 72×72 cho expert. `guarded` là tùy chọn thử nghiệm, chưa đạt cổng tương đương toàn cohort. |
| Tín hiệu | POS + CHROM; FacePhys.rlap, ME-chunk.pure, ME-flow.pure, trọng số giữ nguyên và self-warm riêng từng video. Nhánh public thêm CNN padding-fixed được lọc đầu vào 0,1 s. |
| Phổ | Cửa sổ 10 s, bước 5 s, nội suy 30 Hz, detrend λ=100, Butterworth 0,7–3 Hz, Hann periodogram nfft=4096. Giữ toán tử BTC/public tương ứng đã khóa. |
| BTC | Gộp bằng chứng hai phương pháp cổ điển và ba expert; prior + beta + offset theo lớp bitrate nguồn và cờ Task. |
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

[Release R1 hiện hành: checkpoint + reader](https://github.com/Khangle-2006/SPS_Challange/releases/tag/inference-assets-r1-20261124) được bootstrap tự tải và kiểm. **CLI dùng `models/` đi cùng code**, gói tải phải trùng tham số đó. Release cũ được giữ riêng để tra cứu. [Chi tiết](docs/INFERENCE_ASSETS.md).

## Cấu trúc code và tái lập

| Thư mục / file | Vai trò |
|---|---|
| `predict.py`, `runtime/` | Video đơn → PR của nhánh BTC, không fit lúc suy luận |
| `models/` | Reader BTC deploy + 5 fold, tham số public P4m; không chứa nhãn hay phổ cá nhân |
| `environments/`, `tools/bootstrap_video.py` | Dependency đã khóa và cài môi trường |
| `configs/`, `tools/repro.py` | Khai báo đường dẫn riêng và chuẩn bị tái lập nghiên cứu |
| `reports/`, `ops/` | Mã thí nghiệm, bản gốc giữ hash; tài sản cá nhân không kèm repo |
| `docs/RESULTS.md`, `docs/SOURCE_MANIFEST.json` | Bảng mới và nguồn/hash |
| `submission/` | Mã đóng gói; không đồng nghĩa đã nộp bài |

[GETTING_STARTED](docs/GETTING_STARTED.md) hướng dẫn môi trường nghiên cứu và `tools/repro.py init/doctor/prepare/plan`; [ASSETS](docs/ASSETS.md) liệt kê cache/checkpoint riêng cần có để lặp bảng. Chạy video mới không đòi các tài sản riêng này. Public/P4m vẫn là đường nghiên cứu, chưa gộp vào lệnh `predict.py`.

Không huấn luyện lại, không mở rộng lưới và không mở holdout trong R1. Gói này không bảo đảm MAE dev cho video bất kỳ. [Nguồn và giấy phép bên thứ ba](THIRD_PARTY_NOTICES.md).
