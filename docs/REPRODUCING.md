# Reproduction guide

**Cài mới:** dùng [GETTING_STARTED.md](GETTING_STARTED.md), [ASSETS.md](ASSETS.md) và [kết quả kiểm môi trường mới](CLEAN_ENV_VALIDATION.md). Các phần dưới giữ bối cảnh snapshot lịch sử; requirements-research.txt không thay thế bốn profile mới.

## 1. Chọn mức tái lập

| Mức | Có thể làm từ repo này? | Cần thêm |
|---|---|---|
| Kiểm source/hash/syntax và scorer tổng hợp | Có | Python, NumPy, SciPy |
| POS baseline từ RGB CSV của bạn | Có | Timestamp và valid mask đúng; không cần nhãn |
| POS MATLAB/Octave từ ROI MAT | Có code | Octave hoặc MATLAB; `roi_data` theo schema bên dưới |
| CPU/GPU inference checkpoint public | Có code lịch sử | Checkpoint, cache/source, receipt frontend, môi trường PyTorch |
| Lặp đúng bảng BTC/public của đội | Có thuật toán/report code, **không phải gói chạy một lệnh** | Dữ liệu được cấp quyền, split/manifest/reference, các checkpoint, các receipt hash gốc |
| Huấn luyện lại các sweep | Có snapshot code | Tất cả mục trên + Linux/FFmpeg/GPU, preregistration và budget riêng |

Không tự chạy lại controller trong project gốc. Các thí nghiệm của đội đã kết thúc; tài liệu này hướng dẫn người có quyền tạo một môi trường tái lập riêng. Không tắt assertion để vượt manifest/hash/split không khớp.

## 2. Môi trường

Quick start trong README chỉ dùng NumPy/SciPy. Môi trường research chủ yếu Linux (có `fcntl`, `taskset`, `nice`, `du`, FFmpeg/ffprobe); Windows chạy được baseline nhưng không tự chạy được mọi controller Linux.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research.txt
ffmpeg -version
ffprobe -version
python -c "import torch, cv2, av, mediapipe; print(torch.__version__)"
```

`requirements-research.txt` là danh sách khởi điểm, **chưa kiểm một môi trường sạch chạy đủ mọi nhánh**. Các `requirements*.txt` gốc được giữ tại thư mục tương ứng; chọn bản khớp thí nghiệm. Bản `submission/requirements.txt` dùng wheel torch CPU; GPU cần wheel phù hợp CUDA và driver, không thay âm thầm rồi tuyên bố bitwise parity.

## 3. Chuẩn bị tài sản được cấp quyền

Không tải dữ liệu bằng script trong repo. Lấy dữ liệu từ chủ sở hữu theo điều kiện sử dụng. Giữ raw immutable và lưu output ở thư mục riêng.

| Tài sản | Nội dung cần có | Script tiêu thụ |
|---|---|---|
| UBFC2 | `subject*/vid.avi`, `ground_truth.txt` | `ops/deep_hr/data_adapters.py` |
| MCD | Video các camera, PPG gốc, bảng ánh xạ và receipt đồng hồ | Cùng adapter; không suy timestamp chỉ từ FPS container |
| BTC dev | Video, bảng PR, person mapping và `btc_split.json` được khóa | `reports/round1_v4_20261105/btc_zeroshot/` |
| AiPEX | ROI và HR trung bình đúng người/lần quay | `ops/aipex_diagnostic_20261031/`, `aipex_data.py` |
| T8 oval cache | `crop/rgb.npy`, `meta.npz`, `manifest.json`, `windows.json`, receipt mỗi condition | `ops/t8_oval_20261031/prepare_t8_cache.py` |
| Training index | Fold theo người, hash cache, eligibility, schedule | `ops/t8_oval_training_20261031/t8_plan.py` |
| Checkpoint | `final.pt` cuối epoch + config/provenance/frontend hashes | Evaluator tương ứng với chính phiên bản model |
| Evaluation artifacts | Reference canonical + raw copy, prediction CSV, failure/support masks | `t8_evaluate.py`, aggregate/report code |

Checkpoint không có trong Git. Không có link download công khai đã xác minh. Yêu cầu maintainer cấp asset bundle đúng quyền trước khi chạy full reproduction. Không tự dùng checkpoint fold0 cho mọi fold; không dùng fold hoặc label validation trong fit/calibration/prior.

Snapshot legacy dùng các root `/mnt/samsung_t5/khangle/facevitals`, `/data_pool/khangle/facevitals` và `/data/khangle/facevitals`. Trên máy tái lập, đặt bản sao vào layout tương ứng trong một môi trường cô lập hoặc tạo một **phiên bản cấu hình mới được ghi rõ**. `code/t8_oval_20261031` là bundle training của `ops/t8_oval_training_20261031`; `code/t8_polarity_v3_20261102` tương ứng `ops/t8_polarity_v3_20261102`. Root reports phải giữ quan hệ thư mục hiện có. Không đổi code/đường dẫn rồi vẫn dùng hash frozen cũ như thể không thay gì.

## 4. Chạy baseline và scorer độc lập

```bash
python tools/pos_trace.py --input /path/to/rgb.csv --output output/prediction.csv
python score_round1.py --help
python selection_round1.py --help
python -m unittest test_score_round1
python -m unittest discover -s tools -p 'test_*.py'
```

Scorer là công cụ nội bộ, không phải file `app.py` của ban tổ chức. Input scorer:

```text
reference_signal.csv:
  subject_id,recording_id,window_start_s,video_support_valid,
  ppg_timestamps_json,ppg_values_json
processed_output.csv:
  subject_id,recording_id,window_start_s,hr_bpm,abstain
```

Reference manifest tạo độc lập với prediction. Cửa sổ thiếu/không hợp lệ phải được báo, không chọn giao thuận lợi để làm đẹp số. Scorer public waveform không trực tiếp là scorer nhãn PR cấp video BTC.

## 5. T8 / V3 / padding-fixed

Luồng: raw + PPG → adapter/clock → canonical windows → encode toàn frame → oval crop36 → cache/index → train theo fold người → final checkpoint → inference → cumsum/detrend/filter/HR → aggregate.

Ví dụ CLI có thật trong code (chỉ chạy sau khi chuẩn bị input và các guard):

```bash
python ops/t8_oval_20261031/prepare_t8_cache.py --dataset UBFC2 \
  --root /authorized/UBFC2 --output /scratch/cache/UBFC2 \
  --reuse-root /scratch/legacy-cache --budget-bytes 6000000000 --smoke
python ops/t8_oval_training_20261031/t8_plan.py \
  --ubfc /scratch/cache/UBFC2 --mcd /scratch/cache/MCD \
  --folds /authorized/folds_frozen.json --output /scratch/training_index.json
```

Đây là lệnh theo schema, **chưa phải xác nhận các path ví dụ tồn tại**. MCD cần `--audit` trỏ receipt hợp lệ. Bản T8 `t8_run.py` gốc chứa queue cũ, không dùng để thay V3 đã sửa dấu hoặc final padding-fixed. Tra [index thí nghiệm](EXPERIMENTS.md) để chọn đúng runner.

- V3: `ops/t8_polarity_v3_20261102/`; reuse 15 control, train 15 joint sửa dấu. Các source tests nằm cùng folder. Cần raw/canonical reference hash và parity, không bỏ qua mismatch.
- Padding implementation và regression: `reports/round1_v4_20261105/tsm_padding/replicate_tsm.py`, `test_static_tsm.py`; pilot repair ở `padding_pilot_20261107/`; final folds ở `reports/round1_execution_20261110/final_priorities_20261111/`.
- V5: `v5_multiview_20261112/`; không đồng nhất với 5-run adaptation sweep thất bại trước đó.
- Evaluation phải dùng đúng model frontend và final epoch. Không chọn checkpoint theo outer validation.

Ví dụ kiểm ảnh tĩnh (đọc implementation của checkpoint trước khi chọn `--padding`):

```bash
cd reports/round1_v4_20261105/tsm_padding
python test_static_tsm.py
python check_checkpoint_static.py /authorized/final.pt --padding replicate
```

Một số bài test import thư viện hoặc frozen artifact gốc; nếu thiếu, báo dependency thiếu, không tự coi test skip là PASS.

## 6. BTC zero-shot, calibration, C5′ và audit

`btc_zeroshot/` giữ preprocessing/inference/report. `round1_bayes_20261108/` giữ model nhiều ROI, fitted prior, calibration và report; `round1_execution_20261110/` giữ C5′, AdaBN, adaptation, PBV/G và các audit cuối.

Đọc mỗi `README.md` thư mục và `docs/ENTRYPOINTS.json`: đây là index được lấy bằng AST, không import/launch worker. Nhiều file dùng constants thay vì CLI; sửa root phải tạo revision riêng. Các controller yêu cầu receipt approval/budget/frozen/source index đã tồn tại; bản công bố **không dựng receipt phê duyệt giả** để chạy được.

Thứ tự để tái lập bảng: khóa person split → export dev-only manifest → preprocessing + support → prediction từng checkpoint/condition → join chính xác key người/video/cửa sổ → calibration bằng train-fold → person-first metrics → paired bootstrap → coverage/failure/static gates. Chỉ chạy report khi đủ dependency. Mỗi BTC table cần constant train-fold và saved calibrated comparator trên cùng video.

C5′: dùng `c5prime_v2.py`/phiên bản được caller hiện tại import; `-bf 0` là condition riêng. C5 cũ, C5′default và C5′bf0 là ba recipe khác nhau. Giữ hash bitstream, command, frame mapping/PTS và bitrate đo, không chỉ cấu hình mục tiêu. Muốn đọc đúng revision cuối xem import trong `common_repair.py` và `ladder_physical_v3.py`; output container-clock lỗi ban đầu không là kết quả PPG chính.

## 7. MATLAB / Octave baseline

MAT input cần `roi_data.t` (N×1 giây, tăng), `roi_data.rgb` (N×3), `roi_data.valid` (N×1). Không đọc video trong script .m; ROI phải được xuất trước.

```bash
octave --traditional --quiet --eval "addpath('reports/round1_execution_20261110'); process_round1('roi_data.mat','output/processed_output');"
```

Tạo thư mục output trước, dùng prefix chưa tồn tại. Script sinh CSV và `.mat -v7`. Đây là POS baseline đã kiểm Octave, không phải estimator BTC cuối hoặc MATLAB E2E pass.

## 8. Phạm vi công bố

`SOURCE_MANIFEST.json` kiểm **từng file source đã phát hành**, không giả làm manifest hoàn chỉnh của máy chủ. Các helper SSH/auth/monitor không phát hành. Những frozen bundle cũ có hash tính trên cả directory sẽ cần asset bundle gốc để replay y hệt; không tái tạo một hash khác rồi gọi là hash cũ.

Không có job nào tự chạy lúc clone/import quick-start. Không mở holdout. Không có token trong repo. Đọc `VALIDATION.md` để biết chính xác những kiểm tra phát hành đã chạy.
