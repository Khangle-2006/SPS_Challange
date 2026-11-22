# Cài mới và chạy lại

Hướng dẫn này dành cho một checkout mới. Không cần tài khoản máy chủ của đội để
chạy demo và kiểm mã. Muốn lặp bảng BTC/public cần thêm bundle dữ liệu được cấp
quyền: xem [ASSETS.md](ASSETS.md). Việc cài môi trường thành công **không** chứng
minh tái lập được các MAE đã công bố.

## 1. Môi trường được hỗ trợ

Linux x86_64, Python **3.12** cho các script nghiên cứu. Windows dùng WSL2 cho
nhánh này; baseline RGB CSV không phụ thuộc Linux. Cần Git và `python3.12-venv`.
Frontend cần `ffmpeg`, `ffprobe`, libGL/EGL và libglib của hệ điều hành. Ví dụ
Ubuntu 24.04: `sudo apt install python3.12-venv ffmpeg libgl1 libegl1 libglib2.0-0t64`.
Lệnh cài hệ thống do người quản trị thực hiện, không có trong bootstrap.

```bash
git clone https://github.com/Khangle-2006/SPS_Challange.git
cd SPS_Challange
python3.12 tools/setup_env.py core
.venv/core/bin/python -m pip check
.venv/core/bin/python tools/smoke_environment.py core
.venv/core/bin/python tools/verify_source.py
.venv/core/bin/python -m unittest discover -s tools -p 'test_*.py'
.venv/core/bin/python -m unittest test_score_round1
```

`setup_env.py` tạo venv không dùng site-packages của Python hệ thống, cài từ
requirements theo profile rồi chạy `pip check`. Không ghi đè venv đã có; nếu cài
lỗi, giữ nguyên log và dùng `--destination /duong/dan/moi` khi thử lại. Receipt
và danh sách tất cả phiên bản đã resolve ở `output/setup/<profile>/` (không lên Git).

## 2. Demo chạy ngay, không tải dữ liệu

```bash
.venv/core/bin/python tools/pos_trace.py --make-demo output/demo_rgb.csv
.venv/core/bin/python tools/pos_trace.py --input output/demo_rgb.csv --output output/demo_predictions.csv
```

Đầu vào tổng hợp có nhịp 90 bpm. Kết quả gần 90 bpm xác nhận đường POS baseline;
**không phải reader_v4/P4m và không đo hiệu năng trên người**. File output đã tồn
tại thì CLI từ chối ghi đè; đổi tên output. Có thể dùng RGB CSV của bạn với cột
`timestamp_s,red,green,blue,valid`, thời gian bắt đầu 0, tăng nghiêm ngặt, ≥10 s.

## 3. Các profile riêng cho nghiên cứu

```bash
python3.12 tools/setup_env.py frontend
python3.12 tools/setup_env.py torch-cpu
python3.12 tools/setup_env.py experts

.venv/frontend/bin/python tools/smoke_environment.py frontend
.venv/torch-cpu/bin/python tools/smoke_environment.py torch-cpu
.venv/experts/bin/python tools/smoke_environment.py experts
```

| Profile | File requirements | Công việc |
|---|---|---|
| core | `environments/core.txt` | POS/CHROM, reader, chấm, test |
| frontend | `environments/frontend.txt` | MediaPipe FaceMesh, OpenCV, PyAV |
| torch-cpu | `environments/torch-cpu.txt` | CNN của đội; wheel CPU từ PyTorch |
| experts | `environments/experts.txt` | open-rppg 0.1.1, Keras/JAX CPU |

Không cài gộp bốn file vào một venv. MediaPipe 0.10.21 dùng NumPy 1.x trong
profile này; experts dùng NumPy/JAX riêng. Trên Linux x86_64/Python 3.12, bootstrap dùng `environments/locks/*-linux-py312.txt`
khóa cả dependency gián tiếp. Trên nền tảng khác nó dùng file profile trực tiếp,
chưa có chứng nhận nghiên cứu. Bản resolve và phạm vi kiểm ở [CLEAN_ENV_VALIDATION.md](CLEAN_ENV_VALIDATION.md).
Đây là môi trường mới được kiểm chức năng, không tuyên bố giống byte môi trường
thí nghiệm lịch sử. Các requirements ở snapshot cũ chỉ là hồ sơ lịch sử.

`smoke_environment.py` không đọc dataset: kiểm POS/CHROM trên sóng tổng hợp và
ảnh tĩnh, FaceMesh trên ảnh đen, PyTorch CPU, cùng ba expert trên tensor màu tổng
hợp. Kiểm expert còn so đúng SHA256 hai file weights. Không có ý nghĩa MAE dev.

## 4. Cấu hình đường dẫn, kiểm tài sản trước khi chạy

Tải và kiểm [gói checkpoint/reader](INFERENCE_ASSETS.md). Nếu chạy trên P100,
dùng các [đường dẫn dataset hiện có](SERVER_DATASETS.md); không cần tải lại.
Gói công khai không chứa split/cache/manifest người tham gia, nên các stage
tái lập bảng vẫn cần bundle nội bộ đúng phiên bản.

```bash
python3.12 tools/repro.py init
# Sửa configs/paths.local.json: workspace, assets_manifest, đường dẫn Python.
# Nhận bundle + assets.json từ maintainer theo docs/ASSETS.md.
python3.12 tools/repro.py doctor --stage btc-reader-v3
python3.12 tools/repro.py prepare
python3.12 tools/repro.py plan btc-reader-v3
```

`init` ghi đường dẫn tuyệt đối đúng checkout hiện tại, không cần thay username
của máy chủ trong hàng trăm file. Nếu chuyển checkout sang máy khác, tạo lại
config. Đường dẫn tương đối trong config được tính từ **thư mục chứa config**.
Workspace phải nằm ngoài checkout, chưa tồn tại, và không phải root sản xuất.

`doctor` kiểm Python, hash từng file trong bundle và scope `dev-only`; thiếu thì
exit 2 với tên tài sản cần bổ sung. `complete_for` do maintainer kê khai là danh
sách bước bundle hỗ trợ, **không phải chứng nhận đã chạy end-to-end**. Lệnh
`doctor` không đọc nhãn để chấm và không chạy mô hình.

`prepare` kiểm hash archive rồi chép code và các tài sản liệt kê sang workspace
riêng. Chỉ thay prefix đường dẫn trong **bản làm việc**, giữ cấu trúc thư mục.
Nó tạo `RELOCATION.json` có hash trước/sau từng file. Không sửa snapshot trong
Git, không sửa thuật toán, không xóa hay ghi đè nguồn. Không sao chép bất kỳ
holdout nào. Chỉ dùng bundle dev-only đã được maintainer xác nhận.

Các prefix `/mnt/samsung_t5/khangle/facevitals`, `/data_pool/khangle/facevitals`,
`/data/khangle/facevitals` được ánh xạ về workspace mới. Code T8 oval và polarity
được ánh xạ sang thư mục `ops/` tương ứng; các interpreter cũ ánh xạ sang
profile mới. Nếu metadata chứa root dữ liệu khác, thêm cặp prefix tuyệt đối vào
`path_map` trong config. `relocate_json_paths=true` chỉ dùng cho JSON metadata
đường dẫn. File nhị phân, checkpoint và cache được chép nguyên byte.

Hash frozen cũ **không** được cập nhật để che việc relocation. Nếu script còn
assert hash nguồn/metadata cũ, dừng và nhờ maintainer cấp receipt relocation
đúng phiên bản; không bỏ assertion. Một số controller lịch sử không thuộc phạm
vi launcher; không chạy chúng để "sửa" workspace.

## 5. Chạy một bước đã được cấp đủ tài sản

Sau khi kiểm `plan`, đặt `allow_dev_reproduction=true` trong config riêng:

```bash
python3.12 tools/repro.py run btc-reader-v3
# Ví dụ sau khi đã có đầy đủ cache và metadata đúng bảng dev:
python3.12 tools/repro.py plan btc-experts run FacePhys.rlap
python3.12 tools/repro.py run btc-experts run FacePhys.rlap
python3.12 tools/repro.py plan btc-score eval FacePhys.rlap,ME-chunk.pure,ME-flow.pure
```

Đây là lệnh **tái lập dev**, không phải inference một video mới hay holdout.
Các bước `eval` có fit prior/beta trong các fold dev như code gốc; không phải
zero-shot hoàn toàn. Không tự chạy train/controller, không tự mở holdout.
Launcher đặt CPU-only, nice 19, một luồng BLAS; yêu cầu RAM available ≥15 GiB.
Số worker ở tham số CLI vẫn do bạn chọn; bắt đầu bằng 1. Không chạy nguyên batch
tất cả điều kiện chỉ để thử cài đặt.

| Stage | Script | Đầu vào chính → đầu ra |
|---|---|---|
| btc-reader-v3 | `a11_reader_v3.py` | Numeric dev, bảng metadata/baseline → đánh giá v1/v2/v3 |
| btc-front | `x01_extract.py` | Manifest dev, video được cấp quyền → blocks/crop theo điều kiện |
| box-crops | `x10_boxcrop.py` | Video, metadata, frontend → crop hộp 72×72 |
| btc-spectra | `x03_btc_spectra.py` | Blocks/crop và clock → phổ BTC |
| btc-experts | `y05_openrppg.py` | Crop x10, mapping dev, metadata → phổ từng expert |
| btc-conditions / btc-score | `y11_openrppg_conditions.py` | Crop/phổ theo điều kiện → inference / bảng reader_v4 |
| public-cnn | `x02_infer.py` | Crop, fold, checkpoint tương ứng → phổ CNN/POS/CHROM |
| public-experts | `y12_openrppg_public.py` | Crop public → phổ expert native |
| public-conditions | `y13_openrppg_public_conditions.py` | Crop public theo điều kiện → phổ expert |
| public-score | `z13_public_p4_prefilter.py` | Phổ đủ kênh và POOL → bảng P4/P4m/N4/N4m |

CLI gốc nằm trong docstring mỗi script; `plan` in interpreter, cwd và argv mà
không chạy. Đầu ra ở `workspace/reports/claude_research_20261119_constraints/out`
và `logs`. Một số script gốc tiêu thụ rồi xóa **crop tạm trong workspace**
(x02); vì vậy prepare chép tài sản, không symlink đến cache gốc. Dự toán dung
lượng trước khi chép video/cache lớn. Output mới không được thêm vào Git.

## 6. Phạm vi đã kiểm

Xem [receipt tổng hợp của môi trường mới](CLEAN_ENV_VALIDATION.md). Chưa có phép
chạy đủ raw video → frontend → CNN/expert → reader_v4/P4m trên một máy mới với
bundle riêng. README không coi demo tổng hợp hoặc unit test là bằng chứng cho
phép chạy đầy đủ đó. Checkpoint và tham số hiện có trong Release, nhưng còn cần đúng bundle dữ liệu,
split/cache và hoàn tất kiểm tích hợp; chưa thể hứa "clone rồi tái lập mọi bảng".
