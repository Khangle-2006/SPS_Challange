# Chọn video và lấy PR

Lệnh mới `predict.py` nối video → FaceMesh/crop → POS + CHROM + ba expert
→ BTC joint → JSON/CSV. Nó không cần `private-assets/assets.json`, nhãn PR,
dataset của đội, GPU hay đường dẫn `/mnt/...` cố định.

## Cài một lần

Máy được hỗ trợ: Linux x86_64, Python 3.12, tối thiểu 15 GiB RAM **còn trống**.
Cần Internet khi cài dependency; sau khi cài xong, suy luận không gọi dịch vụ từ xa.
Lệnh apt dưới đây dành cho Ubuntu 24.04 (máy kiểm dùng Python 3.12.3).
Người quản trị cài các gói hệ thống nếu máy chưa có:

```bash
sudo apt-get update
sudo apt-get install git python3.12-venv ffmpeg libgl1 libegl1 libglib2.0-0t64
git clone https://github.com/Khangle-2006/SPS_Challange.git
cd SPS_Challange
python3.12 tools/bootstrap_video.py
```

Bootstrap tạo ba môi trường riêng (`core`, `frontend`, `experts`), kiểm đúng
phiên bản đã khóa và SHA256 hai file trọng số cùng trạng thái khởi tạo của ba
expert trong open-rppg. Trạng thái được reset trước mỗi video, rồi expert xử lý chính video đó hai lượt và lấy lượt hai (self-warm). Video chạy trước không được ảnh hưởng video chạy sau.
Cấu hình reader nhẹ nằm sẵn trong `models/`; không cần tải checkpoint CNN của
đội để tính PR của nhánh BTC. Bootstrap vẫn tự tải đủ gói R1 để có sẵn 5 checkpoint public; kiểm hash và reader đúng checkout. Môi trường sai phiên bản sẽ báo lỗi, không sửa ngầm.

## Chạy video của bạn

```bash
python3.12 predict.py \
  --video /duong/dan/video.mp4 \
  --activity Resting \
  --output output/video_cua_toi
```

`--activity` bắt buộc: chọn `Resting` (nghỉ), `Task` (làm tác vụ) hoặc `Walking`
(đi bộ) theo lúc quay. Reader có offset theo hoạt động nên không được đoán cờ
này từ kết quả hay tên file. `Resting` và `Walking` cùng thuộc nhóm không-Task
trong phép tính đã khóa. Khi không biết hoạt động, chưa có một cấu hình được
kiểm chứng cho cờ “không biết”.

Dùng video gốc khoảng 20–30 giây, có khuôn mặt nhìn thấy. Khoảng timestamp từ
khung đầu đến khung cuối phải ít nhất 10 giây; file ghi thời lượng đúng 10 giây
có thể thiếu khung cuối cho cửa sổ này. Không đệm để bù. Nếu video đã giảm
độ phân giải/nén lại, cần truyền bitrate **stream video nguồn**:

```bash
python3.12 predict.py --video /path/derivative.mp4 --activity Task \
  --source-bitrate 7700000 --output output/derivative
```

Nếu ffprobe không có bitrate stream, lệnh dừng và yêu cầu giá trị này; không
thay bằng tổng bitrate file hay chọn loại nguồn tùy ý. Video HDR dùng chuỗi
HLG/PQ → SDR đã lưu; xoay display metadata chỉ thực hiện một lần. Thời gian lấy
từ PTS thật, không lấy chỉ số frame chia fps mặc định.

Mặc định `reference` dùng đường một mặt đã kiểm khớp. Chỉ dùng mặc định cho
video một người. `guarded` là tùy chọn thử nghiệm giữ danh tính; nó chưa qua
cổng tương đương từng video (một trong ba video kiểm lệch +22 bpm khi phục hồi
một frame). Không xem nó là chế độ đã xác nhận cho kết quả nghiên cứu. Nếu
chủ động thử video nhiều người, truyền hộp người cần đo trong tọa độ pixel
của ảnh sau xoay display:

```bash
python3.12 predict.py --video /path/group.mp4 --activity Resting \
  --frontend guarded --seed-box 100 60 340 380 --output output/target
```

Không tự chọn lại người khi mất dấu. Khi không đủ cửa sổ hợp lệ, kết quả gắn cờ
`abstain`. Không dùng video thiếu mặt/ảnh tĩnh để diễn giải một số prior thành
mạch đo được.

## Đọc kết quả

- `result.json`: `pr_bpm`, `abstain`, hoạt động, loại nguồn, độ phủ frame,
  số cửa sổ, hash video/mã/reader/trọng số và thời gian chạy.
- `result.csv`: một dòng kết quả tiện đưa vào bảng.
- `frontend.log`, `experts.log`, `core.log`, `decode.log`: log các bước.
- `trace.jsonl`: nhánh native/switch/held/invalid và hộp theo frame.
- `trace.npz`, `crop72.npy`: dữ liệu trung gian **riêng tư**. Mỗi thư mục kết quả
  có `.gitignore` riêng; không đưa chúng lên GitHub.

Khi `abstain=true`, `pr_bpm=null`. Trường `branch` cho biết nguồn tạo PR hoặc nhánh bù bitrate thấp. Dùng `--reader r1` để chạy reader_v4 lịch sử.

## Tái lập kết quả nghiên cứu khác với chạy video mới

Mặc định dùng model fit một lần trên toàn bộ 218 video dev bằng đúng hàm fit,
beta grid và offset đã chọn. Đây là model triển khai, không được dùng nó để
chấm lại dev rồi gọi là OOF. Để chấm một video dev cần đúng fold người:

```bash
python3.12 predict.py --video /path/authorized_dev_video.mp4 --activity Resting \
  --frontend reference --fold 2 --output output/dev_fold2
```

`reference` dùng FaceMesh một mặt và crop lịch sử, dành riêng cho tái lập.
Nó không cung cấp bộ gác nhiều người. Không chọn chế độ theo PR dự đoán hoặc nhãn; mặc định reference được
đặt chung cho mọi video một mặt. Bộ gác khôi phục/loại một số frame nên có thể đổi dự đoán; cổng
khớp từng video của lần tích hợp 22/11 còn chưa đạt. Không đổi ngưỡng đó trong
bản đóng gói này. Số MAE lịch sử 8,350 còn dùng expert giữ trạng thái từ video
trước; không phải kết quả của CLI R1 self-warm. Điểm dev R1 là 8,333 bpm; con số này vẫn
không phải cam kết sai số cho video mới hoặc điểm trên toàn bộ frontend guarded.

BTC joint mặc định đạt 8,020 bpm trên dev ngoài fold. Trên video public dùng `predict_public.py`.

Dataset trên server: [SERVER_DATASETS.md](SERVER_DATASETS.md). Chỉ dùng video
dev trong split đã cấp quyền khi tái lập; **holdout vẫn khóa**. CLI không tải
dataset, đọc Excel nhãn, quét root dữ liệu hoặc tự chấm holdout.
