# Đóng băng hai ứng viên reader — 16/11/2026

**reader_v1 và reader_v2 được đóng băng mã, model JSON/MAT và SHA256.** Đây là khóa hai ứng viên để chờ chọn cấu hình cuối, không phải tuyên bố đã qua cổng và không cấp quyền chạy holdout.

Người dùng xác nhận kiểm tra dữ liệu dev qua snapshot đã khóa là đủ, không cần chờ tìm lại XLSX gốc. Cơ sở là **218/218 dòng dev khớp ID video, mã người suy từ ID, hoạt động và PR**; nhãn truth trong các bảng kết quả cũng khớp manifest khóa. Không dùng bảng mới để thay nhãn dev.

## Xuất xứ bảng

- Đã tải lại trực tiếp từ cổng Google Sheets BTC lúc **18:59:46 ICT ngày 16/11/2026** (11:59:46 UTC), không chỉnh nội dung.
- Hash bản hiện hành của BTC: `284392dabe6cef27a181766d0f4d19f04a06af84a0b34efc8c945a07bc8a661f` — trùng bản đang có và lần tải 18:51:10 ICT cùng ngày. Receipt ở FREEZE_FETCH.json và hồ sơ audit trước đó.
- Hash XLSX gốc ghi ban đầu: `e52a9b190bf91f8a5d6d476b8374b8904f47328a668bfbe7cd60f3c14fafc49a`. Ngày tạo/tải chính xác của file gốc chưa xác minh lại được vì không tìm lại file; không suy ngày tải từ tên thư mục. Hồ sơ trích dữ liệu/split đã có trước lần khóa hash ngày 12/11.
- Manifest dev giữ nguyên hash `8a6d1a7db588d71c9cfad420ec6d667efdaad4b0faa8188f940aafe5159a2f7c`, khớp hồ sơ ngày 12/11.

Không có bản XLSX gốc để so byte. Điều này **không làm thay đổi nhãn đã dùng tính số liệu dev**: giá trị phục vụ tính điểm vẫn là snapshot khóa, đã kiểm đối chiếu từng dòng. Không suy luận các ô ngoài 218 dòng dev giống nhau, và không khẳng định nguyên nhân hash toàn file khác. Phạm vi audit này đã được người dùng chấp nhận để đóng băng.

## Khóa hoạt động holdout

`HOLDOUT_ACTIVITY_LOCK.csv` có đúng **72 dòng, hai cột `recording,activity`**. SHA256:

`a0f58cc5b87888aa591ec07de6957d197a954909f85a7db6dfd95d78a664cd76`

Chỉ đọc ID và hoạt động, **không đọc PR, không mở media, không suy luận holdout**. Khi được cho phép đánh giá, bắt buộc kiểm hash file này và dùng cờ đã khóa: Task1/2/3 = Task; Resting/Walking = không-Task. ID/cờ không có trong danh sách phải báo lỗi, không tự đoán hoặc cập nhật. Cờ được khóa trước kết quả holdout.

## Ứng viên và cổng

|Ứng viên|MAE native dev, bpm|Ghi chú|
|---|---:|---|
|reader_v1|10,57015|Không offset hoạt động|
|reader_v2|10,25048|Hai offset đã fit trong train-fold khi đánh giá OOF; model toàn dev chỉ để triển khai|

**Cổng đã ghi trước chưa đạt: 10,25048 > 10 bpm.** Reader_v2 là bản có điểm native dev tốt nhất trong hai ứng viên, không phải bản đã qua cổng. Delta v2−v1 −0,31967, CI95 [−0,85733; +0,22855] vẫn chứa 0. Không chọn lại offset, beta, bandwidth hoặc bất kỳ biến thể nào trên dev.

Mỗi thư mục ứng viên chứa mã, JSON/MAT và receipt static/Octave. JSON và MAT đã kiểm các giá trị prior, beta, grid, offset (nếu có) trùng tuyệt đối. Receipt Octave trên năm ROI thật được giữ; chưa chạy MATLAB thật. Giữ hai thư mục cạnh nhau vì v2 dùng frontend v1 cố định.

## Cách dùng và xác minh

Chạy `python verify_frozen.py` tại root gói để kiểm SHA256 trước khi dùng. Không sửa nội dung gói; thay đổi được cho phép sau này phải tạo phiên bản riêng.

Python v1: `python reader_v1_20261116/reader_v1.py --meta <meta.npz> --model reader_v1_20261116/reader_model_dev218.json --out <prediction.json>`.

Python v2: `python reader_v2_20261116/reader_v2.py --meta <meta.npz> --model reader_v2_20261116/reader_model_dev218_v2.json --task task --out <prediction.json>`; dùng `non-task` khi cờ khóa là Resting/Walking. Cần NumPy/SciPy; dữ liệu đầu vào t/valid/rgb_mean theo frontend đã kiểm.

Octave/MATLAB: v1 `process_round1(roiPath, outputPrefix, modelPath)`; v2 thêm đối số thứ tư `isTask`. Ảnh tĩnh trả prior (v2 cộng offset) kèm abstain, không được diễn giải số dự phòng là mạch đo được.

## Việc tiếp theo

C1/C4 reader_v1 tiếp tục nguyên cấu hình đang chạy; không khởi động lại. Chưa chạy reader_v2 C1/C4; nếu được yêu cầu sau này, dùng đúng offset đã đóng băng, không fit lại. W2/W2b đã đóng theo cổng, hồ sơ T1/T2 được giữ. **Holdout vẫn khóa và chỉ mở khi người dùng xác nhận**, không tự mở vào ngày dự kiến. Chưa có quyết định chọn cấu hình cuối hoặc nộp bài.
