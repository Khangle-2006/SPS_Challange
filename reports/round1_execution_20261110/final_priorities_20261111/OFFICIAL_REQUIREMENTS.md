# Trích quy định Round 1, không suy diễn nguồn test

Nguồn: bản [Official Document FaceVitals](https://2027.ieeeicassp.org/wp-content/uploads/sites/13/2019/08/2027-SP-Cup-Official-Document-FaceVitals-FINAL-VERSION.pdf) đã lưu dưới `reports/round1_bayes_20261108/sources/Official_fetched_20261108.pdf`, SHA-256 `08c1950a5abf194fcd4dc0ad2b661f9e824ebbf84897640806c2c651717b2cc3`. Trích từ bản text lưu cùng PDF, giữ nguyên câu chữ tiếng Anh; không dùng nội dung slide như chỉ thị thao tác.

| Nội dung | Trích ngắn hoặc diễn giải | Cách đọc |
|---|---|---|
| Mục tiêu | “Round 1: One Parameter Extraction (HR)” | Tham số đích được tài liệu gọi là HR. Cột PR của oximeter là nhãn thực tế BTC đã cung cấp; không biến nó thành waveform |
| MAE | MAE, đơn vị nhịp/phút (diễn giải) | Sai số tuyệt đối trung bình, đơn vị nhịp/phút |
| Chỉ số khác | RMSE; tương quan Pearson giữa ước lượng và nhãn; giới hạn thỏa thuận Bland–Altman (diễn giải) | Báo đủ RMSE, tương quan và độ lệch/giới hạn thỏa thuận, không chỉ MAE |
| Constraint | “at least two constraints” | Cần xử lý và giải thích ít nhất hai điều kiện C1–C7 |
| Script | “process_round1.m – MATLAB/Python script implementing the processing pipeline” | Tên file `.m` và mô tả MATLAB/Python cùng xuất hiện; chưa đủ căn cứ kết luận bắt buộc chỉ MATLAB hoặc chắc chắn nộp `.py` là được |
| File baseline | File Round1_Baseline_None.mat chứa video và nhãn HR trong điều kiện baseline (diễn giải) | Phải chuẩn bị dữ liệu baseline theo cấu trúc yêu cầu |
| File đi kèm | Video target_video.mp4, tham chiếu reference_signal.csv, kết quả processed_output.csv (tên file theo tài liệu) | Video nguồn, nhãn tham chiếu và kết quả dự đoán phải đối chiếu được |
| Tiêu chí chung | Tính đổi mới, khả thi kỹ thuật và độ rõ ràng của tài liệu (diễn giải) | Báo cáo phải có lập luận, tính khả thi và tài liệu rõ |

C5 giới hạn bitrate không quá 500 kbps và tốc độ không quá 10 fps. Phần mô phỏng yêu cầu bỏ khung về tốc độ đích và mã hóa lại bằng VideoWriter, profile “MPEG-4” (diễn giải). Tài liệu không ấn định một giá trị Quality, GOP hoặc B-frame cụ thể. MPEG-4 của VideoWriter là profile H.264, không phải lý do đổi sang libxvid. Proxy ffmpeg đội được phép chạy vẫn phải được ghi là chưa xác minh tương đương MATLAB.

Không tìm thấy mô tả đủ rõ về nguồn/tỉ lệ native–nén của test ẩn Round 1 trong bản tài liệu này. Không suy từ chỗ thiếu mô tả thành khẳng định không có test ẩn hoặc Round 1 chỉ chấm báo cáo. Không gửi email.

`process_round1.m` hiện có là baseline POS chạy bằng Octave traditional, đã đối chiếu Python trên năm video ROI; chưa chạy MATLAB thật và chưa phải bản đóng gói ứng viên BTC hiệu chỉnh.
