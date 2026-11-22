# W2b và reader — cập nhật 16/11/2026

**W2 cũ đã dừng. Reader native 10,570 bpm là ứng viên BTC theo quyết định của người dùng.** Đây là kết quả dev, chưa phải holdout. Cổng thành công ≤10 bpm vẫn chưa đạt.

## 1. Phép thử W2

Đã giữ nguyên các lượt hoàn tất và lượt dở; hủy hàng chờ W2 cũ và bước ghép phổ W2.

T1 chỉ kiểm tra mô hình có học thuộc được 16 clip train cố định hay không. Không dùng nó làm bằng chứng tổng quát hóa. Batch hiệu dụng 16, chia hai microbatch 8 để giữ giới hạn VRAM.

|Bước|Weak-CE chế độ eval trên 16 clip|Weak-CE train của bước vừa chạy|MAE readout trên 16 clip (bpm)|
|---:|---:|---:|---:|
|0|5,266863|—|19,403198|
|100|3,423835|3,805221|0,876465|
|200|3,260529|3,578791|0,618042|
|400|3,198261|3,365320|0,389648|

Phổ phẳng có CE 5,056246; phân bố đích lý tưởng có CE 3,157451. CE thấp hơn là tốt hơn. T1 qua điều kiện <4,0 bằng cả loss train và eval. Cặp W2b fold 2 nhãn thật/nhãn xáo đã hoàn tất; nhánh W2b đã đóng vì không qua cổng.

T2, cùng trọng số/BN ban đầu trên một batch:

|Phần mô hình|L2 gradient phase loss|L2 gradient weak CE|Tỉ số weak/phase|
|---|---:|---:|---:|
|Toàn mô hình|3,179056|16,737888|5,265050|
|final_dense_2|2,189098|15,949274|7,285775|

Gradient weak CE không nhỏ hơn phase loss trong batch đã đo. Kết quả này không tự xác định nguyên nhân W2 không học. W2b cập nhật hai optimizer riêng theo yêu cầu, không thay các tham số khác.

**Kết quả đã xác minh lúc 09:05 ICT ngày 16/11:**

|Fold 2 — 42 video/32 người validation|Nhãn thật|Nhãn xáo|
|---|---:|---:|
|MAE validation (bpm)|17,432851|17,672521|
|Weak-CE train epoch 1|5,308940|5,485891|
|Weak-CE train epoch 10|5,262399|5,423419|
|MAE 40 video train cố định, epoch 1|18,594813|20,178596|
|MAE 40 video train cố định, epoch 10|18,727266|18,714513|
|Kiểm tra ảnh tĩnh|10/10 PASS|10/10 PASS|

Nhãn thật chỉ tốt hơn nhãn xáo **0,239670 bpm**, thấp hơn ngưỡng 1 bpm. Chênh lệch thật trừ xáo −0,239670, CI95 [−2,044661; +1,458253]. Weak-CE nhãn thật chỉ giảm **0,046541**, thấp hơn ngưỡng 0,3. Cả hai điều kiện mở rộng đều trượt. Controller đã thoát; các fold 0/1/3/4 W2b không được khởi chạy. Không có kết quả W2b toàn bộ 218 video. Xem [W2B_REPORT.md](W2B_REPORT.md) cho bốn nhóm, baseline trên cùng video và đường loss từng epoch.

## 2. Reader với cổng beta

**Prior và beta được fit lại trong từng điều kiện, trên bốn fold train.** Không dùng lại beta native cho C5. Hàng calibrated ở báo cáo chi tiết là dự đoán native lịch sử trên cùng video, không phải bản hiệu chỉnh fit riêng cho điều kiện nén.

Chỉ giữ beta dương nếu MAE train theo người giảm ít nhất 0,5 bpm so với beta 0. Không đổi cửa sổ, bandwidth, lưới beta hoặc prior. Native giữ nguyên toàn bộ 218 dự đoán; beta từng fold vẫn là 3, 2, 2, 3, 3. Mức giảm trên train lần lượt 1,473; 1,130; 1,819; 1,719; 1,380 bpm.

|Điều kiện|Video/người|Reader sau cổng, MAE bpm|Δ so hằng số [CI95], bpm|Beta 5 fold|
|---|---:|---:|---|---|
|Native|218/158|10,570148|−1,500527 [−2,792748; −0,191443]|3 / 2 / 2 / 3 / 3|
|C5 cũ|218/158|12,073840|+0,003165 [−0,280617; +0,290084]|0 / 0 / 0 / 0 / 0|
|C5′ default|161/109|12,951070|+0,146789 [−0,162080; +0,449541]|0 / 0 / 0 / 0 / 0|
|C5′ bf0|158/106|13,278302|+0,205975 [−0,169851; +0,578616]|0 / 0 / 0 / 0 / 0|

MAE tính trong từng người rồi trung bình. CI bootstrap ghép cặp theo người 10.000 lần, seed 0. Số âm của Δ nghĩa là reader tốt hơn. Các điều kiện nén đều dùng prior, không còn tin phổ. Prior KDE có trung vị hơi khác trung vị PR thô của predictor hằng; vì vậy hai dòng không nhất thiết bằng nhau. Không ép chúng bằng nhau bằng một thay đổi thuật toán mới.

Đây là bằng chứng reader hiện tại không khai thác phổ hiệu quả trong các điều kiện nén đã thử; không chứng minh mọi thông tin mạch đều bị xóa. Các hàng có độ phủ khác nhau không được so trực tiếp để quy nguyên nhân codec. C5′ là bản ffmpeg mô phỏng, chưa xác minh tương đương MATLAB VideoWriter.

MAE, bias, Spearman và bốn nhóm nằm trong [GATED_REPORT.md](GATED_REPORT.md). Dự đoán, beta và độ giảm train từng fold nằm trong GATED_RESULTS.json.

## 3. C1/C4 đang chạy

218 video dev, năm điều kiện: C1 gain 0,5 / 0,25 / 0,125; C4 QVGA / VGA. Bốn shard CPU biến đổi toàn khung, dò mặt lại, giữ đồng hồ nguồn và dùng đúng cổng beta trên. Lúc 09:05 ICT đã có 24/1.090 mục video–điều kiện hoàn tất; cả bốn worker còn chạy, chưa ghi lỗi. Đây là tiến độ xử lý, chưa phải bảng đánh giá đầy đủ.

C1 trong simulate.py còn có codec lưu trữ x264 CRF18; không được diễn giải là tác động nhiễu thuần túy. Kết quả C1/C4 chưa có. Bảng cuối C_REVIEWED_REPORT.md sẽ báo độ phủ, bias, Spearman và Δ ghép cặp với hằng số/reader native trên cùng video.

Holdout 72 video/41 người vẫn khóa, số lần đánh giá bằng 0. Không tinh chỉnh reader thêm.
