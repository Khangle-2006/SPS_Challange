# Reader_v2: thêm hai offset theo hoạt động

Kết quả trên **218 video dev / 158 người**: MAE **10,25048 bpm**, bias **+0,18085 bpm**, Spearman **0,49133**. MAE tính trong từng người rồi lấy trung bình; số thấp hơn tốt hơn. Bias dương nghĩa là dự đoán cao hơn nhãn. Spearman mô tả mức đồng biến giữa dự đoán trung bình và nhãn trung bình theo người, không phải độ chính xác tuyệt đối.

|Đối chứng trên cùng video|MAE (bpm)|Reader_v2 trừ đối chứng [CI95]|
|---|---:|---|
|Hằng số train-fold|12,07068|−1,82020 [−3,03373; −0,59080]|
|Hiệu chỉnh đã lưu|11,37385|−1,12337 [−2,08414; −0,18328]|
|Reader_v1|10,57015|−0,31967 [−0,85733; +0,22855]|
|Reader_v2|10,25048|—|

CI bootstrap ghép cặp theo người, 10.000 lần, seed 0. Reader_v2 hơn hai mốc đầu với CI không chứa 0. So với reader_v1, CI còn chứa 0: mức giảm 0,320 bpm chưa đủ để khẳng định cải thiện chắc chắn. Cổng MAE ≤10 bpm vẫn chưa đạt. Không tự đổi ứng viên hoặc đóng băng cấu hình; chờ người dùng chốt cùng C1/C4.

## Bảng chú thích và cách fit

Bản bảng BTC hiện có cột **PROTOCOL** ở cả Normal và Abnormal. Đã xác nhận thông tin hoạt động hợp lệ cho đủ **72/72 video holdout**, chỉ đọc cột ID và hoạt động. Không đọc PR, mở video hoặc đánh giá holdout. File bảng hiện tại có hash khác bản gốc đã khóa; kiểm tra này chỉ xác nhận metadata có sẵn, không thay nguồn nhãn dev hoặc xác nhận giao diện test ẩn.

Với mỗi fold, reader và hai offset đều học từ bốn fold train. Đầu tiên dùng reader vừa fit để dự đoán chính video train; sau đó tính trung bình có trọng số của `(PR − dự đoán)`, riêng nhóm Task và nhóm không-Task. Mỗi người có tổng trọng số bằng nhau trong từng nhóm. Offset chỉ được cộng vào dự đoán fold validation sau khi đã fit xong. Không fit offset trên kết quả OOF, không cắt ngưỡng dự đoán, không tìm biến thể mới. Đây là offset từ residual train in-sample theo yêu cầu, không phải residual inner-OOF.

|Fold validation|Beta|Offset Task (bpm)|Offset không-Task (bpm)|
|---:|---:|---:|---:|
|0|3|+13,42157|+3,28409|
|1|2|+12,09375|+2,49773|
|2|2|+11,54167|+2,54318|
|3|3|+11,88095|+3,05752|
|4|3|+7,71795|+3,99779|

Tên `offset_rest` trong mã nghĩa là **không-Task**, gồm cả Resting và Walking. Task1/2/3 cùng dùng một cờ Task. Bảng theo Normal-Resting / Abnormal-Resting / Task / Walking, gồm đầy đủ MAE, bias, Spearman và các CI, nằm trong REPORT.md.

## Đối chứng

- **Xáo nhãn theo người, 20 lần:** MAE 12,03634 trong cả 20 lần; beta đều bằng 0. Giữ nguyên chính sách xáo block người trong cùng số video, như reader_v1. Có 13–25 block người-fold giữ nguyên mỗi lần. Null không kiểm tra mọi kiểu hoán vị; độ phân tán bằng 0 không phải bằng chứng độc lập rất mạnh. Không chỉnh mã để ép null ra khoảng 12.
- **Xáo cờ Task, 20 lần:** MAE trung bình 10,60673; phân vị 5–95 là [10,42577; 10,72989], toàn khoảng [10,37530; 10,74075]. Lợi ích bổ sung so với reader_v1 mất đi ở mức trung bình, không phải mọi phép xáo đều cho số tệ hơn reader_v1. Cờ được xáo riêng trong train và validation mỗi fold; reader nền giữ nguyên.
- **Ảnh tĩnh: 20/20 PASS.** Không có phổ, trả prior trung vị cộng offset đã fit theo cờ, kèm `abstain=true`. Giá trị đó là dự đoán dự phòng theo hoạt động, không được gọi là đo được mạch trên ảnh tĩnh. Chi tiết STATIC.json.
- **Octave: 5/5 ROI thật PASS**, kiểm cả hai cờ trên mỗi ROI (10 lần đối chiếu), sai khác Python dưới ngưỡng 0,5 bpm. Chưa chạy MATLAB thật. Chi tiết OCTAVE_PARITY.json.

C1/C4 tiếp tục chạy nguyên cấu hình reader_v1 với cổng beta đã khóa; lần này không đổi worker hoặc tự chạy thêm C1/C4 cho reader_v2. Không dùng kết quả này để mở holdout.

## Chạy lại và triển khai

`python reports/reader_v2_20261116/run_reader_v2.py` dùng native_spectra.npz và PREDICTIONS.json đã lưu ở reader_v1_20261116. Mã không truy cập media holdout hoặc nhãn holdout. `check_schema.py` chỉ kiểm tra hai cột metadata theo phạm vi đã duyệt.

Bản `process_round1.m` được lưu riêng trong thư mục reader_v2_20261116; không ghi đè bản v1. Gọi `process_round1(roiPath, outputPrefix, modelPath, isTask)` với cờ hoạt động tường minh. `reader_model_dev218_v2.json`/MAT là bản fit toàn dev để đóng gói, không dùng tính số OOF trên đây. Octave kiểm năm ROI thật, cả hai nhánh cờ trên mỗi ROI; không thay thế kiểm chứng MATLAB thật.
