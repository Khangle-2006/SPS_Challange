> Archived aggregate report. Historical pending/status sentences describe its creation date, not current status. See the root README for the 15 November disposition. Private artifacts named here are not bundled.

# Phép thử ba đỉnh phổ trên BTC dev

## Câu hỏi và cách đọc

Có phải nhịp gần số PR thật vẫn nằm ở đỉnh thứ hai hoặc thứ ba, nhưng bước chọn đỉnh mạnh nhất đã bỏ qua? Mỗi video được giữ tối đa ba đỉnh, chọn hoàn toàn từ tín hiệu, cách nhau ít nhất 6 bpm. Sau khi khóa danh sách này mới đối chiếu với PR.

**Oracle** nghĩa là dùng đáp án PR để chọn ứng viên gần nhất. Đây là phép thử khả năng của tập ứng viên, không phải mô hình có thể dùng khi chưa biết đáp án. MAE là sai lệch tuyệt đối trung bình, đơn vị bpm; tính trung bình từng người trước để người có nhiều video không lấn át.

**Xáo nhãn** là đổi nguyên khối PR giữa những người có cùng số video, lặp 100 lần. Cột trong ngoặc là phân vị 5–95 của 100 kết quả, không phải khoảng tin cậy 95% của mô hình. Cùng danh sách đỉnh được dùng cho nhãn thật và nhãn xáo.

## 1. Sai số oracle

|Nhóm|Video / người|Video có ứng viên|Oracle thật (bpm)|Oracle xáo: trung bình [P5–P95] (bpm)|
|---|---:|---:|---:|---|
|Toàn bộ|218 / 158|218|7.55|8.23 [7.74; 8.83]|
|Normal — nghỉ|62 / 62|62|6.83|7.52 [6.52; 8.79]|
|Abnormal — nghỉ|76 / 76|76|7.53|7.90 [6.78; 8.97]|
|Làm nhiệm vụ|75 / 19|75|10.71|12.01 [10.64; 13.41]|
|Đi bộ|5 / 5|5|2.00|6.75 [2.50; 11.81]|

## 2. Bao nhiêu video có một đỉnh gần PR?

Phủ ±5 bpm nghĩa là có ít nhất một ứng viên lệch PR không quá 5 bpm. Video không có ứng viên được tính là không phủ. Các con số dưới đây là phần trăm video.

|Nhóm|Phủ±5 thật (%)|Phủ±5 xáo: TB [P5–P95] (%)|Phủ±10 thật (%)|Phủ±10 xáo: TB [P5–P95] (%)|
|---|---:|---|---:|---|
|Toàn bộ|47.71|40.79 [35.78; 46.38]|75.69|69.59 [65.11; 73.90]|
|Normal — nghỉ|46.77|43.81 [32.26; 54.84]|85.48|75.90 [69.27; 83.87]|
|Abnormal — nghỉ|53.95|43.89 [35.53; 52.63]|77.63|74.46 [65.79; 81.58]|
|Làm nhiệm vụ|38.67|34.92 [26.67; 42.67]|64.00|58.52 [52.00; 65.33]|
|Đi bộ|100.00|44.40 [19.00; 100.00]|100.00|83.40 [60.00; 100.00]|

## 3. Hai mốc thực tế trên cùng video có ứng viên

Hằng số là trung vị PR ở bốn fold train, không dùng PR của người đang chấm. Trung vị hiệu chỉnh là phương án đang giữ, đã lưu từ trước. Oracle được phép nhìn đáp án nên không thể đem nó làm kết quả thắng hai phương pháp này.

|Nhóm|Phương pháp|Video / người|MAE (bpm)|CI95 bootstrap theo người|
|---|---|---:|---:|---|
|Toàn bộ|Hằng số train-fold|218 / 158|12.07|[10.58; 13.68]|
|Toàn bộ|Trung vị hiệu chỉnh|218 / 158|11.37|[9.92; 12.89]|
|Normal — nghỉ|Hằng số train-fold|62 / 62|8.66|[7.05; 10.39]|
|Normal — nghỉ|Trung vị hiệu chỉnh|62 / 62|9.18|[7.51; 10.90]|
|Abnormal — nghỉ|Hằng số train-fold|76 / 76|14.20|[11.83; 16.75]|
|Abnormal — nghỉ|Trung vị hiệu chỉnh|76 / 76|12.71|[10.41; 15.12]|
|Làm nhiệm vụ|Hằng số train-fold|75 / 19|14.80|[10.23; 20.29]|
|Làm nhiệm vụ|Trung vị hiệu chỉnh|75 / 19|12.95|[8.74; 18.10]|
|Đi bộ|Hằng số train-fold|5 / 5|12.40|[6.00; 19.20]|
|Đi bộ|Trung vị hiệu chỉnh|5 / 5|10.76|[4.66; 18.13]|

## 4. Cổng đã ghi trước

|Điều kiện|Yêu cầu|Đo được|
|---|---:|---:|
|Oracle tốt hơn trung bình xáo|Ít nhất 4 bpm|0.69 bpm|
|Phủ±5 cao hơn xáo|Ít nhất 15 điểm phần trăm|6.91 điểm phần trăm|

**Kết quả: KHÔNG ĐẠT**. Phải đạt đồng thời cả hai điều kiện.

Không triển khai bộ đọc theo nhánh điều kiện này. Giữ bộ chọn và ngưỡng đã khóa; chờ V5 có kết luận cuối trước khi đóng băng.

## 5. Giới hạn cần giữ trong báo cáo

Phân tầng số video/người: {'1': 135, '2': 4, '6': 6, '3': 13}. Tỷ lệ người đổi khối nhãn trung bình: 97.34%. Nếu một tầng chỉ có một người, khối đó không thể đổi; không giả vờ rằng mọi người đều được xáo.

Phổ được tính từ trace RGB đa ROI đã lưu, không đọc lại video và không dùng holdout. Bộ trích đa ROI này khác canonical oval đã sửa trên public. Kết luận chỉ áp dụng cho tối đa ba đỉnh và cách gộp phổ đã khóa. Không chứng minh mọi thông tin nhịp đã mất, không xác định nguyên nhân do nhãn hay codec. Một oracle thấp cũng không chứng minh có bộ đọc học được cách chọn đúng đỉnh.

V5 và sao lưu P100 tiếp tục độc lập. Không mở thêm GPU hoặc holdout. Định nghĩa, hash, danh sách ứng viên và 100 lần hoán vị nằm trong GATES.md, PREREGISTRATION.json, CANDIDATES_FROZEN.json, RESULTS.json và PERMUTATIONS.json.

## 6. Diễn giải và kiểm tra độc lập

Trên 218 video của 158 người, cả 218 video đều có đủ ba ứng viên. Oracle thật đạt 7,55 bpm (CI95% bootstrap theo người: 6,44–8,75), so với 8,23 bpm khi xáo nhãn. Độ phủ ±5 bpm là 47,71% so với 40,79%; độ phủ ±10 bpm là 75,69% so với 69,59%. Như vậy có một lợi thế nhỏ so với khớp tình cờ, nhưng chưa đạt kích thước hiệu quả đã yêu cầu để xây bộ đọc.

Các p-value hoán vị mô tả là 0,0396 cho oracle và 0,0297 cho phủ ±5. Chúng không thay thế hai ngưỡng hiệu quả đã khóa, không dùng để mở cổng sau khi thấy kết quả. Không nên viết rằng tín hiệu hoàn toàn bằng không hoặc mọi thông tin nhịp đã mất.

Nhóm Walking chỉ có 5 video / 5 người. Không chọn riêng nhóm tốt nhất để thay kết luận toàn bộ. Người xuất hiện ở nhiều loại tình huống được tính trong từng nhóm tương ứng; không cộng số người giữa các nhóm để suy ra tổng.

Chỉ đọc các trace BTC dev đã lưu. Tiến trình chuẩn bị đọc PR để đối chiếu đúng cohort và giữ hai baseline; tiến trình trích đỉnh riêng không nhận PR. Danh sách ứng viên được khóa bằng SHA256 trước bước chấm với PR. Xem EXECUTION_CLARIFICATION.md để phân biệt điều này với tuyên bố mạnh hơn rằng không tiến trình nào từng mở nhãn trước khi trích.

Phổ được lấy mẫu trên lưới 0,5 bpm để tính số; điều này không tạo độ phân giải vật lý 0,5 bpm. Cửa sổ 10 giây chỉ có độ phân giải tần số cơ bản khoảng 6 bpm.

Đã kiểm lại SHA256 của gói kết quả, hash mã nguồn/gates đã khóa và tự tính lại MAE theo người cùng hai tỷ lệ phủ từ danh sách ứng viên. Tất cả khớp. Không tính lại phổ hoặc thay cách chọn đỉnh sau khi xem nhãn.

Hoàn tất lúc 07:32:56 ICT ngày 12/11/2026. Nhánh bộ đọc dừng theo cổng; V5 và sao lưu P100 vẫn là hai công việc riêng đang tiếp tục. Holdout: 0 lần đánh giá.

Tệp tái lập: core.py (trích và thống kê), run.py (điều phối), GATES.md (quy tắc), PREREGISTRATION.json (hash trước chạy), CANDIDATES.json (ba đỉnh mỗi video), CANDIDATES_FROZEN.json (hash trước chấm), PERMUTATIONS.json (100 hoán vị), RESULTS.json và COMPLETION_REVIEW.json. render_report.py chỉ định dạng lại số đã lưu; không chạy lại phép thử.
