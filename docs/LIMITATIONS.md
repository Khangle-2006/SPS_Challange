# Giới hạn và trạng thái bản công bố

1. Kết quả trong README lấy từ thí nghiệm đã chạy; bản công bố code này không chạy lại training, inference trên người hoặc holdout. Kiểm phát hành chỉ là syntax, hash, scorer và smoke test tổng hợp được ghi riêng.
2. Không kèm raw video, PR/PPG cá nhân, danh sách split người, trace sinh lý, hình mặt, checkpoint và manifest chứa thông tin người. Tái lập chính xác cần những tài sản được cấp quyền này; xem hướng dẫn và schemas. Không giả rằng clone là có đủ tài sản.
3. Nhiều script nghiên cứu là snapshot gắn với máy chủ Linux và các receipt được hash. Chúng được giữ nguyên để truy vết, không sửa mọi đường dẫn để làm mất liên hệ với kết quả. Không chạy launcher/controller lịch sử trên máy thật trước khi cấu hình môi trường và output độc lập.
4. Các cấu hình bị loại vẫn có trong mã: V2 chưa launch, joint trước sửa polarity không dùng cho so sánh harmonized, V4 padding gốc trượt ảnh tĩnh, SiNC/weak-label bị loại, V5 không thắng. Không hiểu file còn tồn tại là được khuyến nghị dùng.
5. C5′ chưa xác minh MATLAB VideoWriter; `process_round1.m` chỉ nhận ROI đã tính sẵn và là POS baseline. Octave parity không là MATLAB parity, cũng không là parity của BTC calibrated estimator.
6. Dev được dùng cho nhiều thử nghiệm; CI ghép cặp không xóa thiên lệch lựa chọn nhiều lần. Holdout vẫn đóng. Không có điểm test ẩn hay kết quả cuộc thi chính thức trong repo.
7. Kết quả codec/HLG/rotation chỉ suy ra trong điều kiện có support. Thang stage4 mất tracking, các bậc sau không tách riêng được hiệu ứng lên HR. Chưa có cặp webcam–điện thoại thu đồng thời nên không khẳng định lỗi do khâu thu.
8. Phép thử ba đỉnh phổ không qua cổng chỉ giới hạn bộ trích cố định đã thử; không chứng minh mọi thông tin mạch đều biến mất.
9. Full calibrated estimator chưa có kiểm soát ảnh tĩnh end-to-end được xác nhận. Thành phần V3 có vấn đề static; giữ 11,37 là disposition nghiên cứu, không là chứng nhận sẵn sàng triển khai.
