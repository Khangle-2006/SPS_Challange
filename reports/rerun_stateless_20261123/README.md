# R1: tách trạng thái open-rppg, 23/11/2026

Code khoa học và bảng aggregate của lượt chạy 63 model-condition. [Báo cáo đọc được](../../docs/RESULTS.md). Các CSV không chứa nhãn hoặc dự đoán cá nhân. `PUBLIC_FITS.json` ghi toàn bộ weights/readout theo fold của P4, P4m, N4, N4m. Tham số P4m deploy nằm trong `models/public_p4m_r1.json` ở root repo.

Các script là snapshot của lượt P100, cần private manifests/crop/cache tại các đường dẫn ghi trong code. **Không chạy chúng lên thư mục output đã có**. Không phải lệnh chạy một video mới; lệnh đó là `predict.py` ở root. Tài sản riêng không kèm GitHub và holdout không được đọc.

Trình tự đã chạy: `prepare.py` khóa cohort/hash → kiểm cả reset/self-warm/A-B-A → `worker.py` suy luận từng điều kiện → `score_btc.py` và `public_score_wrapper.py` → `public_freeze_wrapper.py` → `report.py`. Chỉ đổi đường phổ trong hai snapshot z13/z14; không mở rộng grid hoặc đổi toán tử. Native BTC tái sử dụng đủ 218 phổ z17 sau quyết định ADOPT self-warm.

Thư mục triển khai R1 gốc: `/mnt/samsung_t5/khangle/facevitals/reports/rerun_stateless_20261123`. Môi trường inference là `venv_orppg` của W, scoring là `/data_pool/khangle/facevitals/envs/hr`. CPU, nice19, hai luồng BLAS, kiểm RAM/dung lượng trước khởi chạy. Không huấn luyện hoặc mở holdout. Các script điều phối/receipt có PID, đường riêng và output cá nhân được giữ ngoài repo.

Các số cũ có trạng thái liên video chỉ là cột kiểm toán, không phải baseline hiệu năng hợp lệ. Chọn 3/17 mô hình trên phổ cũ chưa được làm lại.
