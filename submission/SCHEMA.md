# FaceVitals Round1 — schema kỹ thuật đề xuất

Nguồn chính thức: [FaceVitals 2027, trang 10–11](https://signalprocessingsociety.org/sites/default/files/2026-10/2027-SP-Cup-Official-Document-FaceVitals.pdf). Trang 10 nêu gói tái lập; trang 11 liệt kê sáu biến MAT và ví dụ cây thư mục. Tài liệu chưa quy định dtype, thứ tự trục, cấu trúc con hoặc phiên bản MAT; các lựa chọn dưới đây là triển khai của nhóm.

| Trường | Layout nhóm triển khai |
|---|---|
| facial_video | uint8 `[H,W,3,T]`, RGB, toàn bộ frame hợp lệ theo adapter; mặc định nhúng trực tiếp v7.3 |
| roi_data | struct: forehead/cheek_left/cheek_right/periocular `[T,K,2]`, pixel x,y; padding NaN, polygon_lengths, valid và timestamps_s. Geometry không bảo đảm pixel là da, nhất là tóc mái. Placeholder ghi rõ bbox_placeholder |
| reference_signal | struct: ppg, timestamps_s, valid, source, fs_hz hiệu dụng; PPG nguyên giá trị và clock adapter |
| processed_signal | struct: rppg, timestamps_s, fs_hz=30, hr_bpm_per_window, window_start_s, abstain, window_index_per_sample |
| metrics | MAE/RMSE/Pearson, bias, LoA, coverage, n_windows/n_evaluated/n_subjects; không ước lượng được thì chuỗi NA |
| params | resolution `[W,H]`, fps_nominal/effective, lux/distance/Fitzpatrick/age/ITA là NA nếu chưa có, constraints_applied, dataset/subject, codec/bitrate, pipeline/git/checkpoint hash và timestamp video |

HDF5 chứa trục đảo `[T,3,W,H]` và MATLAB_class uint8 để MATLAB/mat73 hiển thị `[H,W,3,T]`. Struct được ghi bởi hdf5storage với MATLAB compatibility. Đọc toàn file bằng mat73, kiểm tra dtype/shape và hash RGB từng frame. Octave chỉ kiểm tra nếu sẵn có; không suy ra tương thích Octave từ mat73.

Waveform dự đoán ghép các đoạn 10 giây, mỗi đoạn tích phân sai phân từ 0; `window_index_per_sample` phân biệt đoạn và không cho phép hiểu đường nối là tín hiệu sinh lý liên tục. Cửa sổ reject vẫn có hàng trong CSV và mẫu số coverage. HR tham chiếu tính từ PPG với bộ giải mã cố định; không dùng để chọn peak dự đoán. Model input vẫn Haar36; T3 polygons lưu song song. Metadata thiếu không được suy đoán.

`target_video.mp4` là bản RGB lossless dùng đóng gói, kiểm tra bit-exact với nguồn và MAT. Nó dùng nominal CFR; clock sensor thật nằm ở `video_timestamps.csv` và params.video_timestamps_s. Bitrate trong params là bitrate thực của file đóng gói, **không** đại diện điểm vận hành C5; manifest T5 giữ thông số constraint của video input. MCD native đã nén; raw RGB trong MAT là pixel giải mã, không thể gọi là dữ liệu cảm biến chưa nén.

## Đo lặp lại và giới hạn đồng thuận

Với sai số `d_ij=HR_est-HR_ref`, mô hình `d_ij=mu+b_i+e_ij`. Gọi k là số người, N số cửa sổ hợp lệ, n_i số cửa sổ/người. Tính `MSW=sum_ij(d_ij-mean_i)^2/(N-k)`, `MSB=sum_i n_i(mean_i-mean_all)^2/(k-1)`, `n0=(N-sum_i n_i^2/N)/(k-1)`. Phương sai giữa người `max(0,(MSB-MSW)/n0)`; trong người `MSW`. Bias là mean_all; LoA là bias ±1,96√(variance_between+MSW). Cờ riêng khi phải chặn phương sai âm; NA nếu k<2 hoặc N≤k. Công thức theo trường hợp đại lượng thay đổi và cặp đo đồng thời của [Bland–Altman 2007, trang 575–576](https://www-users.york.ac.uk/~mb55/meas/bland2007.pdf). Giả định residual độc lập trong người và phương sai đồng nhất chưa tự được xác minh; LoA không phải khoảng tin cậy của bias.

## Chạy độc lập

Tạo venv Python3.12 mới, cài requirements với extra index CPU PyTorch; giữ nguyên cây `submission/` và các module `ops/deep_hr/` đi kèm. Đặt checkpoint có SHA đã khóa trong `submission/checkpoints/baseline_seed2027.pt` hoặc truyền `--checkpoint`. CLI:

```
python submission/process_round1.py --video INPUT --reference_csv REF.csv --out_dir NEW_OUTPUT --reference-clock-shared
```

CSV generic cần `timestamps_s,ppg` trên clock video, có thể thêm `--video-timestamps` chứa vector NPY. Cờ shared-clock là tuyên bố từ caller, không phải phép tự đồng bộ. UBFC2/MCD nên dùng `--dataset`, `--root`, `--recording`, thêm `--audit-dir` cho receipt MCD. `--roi-cache` chỉ cache T3 đúng recording/timestamps. `--constraints C4,C5 --constraint-manifest FILE` ghi nhận input đã mô phỏng bởi T5; không áp biến đổi lần nữa. Không truyền cờ constraint cho video native.

## Cần hỏi ban tổ chức

1. Có chấp nhận v7.3, layout và struct con này không? Có evaluator/schema mẫu không?
2. Có cho phép external video path+hash thay raw RGB không? Hiện chỉ là opt-in và chưa xác nhận hợp lệ.
3. Trang10 gọi reference CSV là ground-truth HR; trang11 dùng reference_signal rộng hơn. Có cần cả waveform PPG lẫn HR cửa sổ và định nghĩa cửa sổ cụ thể không?
4. Tên file minh họa `Round1_Baseline_None.mat` khác tên thư mục `Round1_Baseline/`; nên đặt prefix constraint ra sao? Cây trang11 còn có Round4 dù mô tả ba vòng.
5. Tên script trang10 là process_round1.m nhưng mô tả cho phép MATLAB/Python; có chấp nhận process_round1.py cùng thư viện và checkpoint đi kèm không?
6. Unknown metadata có được ghi NA không? Thứ tự tọa độ, mask, padding và convention má trái/phải cần theo subject hay ảnh?
7. RGB đã giải mã và lossless MP4 đóng gói có được chấp nhận cùng clock timestamp riêng không, nhất là video constraint VFR/drop-frame?
8. Ban tổ chức yêu cầu LoA đo lặp lại, coverage/abstain và mẫu số như thế nào? Một video/một người không đủ ước lượng thành phần giữa người.

## S1 public P4m representation (24 November)

The six top-level variables are unchanged. S1 uses `roi_type=oval36_actual_frontend`, `oval_points [T,36,2]`, boxes, valid flags and acquisition times, because P4m uses the whole oval. No distinct forehead/cheek/periocular masks are claimed. `processed_signal.rppg` is the CNN cumsum waveform per accepted window with an explicit source label; HR is the P4m spectral-fusion result. Separate NPZ files preserve all source waveforms privately. `params.pipeline=P4m R1`, matching fold/checkpoint/reader/code SHA, and source/target SHA replace the historical Haar36 identity. No change is made to the six-variable contract or MATLAB axis order.
