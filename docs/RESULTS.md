# R1 — chạy lại sau sửa trạng thái open-rppg

**MODE: selfwarm**. Quyết định được lấy nguyên từ z18: `ADOPT self-warm`; không chọn lại dựa trên lượt R1.

Ba mô hình giữ nguyên trọng số. Trạng thái khởi tạo được sao chép ngay khi tạo Model, khôi phục trước mỗi video/bản ghi/điều kiện. Selfwarm (nếu được chọn) chỉ dùng lần 2 của chính video đó. Không huấn luyện CNN, không mở holdout, không thêm lưới tham số.

## Kiểm trước chạy lớn

| Mô hình | Reset: 2 video | Selfwarm: 2 video | A → B → A (hai MODE) | Self-warm gate, thời gian (s) |
|---|---|---|---|---:|
| FacePhys.rlap | lệch 0,0 | lệch 0,0 | lệch 0,0 | 91.8 |
| ME-chunk.pure | lệch 0,0 | lệch 0,0 | lệch 0,0 | 51.6 |
| ME-flow.pure | lệch 0,0 | lệch 0,0 | lệch 0,0 | 111.8 |

## Cách đọc cổng và khoảng tin cậy

Giữ nguyên cổng của script: `a` = độ phủ ≥90%; `b` = cận dưới CI95 của ΔMAE so hằng số ≤0 (chưa chứng minh tệ hơn hằng số); `c` = cận trên CI95 đó <0 (thắng hằng số). Dấu `-` là chưa đạt. `b` riêng lẻ không phải kiểm định tương đương hay không-thua-kém có biên đã định. Các cổng này khác tiêu chí thành công BTC native ≤10 bpm đã ghi trước. CI và delta đều ghép cặp theo người, 10.000 bootstrap, seed 0.

## BTC dev

Mốc cũ 8,350 bị loại; reset độc lập 9,130. Cột cũ chỉ để kiểm toán, không phải baseline hợp lệ để quảng bá hiệu năng. File z18 ghi 8,390 ở hàng historical vì mã z18 dùng phổ reset thay 17 hàng heavy; đây không phải chính lượt lịch sử 8,350.

| Điều kiện | Video | V4 mới [CI95] | V4 cũ | Dịch MAE | Hằng số | Hiệu chỉnh native cũ | V3 cùng điều kiện | Δ V4−V3 [CI95] | Δ V4−hằng số [CI95] | Cổng |
|---|---:|---|---:|---:|---:|---:|---:|---|---|---|
| native | 218 | 8.333 [7.090; 9.659] | 8.350 | -0.017 | 12.071 | 11.374 | 9.587 | -1.254 [-2.282; -0.283] | -3.737 [-5.095; -2.406] | abc |
| split10 | 218 | 9.480 [8.041; 11.019] | 9.815 | -0.335 | 12.071 | 11.374 | 10.053 | -0.572 [-1.601; +0.426] | -2.590 [-4.002; -1.185] | abc |
| track_c1_g25p | 200 | 9.672 [8.173; 11.252] | 9.739 | -0.067 | 12.501 | 11.580 | 10.417 | -0.745 [-1.731; +0.112] | -2.829 [-4.102; -1.615] | abc |
| aidtrack_c1_g25gv01 | 197 | 12.258 [10.547; 14.129] | 12.274 | -0.016 | 12.614 | 11.727 | 12.206 | +0.052 [-0.957; +1.025] | -0.356 [-1.542; +0.821] | ab- |
| c2_vit | 199 | 8.832 [7.335; 10.506] | 9.141 | -0.310 | 12.426 | 11.459 | 9.523 | -0.692 [-1.403; +0.029] | -3.594 [-5.040; -2.172] | abc |
| c4_vga | 200 | 9.101 [7.529; 10.780] | 9.289 | -0.188 | 12.501 | 11.580 | 9.616 | -0.515 [-1.244; +0.248] | -3.401 [-4.859; -1.932] | abc |
| c4_qvga | 200 | 9.443 [7.877; 11.139] | 9.242 | +0.202 | 12.501 | 11.580 | 9.690 | -0.247 [-1.145; +0.653] | -3.058 [-4.559; -1.545] | abc |
| c6_us1 | 197 | 8.444 [6.892; 10.134] | 8.461 | -0.017 | 12.534 | 11.554 | 10.133 | -1.688 [-2.830; -0.745] | -4.090 [-5.641; -2.554] | abc |
| track_c7_blue | 197 | 10.457 [8.785; 12.236] | 9.959 | +0.498 | 12.687 | 11.739 | 10.506 | -0.050 [-1.147; +1.020] | -2.231 [-3.510; -0.967] | abc |

Giữ logic y11: với mỗi điều kiện, fit trên các video native thuộc tập so ghép cặp và fold train, rồi áp phổ điều kiện mà không fit nhãn điều kiện. Mốc hiệu chỉnh là cấu hình native lịch sử áp lại trên đúng video, không phải fit riêng từng điều kiện. Heavy chỉ là 17 video native đã có: tái sử dụng, không suy luận lại.

## Công khai — P4m và P4

| Điều kiện | Bộ | Cửa sổ | P4 mới [CI95] / cũ | P4m mới [CI95] / cũ | Δ P4m mới−cũ [CI95] | Hằng số | Cổng P4 / P4m |
|---|---|---:|---|---|---|---:|---|
| native | UBFC2 | 193 | 0.722 [+0.494; +1.024] / 0.710 | 0.746 [0.516; 1.051] / 0.754 | -0.007 [-0.036; +0.021] | 14.592 | abc / abc |
| native | MCD | 1959 | 1.908 [+1.481; +2.547] / 1.835 | 1.511 [1.247; 1.820] / 1.586 | -0.075 [-0.122; -0.029] | 11.684 | abc / abc |
| c4_qvga | UBFC2 | 193 | 0.728 [+0.501; +1.028] / 0.717 | 0.727 [0.494; 1.038] / 0.743 | -0.016 [-0.041; +0.011] | 14.592 | abc / abc |
| c4_qvga | MCD | 618 | 2.282 [+1.424; +3.676] / 2.017 | 1.520 [1.158; 1.975] / 1.544 | -0.024 [-0.073; +0.028] | 10.962 | abc / abc |
| c6_us1 | UBFC2 | 193 | 0.730 [+0.495; +1.045] / 0.731 | 0.732 [0.507; 1.024] / 0.756 | -0.024 [-0.055; +0.005] | 14.592 | abc / abc |
| c6_us1 | MCD | 618 | 2.146 [+1.425; +3.308] / 1.965 | 1.641 [1.202; 2.198] / 1.597 | +0.043 [-0.039; +0.146] | 10.962 | abc / abc |
| aidtrack_c1_g25gv01 | UBFC2 | 184 | 15.124 [+11.956; +18.575] / 15.093 | 13.450 [10.776; 16.320] / 14.055 | -0.605 [-1.483; +0.124] | 14.613 | ab- / ab- |
| aidtrack_c1_g25gv01 | MCD | 592 | 9.887 [+7.589; +12.508] / 10.141 | 10.435 [8.187; 12.915] / 10.306 | +0.128 [-0.791; +1.019] | 10.983 | ab- / ab- |
| c2_vit | UBFC2 | 193 | 0.745 [+0.513; +1.047] / 0.724 | 0.750 [0.524; 1.045] / 0.748 | +0.002 [-0.025; +0.028] | 14.592 | abc / abc |
| c2_vit | MCD | 618 | 2.183 [+1.490; +3.218] / 2.232 | 1.734 [1.301; 2.245] / 1.741 | -0.007 [-0.086; +0.067] | 10.962 | abc / abc |
| track_c1_g25p | UBFC2 | 193 | 2.172 [+1.305; +3.183] / 2.241 | 1.532 [0.997; 2.264] / 1.623 | -0.091 [-0.229; +0.015] | 14.592 | abc / abc |
| track_c1_g25p | MCD | 618 | 3.046 [+2.203; +4.066] / 3.006 | 2.443 [1.790; 3.182] / 2.526 | -0.083 [-0.157; -0.009] | 10.962 | abc / abc |
| track_c7_blue | UBFC2 | 192 | 1.506 [+0.919; +2.323] / 1.506 | 1.447 [0.839; 2.286] / 1.711 | -0.264 [-0.554; -0.076] | 14.587 | abc / abc |
| track_c7_blue | MCD | 618 | 4.712 [+3.086; +6.900] / 5.068 | 2.972 [2.283; 3.768] / 3.217 | -0.245 [-0.550; +0.028] | 10.962 | abc / abc |
| c1_g50p | UBFC2 | 193 | 1.263 [+0.898; +1.703] / 1.239 | 0.982 [0.741; 1.272] / 1.005 | -0.023 [-0.073; +0.018] | 14.592 | abc / abc |
| c1_g50p | MCD | 618 | 2.917 [+1.898; +4.552] / 3.027 | 1.869 [1.509; 2.274] / 1.886 | -0.016 [-0.073; +0.042] | 10.962 | abc / abc |
| c1_g12p | UBFC2 | 186 | 4.661 [+3.002; +6.686] / 4.797 | 2.564 [1.723; 3.744] / 2.997 | -0.433 [-0.916; -0.058] | 14.844 | abc / abc |
| c1_g12p | MCD | 616 | 4.130 [+2.889; +5.674] / 4.142 | 3.056 [2.335; 3.868] / 2.945 | +0.111 [-0.067; +0.309] | 10.969 | abc / abc |
| c1_g25gv001 | UBFC2 | 192 | 4.126 [+2.715; +5.855] / 4.162 | 3.082 [2.012; 4.348] / 3.129 | -0.047 [-0.552; +0.552] | 14.583 | abc / abc |
| c1_g25gv001 | MCD | 616 | 4.925 [+3.563; +6.586] / 4.866 | 4.139 [2.899; 5.768] / 4.341 | -0.202 [-0.464; +0.017] | 10.969 | abc / abc |
| c7_white | UBFC2 | 176 | 1.757 [+0.880; +3.155] / 1.808 | 1.154 [0.787; 1.579] / 1.211 | -0.057 [-0.147; +0.033] | 14.854 | abc / abc |
| c7_white | MCD | 614 | 4.474 [+3.132; +6.160] / 4.399 | 3.602 [2.517; 4.972] / 3.711 | -0.109 [-0.986; +0.605] | 10.943 | abc / abc |
| c5_split10 | UBFC2 | 179 | 0.659 [+0.515; +0.849] / 0.637 | 0.646 [0.496; 0.845] / 0.655 | -0.009 [-0.038; +0.018] | 14.656 | abc / abc |
| c5_split10 | MCD | 1929 | 1.761 [+1.486; +2.083] / 1.765 | 1.704 [1.435; 2.022] / 1.768 | -0.064 [-0.112; -0.023] | 11.690 | abc / abc |

Tất cả so sánh old/new dùng đúng giao cửa sổ và ghép cặp theo người; 10.000 bootstrap, seed 0. Cổng a: coverage ≥0,9; b: CI không chứng minh tệ hơn hằng số; c: CI chứng minh tốt hơn hằng số. Không kết luận cải thiện khi CI chứa 0.

## Tham số P4m

| Fold | Trọng số mới (CNN, POS, CHROM, experts) | Trọng số cũ |
|---|---|---|
| fold0 | (1.0, 0.5, 0.0, 1.0) | (1.0, 0.0, 0.25, 1.0) |
| fold1 | (1.0, 0.5, 0.0, 1.0) | (1.0, 0.0, 0.25, 1.0) |
| fold2 | (1.0, 0.0, 0.25, 1.0) | (1.0, 0.0, 0.25, 1.0) |
| fold3 | (1.0, 0.5, 0.0, 1.0) | (1.0, 0.0, 0.25, 1.0) |
| fold4 | (1.0, 0.5, 0.0, 1.0) | (1.0, 0.0, 0.25, 1.0) |
| deploy | (1.0, 0.5, 0.0, 1.0) | (1.0, 0.0, 0.25, 1.0) |

Tham số readout đầy đủ (gamma, sigma, tau, estimator, ref, hằng số train):

```text
P4m public reader, frozen 2026-11-23 21:49:16 (team decision 22 Nov 2026: P4m, documented deviation from the z13 rule)
  channels: CNN (team padding-fixed EfficientPhys+TSM, out-of-fold checkpoint) on crops pre-filtered by a centred moving average of k = ceil(fps/10)
            frames weighted by the valid flags (x13 nat_ma3, x20 under conditions; C5 split keeps its binned 10 fps stream); POS and CHROM from the
            UNFILTERED 6x6 block means; experts = mean of the standardised log spectra of FacePhys.rlap, ME-chunk.pure, ME-flow.pure (y12/y13).
  fusion  : E = w . Z (Z = per-window standardised log spectra on the 42-180 bpm grid), w fitted on the training folds over {0,.5,1,2}^4 (normalised).
  readout : forward-backward with quality weight (y07 marg/point, y02 qscale), (gamma, sigma, tau, estimator) per data set per fold.
  pooled  : training windows = native + c4_qvga, c6_us1, aidtrack_c1_g25gv01, c2_vit, track_c1_g25p, track_c7_blue on the common recordings.
  fold 0 : w (CNN, POS, CHROM, experts) = (1.0, 0.5, 0.0, 1.0); UBFC2 (gamma, sigma, tau, est, ref, const) = 2.0, 15.0, 1.6, median, 4.995398, 96.680; MCD (gamma, sigma, tau, est, ref, const) = 4.0, 6.0, 1.6, median, 4.383784, 83.496
  fold 1 : w (CNN, POS, CHROM, experts) = (1.0, 0.5, 0.0, 1.0); UBFC2 (gamma, sigma, tau, est, ref, const) = 2.0, 15.0, 1.6, median, 5.016544, 93.604; MCD (gamma, sigma, tau, est, ref, const) = 4.0, 6.0, 0.8, median, 4.199450, 84.375
  fold 2 : w (CNN, POS, CHROM, experts) = (1.0, 0.0, 0.25, 1.0); UBFC2 (gamma, sigma, tau, est, ref, const) = 2.0, 15.0, 1.6, median, 4.475250, 95.361; MCD (gamma, sigma, tau, est, ref, const) = 4.0, 6.0, 1.6, median, 3.811568, 86.572
  fold 3 : w (CNN, POS, CHROM, experts) = (1.0, 0.5, 0.0, 1.0); UBFC2 (gamma, sigma, tau, est, ref, const) = 2.0, 15.0, 1.6, median, 5.021681, 97.119; MCD (gamma, sigma, tau, est, ref, const) = 4.0, 6.0, 1.6, median, 4.214000, 87.012
  fold 4 : w (CNN, POS, CHROM, experts) = (1.0, 0.5, 0.0, 1.0); UBFC2 (gamma, sigma, tau, est, ref, const) = 2.0, 15.0, 1.6, median, 5.044070, 97.119; MCD (gamma, sigma, tau, est, ref, const) = 4.0, 6.0, 1.6, median, 4.219725, 82.617
  deploy : w (CNN, POS, CHROM, experts) = (1.0, 0.5, 0.0, 1.0); UBFC2 (gamma, sigma, tau, est, ref, const) = 2.0, 15.0, 1.6, median, 5.016605, 96.680; MCD (gamma, sigma, tau, est, ref, const) = 4.0, 6.0, 1.6, median, 4.258152, 84.814
  check   : max |MAE - z13 P4m| over conditions x data sets = 0.00e+00
  sha256  : a75d4deb6199f7016a5a8736c27a84ebcfccd4352e6a51345a2be9eb68b4c787  out/z14_p4m_frozen.npz
```

## Giới hạn và thời gian

**Việc chọn ba mô hình trong nhóm 17 đã làm trên phổ kiểu cũ; không làm lại lựa chọn/nested selection trong R1.** Mọi kết quả vẫn là dev sau nhiều quyết định nghiên cứu, không phải kiểm định holdout độc lập. Không lấy tham số deploy để báo OOF.

- Reset reference gate: 91,48 giây.
- Self-warm + order-independence gates: 129,75 giây.
- inference: 56687.7 giây.
- score: 265.3 giây.
- freeze_public: 104.3 giây.

Kiểm đóng gói CLI được ghi riêng trong [VIDEO_CLI_VALIDATION.md](VIDEO_CLI_VALIDATION.md). PR R1 không tự merge.

## Phạm vi các mã điều kiện

`native`: video gốc; `split10`/`c5_split10`: luồng crop tách còn ≤10 fps, không phải video toàn khung nén ≤500 kbps; `c4_vga`/`c4_qvga`: giảm độ phân giải; `track_c1_g25p`: giảm sáng gain 0,25 theo simulator đã khóa; `aidtrack_c1_g25gv01`: giảm sáng kèm nhiễu mạnh và trợ giúp tracking; `c2_vit`: biến thể C2 đã khóa; `c6_us1`: biến thể C6 đã khóa; `track_c7_blue`: ánh sáng xanh kèm tracking. Công thức simulator và codec không đổi trong R1.

Cohort điều kiện khác nhau: chỉ dùng delta ghép cặp trên cùng người/video/cửa sổ, không xếp hạng bằng cách trừ hai trung bình ở hai tập khác nhau.

[Lịch sử và các báo cáo trước R1](history/R1_STATE_BUG_20261123.md).
