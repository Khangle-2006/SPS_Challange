# FaceVitals — SP Cup Round 1

Mã nguồn nghiên cứu ước lượng nhịp mạch từ video khuôn mặt, cùng thí nghiệm và kết quả đã đo đến **15/11/2026**. Đây là bản công bố **code + kết quả tổng hợp**, không phải một sản phẩm chẩn đoán hay bản nộp đã hoàn tất.

## Đọc gì trước?

| Bạn muốn làm gì? | Tài liệu |
|---|---|
| Chạy thử không cần dữ liệu người tham gia | [Quick start bên dưới](#chạy-thử-trên-cpu) |
| Hiểu cấu trúc và tìm code từng thí nghiệm | [Experiment map](docs/EXPERIMENTS.md) |
| Tái lập với dữ liệu/checkpoint được cấp quyền | [Reproduction guide](docs/REPRODUCING.md) |
| Xem toàn bộ kết quả chính và các lần thất bại | [Results](docs/RESULTS.md) và [bảng nhóm BTC](docs/results/CPU_REPORT.md) |
| Hiểu nhãn, FPS, cửa sổ, ký hiệu và metric | [Data and protocol](docs/DATA_AND_METRICS.md) |
| Kiểm bản mã nguồn và tham số CLI | [Source manifest](docs/SOURCE_MANIFEST.json), [entrypoint index](docs/ENTRYPOINTS.json) |
| Xem giới hạn còn lại trước triển khai | [Limitations](docs/LIMITATIONS.md) |

## Kết quả chính — đọc cùng điều kiện đánh giá

**BTC native: 218 video / 158 người**, tính lỗi trong từng người trước rồi trung bình giữa người. Đơn vị bpm, thấp hơn tốt hơn. Đây là dev OOF; đã dùng dev cho nhiều quyết định nghiên cứu nên không thay thế một phép thử độc lập.

| Phương pháp | MAE | Trạng thái |
|---|---:|---|
| Hằng số: trung vị PR của train-fold | 12,07068 | Mốc bắt buộc |
| Trung vị POS/CHROM/V3 + hiệu chỉnh tuyến tính theo fold | **11,37385** | Ứng viên nghiên cứu BTC đang giữ; CI cải thiện so hằng số chứa 0 |
| POS | 13,578 | Cổ điển, chưa hiệu chỉnh |
| CHROM | 12,356 | Cổ điển, chưa hiệu chỉnh |
| PBV học chữ ký ở train-fold | 12,159 | Không được chọn |
| Stacking 5 tham số, nested CV | 11,283 | Δ so hiệu chỉnh −0,091; CI [−0,644; +0,442] |
| Bayes đầy đủ | 12,12711 | Không qua cổng |
| Padding-fixed, 5 fold seed 2027 | 17,29208 | Mô hình chính cho public, không chọn cho BTC |
| V5 multiview, 5 fold seed 2027 | 18,69816 | Không qua cổng, kết thúc huấn luyện |
| Padding-fixed + xoay + mask màu | 16,75205 | Δ −0,54003; CI [−2,12754; +1,06003] |
| Hiệu chỉnh refit + readout bù nghiêng 0,7–3 Hz | 10,75341 | Khám phá; **không được chọn**, chưa qua cổng |

Cổng BTC đã ghi trước: MAE ≤10 bpm **và** giảm ≥1 bpm so bản hiệu chỉnh trên cùng cohort **và** CI95% ghép cặp loại 0. **Chưa đạt.** Bản 10,75341 giảm 0,62044 bpm với CI [−1,29593; +0,05546]. Không đổi ngưỡng sau khi xem kết quả.

**Public development, cùng seed 2027:**

| Mô hình | S public ↓ | Kiểm soát ảnh tĩnh | Vai trò |
|---|---:|---|---|
| V3 | 9,08771 | FAIL | Đối chứng lịch sử |
| V4 padding gốc | 5,94765 | FAIL | Ablation, không chọn dù S nhỏ hơn |
| **Padding-fixed** | **6,71043** | PASS | Mô hình chính cho public |
| V5 multiview | 9,23503 | Xem hồ sơ thí nghiệm | Không thay padding-fixed |

`S = (MAE UBFC native + MAE UBFC C5 cũ + MAE MCD native) / 3`.
S này là chỉ số phát triển của đội, **không phải kết quả BTC hoặc điểm chính thức của BTC**.
Padding-fixed: UBFC native 1,02727; UBFC C5 cũ 12,52797; MCD native 6,57604 bpm. Xem [bảng đầy đủ](docs/RESULTS.md).

## Cấu trúc repository

```text
tools/                       Lệnh CPU để chạy thử và kiểm bản phát hành
docs/                        Cách chạy, giao thức, kết quả, giới hạn
ops/deep_hr/                 Adapter, EfficientPhys, decoder, frontend, kiểm thử
ops/t8_*/                    Các phiên bản T8: cache, train, polarity, evaluate
ops/t9_*/                    Tách ảnh hưởng FPS và nén
ops/constraints/             Mô phỏng constraint
reports/round1_v4_20261105/   Code V4, padding, zero-shot BTC và chẩn đoán
reports/round1_bayes_20261108/ Code multi-ROI/Bayes, đối chiếu public
reports/round1_execution_20261110/
  final_priorities_20261111/ Padding-fixed cuối, kiểm codec
  v5_multiview_20261112/     V5 cuối cùng
  spectral_readout_20261112/ Phép thử ba đỉnh phổ, gate fail
  inference_repair_20261113/ Xoay/mask/readout, đánh giá ghép cặp
  causal_audit_20261113/     Batch, đồng hồ, thang suy giảm
submission/                  Đóng gói lịch sử Python, không phải BTC estimator cuối
score_round1.py              Scorer nội bộ, không phải app chấm của BTC
selection_round1.py          So sánh/cổng lựa chọn
```

Giữ các thư mục ngày tháng vì code có import tương đối và kiểm hash. `reports/` trong bản phát hành chứa **code thí nghiệm**, không chứa raw video, nhãn cá nhân hay mọi file output của máy chủ. Tên `fixed` trong EfficientPhys lịch sử không tự có nghĩa đã sửa padding; biến thể replicate nằm trong nhánh padding riêng.

## Chạy thử trên CPU

Python 3.11 được khuyến nghị. Từ thư mục repo:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tools/verify_source.py
python tools/pos_trace.py --make-demo output/demo_rgb.csv
python tools/pos_trace.py --input output/demo_rgb.csv --output output/demo_predictions.csv
python -m unittest test_score_round1
```

Demo là tín hiệu **tổng hợp 90 bpm**, không phải kết quả trên người. `pos_trace.py` dùng đúng numerical POS/Welch baseline đã có, không phải estimator hiệu chỉnh 11,37. Đầu vào CSV: `timestamp_s,red,green,blue,valid`; `valid` bằng 0 hoặc 1. Trace phải là RGB tuyến tính theo thời gian đã xuất từ ROI, không phải nhãn hay waveform PPG. Không dùng sample demo để suy ra chất lượng video thật.

Muốn chạy video, GPU hoặc tái lập thí nghiệm: đọc [REPRODUCING.md](docs/REPRODUCING.md) trước. Cần dữ liệu hợp lệ, manifest/split đúng, checkpoint và môi trường tương ứng; chúng **không được tự tải** khi chạy demo. Không chạy các controller lịch sử như một quick start.

## Những gì đã kiểm và những gì chưa

- Canonical POS sửa lại khớp **4.304/4.304** dự đoán public cũ; MCD 4,67015 bpm.
- Octave 8.4 `--traditional` khớp Python 0 bpm trên 5 ROI video thật cho **POS baseline**. Chưa kiểm MATLAB thật và chưa phải parity của estimator BTC được chọn.
- 20 tensor kiểm batch: sai khác waveform tối đa 1,53482e−6 giữa single/batch; companion/permutation/repeat bằng 0.
- Thang physical-clock 10/10 và A/B xoay/mask 1.488/1.488 hoàn tất. Không suy ra tổn thất chắc chắn do khâu thu; đối chứng webcam/điện thoại đồng thời còn thiếu.
- C5′ dùng ffmpeg là **proxy**, chưa xác minh tương đương MATLAB VideoWriter. Các cohort khác coverage không được so như một thử nghiệm nhân quả ghép cặp.
- Holdout BTC 72 video/41 người **chưa được đánh giá**. D3 và 7 UBFC test vẫn đóng trong giao thức hiện tại. Một số T5 rất cũ có cohort khác; xem chú thích lịch sử.
- Chưa có bằng chứng kiểm soát âm end-to-end của chính estimator hiệu chỉnh. V3 thành phần trượt ảnh tĩnh; không được lấy PASS của padding-fixed gán sang estimator hiệu chỉnh.

## Nguồn và giấy phép

Code dựa một phần trên EfficientPhys/rPPG-Toolbox; giữ [giấy phép upstream](ops/deep_hr/TOOLBOX_LICENSE.txt) và [thông tin nguồn](THIRD_PARTY_NOTICES.md). Không gán một giấy phép MIT mới lên toàn bộ mã có giấy phép khác. Dữ liệu thuộc điều kiện sử dụng riêng của từng bộ; repo không cấp quyền phân phối lại video/nhãn.

Bố cục tài liệu tham khảo cách trình bày của [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) và [PhysFormer](https://github.com/ZitongYu/PhysFormer): cài đặt → dữ liệu → chạy → đánh giá → nguồn. Phần giải thích và kết quả ở đây được viết riêng cho dự án này.
