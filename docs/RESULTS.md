# Kết quả nghiên cứu đã chạy

> **Cập nhật 22/11/2026:** các kết quả dưới đây là bản lịch sử đến 15/11. Pipeline đã chốt (BTC reader_v4 8,35; công khai P4m UBFC2 0,75 / MCD 1,59) nằm ở [README](../README.md#kết-quả-chính) và [PIPELINE.md](PIPELINE.md).

Ngày chốt bản công bố: 15/11/2026. Không có kết quả holdout/test ẩn. Các bảng không cùng cohort/protocol không được xếp hạng chung.

## BTC: tập ghép cặp cuối

218 video, 158 người. Kết quả từng nhóm Normal-Resting, Abnormal-Resting, Task, Walking có trong [CPU tables](results/CPU_REPORT.md). Constant train-fold và calibrated đều được tính lại trên cùng cohort mỗi hàng.

| Thí nghiệm | Kết quả đo | Kết luận |
|---|---|---|
| Constant / calibrated | 12,07068 / 11,37385 bpm | CI delta calibrated − constant [−1,51016; +0,14547], chưa chứng minh thắng |
| POS / CHROM | 13,578 / 12,356 | Không chọn riêng làm mô hình chính |
| Bayes đầy đủ | 12,12711 | Delta constant +0,056, CI [−0,616; +0,732], G2 fail |
| PBV train-fold / G-only | 12,159 / 15,519 | Không thay candidate |
| Stacking ≤5 tham số | 11,283 | Delta calibrated −0,091, CI [−0,644; +0,442] |
| AdaBN | 19,06018 → 19,23028 | Chỉ bác bỏ recalibration đã thử; không chứng minh mọi domain shift không liên quan BN |
| Padding-fixed | 17,29208 | Bias −11,56164, Spearman khoảng 0,047; không thay calibrated |
| V5 multiview | 18,69816 | Delta calibrated +7,32431, CI [5,18292; 9,44631] |
| Xoay / mask / cả hai (padding) | 17,37299 / 16,79073 / 16,75205 | Cả hai delta −0,54003, CI [−2,12754; +1,06003] |
| Calibrated refit, tilt 0,7–3 Hz | 10,75341 | Delta saved −0,62044, CI [−1,29593; +0,05546]; fail cả ba gate |

### Các sweep bị dừng

- V4 phase lượt đầu bị dồn HR lên trần; train loss gần 0,97–0,99, kiểm lỗi riêng, không coi là kết quả model hợp lệ.
- Pilot repair: epoch1 loss 1,60580958 → epoch5 1,41631262, vượt mức giảm 10%; MCD cùng fold 10,80412 vs V3 13,35474, 398 cửa sổ/12 người. Đây là pilot một fold, không phải full OOF.
- Priority SiNC run4/fold3: dừng epoch8, ảnh tĩnh trả 61,52344 bpm với std2,425e−9; **không đo native MAE**. Không ghi MAE =0 hoặc gán score cho run bị loại.
- Weak-label run3/fold2: 42 video/32 người, MAE18,15033 [13,62877;22,87975], cùng fold calibrated9,41745 và constant9,68229. Delta calibrated+8,73289 [4,09618;13,28093]. Static10/10PASS, SD12,61062≥8 nhưng Spearman −0,10903→−0,13537. Bị loại. Runs1/2/5 không khởi chạy.
- LQ-rPPG: thiếu tương thích Mamba/checkpoint; không có zero-shot MAE, không chỉnh architecture để gọi là cùng mô hình.

## Public: so sánh cùng seed

| Phương pháp | S | Vai trò |
|---|---:|---|
| V3 | 9,08771 | Đối chứng |
| V4 padding gốc | 5,94765 | Static FAIL; ablation |
| Padding-fixed | 6,71043 | Public primary |
| V5 | 9,23503 | Không chọn |

| Padding-fixed | Gốc | Thêm mask | Delta [CI95%] |
|---|---:|---:|---|
| MCD native | 6,57604 | 12,76949 | +6,19346 [5,10851;7,35374] |
| MCD C5 cũ | 11,84789 | 17,64287 | +5,79498 [4,57855;7,09243] |
| UBFC native | 1,02727 | 1,05155 | +0,02427 [−0,01256;0,06069] |
| UBFC C5 cũ | 12,52797 | 12,60665 | +0,07868 [−2,46974;2,42126] |
| S | 6,71043 | 8,80923 | +2,09880 [1,15881;2,95038] |

Public gốc: UBFC193/207 cửa sổ,35 người; MCD1959/2023,60 người. C5′ support còn UBFC180 và MCD1929. Bảng [C5_REPORT](results/C5_REPORT.md) dùng **giao cửa sổ** để đối chiếu các recipe và ghi rõ V4 trong đó là padding gốc, không phải final padding-fixed.

## FPS, codec và các phép thử xác minh

| Kiểm tra | Kết quả | Giới hạn |
|---|---|---|
| T9 35 người,193 cửa sổ chung | POS native1,23842; raw10 1,02769; nativeFPS+codec12,94064; C5 25,54185 | Frontend/clock riêng của T9, không trộn với POS MCD |
| T9 interaction | +12,81194 CI[9,84165;15,93529] | Cho thấy tương tác, không cô lập cơ chế codec |
| POS public repair | 4304/4304 exact parity; MCD4,67015 | Blind SNR median−0,24956 vẫn fail interpretation gate |
| BTC default→bf0,451 cửa sổ158 video chung | POS148–152:41,24%→0%; CHROM44,35%→0% | Liên hệ với encoder; không cơ chế nhân quả duy nhất |
| UBFC paired180windows | POS11,11%→0%; CHROM8,33%→0% | Cùng định nghĩa khoảng148–152 |
| MCD paired1929windows | POS20,11%→1,45%; CHROM43,44%→1,50% | B-frame thay đổi nhiều khía cạnh encoder |
| Hồi quy nuisance | Full projection bị loại vì xóa pulse thật; attenuation0,05 probe24/24PASS, retention≥94,9867% | BTCdefault POS45,49/CHROM41,97 vẫn tệ hơn calibrated12,87 trên cùng161video109người |
| Batch dependence | 20/20; single/batch max1,53482e−6; companion/permutation/repeat0 | Trong tập tensor/checkpoint đã kiểm |
| Physical clock ladder | 10/10 hoàn tất; native tái hiện saved predictions | Stage4 mất tracking10/10; inverse90 cứu5/10, bậc sau không tách hiệu ứng |
| Annotation lineage | 218video158people,2118windowconditions,0 bất nhất mapping | WorkbookSHA khác frozen dù dev ID/PR khớp; không thay file gốc |
| Source support | Hai extrapolation đã loại; hai span10,053333s vượt10±0,05 | Không che bằng lưới nội suy đúng10s |
| Static review | Padding2828PASS, V325172FAIL | Là số test, không checkpoint độc lập; calibrated E2E chưa kiểm |
| Octave | 5 real ROI video,0bpm chênh Python | POS baseline, chưa chạy MATLAB thật |

## Phép thử ba đỉnh phổ

218/158: oracle7,54694 vs100 whole-person shuffles mean8,23244; gain0,68550<4. Phủ±5:47,7064% vs40,7936%; gain6,91284pp<15. **Gate fail: không fit reader**. [Bảng nhóm và null distribution](results/SPECTRAL_CANDIDATES.md). Không suy ra mọi tín hiệu mạch đã mất.

## Những kết quả lịch sử cần để riêng

- T5: MCD833/833,UBFC294/294 job complete,0jobfailure. Frontend Haar và band0,75–3Hz; UBFC lịch sử42 gồm7 người test, **không dùng cho T8 selection/S**. Không tái phân loại thành dev35.
- AiPEX limited diagnostic88/88, MAE18,8844. Mean person ratio India1,255242 vs SierraLeone1,247941; delta0,007301 CI[−0,038299;0,053870]. 44 người/nhóm; timing15fps chưa giải quyết, ratio không timing-invariant; không diễn giải nhân quả da/quốc gia.
- T8 joint trước polarity:8 hoàn tất,1 dừng giữa,6 không chạy; chỉ giữ lịch sử, invalid cho harmonized comparison.15 control được giữ,15 V3 joint sửa dấu được chạy riêng. V2 không launch.
- Nhánh72 không có một full matched sweep được chọn. Các pilot/benchmark không là bằng chứng72 thắng36.
- QA-lite đã bị bỏ; không trình bày partial37/119 như full cohort.

## Kết luận hiện tại

Public dùng padding-fixed; BTC giữ saved calibrated. Chưa đạt mục tiêu≤10 và cải thiện≥1bpm có CI loại0. Hoàn tất các phép tính không có nghĩa hoàn tất estimator deployment: còn full-estimator negative controls/parity và đối chứng thu đồng thời. Không có kết luận nhân quả rằng nhãn, da, codec hay acquisition là nguyên nhân duy nhất.

Nguồn số: các báo cáo aggregate kèm repo và các artifact gốc `inference_repair_20261113/report_spatial`, `v5_multiview_20261112`, `FASTTRACK_FINAL_DECISION.json`, `OCTAVE_PARITY.json`. Private per-person inputs/predictions không kèm; mã tạo các bảng vẫn có trong source manifest.


## Bảng đầy đủ giai đoạn V3 / hybrid / 72

- [V3 frozen: mọi nhánh và cả bốn điều kiện](results/v3_frozen_report.md): ba seed, S V3=9,354817; khác bảng **cùng seed2027** S=9,08771 ở trên. MCD native V3=9,359 không phải S.
- [Hybrid POS/model](results/hybrid_report.md): hybrid V3 MCD native4,373 nhưng không qua cổng P95 UBFC-C5; giữ kết quả xấu và tốt cùng nhau.
- [MCD C5 đầy đủ](results/mcd_c5_report.md): chỉ số bổ sung ngoài S.
- [72×72 preflight](results/branch72_report.md): không khởi chạy full sweep, không có kết quả thắng/thua.

Các bảng archived giữ giá trị đã đo. Comment “pending/selected” ở thời điểm cũ không thay disposition15/11. Phiên bản README này không kèm mọi dòng per-person/window hoặc log đào tạo; đó là tài sản riêng, không phải thiếu được điền bằng số giả.
