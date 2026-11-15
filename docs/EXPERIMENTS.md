# Experiment map

Mỗi thư mục giữ snapshot nguồn. **Có source không đồng nghĩa một lệnh không cần asset riêng sẽ chạy được.** Bảng hiện trạng ở README/RESULTS có ưu tiên hơn tên file hay comment trạng thái lịch sử.

| Nhóm | Source | Cách đọc / trạng thái |
|---|---|---|
| Core adapters/model/decoder | [ops/deep_hr](../ops/deep_hr/) | Adapter, waveform/support, EfficientPhys, HR decoder và kiểm thử. Nhiều thử nghiệm trước T8 là lịch sử. |
| Public classical baselines | [ops/p100](../ops/p100/) | POS/CHROM và runners MCD; không phải BTC calibrated estimator. |
| Constraint simulation | [ops/constraints](../ops/constraints/) | C1–C5, frame/codec support, kiểm bitrate. |
| Oval cache T8 | [ops/t8_oval_20261031](../ops/t8_oval_20261031/) | Fullframe encode trước crop; oval36 cache, clock/support. |
| Original T8 train | [ops/t8_oval_training_20261031](../ops/t8_oval_training_20261031/) | Control reused; original joint obsolete sau polarity; không chạy queue này thay V3. |
| V3 corrected polarity | [ops/t8_polarity_v3_20261102](../ops/t8_polarity_v3_20261102/) | 15 joint sửa MCD sign; same folds/seeds với control. |
| V3 aggregation | [ops/t8_polarity_aggregate_20261105](../ops/t8_polarity_aggregate_20261105/) | Ghép15joint+15control; raw-reference hash/sign/time/HR/failure parity. |
| T9 FPS × codec | [ops/t9_fps_codec_v2_20261031](../ops/t9_fps_codec_v2_20261031/) | 35/35 UBFC; report kết quả T9 giữ riêng. |
| AiPEX diagnostic | [ops/aipex_diagnostic_20261031](../ops/aipex_diagnostic_20261031/) | 88 người; timing15fps unresolved. |
| Historical final packaging | [reports/round1_final_20261105/revision_02/code](../reports/round1_final_20261105/revision_02/code/) | MAT export, frozen evaluator/report; trạng thái lịch sử, không phải final BTC runtime. |
| V4 | [reports/round1_v4_20261105](../reports/round1_v4_20261105/) | Phase loss, repair, zero-shot, padding pilot; xem subfolder. |
| Static padding regression | [reports/round1_v4_20261105/tsm_padding](../reports/round1_v4_20261105/tsm_padding/) | replicate_tsm.py, test_static_tsm.py, check_checkpoint_static.py. |
| Padding pilot | [reports/round1_v4_20261105/padding_pilot_20261107](../reports/round1_v4_20261105/padding_pilot_20261107/) | pilot_runner.py và repair_loss.py; giữ epoch5 gate. |
| BTC zero-shot | [reports/round1_v4_20261105/btc_zeroshot](../reports/round1_v4_20261105/btc_zeroshot/) | Dev-only preprocessing/inference, fold-safe calibration, diagnostics. |
| Bayes / multi-ROI | [reports/round1_bayes_20261108](../reports/round1_bayes_20261108/) | Không qua cổng; public crosscheck và POS repair giữ nguyên. |
| A–F execution | [reports/round1_execution_20261110](../reports/round1_execution_20261110/) | C5proxy, AdaBN, train_sweep_fasttrack_v5.py, report code. Queues đã dừng. |
| Final padding / codec | [reports/round1_execution_20261110/final_priorities_20261111](../reports/round1_execution_20261110/final_priorities_20261111/) | Final padding public5fold; codec probes và classical_screen. |
| V5 multiview | [reports/round1_execution_20261110/v5_multiview_20261112](../reports/round1_execution_20261110/v5_multiview_20261112/) | Offset eligibility, augmentation, train_v2.py, evaluation. Final V5 không qua BTC gate. |
| Spectral candidates | [reports/round1_execution_20261110/spectral_readout_20261112](../reports/round1_execution_20261110/spectral_readout_20261112/) | Label-free candidate extraction rồi oracle/shuffle. Gatefail, không fit reader. |
| Visual diagnostics | [reports/round1_execution_20261110/visual_diagnostics_20261112](../reports/round1_execution_20261110/visual_diagnostics_20261112/) | Occlusion/gradient/pulse map/UMAP. Không kèm ảnh mặt. |
| Inference repair | [reports/round1_execution_20261110/inference_repair_20261113](../reports/round1_execution_20261110/inference_repair_20261113/) | Four spatial branches + readout, report_repair.py và report_spatial.py. Không promotion. |
| Causal audit | [reports/round1_execution_20261110/causal_audit_20261113](../reports/round1_execution_20261110/causal_audit_20261113/) | Batch/lineage/ladder_physical_v3.py/compare_stage_exports; clockV3 là authoritative. |
| Legacy packaging | [submission](../submission/) | Historical frozen checkpoint Python export, không coi là selected calibrated implementation. |

## Tra cứu đầy đủ

[ENTRYPOINTS.json](ENTRYPOINTS.json) liệt kê imports và tham số `add_argument` tĩnh của từng Python file. Không chạy module để tạo index. Các thư mục con có README file list; CLI flags ở đó là thông tin schema, không bảo đảm assets private đã có.

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) là danh sách tất cả mã/config/license được đưa lên và SHA256. Helper authentication/transfer/monitor bị loại có lý do ghi riêng. Raw assets, kết quả cá nhân, cache, môi trường cài đặt, papers và transcript không phải source release.
