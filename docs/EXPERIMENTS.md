# Experiment map

Each directory preserves a source snapshot. **Available source does not imply commands can run without private assets.** Current README/RESULTS tables take precedence over filenames or historical status comments.

| Group | Source | Interpretation / status |
|---|---|---|
| Core adapters/model/decoder | [ops/deep_hr](../ops/deep_hr/) | Adapters, waveform/support handling, EfficientPhys, the HR decoder, and tests. Many experiments predating T8 are historical. |
| Public classical baselines | [ops/p100](../ops/p100/) | POS/CHROM and MCD runners; not the calibrated BTC estimator. |
| Constraint simulation | [ops/constraints](../ops/constraints/) | C1–C5, frame/codec support, and bitrate checks. |
| Oval cache T8 | [ops/t8_oval_20261031](../ops/t8_oval_20261031/) | Full-frame encoding before cropping; oval36 cache and clock/support handling. |
| Original T8 train | [ops/t8_oval_training_20261031](../ops/t8_oval_training_20261031/) | The control is reused; the original joint run became obsolete after the polarity correction. Do not run this queue in place of V3. |
| V3 corrected polarity | [ops/t8_polarity_v3_20261102](../ops/t8_polarity_v3_20261102/) | 15 joint runs with the MCD sign corrected; the same folds and seeds as the control. |
| V3 aggregation | [ops/t8_polarity_aggregate_20261105](../ops/t8_polarity_aggregate_20261105/) | Pair 15 joint runs with 15 control runs; check raw-reference hash, sign, time, HR, and failure parity. |
| T9 FPS × codec | [ops/t9_fps_codec_v2_20261031](../ops/t9_fps_codec_v2_20261031/) | 35/35 UBFC; the T9 results report is kept separately. |
| AiPEX diagnostic | [ops/aipex_diagnostic_20261031](../ops/aipex_diagnostic_20261031/) | 88 people; timing15fps remains unresolved. |
| Historical final packaging | [reports/round1_final_20261105/revision_02/code](../reports/round1_final_20261105/revision_02/code/) | MAT export and frozen evaluator/report; historical status, not the final BTC runtime. |
| V4 | [reports/round1_v4_20261105](../reports/round1_v4_20261105/) | Phase loss, repair, zero-shot, padding pilot; xem subfolder. |
| Static padding regression | [reports/round1_v4_20261105/tsm_padding](../reports/round1_v4_20261105/tsm_padding/) | replicate_tsm.py, test_static_tsm.py, check_checkpoint_static.py. |
| Padding pilot | [reports/round1_v4_20261105/padding_pilot_20261107](../reports/round1_v4_20261105/padding_pilot_20261107/) | pilot_runner.py and repair_loss.py; the epoch5 gate is retained. |
| BTC zero-shot | [reports/round1_v4_20261105/btc_zeroshot](../reports/round1_v4_20261105/btc_zeroshot/) | Dev-only preprocessing/inference, fold-safe calibration, diagnostics. |
| Bayes / multi-ROI | [reports/round1_bayes_20261108](../reports/round1_bayes_20261108/) | Did not pass the gate; the public cross-check and POS repair are preserved. |
| A–F execution | [reports/round1_execution_20261110](../reports/round1_execution_20261110/) | C5proxy, AdaBN, train_sweep_fasttrack_v5.py, and report code. The queues have been stopped. |
| Final padding / codec | [reports/round1_execution_20261110/final_priorities_20261111](../reports/round1_execution_20261110/final_priorities_20261111/) | Final padding public5fold; codec probes and classical_screen. |
| V5 multiview | [reports/round1_execution_20261110/v5_multiview_20261112](../reports/round1_execution_20261110/v5_multiview_20261112/) | Offset eligibility, augmentation, train_v2.py, and evaluation. Final V5 did not pass the BTC gate. |
| Spectral candidates | [reports/round1_execution_20261110/spectral_readout_20261112](../reports/round1_execution_20261110/spectral_readout_20261112/) | Label-free candidate extraction followed by oracle/shuffle checks. The gate failed; no reader was fitted. |
| Visual diagnostics | [reports/round1_execution_20261110/visual_diagnostics_20261112](../reports/round1_execution_20261110/visual_diagnostics_20261112/) | Occlusion/gradient/pulse map/UMAP. Face images are not included. |
| Inference repair | [reports/round1_execution_20261110/inference_repair_20261113](../reports/round1_execution_20261110/inference_repair_20261113/) | Four spatial branches plus readout, report_repair.py and report_spatial.py. No promotion. |
| Causal audit | [reports/round1_execution_20261110/causal_audit_20261113](../reports/round1_execution_20261110/causal_audit_20261113/) | Batch/lineage/ladder_physical_v3.py/compare_stage_exports; clockV3 is authoritative. |
| Legacy packaging | [submission](../submission/) | Historical frozen-checkpoint Python export, not the selected calibrated implementation. |

## Complete inventory

[ENTRYPOINTS.json](ENTRYPOINTS.json) lists statically extracted imports and `add_argument` parameters for each Python file. Modules are not executed to build the index. Subdirectory READMEs list files; CLI flags describe the schema, not availability of private assets.

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) lists released code/configuration/licenses and SHA256. Exclusions of authentication/transfer/monitoring helpers are documented separately. Raw assets, individual results, caches, installed environments, papers, and transcripts are not part of the source release.
