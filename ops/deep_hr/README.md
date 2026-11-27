# ops/deep_hr

Adapters, waveform/support handling, EfficientPhys, the HR decoder, and tests. Many experiments predating T8 are historical.

[Reproduction guide](../../docs/REPRODUCING.md) · [Results](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

The parameters below were extracted statically from the code; they are not commands that automatically launch a run. For scripts without a CLI, check the constants, paths, and dependencies at the top of the file. Controllers and site runners require the original receipts and environments; do not bypass their guards.

| File | Declared CLI parameters |
|---|---|
| [EfficientPhys_fixed.py](EfficientPhys_fixed.py) | No argparse interface; read the source/config |
| [EfficientPhys_original.py](EfficientPhys_original.py) | No argparse interface; read the source/config |
| [audit_saved_evaluation.py](audit_saved_evaluation.py) | `directory` |
| [audit_ubfc1.py](audit_ubfc1.py) | `--root`, `--out`, `--smoke-only` |
| [build_t8_bundle.py](build_t8_bundle.py) | No argparse interface; read the source/config |
| [control_train_spectral_matches.py](control_train_spectral_matches.py) | No argparse interface; read the source/config |
| [count_loader_windows.py](count_loader_windows.py) | `--dataset`, `--root`, `--audit-dir`, `--out` |
| [cv_common.py](cv_common.py) | No argparse interface; read the source/config |
| [cv_status.py](cv_status.py) | No argparse interface; read the source/config |
| [data_adapters.py](data_adapters.py) | No argparse interface; read the source/config |
| [decode_hr_v2.py](decode_hr_v2.py) | No argparse interface; read the source/config |
| [diagnose_h264_waveforms.py](diagnose_h264_waveforms.py) | No argparse interface; read the source/config |
| [diagnose_spectral_waveforms.py](diagnose_spectral_waveforms.py) | No argparse interface; read the source/config |
| [diagnose_subject32_matched.py](diagnose_subject32_matched.py) | `--inputs-only` |
| [diagnose_train_compression.py](diagnose_train_compression.py) | No argparse interface; read the source/config |
| [diagnose_validation_regressions.py](diagnose_validation_regressions.py) | No argparse interface; read the source/config |
| [evaluate_cv_loss.py](evaluate_cv_loss.py) | No argparse interface; read the source/config |
| [evaluate_development_models.py](evaluate_development_models.py) | No argparse interface; read the source/config |
| [evaluate_end_to_end_detection.py](evaluate_end_to_end_detection.py) | No argparse interface; read the source/config |
| [evaluate_h264_training.py](evaluate_h264_training.py) | No argparse interface; read the source/config |
| [evaluate_matched_detector_roi.py](evaluate_matched_detector_roi.py) | No argparse interface; read the source/config |
| [evaluate_matched_spectral.py](evaluate_matched_spectral.py) | No argparse interface; read the source/config |
| [evaluate_roi_ablation.py](evaluate_roi_ablation.py) | No argparse interface; read the source/config |
| [evaluate_spectral_ensemble.py](evaluate_spectral_ensemble.py) | No argparse interface; read the source/config |
| [evaluate_spectral_frozen.py](evaluate_spectral_frozen.py) | No argparse interface; read the source/config |
| [evaluate_t2_historical_pos.py](evaluate_t2_historical_pos.py) | `--source`, `--out` |
| [evaluate_t2_pos.py](evaluate_t2_pos.py) | `--out`, `--ubfc-root`, `--reference`, `--mcd-root`, `--audit`, `--e0`, `--condition` |
| [evaluate_t2_saved.py](evaluate_t2_saved.py) | `--base`, `--out` |
| [evaluate_three_seeds.py](evaluate_three_seeds.py) | `--split` |
| [freeze_t8_folds.py](freeze_t8_folds.py) | No argparse interface; read the source/config |
| [hr_decoder.py](hr_decoder.py) | No argparse interface; read the source/config |
| [inference_hr.py](inference_hr.py) | No argparse interface; read the source/config |
| [input_quality.py](input_quality.py) | No argparse interface; read the source/config |
| [inspect_end_to_end_rois.py](inspect_end_to_end_rois.py) | No argparse interface; read the source/config |
| [package_cv_loss.py](package_cv_loss.py) | No argparse interface; read the source/config |
| [package_h264_results.py](package_h264_results.py) | No argparse interface; read the source/config |
| [package_matched_spectral.py](package_matched_spectral.py) | No argparse interface; read the source/config |
| [package_spectral_results.py](package_spectral_results.py) | No argparse interface; read the source/config |
| [ppg_polarity.py](ppg_polarity.py) | No argparse interface; read the source/config |
| [prepare_cv_primary.py](prepare_cv_primary.py) | No argparse interface; read the source/config |
| [prepare_cv_secondary.py](prepare_cv_secondary.py) | No argparse interface; read the source/config |
| [prepare_matched_roi_training.py](prepare_matched_roi_training.py) | No argparse interface; read the source/config |
| [prepare_recording.py](prepare_recording.py) | `--dataset`, `--root`, `--recording`, `--out`, `--manifest`, `--audit-dir`, `--sensor-offset-s` |
| [prepare_t8_cache.py](prepare_t8_cache.py) | `--dataset`, `--audit`, `--budget-bytes`, `--smoke` |
| [replay_band_e0.py](replay_band_e0.py) | `--out` |
| [report_cv_loss.py](report_cv_loss.py) | No argparse interface; read the source/config |
| [report_end_to_end_detection.py](report_end_to_end_detection.py) | No argparse interface; read the source/config |
| [report_h264_experiment.py](report_h264_experiment.py) | `--pos-only` |
| [report_matched_detector_roi.py](report_matched_detector_roi.py) | No argparse interface; read the source/config |
| [report_matched_spectral.py](report_matched_spectral.py) | No argparse interface; read the source/config |
| [report_robustness.py](report_robustness.py) | `root` |
| [report_roi_experiment.py](report_roi_experiment.py) | No argparse interface; read the source/config |
| [report_spectral_ensemble.py](report_spectral_ensemble.py) | No argparse interface; read the source/config |
| [report_spectral_experiment.py](report_spectral_experiment.py) | `--before-only` |
| [report_t2_decoder.py](report_t2_decoder.py) | `--out` |
| [report_train_signal_diagnosis.py](report_train_signal_diagnosis.py) | No argparse interface; read the source/config |
| [spectral_ensemble.py](spectral_ensemble.py) | No argparse interface; read the source/config |
| [spectral_objective.py](spectral_objective.py) | No argparse interface; read the source/config |
| [stress_ubfc.py](stress_ubfc.py) | `--out` |
| [summarize_errors.py](summarize_errors.py) | No argparse interface; read the source/config |
| [t3_landmark_report.py](t3_landmark_report.py) | `--report` |
| [t3_landmark_runner.py](t3_landmark_runner.py) | `--dataset`, `--root`, `--audit`, `--out`, `--protocol`, `--limit` |
| [t8_t1_loader.py](t8_t1_loader.py) | No argparse interface; read the source/config |
| [test_data_boundary.py](test_data_boundary.py) | No argparse interface; read the source/config |
| [test_decode_hr_v2.py](test_decode_hr_v2.py) | No argparse interface; read the source/config |
| [test_ubfc1_diagnostics.py](test_ubfc1_diagnostics.py) | No argparse interface; read the source/config |
| [test_video_landmark_frontend.py](test_video_landmark_frontend.py) | No argparse interface; read the source/config |
| [test_window_support.py](test_window_support.py) | No argparse interface; read the source/config |
| [train_compression_ablation.py](train_compression_ablation.py) | `--seed`, `--mode` |
| [train_cv_loss.py](train_cv_loss.py) | `--fold`, `--loss` |
| [train_efficientphys.py](train_efficientphys.py) | `--root`, `--out`, `--config`, `--smoke-only`, `--resume` |
| [train_h264_ablation.py](train_h264_ablation.py) | `--mode`, `--seed` |
| [train_matched_detector_roi.py](train_matched_detector_roi.py) | `--arm`, `--seed` |
| [train_matched_spectral.py](train_matched_spectral.py) | `--mode`, `--seed` |
| [train_roi_ablation.py](train_roi_ablation.py) | No argparse interface; read the source/config |
| [train_spectral_ablation.py](train_spectral_ablation.py) | `--mode`, `--seed` |
| [ubfc1_diagnostics.py](ubfc1_diagnostics.py) | `--root`, `--audit`, `--out` |
| [verify_adapter_real.py](verify_adapter_real.py) | `--root`, `--subject` |
| [verify_best.py](verify_best.py) | No argparse interface; read the source/config |
| [verify_guarded_inference.py](verify_guarded_inference.py) | No argparse interface; read the source/config |
| [verify_input_quality.py](verify_input_quality.py) | No argparse interface; read the source/config |
| [verify_matched_detector_roi.py](verify_matched_detector_roi.py) | No argparse interface; read the source/config |
| [verify_roi_caches.py](verify_roi_caches.py) | No argparse interface; read the source/config |
| [verify_spectral_ensemble.py](verify_spectral_ensemble.py) | No argparse interface; read the source/config |
| [verify_spectral_objective.py](verify_spectral_objective.py) | No argparse interface; read the source/config |
| [verify_video_roi_frontend.py](verify_video_roi_frontend.py) | No argparse interface; read the source/config |
| [video_landmark_frontend.py](video_landmark_frontend.py) | No argparse interface; read the source/config |
| [video_roi_frontend.py](video_roi_frontend.py) | No argparse interface; read the source/config |
| [wait_t2_mcd500k.py](wait_t2_mcd500k.py) | `--out`, `--root`, `--audit`, `--e0` |
| [window_support.py](window_support.py) | No argparse interface; read the source/config |
