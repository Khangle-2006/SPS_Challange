# ops/deep_hr

Adapter, waveform/support, EfficientPhys, HR decoder và kiểm thử. Nhiều thử nghiệm trước T8 là lịch sử.

[Hướng dẫn chạy](../../docs/REPRODUCING.md) · [Kết quả](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [EfficientPhys_fixed.py](EfficientPhys_fixed.py) | Không có argparse; đọc source/config |
| [EfficientPhys_original.py](EfficientPhys_original.py) | Không có argparse; đọc source/config |
| [audit_saved_evaluation.py](audit_saved_evaluation.py) | `directory` |
| [audit_ubfc1.py](audit_ubfc1.py) | `--root`, `--out`, `--smoke-only` |
| [build_t8_bundle.py](build_t8_bundle.py) | Không có argparse; đọc source/config |
| [control_train_spectral_matches.py](control_train_spectral_matches.py) | Không có argparse; đọc source/config |
| [count_loader_windows.py](count_loader_windows.py) | `--dataset`, `--root`, `--audit-dir`, `--out` |
| [cv_common.py](cv_common.py) | Không có argparse; đọc source/config |
| [cv_status.py](cv_status.py) | Không có argparse; đọc source/config |
| [data_adapters.py](data_adapters.py) | Không có argparse; đọc source/config |
| [decode_hr_v2.py](decode_hr_v2.py) | Không có argparse; đọc source/config |
| [diagnose_h264_waveforms.py](diagnose_h264_waveforms.py) | Không có argparse; đọc source/config |
| [diagnose_spectral_waveforms.py](diagnose_spectral_waveforms.py) | Không có argparse; đọc source/config |
| [diagnose_subject32_matched.py](diagnose_subject32_matched.py) | `--inputs-only` |
| [diagnose_train_compression.py](diagnose_train_compression.py) | Không có argparse; đọc source/config |
| [diagnose_validation_regressions.py](diagnose_validation_regressions.py) | Không có argparse; đọc source/config |
| [evaluate_cv_loss.py](evaluate_cv_loss.py) | Không có argparse; đọc source/config |
| [evaluate_development_models.py](evaluate_development_models.py) | Không có argparse; đọc source/config |
| [evaluate_end_to_end_detection.py](evaluate_end_to_end_detection.py) | Không có argparse; đọc source/config |
| [evaluate_h264_training.py](evaluate_h264_training.py) | Không có argparse; đọc source/config |
| [evaluate_matched_detector_roi.py](evaluate_matched_detector_roi.py) | Không có argparse; đọc source/config |
| [evaluate_matched_spectral.py](evaluate_matched_spectral.py) | Không có argparse; đọc source/config |
| [evaluate_roi_ablation.py](evaluate_roi_ablation.py) | Không có argparse; đọc source/config |
| [evaluate_spectral_ensemble.py](evaluate_spectral_ensemble.py) | Không có argparse; đọc source/config |
| [evaluate_spectral_frozen.py](evaluate_spectral_frozen.py) | Không có argparse; đọc source/config |
| [evaluate_t2_historical_pos.py](evaluate_t2_historical_pos.py) | `--source`, `--out` |
| [evaluate_t2_pos.py](evaluate_t2_pos.py) | `--out`, `--ubfc-root`, `--reference`, `--mcd-root`, `--audit`, `--e0`, `--condition` |
| [evaluate_t2_saved.py](evaluate_t2_saved.py) | `--base`, `--out` |
| [evaluate_three_seeds.py](evaluate_three_seeds.py) | `--split` |
| [freeze_t8_folds.py](freeze_t8_folds.py) | Không có argparse; đọc source/config |
| [hr_decoder.py](hr_decoder.py) | Không có argparse; đọc source/config |
| [inference_hr.py](inference_hr.py) | Không có argparse; đọc source/config |
| [input_quality.py](input_quality.py) | Không có argparse; đọc source/config |
| [inspect_end_to_end_rois.py](inspect_end_to_end_rois.py) | Không có argparse; đọc source/config |
| [package_cv_loss.py](package_cv_loss.py) | Không có argparse; đọc source/config |
| [package_h264_results.py](package_h264_results.py) | Không có argparse; đọc source/config |
| [package_matched_spectral.py](package_matched_spectral.py) | Không có argparse; đọc source/config |
| [package_spectral_results.py](package_spectral_results.py) | Không có argparse; đọc source/config |
| [ppg_polarity.py](ppg_polarity.py) | Không có argparse; đọc source/config |
| [prepare_cv_primary.py](prepare_cv_primary.py) | Không có argparse; đọc source/config |
| [prepare_cv_secondary.py](prepare_cv_secondary.py) | Không có argparse; đọc source/config |
| [prepare_matched_roi_training.py](prepare_matched_roi_training.py) | Không có argparse; đọc source/config |
| [prepare_recording.py](prepare_recording.py) | `--dataset`, `--root`, `--recording`, `--out`, `--manifest`, `--audit-dir`, `--sensor-offset-s` |
| [prepare_t8_cache.py](prepare_t8_cache.py) | `--dataset`, `--audit`, `--budget-bytes`, `--smoke` |
| [replay_band_e0.py](replay_band_e0.py) | `--out` |
| [report_cv_loss.py](report_cv_loss.py) | Không có argparse; đọc source/config |
| [report_end_to_end_detection.py](report_end_to_end_detection.py) | Không có argparse; đọc source/config |
| [report_h264_experiment.py](report_h264_experiment.py) | `--pos-only` |
| [report_matched_detector_roi.py](report_matched_detector_roi.py) | Không có argparse; đọc source/config |
| [report_matched_spectral.py](report_matched_spectral.py) | Không có argparse; đọc source/config |
| [report_robustness.py](report_robustness.py) | `root` |
| [report_roi_experiment.py](report_roi_experiment.py) | Không có argparse; đọc source/config |
| [report_spectral_ensemble.py](report_spectral_ensemble.py) | Không có argparse; đọc source/config |
| [report_spectral_experiment.py](report_spectral_experiment.py) | `--before-only` |
| [report_t2_decoder.py](report_t2_decoder.py) | `--out` |
| [report_train_signal_diagnosis.py](report_train_signal_diagnosis.py) | Không có argparse; đọc source/config |
| [spectral_ensemble.py](spectral_ensemble.py) | Không có argparse; đọc source/config |
| [spectral_objective.py](spectral_objective.py) | Không có argparse; đọc source/config |
| [stress_ubfc.py](stress_ubfc.py) | `--out` |
| [summarize_errors.py](summarize_errors.py) | Không có argparse; đọc source/config |
| [t3_landmark_report.py](t3_landmark_report.py) | `--report` |
| [t3_landmark_runner.py](t3_landmark_runner.py) | `--dataset`, `--root`, `--audit`, `--out`, `--protocol`, `--limit` |
| [t8_t1_loader.py](t8_t1_loader.py) | Không có argparse; đọc source/config |
| [test_data_boundary.py](test_data_boundary.py) | Không có argparse; đọc source/config |
| [test_decode_hr_v2.py](test_decode_hr_v2.py) | Không có argparse; đọc source/config |
| [test_ubfc1_diagnostics.py](test_ubfc1_diagnostics.py) | Không có argparse; đọc source/config |
| [test_video_landmark_frontend.py](test_video_landmark_frontend.py) | Không có argparse; đọc source/config |
| [test_window_support.py](test_window_support.py) | Không có argparse; đọc source/config |
| [train_compression_ablation.py](train_compression_ablation.py) | `--seed`, `--mode` |
| [train_cv_loss.py](train_cv_loss.py) | `--fold`, `--loss` |
| [train_efficientphys.py](train_efficientphys.py) | `--root`, `--out`, `--config`, `--smoke-only`, `--resume` |
| [train_h264_ablation.py](train_h264_ablation.py) | `--mode`, `--seed` |
| [train_matched_detector_roi.py](train_matched_detector_roi.py) | `--arm`, `--seed` |
| [train_matched_spectral.py](train_matched_spectral.py) | `--mode`, `--seed` |
| [train_roi_ablation.py](train_roi_ablation.py) | Không có argparse; đọc source/config |
| [train_spectral_ablation.py](train_spectral_ablation.py) | `--mode`, `--seed` |
| [ubfc1_diagnostics.py](ubfc1_diagnostics.py) | `--root`, `--audit`, `--out` |
| [verify_adapter_real.py](verify_adapter_real.py) | `--root`, `--subject` |
| [verify_best.py](verify_best.py) | Không có argparse; đọc source/config |
| [verify_guarded_inference.py](verify_guarded_inference.py) | Không có argparse; đọc source/config |
| [verify_input_quality.py](verify_input_quality.py) | Không có argparse; đọc source/config |
| [verify_matched_detector_roi.py](verify_matched_detector_roi.py) | Không có argparse; đọc source/config |
| [verify_roi_caches.py](verify_roi_caches.py) | Không có argparse; đọc source/config |
| [verify_spectral_ensemble.py](verify_spectral_ensemble.py) | Không có argparse; đọc source/config |
| [verify_spectral_objective.py](verify_spectral_objective.py) | Không có argparse; đọc source/config |
| [verify_video_roi_frontend.py](verify_video_roi_frontend.py) | Không có argparse; đọc source/config |
| [video_landmark_frontend.py](video_landmark_frontend.py) | Không có argparse; đọc source/config |
| [video_roi_frontend.py](video_roi_frontend.py) | Không có argparse; đọc source/config |
| [wait_t2_mcd500k.py](wait_t2_mcd500k.py) | `--out`, `--root`, `--audit`, `--e0` |
| [window_support.py](window_support.py) | Không có argparse; đọc source/config |
