# ops/p100

POS/CHROM và runners MCD; không phải BTC calibrated estimator.

[Hướng dẫn chạy](../../docs/REPRODUCING.md) · [Kết quả](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [aipex_baseline.py](aipex_baseline.py) | `--root`, `--out`, `--workers` |
| [audit_aipex.py](audit_aipex.py) | Không có argparse; đọc source/config |
| [audit_aipex_timing_gate.py](audit_aipex_timing_gate.py) | Không có argparse; đọc source/config |
| [audit_mcd_clocks.py](audit_mcd_clocks.py) | Không có argparse; đọc source/config |
| [collect_sac_reference.py](collect_sac_reference.py) | Không có argparse; đọc source/config |
| [consolidate_datasets_20261030.py](consolidate_datasets_20261030.py) | Không có argparse; đọc source/config |
| [download_aipex.py](download_aipex.py) | Không có argparse; đọc source/config |
| [download_direct_datasets.py](download_direct_datasets.py) | Không có argparse; đọc source/config |
| [download_mcd.py](download_mcd.py) | `root`, `--metadata-only` |
| [download_ubfc2.py](download_ubfc2.py) | Không có argparse; đọc source/config |
| [finish_mcd_audit.py](finish_mcd_audit.py) | Không có argparse; đọc source/config |
| [gpu_smoke_test.py](gpu_smoke_test.py) | Không có argparse; đọc source/config |
| [hr_baseline.py](hr_baseline.py) | `--root`, `--out`, `--self-test` |
| [mcd_aggregate_mapping.py](mcd_aggregate_mapping.py) | Không có argparse; đọc source/config |
| [mcd_compose_comparison.py](mcd_compose_comparison.py) | `--results`, `--author-image`, `--out`, `--partial` |
| [mcd_crop_ffv1_measure.py](mcd_crop_ffv1_measure.py) | Không có argparse; đọc source/config |
| [mcd_development_gate.py](mcd_development_gate.py) | `--root`, `--audit`, `--output`, `--limit` |
| [mcd_development_gate_v3.py](mcd_development_gate_v3.py) | `--root`, `--audit`, `--output`, `--limit` |
| [mcd_e0_metrics.py](mcd_e0_metrics.py) | Không có argparse; đọc source/config |
| [mcd_e0_report.py](mcd_e0_report.py) | `--output` |
| [mcd_e0_runner.py](mcd_e0_runner.py) | `--root`, `--audit`, `--output`, `--protocol`, `--limit` |
| [mcd_errata_audit.py](mcd_errata_audit.py) | `--root`, `--audit`, `--fold-csv` |
| [mcd_figure5_metrics.py](mcd_figure5_metrics.py) | Không có argparse; đọc source/config |
| [mcd_header_gate.py](mcd_header_gate.py) | `--root`, `--audit` |
| [mcd_lag_reading.py](mcd_lag_reading.py) | Không có argparse; đọc source/config |
| [mcd_lag_runner.py](mcd_lag_runner.py) | `--root`, `--audit`, `--output`, `--wait-gate`, `--limit` |
| [mcd_packet_audit.py](mcd_packet_audit.py) | `--plan`, `--subject` |
| [mcd_plan_development.py](mcd_plan_development.py) | Không có argparse; đọc source/config |
| [mcd_resume_development.py](mcd_resume_development.py) | Không có argparse; đọc source/config |
| [mcd_riff_tail.py](mcd_riff_tail.py) | Không có argparse; đọc source/config |
| [mcd_summarize_live.py](mcd_summarize_live.py) | `--audit` |
| [mcd_sync_prefix_check.py](mcd_sync_prefix_check.py) | Không có argparse; đọc source/config |
| [mcd_timestamp_histogram.py](mcd_timestamp_histogram.py) | Không có argparse; đọc source/config |
| [mcd_v3_rules.py](mcd_v3_rules.py) | Không có argparse; đọc source/config |
| [pilot_mcd_storage.py](pilot_mcd_storage.py) | Không có argparse; đọc source/config |
| [plan_mmpd_subset.py](plan_mmpd_subset.py) | Không có argparse; đọc source/config |
| [refine_mmpd_conditions.py](refine_mmpd_conditions.py) | Không có argparse; đọc source/config |
| [report_t6_e0.py](report_t6_e0.py) | Không có argparse; đọc source/config |
| [report_t6_t5.py](report_t6_t5.py) | `--input`, `--output` |
| [report_t8_crop_gate.py](report_t8_crop_gate.py) | `--input`, `--output` |
| [summarize_e0_branch.py](summarize_e0_branch.py) | `--report` |
| [t7_frontend.py](t7_frontend.py) | `--limit` |
| [t7_frontend_report.py](t7_frontend_report.py) | `root` |
| [t7_pipeline.py](t7_pipeline.py) | Không có argparse; đọc source/config |
| [t7_raw_reference_v2.py](t7_raw_reference_v2.py) | Không có argparse; đọc source/config |
| [t7_report.py](t7_report.py) | `root`, `--raw-v2` |
| [t7_window_diagnostic.py](t7_window_diagnostic.py) | Không có argparse; đọc source/config |
| [t8_crop_gate.py](t8_crop_gate.py) | Không có argparse; đọc source/config |
| [test_mcd_e0.py](test_mcd_e0.py) | Không có argparse; đọc source/config |
| [test_mcd_figure5_metrics.py](test_mcd_figure5_metrics.py) | Không có argparse; đọc source/config |
| [test_mcd_header_gate.py](test_mcd_header_gate.py) | Không có argparse; đọc source/config |
| [test_mcd_lag_reading.py](test_mcd_lag_reading.py) | Không có argparse; đọc source/config |
| [test_mcd_lag_runner.py](test_mcd_lag_runner.py) | Không có argparse; đọc source/config |
| [test_mcd_summarize_live.py](test_mcd_summarize_live.py) | Không có argparse; đọc source/config |
| [test_mcd_v3_rules.py](test_mcd_v3_rules.py) | Không có argparse; đọc source/config |
| [update_storage_policy_mapping.py](update_storage_policy_mapping.py) | Không có argparse; đọc source/config |
| [write_mapping_report_local.py](write_mapping_report_local.py) | Không có argparse; đọc source/config |
