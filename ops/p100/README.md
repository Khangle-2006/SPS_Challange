# ops/p100

POS/CHROM and MCD runners; not the calibrated BTC estimator.

[Reproduction guide](../../docs/REPRODUCING.md) · [Results](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

The parameters below were extracted statically from the code; they are not commands that automatically launch a run. For scripts without a CLI, check the constants, paths, and dependencies at the top of the file. Controllers and site runners require the original receipts and environments; do not bypass their guards.

| File | Declared CLI parameters |
|---|---|
| [aipex_baseline.py](aipex_baseline.py) | `--root`, `--out`, `--workers` |
| [audit_aipex.py](audit_aipex.py) | No argparse interface; read the source/config |
| [audit_aipex_timing_gate.py](audit_aipex_timing_gate.py) | No argparse interface; read the source/config |
| [audit_mcd_clocks.py](audit_mcd_clocks.py) | No argparse interface; read the source/config |
| [collect_sac_reference.py](collect_sac_reference.py) | No argparse interface; read the source/config |
| [consolidate_datasets_20261030.py](consolidate_datasets_20261030.py) | No argparse interface; read the source/config |
| [download_aipex.py](download_aipex.py) | No argparse interface; read the source/config |
| [download_direct_datasets.py](download_direct_datasets.py) | No argparse interface; read the source/config |
| [download_mcd.py](download_mcd.py) | `root`, `--metadata-only` |
| [download_ubfc2.py](download_ubfc2.py) | No argparse interface; read the source/config |
| [finish_mcd_audit.py](finish_mcd_audit.py) | No argparse interface; read the source/config |
| [gpu_smoke_test.py](gpu_smoke_test.py) | No argparse interface; read the source/config |
| [hr_baseline.py](hr_baseline.py) | `--root`, `--out`, `--self-test` |
| [mcd_aggregate_mapping.py](mcd_aggregate_mapping.py) | No argparse interface; read the source/config |
| [mcd_compose_comparison.py](mcd_compose_comparison.py) | `--results`, `--author-image`, `--out`, `--partial` |
| [mcd_crop_ffv1_measure.py](mcd_crop_ffv1_measure.py) | No argparse interface; read the source/config |
| [mcd_development_gate.py](mcd_development_gate.py) | `--root`, `--audit`, `--output`, `--limit` |
| [mcd_development_gate_v3.py](mcd_development_gate_v3.py) | `--root`, `--audit`, `--output`, `--limit` |
| [mcd_e0_metrics.py](mcd_e0_metrics.py) | No argparse interface; read the source/config |
| [mcd_e0_report.py](mcd_e0_report.py) | `--output` |
| [mcd_e0_runner.py](mcd_e0_runner.py) | `--root`, `--audit`, `--output`, `--protocol`, `--limit` |
| [mcd_errata_audit.py](mcd_errata_audit.py) | `--root`, `--audit`, `--fold-csv` |
| [mcd_figure5_metrics.py](mcd_figure5_metrics.py) | No argparse interface; read the source/config |
| [mcd_header_gate.py](mcd_header_gate.py) | `--root`, `--audit` |
| [mcd_lag_reading.py](mcd_lag_reading.py) | No argparse interface; read the source/config |
| [mcd_lag_runner.py](mcd_lag_runner.py) | `--root`, `--audit`, `--output`, `--wait-gate`, `--limit` |
| [mcd_packet_audit.py](mcd_packet_audit.py) | `--plan`, `--subject` |
| [mcd_plan_development.py](mcd_plan_development.py) | No argparse interface; read the source/config |
| [mcd_resume_development.py](mcd_resume_development.py) | No argparse interface; read the source/config |
| [mcd_riff_tail.py](mcd_riff_tail.py) | No argparse interface; read the source/config |
| [mcd_summarize_live.py](mcd_summarize_live.py) | `--audit` |
| [mcd_sync_prefix_check.py](mcd_sync_prefix_check.py) | No argparse interface; read the source/config |
| [mcd_timestamp_histogram.py](mcd_timestamp_histogram.py) | No argparse interface; read the source/config |
| [mcd_v3_rules.py](mcd_v3_rules.py) | No argparse interface; read the source/config |
| [pilot_mcd_storage.py](pilot_mcd_storage.py) | No argparse interface; read the source/config |
| [plan_mmpd_subset.py](plan_mmpd_subset.py) | No argparse interface; read the source/config |
| [refine_mmpd_conditions.py](refine_mmpd_conditions.py) | No argparse interface; read the source/config |
| [report_t6_e0.py](report_t6_e0.py) | No argparse interface; read the source/config |
| [report_t6_t5.py](report_t6_t5.py) | `--input`, `--output` |
| [report_t8_crop_gate.py](report_t8_crop_gate.py) | `--input`, `--output` |
| [summarize_e0_branch.py](summarize_e0_branch.py) | `--report` |
| [t7_frontend.py](t7_frontend.py) | `--limit` |
| [t7_frontend_report.py](t7_frontend_report.py) | `root` |
| [t7_pipeline.py](t7_pipeline.py) | No argparse interface; read the source/config |
| [t7_raw_reference_v2.py](t7_raw_reference_v2.py) | No argparse interface; read the source/config |
| [t7_report.py](t7_report.py) | `root`, `--raw-v2` |
| [t7_window_diagnostic.py](t7_window_diagnostic.py) | No argparse interface; read the source/config |
| [t8_crop_gate.py](t8_crop_gate.py) | No argparse interface; read the source/config |
| [test_mcd_e0.py](test_mcd_e0.py) | No argparse interface; read the source/config |
| [test_mcd_figure5_metrics.py](test_mcd_figure5_metrics.py) | No argparse interface; read the source/config |
| [test_mcd_header_gate.py](test_mcd_header_gate.py) | No argparse interface; read the source/config |
| [test_mcd_lag_reading.py](test_mcd_lag_reading.py) | No argparse interface; read the source/config |
| [test_mcd_lag_runner.py](test_mcd_lag_runner.py) | No argparse interface; read the source/config |
| [test_mcd_summarize_live.py](test_mcd_summarize_live.py) | No argparse interface; read the source/config |
| [test_mcd_v3_rules.py](test_mcd_v3_rules.py) | No argparse interface; read the source/config |
| [update_storage_policy_mapping.py](update_storage_policy_mapping.py) | No argparse interface; read the source/config |
| [write_mapping_report_local.py](write_mapping_report_local.py) | No argparse interface; read the source/config |
