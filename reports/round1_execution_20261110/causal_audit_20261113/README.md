# reports/round1_execution_20261110/causal_audit_20261113

Batch/lineage/ladder_physical_v3.py/compare_stage_exports; clockV3 là authoritative.

[Hướng dẫn chạy](../../../docs/REPRODUCING.md) · [Kết quả](../../../docs/RESULTS.md) · [Experiment map](../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [audit_metadata.py](audit_metadata.py) | Không có argparse; đọc source/config |
| [audit_workbook_selected.py](audit_workbook_selected.py) | Không có argparse; đọc source/config |
| [backup_partial.py](backup_partial.py) | Không có argparse; đọc source/config |
| [backup_progress_v2.py](backup_progress_v2.py) | Không có argparse; đọc source/config |
| [backup_progress_v3.py](backup_progress_v3.py) | Không có argparse; đọc source/config |
| [batch_audit.py](batch_audit.py) | Không có argparse; đọc source/config |
| [common_causal.py](common_causal.py) | Không có argparse; đọc source/config |
| [compare_frontends.py](compare_frontends.py) | Không có argparse; đọc source/config |
| [compare_interventions.py](compare_interventions.py) | Không có argparse; đọc source/config |
| [cross_route.py](cross_route.py) | Không có argparse; đọc source/config |
| [encoder_clock_probe_v4.py](encoder_clock_probe_v4.py) | Không có argparse; đọc source/config |
| [export_causal.py](export_causal.py) | Không có argparse; đọc source/config |
| [ladder.py](ladder.py) | Không có argparse; đọc source/config |
| [ladder_infer.py](ladder_infer.py) | Không có argparse; đọc source/config |
| [ladder_infer_v2.py](ladder_infer_v2.py) | Không có argparse; đọc source/config |
| [ladder_physical_v3.py](ladder_physical_v3.py) | Không có argparse; đọc source/config |
| [ladder_prefix_v3.py](ladder_prefix_v3.py) | Không có argparse; đọc source/config |
| [ladder_recovery_v4.py](ladder_recovery_v4.py) | Không có argparse; đọc source/config |
| [ladder_v2.py](ladder_v2.py) | Không có argparse; đọc source/config |
| [localization_all.py](localization_all.py) | Không có argparse; đọc source/config |
| [metadata_report_v2.py](metadata_report_v2.py) | Không có argparse; đọc source/config |
| [orientation_localization.py](orientation_localization.py) | Không có argparse; đọc source/config |
| [prefix_clock_audit.py](prefix_clock_audit.py) | Không có argparse; đọc source/config |
| [reconcile_public_clock.py](reconcile_public_clock.py) | Không có argparse; đọc source/config |
| [recover_orientation.py](recover_orientation.py) | Không có argparse; đọc source/config |
| [repair_prefix.py](repair_prefix.py) | Không có argparse; đọc source/config |
| [report_causal.py](report_causal.py) | Không có argparse; đọc source/config |
| [review_controller.py](review_controller.py) | Không có argparse; đọc source/config |
| [review_controller_v2.py](review_controller_v2.py) | Không có argparse; đọc source/config |
| [review_controller_v3.py](review_controller_v3.py) | Không có argparse; đọc source/config |
| [review_controller_v4.py](review_controller_v4.py) | Không có argparse; đọc source/config |
| [start_inspected_recovery_v4.py](start_inspected_recovery_v4.py) | Không có argparse; đọc source/config |
