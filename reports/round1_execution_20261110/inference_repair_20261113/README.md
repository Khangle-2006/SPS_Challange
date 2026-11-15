# reports/round1_execution_20261110/inference_repair_20261113

Four spatial branches + readout, report_repair.py và report_spatial.py. Không promotion.

[Hướng dẫn chạy](../../../docs/REPRODUCING.md) · [Kết quả](../../../docs/RESULTS.md) · [Experiment map](../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [audit_baseline.py](audit_baseline.py) | Không có argparse; đọc source/config |
| [backup_repair.py](backup_repair.py) | Không có argparse; đọc source/config |
| [classical_original.py](classical_original.py) | Không có argparse; đọc source/config |
| [classical_original_v2.py](classical_original_v2.py) | Không có argparse; đọc source/config |
| [classical_repair.py](classical_repair.py) | Không có argparse; đọc source/config |
| [classical_repair_v2.py](classical_repair_v2.py) | Không có argparse; đọc source/config |
| [codec_band_v2.py](codec_band_v2.py) | Không có argparse; đọc source/config |
| [codec_four.py](codec_four.py) | Không có argparse; đọc source/config |
| [common_repair.py](common_repair.py) | Không có argparse; đọc source/config |
| [crop_io.py](crop_io.py) | Không có argparse; đọc source/config |
| [freeze.py](freeze.py) | Không có argparse; đọc source/config |
| [infer_original.py](infer_original.py) | Không có argparse; đọc source/config |
| [infer_spatial.py](infer_spatial.py) | Không có argparse; đọc source/config |
| [infer_spatial_v2.py](infer_spatial_v2.py) | Không có argparse; đọc source/config |
| [infer_spatial_v3.py](infer_spatial_v3.py) | Không có argparse; đọc source/config |
| [orientation_audit.py](orientation_audit.py) | `shard`, `--shards` |
| [orientation_audit_v2.py](orientation_audit_v2.py) | `shard`, `--shards` |
| [orientation_recovery_01.py](orientation_recovery_01.py) | Không có argparse; đọc source/config |
| [prepare_chrom_band_v2.py](prepare_chrom_band_v2.py) | Không có argparse; đọc source/config |
| [probe_readout.py](probe_readout.py) | Không có argparse; đọc source/config |
| [pull_legacy.py](pull_legacy.py) | Không có argparse; đọc source/config |
| [readout.py](readout.py) | Không có argparse; đọc source/config |
| [readout_native_summary.py](readout_native_summary.py) | Không có argparse; đọc source/config |
| [readout_public_summary.py](readout_public_summary.py) | Không có argparse; đọc source/config |
| [reference_repair.py](reference_repair.py) | Không có argparse; đọc source/config |
| [render_controller.py](render_controller.py) | Không có argparse; đọc source/config |
| [render_repair.py](render_repair.py) | Không có argparse; đọc source/config |
| [report_controller.py](report_controller.py) | Không có argparse; đọc source/config |
| [report_controller_v2.py](report_controller_v2.py) | Không có argparse; đọc source/config |
| [report_repair.py](report_repair.py) | `--stage` |
| [review_saved_panels.py](review_saved_panels.py) | Không có argparse; đọc source/config |
| [spatial.py](spatial.py) | Không có argparse; đọc source/config |
| [spatial_stream.py](spatial_stream.py) | Không có argparse; đọc source/config |
| [test_spatial.py](test_spatial.py) | Không có argparse; đọc source/config |
| [visual_repair.py](visual_repair.py) | Không có argparse; đọc source/config |
| [write_early_report.py](write_early_report.py) | Không có argparse; đọc source/config |
