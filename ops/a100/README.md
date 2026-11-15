# ops/a100

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../docs/REPRODUCING.md) · [Kết quả](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [audit_label_values.py](audit_label_values.py) | Không có argparse; đọc source/config |
| [check_report.py](check_report.py) | Không có argparse; đọc source/config |
| [compact_ubfc2_20261030.py](compact_ubfc2_20261030.py) | Không có argparse; đọc source/config |
| [encode_cv_secondary.py](encode_cv_secondary.py) | `--rate` |
| [inspect_h264_roi_jump.py](inspect_h264_roi_jump.py) | Không có argparse; đọc source/config |
| [inspect_h264_rois.py](inspect_h264_rois.py) | Không có argparse; đọc source/config |
| [pilot_ffv1_20261030.py](pilot_ffv1_20261030.py) | `--subject`, `--replace-verified` |
| [pos_h264_validation.py](pos_h264_validation.py) | `--split` |
| [prepare_efficientphys.py](prepare_efficientphys.py) | Không có argparse; đọc source/config |
| [prepare_efficientphys_test.py](prepare_efficientphys_test.py) | Không có argparse; đọc source/config |
| [prepare_end_to_end_inputs.py](prepare_end_to_end_inputs.py) | Không có argparse; đọc source/config |
| [prepare_h264_rates.py](prepare_h264_rates.py) | `--rate` |
| [prepare_h264_training.py](prepare_h264_training.py) | Không có argparse; đọc source/config |
| [prepare_h264_validation.py](prepare_h264_validation.py) | Không có argparse; đọc source/config |
| [prepare_roi_ablation.py](prepare_roi_ablation.py) | `--stage` |
| [run_ubfc_cpu.py](run_ubfc_cpu.py) | Không có argparse; đọc source/config |
| [verify_source_checksums.py](verify_source_checksums.py) | Không có argparse; đọc source/config |
