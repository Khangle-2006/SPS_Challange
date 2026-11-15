# ops/deep_hr/reports/matched_spectral_20261019/code/deep_hr

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../../../../../docs/REPRODUCING.md) · [Kết quả](../../../../../../docs/RESULTS.md) · [Experiment map](../../../../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [EfficientPhys_fixed.py](EfficientPhys_fixed.py) | Không có argparse; đọc source/config |
| [diagnose_subject32_matched.py](diagnose_subject32_matched.py) | `--inputs-only` |
| [evaluate_matched_spectral.py](evaluate_matched_spectral.py) | Không có argparse; đọc source/config |
| [hr_baseline.py](hr_baseline.py) | `--root`, `--out`, `--self-test` |
| [package_matched_spectral.py](package_matched_spectral.py) | Không có argparse; đọc source/config |
| [spectral_objective.py](spectral_objective.py) | Không có argparse; đọc source/config |
| [train_efficientphys.py](train_efficientphys.py) | `--root`, `--out`, `--config`, `--smoke-only`, `--resume` |
| [train_matched_detector_roi.py](train_matched_detector_roi.py) | `--arm`, `--seed` |
| [train_matched_spectral.py](train_matched_spectral.py) | `--mode`, `--seed` |
