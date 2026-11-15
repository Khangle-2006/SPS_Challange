# ops/deep_hr/reports/cv_5fold_20261020/code/deep_hr

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../../../../../docs/REPRODUCING.md) · [Kết quả](../../../../../../docs/RESULTS.md) · [Experiment map](../../../../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [EfficientPhys_fixed.py](EfficientPhys_fixed.py) | Không có argparse; đọc source/config |
| [cv_common.py](cv_common.py) | Không có argparse; đọc source/config |
| [encode_cv_secondary.py](encode_cv_secondary.py) | `--rate` |
| [evaluate_cv_loss.py](evaluate_cv_loss.py) | Không có argparse; đọc source/config |
| [hr_baseline.py](hr_baseline.py) | `--root`, `--out`, `--self-test` |
| [package_cv_loss.py](package_cv_loss.py) | Không có argparse; đọc source/config |
| [prepare_cv_primary.py](prepare_cv_primary.py) | Không có argparse; đọc source/config |
| [prepare_cv_secondary.py](prepare_cv_secondary.py) | Không có argparse; đọc source/config |
| [spectral_ensemble.py](spectral_ensemble.py) | Không có argparse; đọc source/config |
| [spectral_objective.py](spectral_objective.py) | Không có argparse; đọc source/config |
| [train_cv_loss.py](train_cv_loss.py) | `--fold`, `--loss` |
| [train_efficientphys.py](train_efficientphys.py) | `--root`, `--out`, `--config`, `--smoke-only`, `--resume` |
| [video_roi_frontend.py](video_roi_frontend.py) | Không có argparse; đọc source/config |
