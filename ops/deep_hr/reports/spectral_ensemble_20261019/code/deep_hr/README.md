# ops/deep_hr/reports/spectral_ensemble_20261019/code/deep_hr

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../../../../../docs/REPRODUCING.md) · [Kết quả](../../../../../../docs/RESULTS.md) · [Experiment map](../../../../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [EfficientPhys_fixed.py](EfficientPhys_fixed.py) | Không có argparse; đọc source/config |
| [evaluate_spectral_ensemble.py](evaluate_spectral_ensemble.py) | Không có argparse; đọc source/config |
| [hr_baseline.py](hr_baseline.py) | `--root`, `--out`, `--self-test` |
| [spectral_ensemble.py](spectral_ensemble.py) | Không có argparse; đọc source/config |
| [train_efficientphys.py](train_efficientphys.py) | `--root`, `--out`, `--config`, `--smoke-only`, `--resume` |
| [verify_spectral_ensemble.py](verify_spectral_ensemble.py) | Không có argparse; đọc source/config |
