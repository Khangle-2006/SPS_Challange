# ops/deep_hr/reports/spectral_experiment_20261015/code/deep_hr

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../../../../../docs/REPRODUCING.md) · [Kết quả](../../../../../../docs/RESULTS.md) · [Experiment map](../../../../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [diagnose_h264_waveforms.py](diagnose_h264_waveforms.py) | Không có argparse; đọc source/config |
| [diagnose_spectral_waveforms.py](diagnose_spectral_waveforms.py) | Không có argparse; đọc source/config |
| [evaluate_spectral_frozen.py](evaluate_spectral_frozen.py) | Không có argparse; đọc source/config |
| [package_spectral_results.py](package_spectral_results.py) | Không có argparse; đọc source/config |
| [spectral_objective.py](spectral_objective.py) | Không có argparse; đọc source/config |
| [train_spectral_ablation.py](train_spectral_ablation.py) | `--mode`, `--seed` |
| [verify_spectral_objective.py](verify_spectral_objective.py) | Không có argparse; đọc source/config |
