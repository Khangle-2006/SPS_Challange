# ops/deep_hr/reports/h264_experiment_20261015/code/deep_hr

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../../../../../docs/REPRODUCING.md) · [Kết quả](../../../../../../docs/RESULTS.md) · [Experiment map](../../../../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [evaluate_h264_training.py](evaluate_h264_training.py) | Không có argparse; đọc source/config |
| [package_h264_results.py](package_h264_results.py) | Không có argparse; đọc source/config |
| [train_h264_ablation.py](train_h264_ablation.py) | `--mode`, `--seed` |
