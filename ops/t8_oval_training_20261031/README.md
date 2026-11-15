# ops/t8_oval_training_20261031

Control reused; original joint obsolete sau polarity; không chạy queue này thay V3.

[Hướng dẫn chạy](../../docs/REPRODUCING.md) · [Kết quả](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [EfficientPhys_fixed.py](EfficientPhys_fixed.py) | Không có argparse; đọc source/config |
| [score_round1.py](score_round1.py) | `--reference`, `--output`, `--metrics` |
| [t8_aggregate.py](t8_aggregate.py) | `--runs`, `--output` |
| [t8_evaluate.py](t8_evaluate.py) | Không có argparse; đọc source/config |
| [t8_plan.py](t8_plan.py) | `--ubfc`, `--mcd`, `--folds`, `--output` |
| [t8_run.py](t8_run.py) | `--index`, `--output`, `--smoke-only` |
| [t8_t1_loader.py](t8_t1_loader.py) | Không có argparse; đọc source/config |
| [test_t8.py](test_t8.py) | Không có argparse; đọc source/config |
| [window_support.py](window_support.py) | Không có argparse; đọc source/config |
