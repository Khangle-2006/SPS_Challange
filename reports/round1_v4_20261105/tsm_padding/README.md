# reports/round1_v4_20261105/tsm_padding

replicate_tsm.py, test_static_tsm.py, check_checkpoint_static.py.

[Hướng dẫn chạy](../../../docs/REPRODUCING.md) · [Kết quả](../../../docs/RESULTS.md) · [Experiment map](../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [audit.py](audit.py) | Không có argparse; đọc source/config |
| [check_checkpoint_static.py](check_checkpoint_static.py) | `checkpoint`, `--padding` |
| [check_original_cpu.py](check_original_cpu.py) | Không có argparse; đọc source/config |
| [paired_summary.py](paired_summary.py) | Không có argparse; đọc source/config |
| [replicate_tsm.py](replicate_tsm.py) | Không có argparse; đọc source/config |
| [test_static_tsm.py](test_static_tsm.py) | Không có argparse; đọc source/config |
