# ops/constraints

C1–C5, frame/codec support, kiểm bitrate.

[Hướng dẫn chạy](../../docs/REPRODUCING.md) · [Kết quả](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [report_t5.py](report_t5.py) | `root` |
| [run_t5.py](run_t5.py) | `--dataset`, `--root`, `--audit`, `--output`, `--shared`, `--limit`, `--budget-gb` |
| [simulate.py](simulate.py) | Không có argparse; đọc source/config |
| [test_simulate.py](test_simulate.py) | Không có argparse; đọc source/config |
