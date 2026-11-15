# reports/round1_v4_20261105/controls_20261107

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../../docs/REPRODUCING.md) · [Kết quả](../../../docs/RESULTS.md) · [Experiment map](../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [common.py](common.py) | Không có argparse; đọc source/config |
| [report_watch.py](report_watch.py) | Không có argparse; đọc source/config |
| [run_controls.py](run_controls.py) | `--watch` |
