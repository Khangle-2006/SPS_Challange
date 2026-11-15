# ops/t8_label_diagnostic_20261102

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../docs/REPRODUCING.md) · [Kết quả](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [loss_probe.py](loss_probe.py) | `--code`, `--index`, `--runs`, `--output` |
| [render_results.py](render_results.py) | Không có argparse; đọc source/config |
| [signal_checks.py](signal_checks.py) | `--index`, `--output` |
| [verify_staged.py](verify_staged.py) | Không có argparse; đọc source/config |
| [verify_staged_v3.py](verify_staged_v3.py) | Không có argparse; đọc source/config |
