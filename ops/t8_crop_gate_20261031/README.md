# ops/t8_crop_gate_20261031

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../docs/REPRODUCING.md) · [Kết quả](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [data_adapters.py](data_adapters.py) | Không có argparse; đọc source/config |
| [hr_baseline.py](hr_baseline.py) | `--root`, `--out`, `--self-test` |
| [mcd_e0_metrics.py](mcd_e0_metrics.py) | Không có argparse; đọc source/config |
| [t8_crop_gate.py](t8_crop_gate.py) | Không có argparse; đọc source/config |
