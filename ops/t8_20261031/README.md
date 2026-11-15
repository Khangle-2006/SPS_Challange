# ops/t8_20261031

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../docs/REPRODUCING.md) · [Kết quả](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [data_adapters.py](data_adapters.py) | Không có argparse; đọc source/config |
| [prepare_recording.py](prepare_recording.py) | `--dataset`, `--root`, `--recording`, `--out`, `--manifest`, `--audit-dir`, `--sensor-offset-s` |
| [prepare_t8_cache.py](prepare_t8_cache.py) | `--dataset`, `--audit`, `--budget-bytes`, `--smoke` |
| [simulate.py](simulate.py) | Không có argparse; đọc source/config |
| [t8_t1_loader.py](t8_t1_loader.py) | Không có argparse; đọc source/config |
| [video_roi_frontend.py](video_roi_frontend.py) | Không có argparse; đọc source/config |
| [window_support.py](window_support.py) | Không có argparse; đọc source/config |
