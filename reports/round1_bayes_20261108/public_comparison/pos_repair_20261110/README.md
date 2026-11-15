# reports/round1_bayes_20261108/public_comparison/pos_repair_20261110

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../../../docs/REPRODUCING.md) · [Kết quả](../../../../docs/RESULTS.md) · [Experiment map](../../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [audit.py](audit.py) | Không có argparse; đọc source/config |
| [pos_corrected.py](pos_corrected.py) | Không có argparse; đọc source/config |
| [process_pos.py](process_pos.py) | `--spatial-npz`, `--start` |
| [summarize_crossed.py](summarize_crossed.py) | Không có argparse; đọc source/config |
| [test_corrected.py](test_corrected.py) | Không có argparse; đọc source/config |
| [validate_corrected.py](validate_corrected.py) | Không có argparse; đọc source/config |
