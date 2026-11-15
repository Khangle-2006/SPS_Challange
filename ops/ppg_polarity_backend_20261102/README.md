# ops/ppg_polarity_backend_20261102

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../docs/REPRODUCING.md) · [Kết quả](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [check_dev_morphology.py](check_dev_morphology.py) | `--index`, `--output` |
| [data_adapters.py](data_adapters.py) | Không có argparse; đọc source/config |
| [morphology_contract.py](morphology_contract.py) | Không có argparse; đọc source/config |
| [ppg_polarity.py](ppg_polarity.py) | Không có argparse; đọc source/config |
| [source_data_adapters.py](source_data_adapters.py) | Không có argparse; đọc source/config |
| [test_backend_polarity.py](test_backend_polarity.py) | Không có argparse; đọc source/config |
