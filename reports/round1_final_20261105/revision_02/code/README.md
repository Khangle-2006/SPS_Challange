# reports/round1_final_20261105/revision_02/code

MAT export, frozen evaluator/report; trạng thái lịch sử, không phải final BTC runtime.

[Hướng dẫn chạy](../../../../docs/REPRODUCING.md) · [Kết quả](../../../../docs/RESULTS.md) · [Experiment map](../../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [benchmark72.py](benchmark72.py) | Không có argparse; đọc source/config |
| [cadence_receipt.py](cadence_receipt.py) | Không có argparse; đọc source/config |
| [collect_receipts.py](collect_receipts.py) | Không có argparse; đọc source/config |
| [cpu_reports.py](cpu_reports.py) | Không có argparse; đọc source/config |
| [create_venv.py](create_venv.py) | Không có argparse; đọc source/config |
| [delivery_checks.py](delivery_checks.py) | Không có argparse; đọc source/config |
| [file_inventory.py](file_inventory.py) | Không có argparse; đọc source/config |
| [final_checks.py](final_checks.py) | Không có argparse; đọc source/config |
| [final_reports.py](final_reports.py) | Không có argparse; đọc source/config |
| [freeze_report.py](freeze_report.py) | `--root` |
| [hybrid.py](hybrid.py) | Không có argparse; đọc source/config |
| [lag_check.py](lag_check.py) | `--root`, `--pos-source` |
| [launch_packaging.py](launch_packaging.py) | Không có argparse; đọc source/config |
| [mcd_c5_report.py](mcd_c5_report.py) | `--root` |
| [metrics.py](metrics.py) | Không có argparse; đọc source/config |
| [prepare_frozen.py](prepare_frozen.py) | Không có argparse; đọc source/config |
| [process_round1.py](process_round1.py) | `--frozen-cache`, `--audit`, `--dataset`, `--recording`, `--branch` |
| [reproduce_report.py](reproduce_report.py) | Không có argparse; đọc source/config |
| [schema_report.py](schema_report.py) | Không có argparse; đọc source/config |
| [setup_packaging.py](setup_packaging.py) | Không có argparse; đọc source/config |
| [source_codec.py](source_codec.py) | Không có argparse; đọc source/config |
| [spectral.py](spectral.py) | Không có argparse; đọc source/config |
| [test_contracts.py](test_contracts.py) | Không có argparse; đọc source/config |
| [verify_downloads.py](verify_downloads.py) | Không có argparse; đọc source/config |
