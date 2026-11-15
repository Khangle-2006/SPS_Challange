# ops/t6_oval_20261031

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../docs/REPRODUCING.md) · [Kết quả](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [data_adapters.py](data_adapters.py) | Không có argparse; đọc source/config |
| [hr_baseline.py](hr_baseline.py) | `--root`, `--out`, `--self-test` |
| [mcd_e0_metrics.py](mcd_e0_metrics.py) | Không có argparse; đọc source/config |
| [mcd_errata_audit.py](mcd_errata_audit.py) | `--root`, `--audit`, `--fold-csv` |
| [oval_frontend.py](oval_frontend.py) | Không có argparse; đọc source/config |
| [qa_mask.py](qa_mask.py) | Không có argparse; đọc source/config |
| [report_t6_e0.py](report_t6_e0.py) | Không có argparse; đọc source/config |
| [report_t6_t5.py](report_t6_t5.py) | `--input`, `--output` |
| [run_e0_qa.py](run_e0_qa.py) | `--limit` |
| [run_t5_qa.py](run_t5_qa.py) | `--audit`, `--dataset`, `--watch`, `--limit` |
| [score_round1.py](score_round1.py) | `--reference`, `--output`, `--metrics` |
| [t8_t1_loader.py](t8_t1_loader.py) | Không có argparse; đọc source/config |
| [test_adapter.py](test_adapter.py) | Không có argparse; đọc source/config |
| [test_oval.py](test_oval.py) | Không có argparse; đọc source/config |
| [test_qa_mask.py](test_qa_mask.py) | Không có argparse; đọc source/config |
| [window_support.py](window_support.py) | Không có argparse; đọc source/config |
