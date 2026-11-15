# ops/t6_qalite_20261031

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../docs/REPRODUCING.md) · [Kết quả](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [hr_baseline.py](hr_baseline.py) | `--root`, `--out`, `--self-test` |
| [mcd_e0_metrics.py](mcd_e0_metrics.py) | Không có argparse; đọc source/config |
| [mcd_errata_audit.py](mcd_errata_audit.py) | `--root`, `--audit`, `--fold-csv` |
| [qa_mask.py](qa_mask.py) | Không có argparse; đọc source/config |
| [run_e0_qa.py](run_e0_qa.py) | `--limit` |
| [test_qa_mask.py](test_qa_mask.py) | Không có argparse; đọc source/config |
