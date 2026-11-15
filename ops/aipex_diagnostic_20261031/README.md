# ops/aipex_diagnostic_20261031

88 người; timing15fps unresolved.

[Hướng dẫn chạy](../../docs/REPRODUCING.md) · [Kết quả](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [diagnostic.py](diagnostic.py) | `--root`, `--cache`, `--output` |
| [hr_baseline.py](hr_baseline.py) | `--root`, `--out`, `--self-test` |
| [report.py](report.py) | `--input`, `--output` |
| [test_contract.py](test_contract.py) | Không có argparse; đọc source/config |
