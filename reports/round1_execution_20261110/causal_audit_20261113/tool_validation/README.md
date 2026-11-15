# reports/round1_execution_20261110/causal_audit_20261113/tool_validation

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../../../docs/REPRODUCING.md) · [Kết quả](../../../../docs/RESULTS.md) · [Experiment map](../../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [compare_stage_exports.py](compare_stage_exports.py) | `left_manifest`, `right_manifest`, `--output`, `--max-array-mib` |
| [self_test.py](self_test.py) | Không có argparse; đọc source/config |
