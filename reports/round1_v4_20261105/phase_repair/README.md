# reports/round1_v4_20261105/phase_repair

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../../docs/REPRODUCING.md) · [Kết quả](../../../docs/RESULTS.md) · [Experiment map](../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [boundary_control.py](boundary_control.py) | Không có argparse; đọc source/config |
| [diagnose.py](diagnose.py) | Không có argparse; đọc source/config |
| [partial_status_summary.py](partial_status_summary.py) | Không có argparse; đọc source/config |
| [pilot_gate.py](pilot_gate.py) | Không có argparse; đọc source/config |
| [pilot_runner.py](pilot_runner.py) | `--receipt`, `--preflight-only`, `--remaining` |
| [plot_diagnostic.py](plot_diagnostic.py) | Không có argparse; đọc source/config |
| [repair_loss.py](repair_loss.py) | Không có argparse; đọc source/config |
| [test_epoch5_gate.py](test_epoch5_gate.py) | Không có argparse; đọc source/config |
| [validate_repair.py](validate_repair.py) | Không có argparse; đọc source/config |
