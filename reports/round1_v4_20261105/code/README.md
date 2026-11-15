# reports/round1_v4_20261105/code

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../../docs/REPRODUCING.md) · [Kết quả](../../../docs/RESULTS.md) · [Experiment map](../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [cleanup_t9_regenerable.py](cleanup_t9_regenerable.py) | Không có argparse; đọc source/config |
| [diagnostic_13_noise.py](diagnostic_13_noise.py) | Không có argparse; đọc source/config |
| [diagnostics_11_12.py](diagnostics_11_12.py) | Không có argparse; đọc source/config |
| [frozen_t3.py](frozen_t3.py) | Không có argparse; đọc source/config |
| [organiser_trace_eval.py](organiser_trace_eval.py) | Không có argparse; đọc source/config |
| [phase_evaluate.py](phase_evaluate.py) | Không có argparse; đọc source/config |
| [phase_loss.py](phase_loss.py) | Không có argparse; đọc source/config |
| [phase_train.py](phase_train.py) | `--receipt`, `--preflight-only` |
| [pull_ubfc_direct.py](pull_ubfc_direct.py) | Không có argparse; đọc source/config |
| [test_phase.py](test_phase.py) | Không có argparse; đọc source/config |
| [validate_phase_cpu.py](validate_phase_cpu.py) | Không có argparse; đọc source/config |
