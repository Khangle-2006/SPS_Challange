# reports/round1_v4_20261105/padding_pilot_20261107

pilot_runner.py và repair_loss.py; giữ epoch5 gate.

[Hướng dẫn chạy](../../../docs/REPRODUCING.md) · [Kết quả](../../../docs/RESULTS.md) · [Experiment map](../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [pilot_gate.py](pilot_gate.py) | Không có argparse; đọc source/config |
| [pilot_runner.py](pilot_runner.py) | `--receipt`, `--preflight-only`, `--remaining` |
| [repair_loss.py](repair_loss.py) | Không có argparse; đọc source/config |
| [replicate_tsm.py](replicate_tsm.py) | Không có argparse; đọc source/config |
| [test_static_tsm.py](test_static_tsm.py) | Không có argparse; đọc source/config |
| [validate_repair.py](validate_repair.py) | Không có argparse; đọc source/config |
