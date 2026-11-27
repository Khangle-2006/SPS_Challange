# reports/round1_final_20261105/revision_02/code

MAT export and frozen evaluator/report; historical status, not the final BTC runtime.

[Reproduction guide](../../../../docs/REPRODUCING.md) · [Results](../../../../docs/RESULTS.md) · [Experiment map](../../../../docs/EXPERIMENTS.md)

The parameters below were extracted statically from the code; they are not commands that automatically launch a run. For scripts without a CLI, check the constants, paths, and dependencies at the top of the file. Controllers and site runners require the original receipts and environments; do not bypass their guards.

| File | Declared CLI parameters |
|---|---|
| [benchmark72.py](benchmark72.py) | No argparse interface; read the source/config |
| [cadence_receipt.py](cadence_receipt.py) | No argparse interface; read the source/config |
| [collect_receipts.py](collect_receipts.py) | No argparse interface; read the source/config |
| [cpu_reports.py](cpu_reports.py) | No argparse interface; read the source/config |
| [create_venv.py](create_venv.py) | No argparse interface; read the source/config |
| [delivery_checks.py](delivery_checks.py) | No argparse interface; read the source/config |
| [file_inventory.py](file_inventory.py) | No argparse interface; read the source/config |
| [final_checks.py](final_checks.py) | No argparse interface; read the source/config |
| [final_reports.py](final_reports.py) | No argparse interface; read the source/config |
| [freeze_report.py](freeze_report.py) | `--root` |
| [hybrid.py](hybrid.py) | No argparse interface; read the source/config |
| [lag_check.py](lag_check.py) | `--root`, `--pos-source` |
| [launch_packaging.py](launch_packaging.py) | No argparse interface; read the source/config |
| [mcd_c5_report.py](mcd_c5_report.py) | `--root` |
| [metrics.py](metrics.py) | No argparse interface; read the source/config |
| [prepare_frozen.py](prepare_frozen.py) | No argparse interface; read the source/config |
| [process_round1.py](process_round1.py) | `--frozen-cache`, `--audit`, `--dataset`, `--recording`, `--branch` |
| [reproduce_report.py](reproduce_report.py) | No argparse interface; read the source/config |
| [schema_report.py](schema_report.py) | No argparse interface; read the source/config |
| [setup_packaging.py](setup_packaging.py) | No argparse interface; read the source/config |
| [source_codec.py](source_codec.py) | No argparse interface; read the source/config |
| [spectral.py](spectral.py) | No argparse interface; read the source/config |
| [test_contracts.py](test_contracts.py) | No argparse interface; read the source/config |
| [verify_downloads.py](verify_downloads.py) | No argparse interface; read the source/config |
