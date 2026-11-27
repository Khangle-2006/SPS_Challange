# reports/round1_v4_20261105/btc_zeroshot

Dev-only preprocessing/inference, fold-safe calibration, diagnostics.

[Reproduction guide](../../../docs/REPRODUCING.md) · [Results](../../../docs/RESULTS.md) · [Experiment map](../../../docs/EXPERIMENTS.md)

The parameters below were extracted statically from the code; they are not commands that automatically launch a run. For scripts without a CLI, check the constants, paths, and dependencies at the top of the file. Controllers and site runners require the original receipts and environments; do not bypass their guards.

| File | Declared CLI parameters |
|---|---|
| [build_split.py](build_split.py) | No argparse interface; read the source/config |
| [control_summary.py](control_summary.py) | No argparse interface; read the source/config |
| [finalize_dev.py](finalize_dev.py) | No argparse interface; read the source/config |
| [infer_dev.py](infer_dev.py) | No argparse interface; read the source/config |
| [preprocess.py](preprocess.py) | `--limit` |
| [preprocess_pilot_v1.py](preprocess_pilot_v1.py) | `--limit` |
| [report_watch.py](report_watch.py) | No argparse interface; read the source/config |
| [summarize_dev.py](summarize_dev.py) | No argparse interface; read the source/config |
| [test_contract.py](test_contract.py) | No argparse interface; read the source/config |
| [test_gate.py](test_gate.py) | No argparse interface; read the source/config |
| [test_inference.py](test_inference.py) | No argparse interface; read the source/config |
| [test_summary.py](test_summary.py) | No argparse interface; read the source/config |
| [validate_media.py](validate_media.py) | No argparse interface; read the source/config |
