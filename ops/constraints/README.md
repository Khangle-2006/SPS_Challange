# ops/constraints

C1–C5, frame/codec support, and bitrate checks.

[Reproduction guide](../../docs/REPRODUCING.md) · [Results](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

The parameters below were extracted statically from the code; they are not commands that automatically launch a run. For scripts without a CLI, check the constants, paths, and dependencies at the top of the file. Controllers and site runners require the original receipts and environments; do not bypass their guards.

| File | Declared CLI parameters |
|---|---|
| [report_t5.py](report_t5.py) | `root` |
| [run_t5.py](run_t5.py) | `--dataset`, `--root`, `--audit`, `--output`, `--shared`, `--limit`, `--budget-gb` |
| [simulate.py](simulate.py) | No argparse interface; read the source/config |
| [test_simulate.py](test_simulate.py) | No argparse interface; read the source/config |
