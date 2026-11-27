# ops/t8_polarity_v3_20261102

15 joint runs with the MCD sign corrected; the same folds and seeds as the control.

[Reproduction guide](../../docs/REPRODUCING.md) · [Results](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

The parameters below were extracted statically from the code; they are not commands that automatically launch a run. For scripts without a CLI, check the constants, paths, and dependencies at the top of the file. Controllers and site runners require the original receipts and environments; do not bypass their guards.

| File | Declared CLI parameters |
|---|---|
| [EfficientPhys_fixed.py](EfficientPhys_fixed.py) | No argparse interface; read the source/config |
| [polarity_evaluate.py](polarity_evaluate.py) | No argparse interface; read the source/config |
| [polarity_guard.py](polarity_guard.py) | No argparse interface; read the source/config |
| [polarity_loader.py](polarity_loader.py) | No argparse interface; read the source/config |
| [ppg_polarity.py](ppg_polarity.py) | No argparse interface; read the source/config |
| [score_round1.py](score_round1.py) | `--reference`, `--output`, `--metrics` |
| [t8_aggregate.py](t8_aggregate.py) | `--runs`, `--output` |
| [t8_evaluate.py](t8_evaluate.py) | No argparse interface; read the source/config |
| [t8_plan.py](t8_plan.py) | `--ubfc`, `--mcd`, `--folds`, `--output` |
| [t8_run.py](t8_run.py) | `--index`, `--output`, `--original-runs`, `--project-budget-receipt`, `--lag-verification-receipt`, `--preflight-only` |
| [t8_t1_loader.py](t8_t1_loader.py) | No argparse interface; read the source/config |
| [test_polarity_v3.py](test_polarity_v3.py) | No argparse interface; read the source/config |
| [test_t8.py](test_t8.py) | No argparse interface; read the source/config |
| [window_support.py](window_support.py) | No argparse interface; read the source/config |
