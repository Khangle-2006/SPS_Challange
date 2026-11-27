# reports/round1_v4_20261105/phase_repair

Research code snapshot; see the root README for its role and results.

[Reproduction guide](../../../docs/REPRODUCING.md) · [Results](../../../docs/RESULTS.md) · [Experiment map](../../../docs/EXPERIMENTS.md)

The parameters below were extracted statically from the code; they are not commands that automatically launch a run. For scripts without a CLI, check the constants, paths, and dependencies at the top of the file. Controllers and site runners require the original receipts and environments; do not bypass their guards.

| File | Declared CLI parameters |
|---|---|
| [boundary_control.py](boundary_control.py) | No argparse interface; read the source/config |
| [diagnose.py](diagnose.py) | No argparse interface; read the source/config |
| [partial_status_summary.py](partial_status_summary.py) | No argparse interface; read the source/config |
| [pilot_gate.py](pilot_gate.py) | No argparse interface; read the source/config |
| [pilot_runner.py](pilot_runner.py) | `--receipt`, `--preflight-only`, `--remaining` |
| [plot_diagnostic.py](plot_diagnostic.py) | No argparse interface; read the source/config |
| [repair_loss.py](repair_loss.py) | No argparse interface; read the source/config |
| [test_epoch5_gate.py](test_epoch5_gate.py) | No argparse interface; read the source/config |
| [validate_repair.py](validate_repair.py) | No argparse interface; read the source/config |
