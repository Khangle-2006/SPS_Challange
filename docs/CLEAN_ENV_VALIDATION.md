# Clean-environment validation — 22 November 2026

Update on 23 November: real-video validation of the new CLI is in
[VIDEO_CLI_VALIDATION.md](VIDEO_CLI_VALIDATION.md). This page preserves earlier
installation/demo evidence and does not replace the new video checks.

**Four new venvs** were created without system `site-packages` or previous experiment
environments on Linux x86_64/Python 3.12.3. Runs used CPU and nice 19, with no participant videos,
labels, or holdout reads. These are installation and functional checks, **not
BTC/public MAE reproduction or end-to-end validation of the selected pipeline**.

Receipt with private paths removed: [clean_env_20261122.json](validation/clean_env_20261122.json).
Complete dependency locks: [environments/locks](../environments/locks).

## Measured results

| Environment / check | Result |
|---|---|
| Fresh installation and `pip check` | All 4 profiles PASS; no dependency conflicts |
| core: POS + CHROM on synthetic 90 bpm RGB | Both peaks at 90.0 bpm; static input produces no valid spectrum |
| frontend: MediaPipe 0.10.21, OpenCV 4.11.0, PyAV 14.2.0 | Imports and FaceMesh on a black image PASS; 0 faces |
| torch-cpu: PyTorch 2.5.1+cpu | CPU tensor computation PASS; no CUDA runtime |
| experts: open-rppg 0.1.1 / JAX 0.12.1 / Keras 3.15.1 | FacePhys.rlap, ME-chunk.pure, ME-flow.pure each return 360 finite BVP samples from synthetic color tensors |
| Two expert weight files | SHA256 matches [ASSETS.md](ASSETS.md) |
| Demo and setup tools | 7/7 unit tests PASS |
| `test_score_round1` | 10/10 unit test PASS |
| `verify_source.py` | 1,020 archived file hashes match; Python source parses |
| Additional core venv from the final lock | PASS; POS/CHROM and static smoke checks PASS |
| `doctor` without a private bundle | Exit 2 as designed, reporting a missing asset manifest |

The three experts ran sequentially in one check: total time **1 minute 45.37 seconds**,
peak RSS **2,713,808 KiB** from `/usr/bin/time -v`. This is not a
real-video runtime benchmark. Finite BVP does not demonstrate accurate pulse measurement.

Setup tests cover nonrecursive prefix replacement, rejection of paths outside the workspace,
wrong bundle scope/hash or duplicate destinations, isolated workspace creation, source preservation,
overwrite refusal, and detection of assets modified after prepare.

## Validation commands

After the installation steps in [GETTING_STARTED.md](GETTING_STARTED.md):

```bash
.venv/core/bin/python -m unittest discover -s tools -p 'test_*.py'
.venv/core/bin/python -m unittest test_score_round1
.venv/core/bin/python tools/verify_source.py
.venv/core/bin/python tools/smoke_environment.py core
.venv/frontend/bin/python tools/smoke_environment.py frontend
.venv/torch-cpu/bin/python tools/smoke_environment.py torch-cpu
.venv/experts/bin/python tools/smoke_environment.py experts
```

## Remaining limitations

- Research tables have not been rerun with private bundles in the new environments.
  NumPy/SciPy and runtime versions may differ from historical environments; numerical
  parity must be checked before claiming reproduction of README MAEs.
- The team's five private CNN checkpoints have not been loaded into the new environment. Torch CPU
  smoke tests check runtime; expert smoke tests load only hash-verified upstream weights.
- FaceMesh on a black image does not validate decoding, timestamps, ROI, target locking, or
  the real-video frontend. It does not replace C3 checks or 218-video parity.
- The launcher was tested with a small synthetic bundle. Not every historical dependency
  has been validated after relocation; remaining hash assertions require
  version-matched receipts, not removal.
- Windows/WSL2 was not tested in this run. These environment locks target
  Linux x86_64/Python 3.12; other platforms remain unverified.
- No personal data, private checkpoints, holdout labels, or experiment caches
  are released. Asset request procedures are in [ASSETS.md](ASSETS.md).

This historical PR added installation, asset checking, and workspace preparation; it did not change
algorithms, train models, open holdout, or merge automatically.
