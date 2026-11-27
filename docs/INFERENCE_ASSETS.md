# R1 checkpoints and readers — 24 November 2026

Current package: [inference-assets-r1-20261124](https://github.com/Khangle-2006/SPS_Challange/releases/tag/inference-assets-r1-20261124).
`python3.12 tools/bootstrap_video.py` installs environments, downloads this package, checks the ZIP and every file's SHA256, and compares readers against the checkout. No manual checkpoint search is needed.

## Standalone download

```bash
python3.12 tools/fetch_inference_assets.py
python3.12 tools/check_inference_assets.py checkpoints/r1/facevitals-inference-assets-r1-20261124 \
  --archive checkpoints/r1/facevitals-inference-assets-r1-20261124.zip
```

Use `--destination /path/new-folder` for a standalone download. Bootstrap and replay default to `checkpoints/r1/`. Existing valid assets are reused; incorrect hashes and partial downloads are retained for inspection, not overwritten. Retry partial downloads in a new directory.

## Contents

| Component | Role |
|---|---|
| `checkpoints/padding_fixed_fold0..4_seed2027.pt` | All five public checkpoints, epoch 30; bytes preserved from tensor-verified exports; do not use fold 0 for everyone |
| `readers/btc_reader_v4_deploy.json` | New-video configuration, fitted once on dev |
| `readers/btc_reader_v4_folds.json` | Five matching fold configurations for OOF dev scoring |
| `readers/public_p4m_r1.json` | Public R1 fusion weights and readouts, per fold and deployment |
| `code/` | EfficientPhys/padding architecture and reader mathematics needed for correct loading |
| `MANIFEST.json`, `CONFIG.json` | Per-file SHA256, provenance, and self-warm rules |

FacePhys.rlap, ME-chunk.pure, and ME-flow.pure are installed with `open-rppg==0.1.1`, which includes weights. Bootstrap checks both weight files and `state.pkl` against frozen hashes. No other download is needed. BTC does not use the padding-fixed CNN; all checkpoints are nevertheless downloaded for public/research workflows.

The CLI uses readers in `models/`, without overwriting them from the Release. The downloader requires ZIP readers to match `models/` byte-for-byte. Before each video, experts reset state, process that video twice, and retain only pass 2.

## Load and validate public CNNs (optional)

```bash
python3.12 tools/setup_env.py torch-cpu
.venv/torch-cpu/bin/python tools/check_inference_assets.py \
  checkpoints/r1/facevitals-inference-assets-r1-20261124 --smoke
```

This additional environment is unnecessary for BTC `predict.py` or readout replay. Smoke tests strictly load all five checkpoints, check finite output and flat waveforms for static inputs; they do not replace real-video validation.

Checkpoints accept `[batch,301,3,36,36]` and output 300 differences; cumsum is required before readout. Public P4m uses prefiltered crops and matching fold parameters. Running BTC CLI on webcam input is not equivalent to P4m. See [Pipeline](PIPELINE.md) and [server reproduction](REPRODUCE_SERVER.md).

The package excludes faces, videos, individual labels, ID-bearing splits, optimizer state, credentials, and holdout. Access-controlled data remain on the server. The older `inference-assets-20261123` package is retained for auditing; **do not use its reader/P4m parameters for R1**. See [historical description](history/INFERENCE_ASSETS_20261123.md).
