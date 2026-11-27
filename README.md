# FaceVitals: SP Cup 2027, Round 1

Estimate pulse rate (PR, bpm) from face videos. For BTC, `predict.py` uses **BTC joint** by default. For UBFC2 and MCD, use **P4m** through `predict_public.py`.

## Clone and run a video

Requires Linux x86_64, Python 3.12, `ffmpeg`/`ffprobe`, and at least **15 GiB of available RAM**:

```bash
git clone https://github.com/Khangle-2006/SPS_Challange.git
cd SPS_Challange
python3.12 tools/bootstrap_video.py
python3.12 predict.py --video /path/video.mp4 --activity Resting --output output/my_video
```

Bootstrap installs three environments with pinned dependencies, downloads all five public checkpoints, and verifies the readers and weights with SHA256. Internet access is required for the initial setup; no files need to be copied from the authors' private directories.

Read `output/my_video/result.json` or `result.csv`. Set `--activity` to `Resting`, `Task`, or `Walking`, according to the activity during recording. The default `reference` mode is intended for single-person videos. If there is insufficient evidence, the output contains `abstain=true` and `pr_bpm=null`; see the `reason` field for details. The output directory must be new.

See **[RUN_VIDEO.md](docs/RUN_VIDEO.md)** for system packages, source bitrate handling, HDR videos, fold selection, and the multiple-face guard. This command runs on CPU and does not require PR labels, the team's dataset, or the team's CNN checkpoints. Bootstrap creates three environments with pinned versions; the three experts are supplied by `open-rppg==0.1.1`, and the lightweight readers are included in `models/`.

## Reproduce BTC joint on dev

With authorized project data, the following commands replay the reader on RGB traces and three expert spectra using the correct person folds, checking every prediction across all 218 videos. New videos use the deployment parameters in `models/btc_joint/`.

```bash
.venv/core/bin/python tools/btc_joint_assets.py create --reports /path/to/facevitals/reports --manifest output/joint_assets/assets.local.json --purpose both
.venv/core/bin/python tools/reproduce_btc_joint.py --reports /path/to/facevitals/reports --assets-manifest output/joint_assets/assets.local.json
```

The code to fit and export the router and low-bitrate fallback is in `training/btc_joint/`.
The `tools/fit_btc_joint.py --verify-shipped` command checks the models and 436 OOF/deployment predictions
before exporting to a new directory. See the [input inventory and complete fit/replay commands](docs/BTC_JOINT_REPRODUCING.md).

## BTC joint results

Each video produces one pulse-rate estimate. Mean absolute error (MAE) is computed within each person and then averaged across people. Values are in beats per minute (bpm); lower is better.

| BTC split | Videos with predictions / total | People scored / total | MAE (bpm) |
|---|---:|---:|---:|
| Dev, person-disjoint cross-validation | 218/218 | 158/158 | **8.020** |
| Previously evaluated holdout | 71/72 | 40/41 | **9.704** |

The 95% confidence interval for dev MAE is 6.790–9.350 bpm. One holdout video lacked valid windows to produce a PR estimate.

### BTC dev under the seven constraints C1–C7

Each row compares BTC joint with BTC R1 on the same videos with predictions, out of 218 dev videos. Δ is joint MAE minus R1 MAE (bpm). The 95% confidence intervals use 10,000 paired bootstrap samples at the person level, with seed 0.

| Condition | Videos scored | BTC joint | BTC R1 | Δ joint − R1 [CI95] |
|---|---:|---:|---:|---:|
| C1 · Reduced light, Poisson noise | 200 | 9.49 | 9.67 | −0.18 [−0.55; +0.16] |
| C1 · Reduced light, Gaussian noise | 197 | 12.28 | 12.26 | +0.02 [−0.57; +0.57] |
| C2 · Simulated hypopigmented patches | 199 | 8.91 | 8.83 | +0.08 [−0.37; +0.61] |
| C3 · Two faces side by side | 216 | 9.03 | — | — |
| C3 · Neighboring face scaled to 1.3× | 212 | 9.05 | — | — |
| C4 · VGA | 200 | 8.73 | 9.10 | −0.37 [−0.73; −0.03] |
| C4 · QVGA | 200 | 9.12 | 9.44 | −0.32 [−0.88; +0.17] |
| C5 · Crop transmission, `split10` | 218 | 9.30 | 9.48 | −0.18 [−0.50; +0.13] |
| C6 · Sharpening | 197 | 8.36 | 8.44 | −0.09 [−0.40; +0.23] |
| C7 · Blue mask | 197 | 10.51 | 10.46 | +0.06 [−0.35; +0.46] |

BTC joint has lower MAE than R1 in 5 of the 8 configurations with paired results. Only C4 VGA has a difference CI95 entirely below zero. C1 with Gaussian noise has the largest error (12.28 bpm), close to the constant-prediction baseline of 12.61 bpm on the same 197 videos.

The two C3 layouts are new tests on BTC. The identity guard switched to the neighboring person in **0/365,426 valid frames**. With the same guard enabled for both single-face and two-face videos, error increased by 0.84 bpm for the side-by-side layout (215 paired videos) and 1.02 bpm for the larger neighboring face (211 videos). Both exceed the 0.05 bpm threshold.

<details>
<summary>Constant baseline and earlier calibration on the same BTC videos (MAE, bpm)</summary>

| Condition | Training-fold constant | Native calibration |
|---|---:|---:|
| C1 · Reduced light, Poisson noise | 12.50 | 11.58 |
| C1 · Reduced light, Gaussian noise | 12.61 | 11.73 |
| C2 · Simulated hypopigmented patches | 12.43 | 11.46 |
| C3 · Two faces side by side | 12.12 | 11.45 |
| C3 · Neighboring face scaled to 1.3× | 11.96 | 11.38 |
| C4 · VGA | 12.50 | 11.58 |
| C4 · QVGA | 12.50 | 11.58 |
| C5 · Crop transmission, `split10` | 12.07 | 11.37 |
| C6 · Sharpening | 12.53 | 11.55 |
| C7 · Blue mask | 12.69 | 11.74 |

</details>

## P4m on UBFC2 and MCD

P4m estimates pulse rate in 10-second windows. For one PR score per video, as in BTC, take the median predictions and the median fingertip-PPG reference rates over the same valid windows.

| Dev dataset, original videos | Videos / people | Window MAE (bpm) | Video MAE (bpm) |
|---|---:|---:|---:|
| UBFC2 | 35 / 35 | 0.746 | **0.515** |
| MCD FullHD | 119 / 60 | 1.511 | **0.648** |

### UBFC2 and MCD under the constraints

The table below uses **P4m results evaluated in R1**. Each cell shows **MAE (bpm) · number of scored 10-second windows**. P4m computes errors per window and then aggregates them by person. The BTC table above scores one prediction per video.

| Condition | UBFC2 | MCD FullHD |
|---|---:|---:|
| C1 · Reduced light, Poisson, gain 0.50 | 0.982 · 193 | 1.869 · 618 |
| C1 · Reduced light, Poisson, gain 0.25 | 1.532 · 193 | 2.443 · 618 |
| C1 · Reduced light, Poisson, gain 0.12 | 2.564 · 186 | 3.056 · 616 |
| C1 · Reduced light, Gaussian, gain 0.25, variance 0.001 | 3.082 · 192 | 4.139 · 616 |
| C1 · Reduced light, Gaussian, gain 0.25, variance 0.01 | 13.450 · 184 | 10.435 · 592 |
| C2 · Simulated hypopigmented patches | 0.750 · 193 | 1.734 · 618 |
| C4 · QVGA | 0.727 · 193 | 1.520 · 618 |
| C5 · Crop transmission, `split10` | 0.646 · 179 | 1.704 · 1,929 |
| C6 · Sharpening | 0.732 · 193 | 1.641 · 618 |
| C7 · Blue mask | 1.447 · 192 | 2.972 · 618 |
| C7 · White mask | 1.154 · 176 | 3.602 · 614 |

**C3 on UBFC2** was evaluated before the R1 model-state repair. Across 193 paired windows from 35 people, single-face MAE was **0.755 bpm**. With two faces side by side, MAE was **0.775** (Δ +0.020). With the neighboring face scaled to 1.3×, MAE was **0.771** (Δ +0.016). Both meet the 0.05 bpm difference threshold. The earlier report only includes C3 for UBFC2.

[Confidence intervals, constraint definitions, and C5 validation limits](docs/RESULTS.md) · [Server reproduction guide](docs/REPRODUCE_SERVER.md).

## Pipeline

| Stage | Processing |
|---|---|
| Frontend | Actual presentation timestamps (PTS), single-face FaceMesh, a 36×36 oval crop for color signals, and a 72×72 bounding-box crop for the experts. For multiple faces, `guarded` mode tracks the target person. Dataset-specific C3 results are listed above. |
| Signals | POS + CHROM; FacePhys.rlap, ME-chunk.pure, and ME-flow.pure with fixed weights and separate self-warm initialization for each video. The public branch adds the CNN with corrected padding and a 0.1 s input prefilter. |
| Spectra | 10 s windows; BTC joint uses a 2.5 s stride for POS/CHROM and 5 s for the three experts; BTC R1 uses a 5 s stride, and public R1 uses 10 s. Interpolation at 30 Hz, detrending with λ=100, a 0.7–3 Hz Butterworth filter, and a Hann periodogram with nfft=4096. The respective frozen BTC/public operators are preserved. |
| BTC | BTC joint chooses among the fused PR and five individual PR estimates using spectral quality, agreement, bitrate, and the Task flag. When bitrate is below 4 Mb/s and the low-bitrate branch beta is zero, it uses a correction based on capture metadata, face-box motion, and color/spectral signals. |
| Public | P4m combines four channels with a temporal readout; fitting pools native data and six conditions within the training folds. Weights and readout are fold-specific; deployment fits are not used to score dev. |

[Architecture and parameter details](docs/PIPELINE.md). [Raw-video CLI validation](docs/VIDEO_CLI_VALIDATION.md).

## Datasets and checkpoints

The datasets are available on **P100** and require an account with access permission:

| Dataset | Path |
|---|---|
| BTC | `/mnt/samsung_t5/khangle/facevitals/datasets/FaceVitals_BTC_20261107` |
| UBFC-rPPG | `/data_pool/khangle/facevitals/datasets/UBFC-rPPG` |
| MCD-rPPG | `/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG` |
| AiPEX | `/data_pool/khangle/facevitals/datasets/AiPEX` |

[Permissions, splits, and data paths](docs/SERVER_DATASETS.md). Use only authorized dev manifests; **the BTC holdout of 72 videos/41 people and the 7 UBFC2 test participants remain locked**. Do not scan an entire dataset root for scoring.

Bootstrap automatically downloads and verifies the [current R1 release: checkpoints and readers](https://github.com/Khangle-2006/SPS_Challange/releases/tag/inference-assets-r1-20261124). **BTC joint** is included in the repository under `models/btc_joint/`. The R1 package supplies the public checkpoints and R1 parameters. Older releases are retained separately for reference. See [details](docs/INFERENCE_ASSETS.md).

## Code structure and reproduction

| Directory / file | Role |
|---|---|
| `predict.py`, `predict_public.py`, `runtime/` | Single video → BTC PR or public P4m HR; no fitting during inference |
| `models/` | BTC deployment and five-fold readers, plus public P4m parameters; no individual labels or spectra |
| `training/btc_joint/`, `tools/fit_btc_joint.py` | Frozen BTC joint fitting recipe, separate export, and parity checks against released models |
| `environments/`, `tools/bootstrap_video.py` | Pinned dependencies and environment setup |
| `configs/`, `tools/repro.py` | Private path configuration and preparation for research reproduction |
| `reports/`, `ops/` | Experiment code with original source hashes preserved; individual data assets are not included |
| `docs/RESULTS.md`, `docs/SOURCE_MANIFEST.json` | Updated results and source/hash inventory |
| `submission/` | Packaging code; its presence does not mean a submission has been made |

[GETTING_STARTED](docs/GETTING_STARTED.md) covers the research environments and `tools/repro.py init/doctor/prepare/plan`; [ASSETS](docs/ASSETS.md) lists the private caches/checkpoints required to reproduce the tables. Running a new video does not require those private assets. Public/P4m uses the separate `predict_public.py` command; raw-video validation on 3 recordings produced 26 windows matching R1 with a 0.000 bpm difference. See the [Round 1 package and limitations](docs/SUBMISSION_ROUND1.md).

The R1 protocol excludes retraining, grid expansion, and opening the holdout. This package does not guarantee dev-level MAE on arbitrary videos. See [third-party sources and licenses](THIRD_PARTY_NOTICES.md).
