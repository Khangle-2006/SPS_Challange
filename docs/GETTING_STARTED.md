> **Run the current version:** see [REPRODUCE_SERVER.md](REPRODUCE_SERVER.md) or `tools/bootstrap_video.py` + `predict.py`. The guide below covers historical research workspaces. Archived stages y05/y11/y12/y13 retain the documented state bug; do not use them to reproduce or advertise R1 results.

# Fresh setup and reproduction

**Only need PR from a video?** Use [RUN_VIDEO.md](RUN_VIDEO.md):
`python3.12 tools/bootstrap_video.py`, then `python3.12 predict.py --video ...
--activity Resting --output output/run1`. The following sections cover research reproduction
and additional profiles, not required steps for the video command.

This guide targets a fresh checkout. No team server account is needed
for the demo or source checks. Reproducing BTC/public tables requires an authorized
data bundle: see [ASSETS.md](ASSETS.md). Successful environment installation **does not**
establish reproduction of the reported MAEs.

## 1. Supported environment

Research scripts target Linux x86_64 and Python **3.12**. On Windows, use WSL2
for this workflow; the RGB CSV baseline is platform-independent. Git and `python3.12-venv` are required.
The frontend needs system `ffmpeg`, `ffprobe`, libGL/EGL, and libglib. For example:
Ubuntu 24.04: `sudo apt install python3.12-venv ffmpeg libgl1 libegl1 libglib2.0-0t64`.
System installation commands are run by an administrator and are not part of bootstrap.

```bash
git clone https://github.com/Khangle-2006/SPS_Challange.git
cd SPS_Challange
python3.12 tools/setup_env.py core
.venv/core/bin/python -m pip check
.venv/core/bin/python tools/smoke_environment.py core
.venv/core/bin/python tools/verify_source.py
.venv/core/bin/python -m unittest discover -s tools -p 'test_*.py'
.venv/core/bin/python -m unittest test_score_round1
```

`setup_env.py` creates a venv without system site-packages, installs profile-specific
requirements, and runs `pip check`. It refuses existing venvs; after an installation
failure, retain logs and retry with `--destination /path/to/new/environment`. Receipts
and all resolved versions are in `output/setup/<profile>/` (excluded from Git).

## 2. Immediate demo without downloading data

```bash
.venv/core/bin/python tools/pos_trace.py --make-demo output/demo_rgb.csv
.venv/core/bin/python tools/pos_trace.py --input output/demo_rgb.csv --output output/demo_predictions.csv
```

The synthetic input has a 90 bpm pulse. A result near 90 bpm validates the POS baseline;
**it is not reader_v4/P4m or a human-performance measurement**. The CLI refuses existing
outputs; choose a new output name. Your own RGB CSV can use columns
`timestamp_s,red,green,blue,valid`, starting at 0 with strictly increasing time spanning ≥10 s.

## 3. Separate research profiles

```bash
python3.12 tools/setup_env.py frontend
python3.12 tools/setup_env.py torch-cpu
python3.12 tools/setup_env.py experts

.venv/frontend/bin/python tools/smoke_environment.py frontend
.venv/torch-cpu/bin/python tools/smoke_environment.py torch-cpu
.venv/experts/bin/python tools/smoke_environment.py experts
```

| Profile | Requirements file | Purpose |
|---|---|---|
| core | `environments/core.txt` | POS/CHROM, readers, scoring, tests |
| frontend | `environments/frontend.txt` | MediaPipe FaceMesh, OpenCV, PyAV |
| torch-cpu | `environments/torch-cpu.txt` | Team CNN; CPU wheels from PyTorch |
| experts | `environments/experts.txt` | open-rppg 0.1.1, Keras/JAX CPU |

Do not combine all four requirements files in one venv. MediaPipe 0.10.21 uses NumPy 1.x
in this profile; experts have separate NumPy/JAX dependencies. On Linux x86_64/Python 3.12, bootstrap uses `environments/locks/*-linux-py312.txt`
to pin transitive dependencies too. Other platforms use the profile requirements directly
and are not research-validated. Resolved versions and validation scope are in [CLEAN_ENV_VALIDATION.md](CLEAN_ENV_VALIDATION.md).
These new environments have functional validation; byte-identical reproduction of historical
experiment environments is not claimed. Requirements in old snapshots are historical records.

`smoke_environment.py` reads no datasets: it checks POS/CHROM on synthetic signals and
static input, FaceMesh on a black image, PyTorch CPU, and three experts on synthetic color
tensors. Expert checks also verify both weight-file SHA256 values. These are not dev MAE tests.

## 4. Configure paths and check assets before running

Download and verify the [checkpoint/reader package](INFERENCE_ASSETS.md). On P100,
use the [existing dataset paths](SERVER_DATASETS.md); no redownload is needed.
The public package excludes participant splits/caches/manifests, so table reproduction
still requires the correctly versioned internal bundle.

```bash
python3.12 tools/repro.py init
# Edit configs/paths.local.json: workspace, assets_manifest, Python paths.
# Obtain the bundle and assets.json from the maintainer as described in docs/ASSETS.md.
python3.12 tools/repro.py doctor --stage btc-reader-v3
python3.12 tools/repro.py prepare
python3.12 tools/repro.py plan btc-reader-v3
```

`init` writes absolute paths for the current checkout; there is no need to replace server
usernames in hundreds of files. Regenerate the config when moving the checkout.
Relative configuration paths resolve from the **configuration directory**.
The workspace must be outside the checkout, not yet exist, and not be a production root.

`doctor` checks Python, each bundle file's hash, and `dev-only` scope; missing assets
produce exit 2 with their names. Maintainer-declared `complete_for` lists
supported stages; **it does not certify end-to-end execution**. The
`doctor` command neither reads labels for scoring nor runs models.

`prepare` checks archive hashes, then copies code and listed assets into a separate
workspace. Path prefixes change only in the **working copy**, preserving the directory layout.
It creates `RELOCATION.json` with before/after hashes for every file. It does not modify
Git snapshots or algorithms, delete or overwrite sources, or copy any
holdout data. Use only a maintainer-confirmed dev-only bundle.

Prefixes `/mnt/samsung_t5/khangle/facevitals`, `/data_pool/khangle/facevitals`,
and `/data/khangle/facevitals` map to the new workspace. T8 oval and polarity code
map to the corresponding `ops/` directories; old interpreters map to
new profiles. For other data roots in metadata, add absolute prefix pairs to
`path_map` in the configuration. Use `relocate_json_paths=true` only for JSON
path metadata. Binaries, checkpoints, and caches are copied byte-for-byte.

Old frozen hashes are **not** updated to conceal relocation. If a script still
asserts an old source/metadata hash, stop and request a version-matched relocation
receipt from the maintainer; do not remove assertions. Some historical controllers
are outside the launcher's scope; do not run them to "repair" the workspace.

## 5. Run a stage with complete authorized assets

After inspecting `plan`, set `allow_dev_reproduction=true` in the private configuration:

```bash
python3.12 tools/repro.py run btc-reader-v3
# Example after obtaining complete caches and metadata for the correct dev table:
python3.12 tools/repro.py plan btc-experts run FacePhys.rlap
python3.12 tools/repro.py run btc-experts run FacePhys.rlap
python3.12 tools/repro.py plan btc-score eval FacePhys.rlap,ME-chunk.pure,ME-flow.pure
```

This is **dev reproduction**, not new-video or holdout inference.
The `eval` stages fit priors/beta within dev folds as in the original code; they are not
fully zero-shot. Do not automatically run training/controllers or open holdout.
The launcher uses CPU only, nice 19, and one BLAS thread; available RAM must be ≥15 GiB.
Choose the CLI worker count yourself; start with 1. Do not run a full batch
of conditions merely to test installation.

| Stage | Script | Main input → output |
|---|---|---|
| btc-reader-v3 | `a11_reader_v3.py` | Numeric dev, metadata/baseline tables → v1/v2/v3 evaluation |
| btc-front | `x01_extract.py` | Dev manifest, authorized videos → condition-specific blocks/crops |
| box-crops | `x10_boxcrop.py` | Video, metadata, frontend → 72×72 box crops |
| btc-spectra | `x03_btc_spectra.py` | Blocks/crops and clock → BTC spectra |
| btc-experts | `y05_openrppg.py` | x10 crops, dev mapping, metadata → per-expert spectra |
| btc-conditions / btc-score | `y11_openrppg_conditions.py` | Condition crops/spectra → inference / reader_v4 tables |
| public-cnn | `x02_infer.py` | Crops, folds, matching checkpoints → CNN/POS/CHROM spectra |
| public-experts | `y12_openrppg_public.py` | Public crops → native expert spectra |
| public-conditions | `y13_openrppg_public_conditions.py` | Condition-specific public crops → expert spectra |
| public-score | `z13_public_p4_prefilter.py` | Complete channel spectra and POOL → P4/P4m/N4/N4m tables |

Original CLIs are in script docstrings; `plan` prints the interpreter, cwd, and argv
without executing them. Outputs are in `workspace/reports/claude_research_20261119_constraints/out`
and `logs`. Some original scripts consume and delete **temporary workspace crops**
(x02), so prepare copies assets instead of symlinking original caches. Estimate disk
requirements before copying large videos/caches. Do not commit new outputs.

## 6. Validation scope

See the [clean-environment validation receipt](CLEAN_ENV_VALIDATION.md). A full
raw-video → frontend → CNN/experts → reader_v4/P4m run on a fresh machine with
a private bundle has not been completed. Synthetic demos and unit tests are not evidence
of that full run. Checkpoints and parameters are now in the Release, but the correct data bundle,
splits/caches, and integration validation are still required; cloning does not yet reproduce every table.
