# From a fresh clone to PR and R1 dev results

BTC joint is now the `predict.py` default. To fit/export or replay its
8.020 bpm result, use [BTC_JOINT_REPRODUCING.md](BTC_JOINT_REPRODUCING.md).
The `reproduce_r1.py` command below still checks reader_v4 R1 and P4m, not joint.

## 1. Machine requirements

Linux x86_64, Python 3.12, ffmpeg/ffprobe, at least 15 GiB available RAM, about 3 GiB for three environments, plus video-output space. Install missing system packages:

```bash
sudo apt-get install git ffmpeg python3.12-venv libgl1 libegl1 libglib2.0-0t64
```

The apt command targets Ubuntu 24.04. If Python 3.12 is unavailable in OS repositories, ask an administrator to install it; do not substitute arbitrary versions in locks. No GPU is required. Initial setup needs GitHub/PyPI Internet access.

## 2. Clone and obtain all assets

```bash
git clone https://github.com/Khangle-2006/SPS_Challange.git
cd SPS_Challange
python3.12 tools/bootstrap_video.py
```

Bootstrap creates `core`, `frontend`, and `experts`, checks pins/pip compatibility, weights, and readers, and downloads the R1 Release with all five public checkpoints. It does not scan datasets or open holdout.

## 3. Select a video and obtain PR

```bash
python3.12 predict.py --video /absolute/path/video.mp4 --activity Resting --output output/video_001
```

Use the recorded activity: Resting/Task/Walking. Read `output/video_001/result.json` or CSV. `pr_bpm` is the prediction; `abstain=true` means no measurement is reported. Every run requires a new output directory. If stream bitrate is unavailable, supply `--source-bitrate` using the **source** bitrate, not derivative bitrate after compression/resizing. Examples and frontend limits: [RUN_VIDEO](RUN_VIDEO.md).

## 4. Reproduce dev results on P100

```bash
.venv/core/bin/python tools/reproduce_r1.py --output output/r1_replay
```

Default server-root is `/mnt/samsung_t5/khangle/facevitals`; authorized copies elsewhere use `--server-root /path/copy`. The private cache is `reports/repo_ready_20261124/dev_replay`. Public configurations contain hashes/inventories, not labels. Read permission is required. Missing paths stop execution with an explicit error; no dataset download or holdout scan occurs.

Replay uses readers in the clone and frozen parameters, starting from spectra/evidence tensors before the reader. Saved predictions are only parity assertions, not outputs. No refitting. BTC includes all 218 native videos/158 people; public includes 24 rows from 12 conditions × 2 datasets. `RESULTS.csv` and `REPLAY_COMPLETE.json` record person-level MAE and prediction differences. This checks reader/readout reproduction, not re-encoding and rerunning CNNs on every video.

| Native | Required reproduced MAE, bpm |
|---|---:|
| BTC reader_v4 | 8,333268430 |
| UBFC2 P4m | 0,746442522 |
| MCD P4m | 1,510869770 |

To score an original BTC dev video, use `predict.py --fold N` with its person fold. Without `--fold`, deployment parameters are used and **must not** be compared as OOF. Authorized raw data/mapping/clocks are in [SERVER_DATASETS](SERVER_DATASETS.md). R1 scripts and nine BTC condition rows are retained in `reports/rerun_stateless_20261123`; this compact replay does not rerun eight BTC conditions or the full public frontend.

## 5. Evidence scope

These are dev results after many experiments. The old 8.350 result was invalidated by cross-video state and is not a current baseline. Selection of 3/17 experts on older spectra has not been repeated. The guarded frontend remains experimental. Arbitrary-video error is not guaranteed to match dev means. BTC holdout and 7 UBFC2 test people remain locked.
