# Choose a video and estimate PR

The `predict.py` command connects video → FaceMesh/crops → POS + CHROM + three experts
→ BTC joint → JSON/CSV. It does not require `private-assets/assets.json`, PR labels,
the team's dataset, a GPU, or fixed `/mnt/...` paths.

## One-time setup

Supported platform: Linux x86_64, Python 3.12, at least 15 GiB of **available** RAM.
Internet access is needed during installation; inference then uses no remote services.
The apt commands below target Ubuntu 24.04 (validation used Python 3.12.3).
An administrator should install any missing system packages:

```bash
sudo apt-get update
sudo apt-get install git python3.12-venv ffmpeg libgl1 libegl1 libglib2.0-0t64
git clone https://github.com/Khangle-2006/SPS_Challange.git
cd SPS_Challange
python3.12 tools/bootstrap_video.py
```

Bootstrap creates separate `core`, `frontend`, and `experts` environments, verifies pinned
versions, and checks the SHA256 of both weight files and the initial states of the three
open-rppg experts. State resets before every video; experts process the same video twice
and use pass 2 (self-warm). Earlier videos must not affect later ones.
Lightweight readers are included in `models/`; BTC does not require the team's CNN checkpoint.
Bootstrap still downloads the full R1 package with five public checkpoints and verifies
hashes and readers against the checkout. Version mismatches raise errors, not silent repairs.

## Run your video

```bash
python3.12 predict.py \
  --video /path/to/video.mp4 \
  --activity Resting \
  --output output/my_video
```

`--activity` is required: choose `Resting`, `Task`, or `Walking` based on the recording.
The reader has activity-dependent offsets; do not infer the flag from results or filenames.
`Resting` and `Walking` share the non-Task group in the frozen computation.
No validated configuration exists for an unknown activity.

Use an original video of about 20–30 seconds with a visible face. The first-to-last-frame
timestamp span must be at least 10 seconds; a file reporting exactly 10 seconds can lack
the required endpoint. Do not pad missing support. For resized/recompressed input,
provide the **source video-stream bitrate**:

```bash
python3.12 predict.py --video /path/derivative.mp4 --activity Task \
  --source-bitrate 7700000 --output output/derivative
```

If ffprobe cannot provide the stream bitrate, the command stops and requests it.
Do not substitute total file bitrate or an arbitrary source class. HDR uses the recorded
HLG/PQ → SDR conversion; display rotation is applied once. Timing uses actual PTS,
not frame indices divided by an assumed fps.

The default `reference` frontend uses the parity-checked single-face path.
Use it only for single-person videos. `guarded` is an experimental identity-preserving
option that has not passed per-video equivalence: one of three validation videos shifted
+22 bpm after a frame was recovered. It is not a validated research-results mode.
To explicitly try a multiple-person video, provide the target box in pixel coordinates
after display rotation:

```bash
python3.12 predict.py --video /path/group.mp4 --activity Resting \
  --frontend guarded --seed-box 100 60 340 380 --output output/target
```

The target is not automatically reselected after tracking is lost. Too few valid windows
produce `abstain`. Do not interpret a prior from a faceless video or static image as a measured pulse.

## Read the results

- `result.json`: `pr_bpm`, `abstain`, activity, source class, frame coverage, window count,
  video/code/reader/weight hashes, and runtime.
- `result.csv`: one result row for tabulation.
- `frontend.log`, `experts.log`, `core.log`, `decode.log`: stage logs.
- `trace.jsonl`: native/switch/held/invalid branch and box per frame.
- `trace.npz`, `crop72.npy`: **private** intermediate data. Each output directory has its
  own `.gitignore`; do not upload these files to GitHub.

When `abstain=true`, `pr_bpm=null`. The `branch` field identifies the PR source or
low-bitrate fallback. Use `--reader r1` for historical reader_v4.

## Research reproduction versus new-video inference

The default deployment model was fitted once on all 218 dev videos using the selected fit
function, beta grid, and offsets. Do not score dev with it and call the result OOF.
A dev video needs its correct person fold:

```bash
python3.12 predict.py --video /path/authorized_dev_video.mp4 --activity Resting \
  --frontend reference --fold 2 --output output/dev_fold2
```

`reference` uses single-face FaceMesh and historical crops for reproduction, without a
multiple-person guard. Do not select modes using predicted PR or labels; reference is the
shared default for single-face videos. Guard recovery/rejection can change predictions;
per-video parity in the 22 November integration did not pass. That threshold is unchanged
in this package. Historical MAE 8.350 used expert state carried across videos, not R1
self-warm CLI behavior. R1 dev MAE is 8.333 bpm; it does not guarantee new-video performance
or results for the full guarded frontend.

Default BTC joint OOF dev MAE is 8.020 bpm. Use `predict_public.py` for public videos.

The public/P4m branch needs the CNN environment after bootstrap:

```bash
python3.12 tools/setup_env.py torch-cpu
python3.12 predict_public.py --video /path/video.mp4 --dataset UBFC2 --output output/public_video
```

Choose `--dataset MCD` for the MCD readout. Benchmarks require the matching fold and
acquisition clock described in [SUBMISSION_ROUND1.md](SUBMISSION_ROUND1.md).

Server datasets: [SERVER_DATASETS.md](SERVER_DATASETS.md). Use only authorized dev splits
for reproduction; **holdout remains locked**. The CLI does not download datasets,
read label spreadsheets, scan dataset roots, or automatically score holdout.
