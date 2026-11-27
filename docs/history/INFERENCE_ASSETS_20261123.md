# Checkpoints and reader configurations — 23 November 2026 export

This package exports existing weights and parameters without training or refitting.
These are **inference/research-reproduction assets**, not a raw-video →
result package that has passed all integration gates.

> This ZIP is retained for historical reproduction. Its readers were fitted from experts that
> could carry state between videos; do not substitute them for the new CLI's `models/`.
> `predict.py` uses repository R1 parameters recomputed with independent per-video self-warm.

## Download

[GitHub Release inference-assets-20261123](https://github.com/Khangle-2006/SPS_Challange/releases/tag/inference-assets-20261123)

```bash
mkdir -p checkpoints/download
cd checkpoints/download
curl -fLO https://github.com/Khangle-2006/SPS_Challange/releases/download/inference-assets-20261123/facevitals-inference-assets-20261123.zip
curl -fLO https://github.com/Khangle-2006/SPS_Challange/releases/download/inference-assets-20261123/SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt
unzip facevitals-inference-assets-20261123.zip
cd ../..

python3 tools/check_inference_assets.py checkpoints/download/facevitals-inference-assets-20261123
.venv/torch-cpu/bin/python tools/check_inference_assets.py checkpoints/download/facevitals-inference-assets-20261123 --smoke
```

On Windows, download ZIP/SHA256SUMS from the Release page and use `Get-FileHash`
to verify SHA256 before extraction. Install CPU environments following
[GETTING_STARTED.md](https://github.com/Khangle-2006/SPS_Challange/blob/main/docs/GETTING_STARTED.md).

## Package contents

| Directory/file | Contents |
|---|---|
| `checkpoints/padding_fixed_fold{0..4}_seed2027.pt` | Five epoch-30 EfficientPhys checkpoints with TSM replicate padding |
| `readers/btc_reader_v4_folds.json` | Five train-fold reader_v4 fits: KDE prior, source-class beta, source-class × Task offsets |
| `readers/public_p4m_parameters.json` | P4m fusion/readout parameters for folds 0–4 and `deploy`, separately for UBFC2/MCD |
| `CONFIG.json` | Architecture, input dimensions, frequency grid, channel order, source-class rules, limitations |
| `code/` | Model/padding definitions and fixed readout mathematics for correct loading |
| `MANIFEST.json` | Per-file SHA256, source hashes, and export transformations |
| `VALIDATION.json` | Weight-tensor checks, synthetic outputs, and parameter round trips |
| `TOOLBOX_LICENSE.txt` | Upstream EfficientPhys/rPPG-Toolbox code license |

Exported checkpoints retain only `model` and `epoch`, omitting training configuration. All weight
tensors are unchanged and compared with `torch.equal`. Removing metadata makes export-file SHA256
**different** from original `final.pt`; both hashes are in the manifest without replacing original
hashes in frozen records.

The package excludes videos, crops, PR/PPG labels, person IDs, per-video/window predictions,
ID-bearing splits, optimizers, passwords, and holdout. P4m exports only 18 parameter arrays
`w_*`/`ro_*`, not the complete research file `z14_p4m_frozen.npz`.
KDE priors are learned aggregate parameters, not individual label lists.

The three experts are not repackaged: install `open-rppg==0.1.1` via the `experts` profile.
Both upstream weight hashes are checked by `tools/smoke_environment.py`.

## Load checkpoints and parameters

Example from the repository after extraction:

```python
from pathlib import Path
import json, sys, torch

assets = Path("checkpoints/download/facevitals-inference-assets-20261123")
sys.path.insert(0, str(assets / "code"))
from EfficientPhys_fixed import EfficientPhys
from replicate_tsm import use_replicate_padding

fold = 0  # use the sample's correct fold for OOF scoring
model = use_replicate_padding(EfficientPhys(frame_depth=10, img_size=36))
payload = torch.load(assets / "checkpoints" / f"padding_fixed_fold{fold}_seed2027.pt",
                     map_location="cpu", weights_only=True)
model.load_state_dict(payload["model"], strict=True)
model.eval()
btc_folds = json.loads((assets / "readers/btc_reader_v4_folds.json").read_text())
p4m = json.loads((assets / "readers/public_p4m_parameters.json").read_text())
```

The CNN accepts normalized clips `[batch, 301, 3, 36, 36]` and returns 300 differences; apply
`cumsum` before readout. Public crops require moving-average filtering according to
`CONFIG.json` before interpolation/normalization. Different crops/clocks do not
guarantee reproduction of reported results.

The BTC reader uses a 42–180 bpm grid in 0.5 bpm steps. JSON keys are strings: before calling
`code/reader_v4.py:predG`, convert `prior` to a NumPy array and `beta`/`off` keys to
integers. Source class is **original video** stream bitrate <4,000,000 bit/s; Task
means activity beginning with `Task`. Do not reclassify using condition-video bitrate/dimensions.
`CONFIG.json` records the correct evidence ordering and offset cells.

P4m uses its own FFT grid, not BTC's 0.5 bpm grid. `w_*` order is CNN, POS,
CHROM, mean of three experts. `ro_*` order is gamma, sigma, tau, median flag,
quality reference, constant baseline. `code/public_fixed_math.py` supplies
`qscale`, `marg`, `point`; run forward–backward separately per recording. Do not concatenate
windows from different people. `deploy` is only for new data;
OOF scoring requires the matching `foldN`.

## Missing components and correct use

This describes the **historical Release `inference-assets-20261123`**; its
bytes/hashes remain unchanged. New-video commands use readers in the checkout's `models/`;
see [RUN_VIDEO.md](../RUN_VIDEO.md). CNN checkpoints in this Release
serve public/research workflows, not BTC reader_v4.

- **This Release has no all-dev BTC `deploy` fit.** It contains exactly the
  five existing train-fold fits; do not average them and call that a validated
  deployment configuration. Exporting did not fit models or open holdout.
- The new frontend's 22 November integration gate still exceeded output-difference limits.
  This is therefore not a complete submission. These parameters reproduce the existing
  research branch without changing or bypassing the failed gate.
- Table reproduction also needs authorized data and matching splits/mappings/caches/clocks.
  These files are not public Release assets. Do not use another fold's checkpoint
  or all-dev parameters to report validation scores.
- No new blanket license is assigned to team weights/code. Preserve upstream terms
  and dataset agreements; see repository `THIRD_PARTY_NOTICES.md`.

P100 dataset locations are documented in
[SERVER_DATASETS.md](https://github.com/Khangle-2006/SPS_Challange/blob/main/docs/SERVER_DATASETS.md).
