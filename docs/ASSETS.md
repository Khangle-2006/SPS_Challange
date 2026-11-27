# Data, checkpoints, and reproduction bundles

The repository contains code and aggregate results; it does not grant permission to use or redistribute
datasets. Labels, demographic tables, and holdout are not downloaded automatically.

**Downloadable package:** [checkpoints and reader configurations](INFERENCE_ASSETS.md).
Server datasets: [SERVER_DATASETS.md](SERVER_DATASETS.md).

**Current BTC joint:** the dedicated fit/replay inventory, array schemas, and SHA256
manifest commands are in [BTC_JOINT_REPRODUCING.md](BTC_JOINT_REPRODUCING.md).
The `numeric_dev` and `reader_v3` entries below are archived; they do not replace
the `export_extracted`/native/shifted/reference bundle required by BTC joint.

## Where to obtain assets

| Asset | Source / request procedure | Required contents and checks |
|---|---|---|
| UBFC-rPPG Dataset 2 | [Authors' official site](https://sites.google.com/view/ybenezeth/ubfcrppg) | Videos `subject*/vid.avi`, `ground_truth.txt`; request the team's 35-person dev split to reproduce the tables; exclude the 7 test participants |
| MCD | Ask the maintainer for the source and terms of the **exact MCD version used by the team**; a public download link has not been verified | FullHD view, PPG, session/person mapping, retained-frame timestamps, and clock receipts. Do not substitute a different dataset with the same name |
| BTC dev | Organizer portal under the team's permission; request a **dev-only** bundle from the maintainer | 218 scored videos, 158 people; PR labels, person/fold, activity, source metadata, and exclusions. Exclude the 72-video/41-person holdout |
| AiPEX | Organizer source and terms; not needed for reader_v4/P4m inference | Required only for historical training reproduction, not the quick start |
| Three experts | [open-rppg 0.1.1 on PyPI](https://pypi.org/project/open-rppg/0.1.1/) | Install the experts profile; weights are included. Check SHA with smoke tests; do not download weights from unverified links |
| Team CNN with corrected padding | [Release inference-assets-r1-20261124](https://github.com/Khangle-2006/SPS_Challange/releases/tag/inference-assets-r1-20261124) | Five model/epoch-only exports for folds 0–4, seed 2027; unchanged tensors, export and source SHAs in the manifest |
| Research readers and caches | BTC CLI readers are in `models/`; the R1 release contains current deployment/fold/P4m parameters; request splits/caches separately | Only parameters are public; per-video predictions/labels remain private. Do not mix CLI models with spectra from earlier runs without expert resets |

Do not substitute fold-0 checkpoints for all folds. Frozen readout files can
contain labels and per-window predictions (e.g. `z14_p4m_frozen.npz`); **do not** upload
the full files to public GitHub Releases. The release extracts only inference parameters with separate hashes;
it excludes the original label/ID/prediction arrays.

The team definitions record these two open-rppg weights:

```text
FacePhys.rlap.weights.h5  6cd99510bcc5771f9cebcdb0f47a5cb3692d19f1e3d6e2444af5e8fac0ee3c07
ME.pure.weights.h5        2135995cb4161d1d4c7c3aa45d333dc6b1f5818b0652d11da9fc0fb0b6e11bf0
```

ME-chunk.pure and ME-flow.pure share a weight file but use different processing modes. The code
license does not establish all rights to weights/training data; review
`THIRD_PARTY_NOTICES.md` and upstream terms before redistribution.

## Maintainer request checklist

Specify the stage, OS/Python version, and repository commit when requesting a bundle. Do not
post personal data, credentials, or videos in public issues. Table reproduction requires:

1. `inference_repair_20261113/MANIFEST.json`, `FROZEN.json`, and their referenced
   references, source clocks/caches; public dev/BTC dev only.
2. Numeric BTC `w2_btc_20261116/numeric_dev/VIDEOS.json`, `blocks/*.npz`,
   the baseline tables used, and private recording mapping (needed by y05/y11).
3. `claude_research_20261118/out/dev_video_meta.csv` and dependencies of
   `lib_btc`/the reader; use **source** metadata, not crop/constraint dimensions.
4. Crops/spectra for selected stages (`out/x10`, `x15`, `y01_spectra.pkl`,
   public x13/x20/y12/y13...), plus completion/clock/hash receipts.
5. Five public checkpoints and readers with correct OOF/deployment scope; do not fit on
   validation people or use all-dev models to report OOF dev performance.
6. A complete file inventory, SHA256, and `complete_for` declaring stages with all dependencies.

Cache-based stages do not necessarily require every raw video. The bundle
must state its starting point and the tables it can reproduce; a partial cache is not
a complete bundle for all tables.

## Private asset manifest

Use `configs/assets.example.json` as a template **outside Git**, for example
`../private-assets/assets.json`. Each file is declared as follows:

```json
{
  "schema_version": 1,
  "scope": "dev-only",
  "complete_for": ["btc-reader-v3"],
  "files": [
    {
      "source": "received/VIDEOS.json",
      "destination": "reports/w2_btc_20261116/numeric_dev/VIDEOS.json",
      "sha256": "<64 hexadecimal characters supplied by the maintainer>",
      "kind": "metadata",
      "relocate_json_paths": false
    }
  ]
}
```

This example shows only **one** entry and is not enough to run the stage. `source` is relative
to the manifest directory. `destination` is relative to the workspace and must not
contain `..`, be absolute, or overwrite code. List **every file**; do not
point at a raw-data directory containing holdout. SHAs must come from the authorized bundle; do not
invent hashes to make doctor pass.

Only JSON containing server paths needs `relocate_json_paths=true`. Do not change
labels, folds, activity, or frozen reference hashes. Resolve hash failures
with traceable receipts, not silent edits. Do not deserialize pickle from
untrusted sources; historical scripts use pickle and accept only authorized artifacts
with hashes verified against the maintainer's records.

`tools/repro.py prepare` copies binaries unchanged, records hashes before/after metadata
path relocation, and retains the source. Private configurations, assets, outputs, and venvs are
gitignored; always check `git status` before committing.
