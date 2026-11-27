# Dataset locations on P100

The paths below were verified to exist on 23 November 2026. They are paths
on **server P100**, not Internet downloads or directories created by `git
clone`. A data-owner-authorized SSH/server account is required.

| Dataset | Actual P100 path |
|---|---|
| BTC / FaceVitals | `/mnt/samsung_t5/khangle/facevitals/datasets/FaceVitals_BTC_20261107` |
| UBFC-rPPG | `/data_pool/khangle/facevitals/datasets/UBFC-rPPG` |
| MCD-rPPG | `/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG` |
| AiPEX | `/data_pool/khangle/facevitals/datasets/AiPEX` |

`/data_pool/khangle/facevitals/datasets/MCD-rPPG` is a symlink to the Samsung
MCD path above. Do not redownload/copy the dataset on the same server.

Dataset directories have broad read/write permissions, but parent directories `/data_pool` and
`/data_pool/khangle` currently permit traversal only by the owner/`datapool` group. For
`Permission denied`, ask an administrator for the appropriate account/group permissions;
path existence does not imply access for every account. This packaging run
did not change server permissions.

## Use only authorized dev data

Dataset roots contain raw data, **not an authorized scoring list**. Do not
use `find`/glob over an entire root to build a test set. Use the frozen dev manifest
and split; BTC holdout 72 videos/41 people and 7 UBFC2 test people remain closed.

Internal records to request from the maintainer for reproduction:

| Purpose | Internal location |
|---|---|
| BTC dev manifest | `/mnt/samsung_t5/khangle/facevitals/reports/round1_v4_20261105/btc_zeroshot/dev_manifest.json` |
| Public/BTC windows and audit folds | `/mnt/samsung_t5/khangle/facevitals/reports/round1_execution_20261110/inference_repair_20261113/MANIFEST.json` |
| Constraint experiment caches/spectra | `/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints/out` |

Manifests/caches can contain labels or IDs; do not upload them to GitHub. Public
checkpoint/reader bundles contain only weights and parameters and do not replace
these internal records. For another machine, request an authorized dev copy from the data owner
and configure paths using `docs/GETTING_STARTED.md`.
