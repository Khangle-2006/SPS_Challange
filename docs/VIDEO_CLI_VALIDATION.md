# Real-video CLI validation — R1, 23 November 2026

## Current code and models

The three experts use **self-warm**: reset to the state copied immediately after initialization,
process the same video twice, and take pass 2. No state crosses videos.
The deployment reader was refitted once on 218 dev videos; five fold models reproduce all
218 new OOF predictions with zero difference. Deployment models must not score dev.

BTC native: **8.33327 bpm**, CI95 [7.08958; 9.65911], 218 videos/158 people.
Versus reader_v3: Δ −1.25379 [−2.28246; −0.28346]. These are dev results after many
experiments. Selection of 3/17 experts on historical spectra was not rerun.
[Complete tables](RESULTS.md).

## Completed checks

| Check | Result |
|---|---|
| Reset versus reference, 2 videos × 3 experts | Spectral difference 0.0 |
| Self-warm versus reference, 2 videos × 3 experts | Spectral difference 0.0 |
| A → B → A, all 3 experts | Spectrum A difference 0.0 |
| 63 BTC/public inference runs | Complete; output counts and window masks match the previous run |
| 3 original dev videos, reference CLI with correct person fold | 3/3 PASS, PR difference versus OOF **0 bpm** |
| Repeated face image | `abstain=true`, `pr_bpm=null` |
| Faceless video | `abstain=true`; no prior reported as a measurement |
| Video too short | Exit 1; no padding to fabricate a result |
| Unit test runtime/tools | 16/16 PASS |
| Unit test scorer | 10/10 PASS |

The three raw-video runs took **206.79 / 184.77 / 173.26 seconds**; all raw and negative
checks took 589.13 seconds on the test machine. This is not an SLA for arbitrary videos/hardware.
[Aggregate receipt without individual IDs/labels/predictions](../reports/rerun_stateless_20261123/CLI_VALIDATION_AGGREGATE.json).

R1 reused the three isolated, lock-installed core/frontend/experts environments in clone NC
from previous validation; **fresh environment installation for R1 is not claimed**.
Source/configuration/models were checked with `tools/verify_source.py` before pushing.
Commands and dependencies: [RUN_VIDEO.md](RUN_VIDEO.md).

## Unchanged limitations

The CLI defaults to `reference` for single-person video. The `guarded` frontend previously
changed a frame and shifted PR; its equivalence gate did not pass. R1 fixes expert
state only, without changing detection/landmarks or promoting `guarded` to default.
Static-input validation here covers the whole CLI with input rejection; it does not establish
that every standalone CNN passes negative checks without that layer.

BTC holdout and 7 UBFC2 test people were not opened. No retraining.
[The 9.130 reset validation and bug history are preserved in the earlier commit](history/R1_STATE_BUG_20261123.md).


## Fresh clone and R1 assets — 24 November 2026

A separate clone and three fresh environments were created from locks, without research venv reuse.
Bootstrap checks pins/pip compatibility, expert weights, and readers. R1 assets download
automatically from GitHub; the ZIP and 17 files are SHA256-verified, and readers match the checkout byte-for-byte.

A fixed dev video was run from raw in both deployment and matching OOF-fold
modes. Both match server results by **0 bpm**, taking 196.78 and 195.76 seconds.
Static/faceless inputs abstain; short videos error without padding. These are two runs
on one video, not a raw-video rerun of all dev.

`tools/reproduce_r1.py` replays BTC native readers for all 218 videos and
public readouts for 24 rows from pre-reader evidence, without fitting or using
saved predictions as outputs. Every prediction is checked against frozen references; public metrics
use only windows paired with native, as in the R1 table. See [guide](REPRODUCE_SERVER.md).

Packaging issues found during testing remain in separate records: a copied configuration SHA
had an extra character (replaced from the original receipt); initial replay scored one extra MCD
window without a native pair (corrected to the R1 scoring mask). Neither models nor research
results were changed to fix packaging. Historical MAT code requires separate dependencies in
`submission/requirements.txt`; these are not PR CLI dependencies.

[Fresh-clone and 25-row replay receipt](validation/repo_ready_20261124.json). All 16 tools and 10 scorer tests PASS. Historical MAT schema was added; MAT/HDF5 tests require a separate dependency profile and are not included in these 26 CLI/scorer tests.
