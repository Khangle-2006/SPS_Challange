# Freeze two reader candidates — November 16, 2026

**reader_v1 and reader_v2 have frozen code, JSON/MAT models, and SHA256 hashes.** This locks two candidates pending final configuration selection; it does not claim that the gate has passed or authorize holdout evaluation.

The user confirmed that verifying dev data against the locked snapshot was sufficient, without waiting to recover the original XLSX. **All 218/218 dev rows match in video ID, subject ID derived from the video ID, activity, and PR**; truth labels in the result tables also match the locked manifest. No new table replaces dev labels.

## Table provenance

- Downloaded again directly from the organizers' Google Sheets portal at **18:59:46 ICT on November 16, 2026** (11:59:46 UTC), without content changes.
- Hash of the organizers' current file: `284392dabe6cef27a181766d0f4d19f04a06af84a0b34efc8c945a07bc8a661f` — matches the existing copy and the download at 18:51:10 ICT that day. See FREEZE_FETCH.json and the earlier audit records.
- Originally recorded XLSX hash: `e52a9b190bf91f8a5d6d476b8374b8904f47328a668bfbe7cd60f3c14fafc49a`. The original file could not be recovered, so its exact creation/download date remains unverified; do not infer a download date from folder names. Data extraction/split records predate the November 12 hash lock.
- The dev manifest retains hash `8a6d1a7db588d71c9cfad420ec6d667efdaad4b0faa8188f940aafe5159a2f7c`, matching the November 12 record.

The original XLSX is unavailable for byte comparison. This **does not change the labels used for dev metrics**: scoring still uses the locked snapshot, verified row by row. This audit does not establish equality outside the 218 dev rows or explain the whole-file hash difference. The user accepted this audit scope for freezing.

## Holdout activity lock

`HOLDOUT_ACTIVITY_LOCK.csv` contains exactly **72 rows and two columns, `recording,activity`**. SHA256:

`a0f58cc5b87888aa591ec07de6957d197a954909f85a7db6dfd95d78a664cd76`

Only IDs and activity were read: **no PR labels, media access, or holdout inference**. When evaluation is authorized, verify this file's hash and use the locked flags: Task1/2/3 = Task; Resting/Walking = non-Task. Unknown IDs/flags must raise an error rather than be guessed or updated. Flags were locked before holdout results.

## Candidates and gates

| Candidate | Native dev MAE, bpm | Notes |
|---|---:|---|
| reader_v1 | 10.57015 | No activity offset |
| reader_v2 | 10.25048 | Two offsets fitted within each training fold for OOF evaluation; the full-dev model is for deployment only |

**The preregistered gate was not met: 10.25048 > 10 bpm.** Reader_v2 has the best native dev score of the two candidates, but has not passed the gate. The v2−v1 difference is −0.31967, with CI95 [−0.85733; +0.22855], which includes 0. Do not reselect offsets, beta, bandwidth, or any variant on dev.

Each candidate directory contains code, JSON/MAT files, and static/Octave receipts. Prior, beta, grid, and offset values (where applicable) match exactly between JSON and MAT. Octave receipts for five real ROIs are retained; actual MATLAB has not been run. Keep both directories together because v2 uses the fixed v1 frontend.

## Usage and verification

Run `python verify_frozen.py` at the package root to verify SHA256 hashes before use. Do not edit package contents; any subsequently authorized change requires a separate version.

Python v1: `python reader_v1_20261116/reader_v1.py --meta <meta.npz> --model reader_v1_20261116/reader_model_dev218.json --out <prediction.json>`.

Python v2: `python reader_v2_20261116/reader_v2.py --meta <meta.npz> --model reader_v2_20261116/reader_model_dev218_v2.json --task task --out <prediction.json>`; use `non-task` for locked Resting/Walking flags. Requires NumPy/SciPy and t/valid/rgb_mean inputs from the verified frontend.

Octave/MATLAB: v1 `process_round1(roiPath, outputPrefix, modelPath)`; v2 adds a fourth argument, `isTask`. Static images return the prior (plus offset in v2) with abstention; this fallback value must not be interpreted as a measured pulse.

## Next steps

C1/C4 reader_v1 continues with its running configuration; do not restart it. Reader_v2 C1/C4 has not been run; if later requested, use the frozen offsets without refitting. W2/W2b are closed under their gates, and T1/T2 records are retained. **The holdout remains locked until explicit user confirmation**, and must not open automatically on a planned date. No final configuration or submission decision has been made.
