# Data, clocks, and metrics

## What is a sample?

| Dataset | Data | PR/HR reference | Caveat |
|---|---|---|---|
| BTC | Normal/Abnormal phone videos, multiple activities per person | One video-level PR from the BTC table | No synchronized PPG for a waveform target; exact PR measurement time is uncertain |
| UBFC-rPPG | Face video and `ground_truth.txt` | PPG, HR, and dataset timestamps | UBFC1 and UBFC2 have different schemas; label rows are not necessarily video frames |
| MCD-rPPG | FullHD/USB/Iriun and contact PPG | PPG has a separate clock | Measured FullHD files are 640×480 MPEG-4 Part 2; camera names do not guarantee resolution |
| AiPEX | Three ROI crops from one recording | Person-level mean HR | 88 people, not 264; the assumed 15 fps remains incompletely verified |

BTC dev contains 221 available videos from 161 people; the primary paired report uses 218 eligible videos/158 people. Holdout 72/41 is locked. Do not include holdout in statistics, normalization, priors, or QC. Two missing videos were deferred; do not fabricate replacements.

### FPS and timestamps

`fps` means frames per second. Container FPS, sensor FPS, and the interpolation grid are different quantities. Models typically use 301 image samples for 10 s at 30 Hz, leaving 300 after differencing. This **does not** establish 10 s of source support: check the time span, support, gaps, and valid mask. Do not bridge large gaps or extrapolate a missing endpoint.

MCD119 source: 640×480 MPEG-4 Part 2; median total-file bitrate using source-clock duration is 1446.118 kbps. UBFC source FPS in T9 was about 23.18–29.97; historical `h26430` means retaining all frames, not necessarily exactly 30 fps. BTC varies in FPS/rotation/HLG/10-bit encoding, so per-file metadata is required.

Canonical MCD PPG is multiplied by −1 **exactly once**; UBFC uses +1. Sign correction is not time synchronization. V5 offsets estimate phase, not physical synchronization of camera exposures.

## Terms

| Name | Meaning |
|---|---|
| PR | Pulse rate in beats per minute; BTC references come from a pulse-measuring device |
| HR | Heart rate in beats per minute; used in decoders/research. PR is not automatically equivalent to ECG HR |
| bpm | Beats per minute; 1.5 Hz corresponds to 90 bpm |
| PPG | Contact optical waveform; amplitude is not measured in bpm |
| rPPG | Pulse signal inferred from video color changes |
| SpO₂ | Peripheral oxygen saturation (%); not a target of this Round 1 |
| ROI | Image region used for signal extraction, such as forehead, cheeks, or face oval |
| POS / CHROM / G / PBV | Signal extraction methods for RGB traces; G uses only green |
| native | No additional experimental compression; the source may already be compressed |
| C5 / C5′ | Low-FPS + low-bitrate condition. C5′ here is an ffmpeg proxy, not MATLAB parity |
| bf0 | Encoding without B-frames; does not establish removal of all artifacts |
| d_k | Model difference output; the decoder applies cumsum before detrending/filtering/spectral analysis |
| OOF | Predictions for validation-fold people excluded from the corresponding fit |
| CI95% | Bootstrap confidence interval; a difference CI containing 0 does not establish improvement in that test |
| Static PASS | Repeating one image must make the decoder reject HR. PASS applies only to the tested path |

## Formulas

For person p with n_p videos: `MAE_p = mean_v |prediction_pv − PR_pv|`.
`MAE = mean_p MAE_p`: equal weight per person, without allowing people with more Task videos to dominate.
`bias = mean_p mean_v (prediction_pv − PR_pv)`: negative means underprediction.
Reported delta is `new method − baseline`; a negative delta is better.

Current bootstrap: resample **people**, retaining all observations and paired methods, 10,000 times with seed 0. Percentiles 2.5 and 97.5 form CI95%. Do not bootstrap windows as independent observations. Historical T9 uses seed 20261031, reported separately.

`S_public = (MAE_UBFC_native + MAE_UBFC_C5_old + MAE_MCD_native)/3`.
P95 can refer to window errors or person-level MAEs; reports specify which, and these cannot be interchanged. Report video and window coverage; abstention is not zero error. People may occur in multiple task groups, so group counts cannot be summed to obtain the cohort size.
