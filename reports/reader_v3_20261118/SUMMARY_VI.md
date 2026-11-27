# Reader v3 — November 20, 2026 update

**Completed:** reader_v3 on C1/C4/C5, integration of the public b04 readout, and static-image checks through real checkpoints. All work ran on CPU, without training or holdout access.

The full report includes four groups, all controls, bias, Spearman, CIs, coverage, and reproduction commands: [INTEGRATION_REPORT_20261120.md](INTEGRATION_REPORT_20261120.md).

## BTC dev

| Condition | Videos/subjects | Reader v3 MAE | Constant, same videos |
|---|---:|---:|---:|
| Native | 218/158 | 9,587 | 12,071 |
| C1 gain 0,5 | 218/158 | 12,390 | 12,071 |
| C1 gain 0,25 | 218/158 | 12,873 | 12,071 |
| C1 gain 0,125 | 218/158 | 12,415 | 12,071 |
| C4 QVGA | 218/158 | 9,927 | 12,071 |
| C4 VGA | 218/158 | 10,110 | 12,071 |
| Historical C5 | 218/158 | 12.488 | 12.071 |
| C5′ default | 161/109 | 14.932 | 12.804 |
| C5′ bf0 | 158/106 | 12,298 | 13,072 |

MAE is averaged within subjects first, then across subjects, in bpm. The five native training-fold models are unchanged; beta/offsets are not refitted per condition. Video type uses source dimensions, not QVGA/VGA dimensions.

**C4 retains an advantage over the constant:** QVGA Δ −2.144, CI95 [−3.249; −1.073]; VGA Δ −1.960 [−3.057; −0.886]. Both differences from native still have CIs containing 0. QVGA has 23 abstained videos and VGA has 22, using prior + offset; do not describe all 218 videos as measured pulse.

C1 and historical C5 do not clearly outperform the constant. C5′ default is worse: Δ +2.128 [0.252; 4.177]. C5′ bf0 has Δ −0.774 [−2.124; 0.547], without evidence of improvement over the constant. Conditions with different coverage cannot support causal comparisons. C1 also includes the CRF18 storage codec; C5′ is an ffmpeg proxy, not verified against MATLAB VideoWriter.

## Public

Spectral fusion + b04 tracking, fitted on the other four subject folds, with offline processing:

| Condition | UBFC MAE | MCD MAE |
|---|---:|---:|
| Native | 0,702 | 2,319 |
| Historical C5 | 7.490 | 5.820 |
| C5′ default | 5.926 | 5.234 |
| C5′ bf0 | 1,105 | 2,334 |

There are 35 UBFC subjects and 60 MCD subjects; see the full report for window counts by condition. These results use saved public spectra and cannot be directly compared with older tables using different denominators. MCD native improves over padding on the same windows: Δ −4.257 [−5.559; −3.018]; UBFC native Δ −0.326 [−0.776; +0.011], with a CI still containing 0. Not every cell improves: UBFC C5′ default Δ +0.271 [−0.853; +1.528].

## Validation

- Independently recomputed 160 BTC metric rows, including MAE, bias, Spearman, and paired CIs with 10,000 samples, seed 0.
- All 8,522 public predictions, covering fusion and tracking, match the source b04 results exactly. The deployment module also matches the b04 function for every sequence.
- All 25/25 static images from real dev ROIs, across all five padding-fixed checkpoints, abstain in both readout modes. Ten non-finite CHROM cases are correctly blocked; NaN is not converted to HR. This does not test face detection from raw video.
- All 21 result files downloaded locally match remote SHA256 hashes. Three processes completed with exit code 0; no workers from this run remain.

## Configuration and limitations

Earlier native reproduction: v1 10.57015; v2 10.25048; v3 9.58706 on 218 videos/158 subjects, matching the source within <1e-9. V3 passes the numerical dev gate but is exploratory after many trials; its CI against v2 still contains 0. No automatic submission selection or holdout access is authorized.

The November 18 v1/v2/v3 bundle remains unchanged: `READER_CANDIDATES_FROZEN_20261118.zip`, SHA256 `3ca6f9db80903b2878874f48b6405266cf97ed9c95f7ad36bb8c62ce166cd080`. Original v3 model SHA256: `4570483ec06b44c9908fffeb82a19b9e05936fad305a5357d07359d54d1f55ae`. V3 has not been ported to Octave/MATLAB; earlier Octave checks cover v1/v2 only.

The user approved a shared cap of **120 GB**, with 500 GB for the project. Before the run: 110.418 GB including reserves; project 354.942 GB. The new remote directory used about 2.52 MB after computation. No data were deleted and no other jobs were stopped. The 72-video/41-subject holdout remains locked pending separate confirmation.
