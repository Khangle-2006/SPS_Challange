> Historical 5 November aggregate report, all three seeds where stated. Not the later single-seed padding-fixed comparison. Candidate/status statements are superseded by the root README (15 November). Private receipts/checkpoints named below are not bundled.

# MCD-C5 matched evaluation

Fixed reading declared before results: deltaCIupper<0 supports the UBFC-specific C5 pattern; otherwise V3 has not demonstrated strong-compression robustness.

| Branch | Condition | MAE [95% CI] | P95 [95% CI] | Bias [95% CI] | Coverage [95% CI] |
|---|---|---:|---:|---:|---:|
| CHROM | UBFC clean | 1.106 [0.654, 1.686] | 3.161 [1.244, 5.580] | 0.046 [-0.444, 0.570] | 1.000 [1.000, 1.000] |
| CHROM | UBFC C5 | 24.276 [21.068, 27.511] | 46.926 [41.776, 52.002] | -2.753 [-8.333, 2.930] | 1.000 [1.000, 1.000] |
| CHROM | MCD native | 8.324 [7.032, 9.735] | 25.726 [22.634, 29.152] | -1.586 [-3.373, 0.282] | 0.997 [0.994, 0.999] |
| CHROM | MCD-C5 | 25.689 [23.784, 27.575] | 66.149 [62.878, 69.495] | 13.018 [9.646, 16.375] | 0.997 [0.994, 0.999] |
| POS | UBFC clean | 1.226 [0.620, 2.025] | 4.112 [1.568, 7.537] | -0.398 [-1.199, 0.225] | 1.000 [1.000, 1.000] |
| POS | UBFC C5 | 25.832 [22.323, 29.098] | 48.321 [42.962, 53.504] | -1.441 [-6.480, 3.653] | 1.000 [1.000, 1.000] |
| POS | MCD native | 4.670 [3.502, 6.017] | 16.663 [12.946, 20.811] | -0.795 [-1.880, 0.381] | 0.997 [0.994, 0.999] |
| POS | MCD-C5 | 18.723 [16.737, 20.716] | 53.832 [49.323, 58.382] | 2.233 [-0.458, 5.070] | 0.997 [0.994, 0.999] |
| constant_joint | UBFC clean | 15.623 [12.601, 18.834] | 21.057 [17.463, 24.736] | -8.130 [-13.267, -2.777] | 1.000 [1.000, 1.000] |
| constant_joint | UBFC C5 | 15.623 [12.601, 18.834] | 21.057 [17.463, 24.736] | -8.130 [-13.267, -2.777] | 1.000 [1.000, 1.000] |
| constant_joint | MCD native | 12.134 [10.306, 14.108] | 19.635 [17.717, 21.632] | 4.753 [1.518, 8.038] | 0.997 [0.994, 0.999] |
| constant_joint | MCD-C5 | 12.134 [10.306, 14.108] | 19.635 [17.717, 21.632] | 4.753 [1.518, 8.038] | 0.997 [0.994, 0.999] |
| constant_ubfc | UBFC clean | 14.441 [11.832, 17.275] | 19.479 [16.453, 22.632] | -0.000 [-5.161, 5.366] | 1.000 [1.000, 1.000] |
| constant_ubfc | UBFC C5 | 14.441 [11.832, 17.275] | 19.479 [16.453, 22.632] | -0.000 [-5.161, 5.366] | 1.000 [1.000, 1.000] |
| constant_ubfc | MCD native | 15.883 [13.655, 18.282] | 23.723 [21.369, 26.149] | 12.882 [9.699, 16.125] | 0.997 [0.994, 0.999] |
| constant_ubfc | MCD-C5 | 15.883 [13.655, 18.282] | 23.723 [21.369, 26.149] | 12.882 [9.699, 16.125] | 0.997 [0.994, 0.999] |
| ubfc_only | UBFC clean | 0.992 [0.553, 1.532] | 2.972 [1.284, 5.106] | -0.179 [-0.608, 0.200] | 1.000 [1.000, 1.000] |
| ubfc_only | UBFC C5 | 15.129 [11.007, 19.452] | 31.821 [24.653, 38.990] | -13.947 [-18.361, -9.704] | 1.000 [1.000, 1.000] |
| ubfc_only | MCD native | 12.622 [9.828, 15.540] | 26.687 [22.721, 30.576] | -8.625 [-11.598, -5.840] | 0.997 [0.994, 0.999] |
| ubfc_only | MCD-C5 | 16.473 [14.115, 18.810] | 48.890 [43.229, 54.771] | 0.130 [-2.630, 3.026] | 0.997 [0.994, 0.999] |
| v3_joint | UBFC clean | 1.248 [0.638, 2.042] | 4.176 [1.579, 7.593] | -0.596 [-1.370, 0.020] | 1.000 [1.000, 1.000] |
| v3_joint | UBFC C5 | 17.457 [12.762, 22.379] | 35.178 [27.849, 42.597] | -16.277 [-21.353, -11.462] | 1.000 [1.000, 1.000] |
| v3_joint | MCD native | 9.359 [7.244, 11.573] | 24.235 [19.795, 28.656] | -5.253 [-7.341, -3.293] | 0.997 [0.994, 0.999] |
| v3_joint | MCD-C5 | 13.895 [11.806, 16.001] | 45.297 [39.772, 50.988] | -1.495 [-3.693, 0.783] | 0.997 [0.994, 0.999] |

{
  "reading": {
    "delta": -2.577591784089133,
    "CI95": [
      -3.241393896346632,
      -1.9200978442910541
    ],
    "reading": "better"
  },
  "UBFC_specific_pattern_supported": true,
  "unique_MCD_windows": 2023,
  "new_losses": 0
}

MCD-C5 is a separate diagnostic condition and never entersS or branch selection. All15 existing checkpoints per branch are reused through their verified complete evaluation artifacts. Reference windows and HR are identical toMCDnative. Frame/support losses and classical failures remain explicit.

MCD source MPEG-4 followed by H.264 is double compression; this premise must be confirmed by SOURCE_CODEC_VERIFICATION.json, not inferred from the filename. A favorable MCD-C5 result supports the specified dataset-specific pattern without proving the physical cause of UBFC degradation.
