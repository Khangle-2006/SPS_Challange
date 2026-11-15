> Historical 5 November aggregate report, all three seeds where stated. Not the later single-seed padding-fixed comparison. Candidate/status statements are superseded by the root README (15 November). Private receipts/checkpoints named below are not bundled.

# Hybrid POS/model OOF results

| Branch | Condition | MAE [95% CI] | P95 [95% CI] | Bias [95% CI] | Coverage [95% CI] |
|---|---|---:|---:|---:|---:|
| ubfc_only | UBFC clean | 0.992 [0.553, 1.532] | 2.972 [1.284, 5.106] | -0.179 [-0.608, 0.200] | 1.000 [1.000, 1.000] |
| ubfc_only | UBFC C5 | 15.129 [11.007, 19.452] | 31.821 [24.653, 38.990] | -13.947 [-18.361, -9.704] | 1.000 [1.000, 1.000] |
| ubfc_only | MCD native | 12.622 [9.828, 15.540] | 26.687 [22.721, 30.576] | -8.625 [-11.598, -5.840] | 0.997 [0.994, 0.999] |
| ubfc_only | MCD-C5 | 16.473 [14.115, 18.810] | 48.890 [43.229, 54.771] | 0.130 [-2.630, 3.026] | 0.997 [0.994, 0.999] |
| v3_joint | UBFC clean | 1.248 [0.638, 2.042] | 4.176 [1.579, 7.593] | -0.596 [-1.370, 0.020] | 1.000 [1.000, 1.000] |
| v3_joint | UBFC C5 | 17.457 [12.762, 22.379] | 35.178 [27.849, 42.597] | -16.277 [-21.353, -11.462] | 1.000 [1.000, 1.000] |
| v3_joint | MCD native | 9.359 [7.244, 11.573] | 24.235 [19.795, 28.656] | -5.253 [-7.341, -3.293] | 0.997 [0.994, 0.999] |
| v3_joint | MCD-C5 | 13.895 [11.806, 16.001] | 45.297 [39.772, 50.988] | -1.495 [-3.693, 0.783] | 0.997 [0.994, 0.999] |
| hybrid_ubfc_only | UBFC clean | 0.717 [0.460, 1.040] | 1.947 [1.011, 3.101] | 0.111 [-0.151, 0.428] | 1.000 [1.000, 1.000] |
| hybrid_ubfc_only | UBFC C5 | 17.302 [13.162, 21.578] | 36.551 [29.700, 43.242] | -11.129 [-15.454, -6.895] | 1.000 [1.000, 1.000] |
| hybrid_ubfc_only | MCD native | 7.598 [5.755, 9.645] | 21.177 [17.479, 24.974] | -5.816 [-7.874, -3.960] | 0.997 [0.994, 0.999] |
| hybrid_ubfc_only | MCD-C5 | 16.689 [14.323, 19.057] | 48.056 [42.622, 53.563] | 1.358 [-1.430, 4.320] | 0.997 [0.994, 0.999] |
| hybrid_v3_joint | UBFC clean | 0.923 [0.480, 1.598] | 2.892 [1.102, 5.759] | -0.214 [-0.887, 0.255] | 1.000 [1.000, 1.000] |
| hybrid_v3_joint | UBFC C5 | 18.960 [14.798, 23.273] | 39.566 [32.713, 46.256] | -12.292 [-16.417, -8.292] | 1.000 [1.000, 1.000] |
| hybrid_v3_joint | MCD native | 4.373 [3.313, 5.596] | 17.148 [13.670, 20.807] | -2.269 [-3.340, -1.325] | 0.997 [0.994, 0.999] |
| hybrid_v3_joint | MCD-C5 | 14.472 [12.355, 16.619] | 45.621 [40.043, 51.352] | -0.150 [-2.413, 2.243] | 0.997 [0.994, 0.999] |

## hybrid_ubfc_only versus ubfc_only

```json
{
  "UBFC clean": {
    "delta": -0.2746930803571429,
    "CI95": [
      -0.7048758370535715,
      -0.007461983816964279
    ],
    "reading": "better"
  },
  "UBFC C5": {
    "delta": 2.173479352678571,
    "CI95": [
      0.9766479492187481,
      3.412395368303569
    ],
    "reading": "worse"
  },
  "MCD native": {
    "delta": -5.02382626850547,
    "CI95": [
      -6.7005959581747705,
      -3.4815784235883
    ],
    "reading": "better"
  },
  "MCD-C5": {
    "delta": 0.2159850431378203,
    "CI95": [
      -0.4879331450431967,
      0.8962494475128735
    ],
    "reading": "no detectable difference"
  },
  "S": {
    "delta": -1.0416799987280143,
    "CI95": [
      -1.737996561696055,
      -0.3772631057772864
    ]
  }
}
```

## hybrid_v3_joint versus v3_joint

```json
{
  "UBFC clean": {
    "delta": -0.3249162946428571,
    "CI95": [
      -0.7710239955357142,
      -0.032643345424107184
    ],
    "reading": "better"
  },
  "UBFC C5": {
    "delta": 1.503069196428573,
    "CI95": [
      -0.33181675502232194,
      3.3199393136160715
    ],
    "reading": "no detectable difference"
  },
  "MCD native": {
    "delta": -4.986273285280481,
    "CI95": [
      -6.753928725021514,
      -3.3607228818907853
    ],
    "reading": "better"
  },
  "MCD-C5": {
    "delta": 0.5768006329263642,
    "CI95": [
      -0.011863041534573907,
      1.155614278125936
    ],
    "reading": "no detectable difference"
  },
  "S": {
    "delta": -1.2693734611649212,
    "CI95": [
      -2.1566073793956417,
      -0.41954114014785
    ]
  }
}
```

## Adoption gates versus v3_joint

```json
{
  "hybrid_ubfc_only": {
    "S_relative_reduction": 0.08721521554794254,
    "UBFC_clean_relative_increase": -0.42557710580738917,
    "UBFC_C5_P95_delta": 1.372767857142854,
    "pass_S": false,
    "pass_clean": true,
    "pass_P95": false,
    "pass_all": false
  },
  "hybrid_v3_joint": {
    "S_relative_reduction": 0.13569196256102778,
    "UBFC_clean_relative_increase": -0.26035436811804813,
    "UBFC_C5_P95_delta": 4.387625558035708,
    "pass_S": true,
    "pass_clean": true,
    "pass_P95": false,
    "pass_all": false
  },
  "ubfc_only": {
    "S_relative_reduction": -0.024137044218270454,
    "UBFC_clean_relative_increase": -0.20546643563803024,
    "UBFC_C5_P95_delta": -3.3567940848214306,
    "pass_S": false,
    "pass_clean": true,
    "pass_P95": true,
    "pass_all": false
  }
}
```

Rule fixed before results. SNR is bandpeak±6bpm power divided by remaining0.7–3Hz bandpower. Ties selectmodel. No labels enterselection. There is no existing numeric SNR threshold in V3; unchanged support and finite/nonflat gates remain, with abstention when neither source is valid. Coverage is retained for all candidate windows. P95 is meanpersonwindowP95 afterseed averaging.
