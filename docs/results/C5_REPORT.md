> Archived aggregate report. Historical pending/status sentences describe its creation date, not current status. See the root README for the 15 November disposition. Private artifacts named here are not bundled.

# Comparison of old C5 and C5′

All C5′ runs are complete. Do not select methods from partial tables.

C5′ is an ffmpeg simulation without validated MATLAB VideoWriter equivalence. `default` retains default GOP/B-frames; `bf0` disables B-frames. Errors are averaged within each person, then across people; CIs use 10,000 person bootstraps, seed 0. Negative calibration deltas are better. P95_person_MAE is the 95th percentile of person MAEs, not window-error P95.

## UBFC/MCD — common windows across all four conditions

V3/V4 use matching folds and seed2027, not BTC's 15-checkpoint median. During partial runs, this intersection included only recordings completed in both C5′ conditions.

| Dataset | Condition | Method | People/windows | MAE [CI95%] | Bias |
|---|---|---|---:|---:|---:|
| UBFC2 | native | POS | 35/180 | 1.441 [0.650; 2.521] | -0.612 |
| UBFC2 | native | CHROM | 35/180 | 1.178 [0.649; 1.863] | -0.035 |
| UBFC2 | native | V3 | 35/180 | 1.408 [0.658; 2.469] | -0.711 |
| UBFC2 | native | V4 | 35/180 | 1.053 [0.565; 1.677] | -0.194 |
| UBFC2 | c5 | POS | 35/180 | 25.730 [22.197; 29.112] | -1.207 |
| UBFC2 | c5 | CHROM | 35/180 | 23.999 [20.644; 27.415] | -3.210 |
| UBFC2 | c5 | V3 | 35/180 | 17.357 [12.776; 22.019] | -15.631 |
| UBFC2 | c5 | V4 | 35/180 | 10.963 [7.503; 14.678] | -8.806 |
| UBFC2 | c5prime_default | POS | 35/180 | 23.323 [19.302; 27.349] | -7.263 |
| UBFC2 | c5prime_default | CHROM | 35/180 | 20.901 [17.930; 23.979] | -3.902 |
| UBFC2 | c5prime_default | V3 | 35/180 | 9.928 [6.045; 14.458] | -8.417 |
| UBFC2 | c5prime_default | V4 | 35/180 | 5.545 [2.992; 8.750] | -4.343 |
| UBFC2 | c5prime_bf0 | POS | 35/180 | 3.152 [1.722; 4.904] | -1.958 |
| UBFC2 | c5prime_bf0 | CHROM | 35/180 | 4.235 [2.292; 6.553] | -3.087 |
| UBFC2 | c5prime_bf0 | V3 | 35/180 | 2.572 [1.210; 4.215] | -1.256 |
| UBFC2 | c5prime_bf0 | V4 | 35/180 | 1.756 [0.824; 2.922] | -0.750 |
| MCD | native | POS | 60/1929 | 4.669 [3.497; 6.005] | -0.802 |
| MCD | native | CHROM | 60/1929 | 8.297 [6.989; 9.714] | -1.581 |
| MCD | native | V3 | 60/1929 | 9.115 [7.077; 11.321] | -5.532 |
| MCD | native | V4 | 60/1929 | 6.205 [4.718; 7.896] | -2.202 |
| MCD | c5 | POS | 60/1929 | 18.669 [16.673; 20.701] | +2.188 |
| MCD | c5 | CHROM | 60/1929 | 25.679 [23.772; 27.694] | +12.966 |
| MCD | c5 | V3 | 60/1929 | 13.976 [11.904; 16.116] | -1.327 |
| MCD | c5 | V4 | 60/1929 | 12.533 [10.299; 14.870] | +1.894 |
| MCD | c5prime_default | POS | 60/1929 | 22.020 [19.781; 24.264] | +10.517 |
| MCD | c5prime_default | CHROM | 60/1929 | 38.025 [35.487; 40.517] | +31.032 |
| MCD | c5prime_default | V3 | 60/1929 | 9.708 [8.130; 11.387] | -5.312 |
| MCD | c5prime_default | V4 | 60/1929 | 7.675 [6.226; 9.219] | -3.021 |
| MCD | c5prime_bf0 | POS | 60/1929 | 7.242 [5.785; 8.823] | -0.804 |
| MCD | c5prime_bf0 | CHROM | 60/1929 | 9.787 [8.441; 11.206] | +1.171 |
| MCD | c5prime_bf0 | V3 | 60/1929 | 5.071 [4.039; 6.175] | -1.860 |
| MCD | c5prime_bf0 | V4 | 60/1929 | 4.559 [3.579; 5.616] | -0.489 |

## BTC — common videos within each condition

Native calibration preserves saved predictions (11.37 on the old cohort). C5′ calibration uses the same two-coefficient formula, refitted on that condition's training folds; it must not be called 11.37 MAE on C5′. All tables include constant and calibrated baselines. Fit C5′ only after all outputs are available.

| Condition | Group | Method | Videos/people | MAE [CI95%] | Δ calibration [CI95%] |
|---|---|---|---:|---:|---:|
| c5 | all | constant | 217/157 | 12.097 [10.595; 13.716] | -0.151 [-0.387; +0.086] |
| c5 | all | calibrated | 217/157 | 12.248 [10.755; 13.865] | +0.000 [+0.000; +0.000] |
| c5 | all | POS | 217/157 | 41.060 [36.991; 45.103] | +28.812 [+24.862; +32.782] |
| c5 | all | CHROM | 217/157 | 50.503 [46.451; 54.501] | +38.255 [+34.096; +42.327] |
| c5 | all | V3 | 217/157 | 19.580 [17.652; 21.643] | +7.332 [+5.205; +9.485] |
| c5 | all | V4 | 217/157 | 19.138 [16.726; 21.766] | +6.891 [+4.703; +9.175] |
| c5 | Normal-Resting | constant | 62/62 | 8.661 [7.048; 10.387] | -0.283 [-0.704; +0.145] |
| c5 | Normal-Resting | calibrated | 62/62 | 8.944 [7.399; 10.562] | +0.000 [+0.000; +0.000] |
| c5 | Normal-Resting | POS | 62/62 | 36.815 [30.198; 43.459] | +27.871 [+21.496; +34.174] |
| c5 | Normal-Resting | CHROM | 62/62 | 43.329 [36.446; 49.993] | +34.385 [+27.451; +41.106] |
| c5 | Normal-Resting | V3 | 62/62 | 17.786 [14.900; 20.770] | +8.842 [+5.547; +12.046] |
| c5 | Normal-Resting | V4 | 62/62 | 16.177 [12.556; 20.204] | +7.233 [+3.702; +11.107] |
| c5 | Abnormal-Resting | constant | 75/75 | 14.280 [11.893; 16.773] | +0.037 [-0.297; +0.369] |
| c5 | Abnormal-Resting | calibrated | 75/75 | 14.243 [11.832; 16.759] | +0.000 [+0.000; +0.000] |
| c5 | Abnormal-Resting | POS | 75/75 | 43.472 [37.158; 49.659] | +29.228 [+23.054; +35.321] |
| c5 | Abnormal-Resting | CHROM | 75/75 | 56.485 [50.586; 62.323] | +42.242 [+36.216; +48.046] |
| c5 | Abnormal-Resting | V3 | 75/75 | 19.296 [16.329; 22.567] | +5.053 [+1.596; +8.469] |
| c5 | Abnormal-Resting | V4 | 75/75 | 20.405 [16.739; 24.405] | +6.161 [+2.683; +9.679] |
| c5 | Task | constant | 75/19 | 14.798 [10.228; 20.289] | -0.456 [-0.762; -0.165] |
| c5 | Task | calibrated | 75/19 | 15.254 [10.565; 20.912] | +0.000 [+0.000; +0.000] |
| c5 | Task | POS | 75/19 | 44.902 [38.674; 51.480] | +29.647 [+19.672; +39.127] |
| c5 | Task | CHROM | 75/19 | 49.793 [42.887; 56.584] | +34.539 [+23.017; +44.783] |
| c5 | Task | V3 | 75/19 | 26.061 [20.741; 32.350] | +10.806 [+7.205; +13.946] |
| c5 | Task | V4 | 75/19 | 23.231 [17.181; 30.206] | +7.976 [+4.958; +11.015] |
| c5 | Walking | constant | 5/5 | 12.400 [6.000; 19.200] | -0.018 [-0.997; +0.953] |
| c5 | Walking | calibrated | 5/5 | 12.418 [6.614; 18.279] | +0.000 [+0.000; +0.000] |
| c5 | Walking | POS | 5/5 | 54.954 [36.918; 72.990] | +42.536 [+25.017; +60.056] |
| c5 | Walking | CHROM | 5/5 | 63.054 [49.390; 75.429] | +50.636 [+32.224; +63.643] |
| c5 | Walking | V3 | 5/5 | 22.991 [13.255; 33.491] | +10.574 [-1.020; +19.181] |
| c5 | Walking | V4 | 5/5 | 18.475 [7.504; 30.940] | +6.057 [-3.608; +15.199] |
| c5prime_bf0 | all | constant | 158/106 | 13.072 [11.135; 15.087] | -0.032 [-0.431; +0.359] |
| c5prime_bf0 | all | calibrated | 158/106 | 13.104 [11.146; 15.168] | +0.000 [+0.000; +0.000] |
| c5prime_bf0 | all | POS | 158/106 | 18.708 [16.122; 21.316] | +5.604 [+2.933; +8.305] |
| c5prime_bf0 | all | CHROM | 158/106 | 16.424 [14.155; 18.702] | +3.320 [+1.305; +5.228] |
| c5prime_bf0 | all | V3 | 158/106 | 18.650 [15.873; 21.505] | +5.546 [+2.923; +8.029] |
| c5prime_bf0 | all | V4 | 158/106 | 18.197 [15.612; 20.879] | +5.093 [+2.543; +7.601] |
| c5prime_bf0 | Normal-Resting | constant | 28/28 | 7.714 [5.750; 9.679] | +0.016 [-0.968; +1.030] |
| c5prime_bf0 | Normal-Resting | calibrated | 28/28 | 7.699 [5.712; 9.783] | +0.000 [+0.000; +0.000] |
| c5prime_bf0 | Normal-Resting | POS | 28/28 | 15.214 [10.861; 19.974] | +7.515 [+2.521; +12.847] |
| c5prime_bf0 | Normal-Resting | CHROM | 28/28 | 14.143 [10.853; 17.662] | +6.444 [+2.930; +9.975] |
| c5prime_bf0 | Normal-Resting | V3 | 28/28 | 14.294 [10.307; 18.825] | +6.595 [+1.803; +11.778] |
| c5prime_bf0 | Normal-Resting | V4 | 28/28 | 14.081 [10.214; 18.184] | +6.382 [+2.335; +10.601] |
| c5prime_bf0 | Abnormal-Resting | constant | 61/61 | 14.869 [12.197; 17.623] | +0.179 [-0.310; +0.661] |
| c5prime_bf0 | Abnormal-Resting | calibrated | 61/61 | 14.690 [11.947; 17.507] | +0.000 [+0.000; +0.000] |
| c5prime_bf0 | Abnormal-Resting | POS | 61/61 | 17.289 [14.048; 20.731] | +2.600 [-0.871; +6.145] |
| c5prime_bf0 | Abnormal-Resting | CHROM | 61/61 | 15.960 [12.829; 19.155] | +1.271 [-1.433; +4.110] |
| c5prime_bf0 | Abnormal-Resting | V3 | 61/61 | 17.479 [13.779; 21.339] | +2.789 [-0.658; +6.294] |
| c5prime_bf0 | Abnormal-Resting | V4 | 61/61 | 17.653 [14.198; 21.150] | +2.963 [-0.630; +6.648] |
| c5prime_bf0 | Task | constant | 65/16 | 16.042 [10.719; 22.260] | -0.942 [-1.642; -0.312] |
| c5prime_bf0 | Task | calibrated | 65/16 | 16.983 [11.533; 23.397] | +0.000 [+0.000; +0.000] |
| c5prime_bf0 | Task | POS | 65/16 | 28.559 [23.198; 34.816] | +11.576 [+7.809; +15.326] |
| c5prime_bf0 | Task | CHROM | 65/16 | 21.038 [14.623; 27.996] | +4.054 [+0.344; +7.727] |
| c5prime_bf0 | Task | V3 | 65/16 | 29.707 [23.185; 37.254] | +12.723 [+9.350; +15.462] |
| c5prime_bf0 | Task | V4 | 65/16 | 26.949 [19.809; 35.108] | +9.965 [+5.547; +14.099] |
| c5prime_bf0 | Walking | constant | 4/4 | 13.250 [5.250; 21.000] | +0.096 [-1.027; +1.130] |
| c5prime_bf0 | Walking | calibrated | 4/4 | 13.154 [6.008; 20.454] | +0.000 [+0.000; +0.000] |
| c5prime_bf0 | Walking | POS | 4/4 | 20.098 [4.000; 36.196] | +6.944 [-9.001; +22.889] |
| c5prime_bf0 | Walking | CHROM | 4/4 | 22.418 [14.388; 32.519] | +9.264 [+5.746; +12.781] |
| c5prime_bf0 | Walking | V3 | 4/4 | 21.307 [11.031; 34.299] | +8.152 [-1.970; +18.274] |
| c5prime_bf0 | Walking | V4 | 4/4 | 21.526 [10.812; 35.252] | +8.372 [-2.190; +18.933] |
| c5prime_default | all | constant | 161/109 | 12.804 [10.908; 14.801] | -0.068 [-0.414; +0.276] |
| c5prime_default | all | calibrated | 161/109 | 12.872 [10.949; 14.863] | +0.000 [+0.000; +0.000] |
| c5prime_default | all | POS | 161/109 | 47.711 [42.875; 52.701] | +34.839 [+30.056; +39.707] |
| c5prime_default | all | CHROM | 161/109 | 43.851 [39.031; 48.811] | +30.979 [+25.969; +35.992] |
| c5prime_default | all | V3 | 161/109 | 20.550 [17.847; 23.426] | +7.678 [+5.222; +10.101] |
| c5prime_default | all | V4 | 161/109 | 20.446 [17.561; 23.376] | +7.574 [+4.720; +10.475] |
| c5prime_default | Normal-Resting | constant | 29/29 | 7.759 [5.862; 9.621] | +0.346 [-0.372; +1.051] |
| c5prime_default | Normal-Resting | calibrated | 29/29 | 7.413 [5.529; 9.460] | +0.000 [+0.000; +0.000] |
| c5prime_default | Normal-Resting | POS | 29/29 | 39.050 [29.756; 48.656] | +31.637 [+22.069; +41.467] |
| c5prime_default | Normal-Resting | CHROM | 29/29 | 38.971 [29.211; 48.553] | +31.558 [+21.858; +41.290] |
| c5prime_default | Normal-Resting | V3 | 29/29 | 18.019 [12.994; 23.957] | +10.606 [+5.508; +16.353] |
| c5prime_default | Normal-Resting | V4 | 29/29 | 14.796 [10.012; 20.485] | +7.383 [+2.676; +12.922] |
| c5prime_default | Abnormal-Resting | constant | 63/63 | 14.413 [11.746; 17.191] | +0.006 [-0.471; +0.473] |
| c5prime_default | Abnormal-Resting | calibrated | 63/63 | 14.406 [11.731; 17.153] | +0.000 [+0.000; +0.000] |
| c5prime_default | Abnormal-Resting | POS | 63/63 | 54.467 [48.030; 60.762] | +40.061 [+33.767; +46.004] |
| c5prime_default | Abnormal-Resting | CHROM | 63/63 | 48.162 [41.332; 54.847] | +33.756 [+26.759; +40.571] |
| c5prime_default | Abnormal-Resting | V3 | 63/63 | 19.715 [16.237; 23.514] | +5.308 [+1.844; +8.686] |
| c5prime_default | Abnormal-Resting | V4 | 63/63 | 20.716 [17.203; 24.607] | +6.310 [+2.410; +10.378] |
| c5prime_default | Task | constant | 65/16 | 16.042 [10.719; 22.260] | -1.111 [-1.555; -0.698] |
| c5prime_default | Task | calibrated | 65/16 | 17.153 [11.637; 23.581] | +0.000 [+0.000; +0.000] |
| c5prime_default | Task | POS | 65/16 | 40.435 [33.341; 47.924] | +23.282 [+12.221; +34.805] |
| c5prime_default | Task | CHROM | 65/16 | 40.242 [33.941; 46.573] | +23.089 [+12.994; +32.760] |
| c5prime_default | Task | V3 | 65/16 | 30.711 [23.645; 38.262] | +13.558 [+9.020; +17.719] |
| c5prime_default | Task | V4 | 65/16 | 28.494 [22.106; 35.359] | +11.341 [+6.068; +16.762] |
| c5prime_default | Walking | constant | 4/4 | 13.250 [5.250; 21.000] | +0.938 [-0.857; +2.144] |
| c5prime_default | Walking | calibrated | 4/4 | 12.312 [5.309; 19.315] | +0.000 [+0.000; +0.000] |
| c5prime_default | Walking | POS | 4/4 | 41.095 [17.532; 69.197] | +28.783 [+11.455; +53.581] |
| c5prime_default | Walking | CHROM | 4/4 | 31.554 [6.348; 63.431] | +19.241 [-9.417; +50.778] |
| c5prime_default | Walking | V3 | 4/4 | 14.728 [4.904; 24.551] | +2.415 [-8.382; +13.872] |
| c5prime_default | Walking | V4 | 4/4 | 34.520 [6.893; 62.147] | +22.208 [-8.352; +52.767] |
| native | all | constant | 218/158 | 12.071 [10.579; 13.678] | +0.697 [-0.145; +1.510] |
| native | all | calibrated | 218/158 | 11.374 [9.916; 12.890] | +0.000 [+0.000; +0.000] |
| native | all | POS | 218/158 | 13.578 [11.603; 15.729] | +2.204 [+0.691; +3.759] |
| native | all | CHROM | 218/158 | 12.356 [10.594; 14.226] | +0.982 [-0.486; +2.472] |
| native | all | V3 | 218/158 | 17.152 [14.869; 19.614] | +5.779 [+3.812; +7.817] |
| native | all | V4 | 218/158 | 16.264 [14.098; 18.532] | +4.890 [+3.219; +6.586] |
| native | Normal-Resting | constant | 62/62 | 8.661 [7.048; 10.387] | -0.521 [-1.881; +0.811] |
| native | Normal-Resting | calibrated | 62/62 | 9.182 [7.507; 10.902] | +0.000 [+0.000; +0.000] |
| native | Normal-Resting | POS | 62/62 | 14.102 [11.288; 17.152] | +4.920 [+2.841; +6.994] |
| native | Normal-Resting | CHROM | 62/62 | 11.430 [9.345; 13.669] | +2.248 [+0.164; +4.403] |
| native | Normal-Resting | V3 | 62/62 | 15.295 [11.923; 18.680] | +6.113 [+2.764; +9.402] |
| native | Normal-Resting | V4 | 62/62 | 13.541 [10.630; 16.732] | +4.359 [+1.373; +7.343] |
| native | Abnormal-Resting | constant | 76/76 | 14.197 [11.829; 16.750] | +1.489 [+0.192; +2.749] |
| native | Abnormal-Resting | calibrated | 76/76 | 12.708 [10.410; 15.121] | +0.000 [+0.000; +0.000] |
| native | Abnormal-Resting | POS | 76/76 | 12.773 [9.584; 16.332] | +0.064 [-2.414; +2.522] |
| native | Abnormal-Resting | CHROM | 76/76 | 12.653 [9.587; 16.115] | -0.055 [-2.415; +2.503] |
| native | Abnormal-Resting | V3 | 76/76 | 16.897 [13.365; 20.689] | +4.189 [+1.243; +7.342] |
| native | Abnormal-Resting | V4 | 76/76 | 16.641 [13.402; 20.173] | +3.932 [+1.546; +6.410] |
| native | Task | constant | 75/19 | 14.798 [10.228; 20.289] | +1.848 [+0.532; +3.113] |
| native | Task | calibrated | 75/19 | 12.950 [8.744; 18.104] | +0.000 [+0.000; +0.000] |
| native | Task | POS | 75/19 | 15.609 [11.732; 20.197] | +2.659 [+0.092; +5.147] |
| native | Task | CHROM | 75/19 | 14.277 [10.667; 18.656] | +1.327 [-1.201; +4.069] |
| native | Task | V3 | 75/19 | 24.493 [19.020; 30.921] | +11.542 [+8.691; +14.518] |
| native | Task | V4 | 75/19 | 23.619 [17.854; 30.378] | +10.669 [+7.501; +13.564] |
| native | Walking | constant | 5/5 | 12.400 [6.000; 19.200] | +1.640 [-0.630; +4.421] |
| native | Walking | calibrated | 5/5 | 10.760 [4.656; 18.134] | +0.000 [+0.000; +0.000] |
| native | Walking | POS | 5/5 | 5.063 [1.042; 11.357] | -5.697 [-14.030; +2.700] |
| native | Walking | CHROM | 5/5 | 10.293 [3.244; 19.348] | -0.467 [-3.971; +2.731] |
| native | Walking | V3 | 5/5 | 18.519 [5.159; 38.093] | +7.759 [-1.655; +20.628] |
| native | Walking | V4 | 5/5 | 18.871 [6.653; 37.280] | +8.111 [-0.107; +19.398] |

Coverage/window failures are in C5_PUBLIC_COVERAGE.json and C5_BTC_COVERAGE.json. Constant/calibrated methods have video-level predictions, not new windows; zero window counts mean not applicable, not 0% failure. Old/native tables here do not reload waveforms or recompute boundary-window statistics.

No holdout access; thresholds, references, and checkpoints are not changed based on scores.
