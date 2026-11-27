> Archived aggregate report. Historical pending/status sentences describe its creation date, not current status. See the root README for the 15 November disposition. Private artifacts named here are not bundled.

# CPU A1/A2/A4/A5/A6 — reconciled results

Reuse saved outputs without inference/retraining. CI95% uses 10,000 person bootstraps, seed0. All BTC comparisons use identical video intersections; the full-cohort 11.37 baseline is recomputed within each group/intersection. Units: bpm.

## A1: gate still failed

New multi-ROI POS: MCD 13.7281; median blind SNR −3.8202 dB. Audit identified contributions from both ROI traces and the decoder. Repair restored canonical old oval/POS/decoder, matching 4304/4304 public predictions exactly. MCD 4.67015 but SNR −0.24956 dB: SNR still FAIL. Do not interpret Bayes as evidence of signal loss or select the post-hoc 4.03 combination.

Paired old/new/V4 tables and original SNR distributions: ../round1_bayes_20261108/public_comparison/fixed_pos_20261109/REPORT.md, METRICS.json; audit: public_comparison/pos_repair_20261110/REPORT.md.

## A4: label timing

Source: 218 videos; POS/CHROM has 162 videos/111 people supporting all 3 windows, not 218. POS last−first: −0.614 CI[−4.051;2.833]; Task last-window error increases +0.793 CI[−3.037;4.743]. No general evidence supports adding an end-weighted candidate, nor inferring oximeter reading times. MAE/CI/Spearman and both baselines are in A4_TIMING.json.

## A5: PBV and G

### all

| Method | Videos/people | MAE [CI95%] | Δ versus calibrated median [CI95%] |
|---|---:|---:|---:|
| constant | 218/158 | 12.071 [10.579;13.678] | +0.697 [-0.145;+1.510] |
| calibrated | 218/158 | 11.374 [9.916;12.890] | +0.000 [+0.000;+0.000] |
| POS | 218/158 | 13.578 [11.603;15.729] | +2.204 [+0.691;+3.759] |
| CHROM | 218/158 | 12.356 [10.594;14.226] | +0.982 [-0.486;+2.472] |
| G | 218/158 | 15.519 [13.492;17.656] | +4.146 [+2.520;+5.769] |
| PBV_trainfold | 218/158 | 12.159 [10.444;13.958] | +0.785 [-0.578;+2.126] |
| stack5_nested | 218/158 | 11.283 [9.805;12.826] | -0.091 [-0.644;+0.442] |

### Normal-Resting

| Method | Videos/people | MAE [CI95%] | Δ versus calibrated median [CI95%] |
|---|---:|---:|---:|
| constant | 62/62 | 8.661 [7.048;10.387] | -0.521 [-1.881;+0.811] |
| calibrated | 62/62 | 9.182 [7.507;10.902] | +0.000 [+0.000;+0.000] |
| POS | 62/62 | 14.102 [11.288;17.152] | +4.920 [+2.841;+6.994] |
| CHROM | 62/62 | 11.430 [9.345;13.669] | +2.248 [+0.164;+4.403] |
| G | 62/62 | 13.699 [11.164;16.368] | +4.516 [+2.114;+7.014] |
| PBV_trainfold | 62/62 | 11.311 [9.074;13.636] | +2.129 [+0.139;+4.106] |
| stack5_nested | 62/62 | 9.035 [7.478;10.672] | -0.147 [-1.061;+0.678] |

### Abnormal-Resting

| Method | Videos/people | MAE [CI95%] | Δ versus calibrated median [CI95%] |
|---|---:|---:|---:|
| constant | 76/76 | 14.197 [11.829;16.750] | +1.489 [+0.192;+2.749] |
| calibrated | 76/76 | 12.708 [10.410;15.121] | +0.000 [+0.000;+0.000] |
| POS | 76/76 | 12.773 [9.584;16.332] | +0.064 [-2.414;+2.522] |
| CHROM | 76/76 | 12.653 [9.587;16.115] | -0.055 [-2.415;+2.503] |
| G | 76/76 | 15.610 [12.536;18.909] | +2.902 [+0.499;+5.394] |
| PBV_trainfold | 76/76 | 11.867 [9.143;14.819] | -0.842 [-2.986;+1.299] |
| stack5_nested | 76/76 | 12.569 [10.151;15.134] | -0.140 [-0.957;+0.700] |

### Task

| Method | Videos/people | MAE [CI95%] | Δ versus calibrated median [CI95%] |
|---|---:|---:|---:|
| constant | 75/19 | 14.798 [10.228;20.289] | +1.848 [+0.532;+3.113] |
| calibrated | 75/19 | 12.950 [8.744;18.104] | +0.000 [+0.000;+0.000] |
| POS | 75/19 | 15.609 [11.732;20.197] | +2.659 [+0.092;+5.147] |
| CHROM | 75/19 | 14.277 [10.667;18.656] | +1.327 [-1.201;+4.069] |
| G | 75/19 | 20.680 [15.321;27.648] | +7.730 [+5.055;+10.785] |
| PBV_trainfold | 75/19 | 15.500 [11.273;20.778] | +2.549 [+0.559;+4.684] |
| stack5_nested | 75/19 | 13.209 [8.784;18.779] | +0.259 [-0.391;+1.029] |

### Walking

| Method | Videos/people | MAE [CI95%] | Δ versus calibrated median [CI95%] |
|---|---:|---:|---:|
| constant | 5/5 | 12.400 [6.000;19.200] | +1.640 [-0.630;+4.421] |
| calibrated | 5/5 | 10.760 [4.656;18.134] | +0.000 [+0.000;+0.000] |
| POS | 5/5 | 5.063 [1.042;11.357] | -5.697 [-14.030;+2.700] |
| CHROM | 5/5 | 10.293 [3.244;19.348] | -0.467 [-3.971;+2.731] |
| G | 5/5 | 18.389 [6.805;35.027] | +7.629 [-0.157;+17.222] |
| PBV_trainfold | 5/5 | 11.712 [2.641;22.078] | +0.952 [-6.020;+10.819] |
| stack5_nested | 5/5 | 10.417 [2.895;17.939] | -0.343 [-4.367;+4.546] |

constant = training-fold median PR; calibrated = calibrated median (full-cohort baseline 11.37); stack5_nested is an older result, not stacking D with new models.

## A6: bitrate/luma

All 66 stratified rows include old POS, constant, and calibrated methods on identical videos. Sources: ../round1_bayes_20261108/revision_20261109/COMPRESSION_REPORT.md and COMPRESSION_STRATA.csv. Video-level native |error|–bpp Spearman: POS −0.129; calibrated +0.112. Associations are weak and not causal. Bpp/luma are observed metadata, not lux or codec-normalized quality.

A2 quotations and 1b are in DECISIONS.md. A3/B/C/D are incomplete; do not claim exhaustive testing or passed gates.
