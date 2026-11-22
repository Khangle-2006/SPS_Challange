# BTC dev numeric export

218 videos,158 people; no holdout, face images or direct names. Video/person codes are local pseudonyms; re-identification map is not included. Not anonymous medical data: PR and derived physiology remain sensitive.

blocks=round(128*RGB block mean),uint16[T,6,6,3]; decode by /128. fill=round(255*fraction pixels with any nonzero RGB channel),uint8[T,6,6]; decode by /255. Spatial order is row,column. Metadata t/valid/rgb_mean/boxes copied exactly. ROI NPZ copied byte-for-byte. PR is a video scalar, not a synchronized waveform. t in seconds.

CSV recording rows restricted to218 videos; JSON calibration and aggregate tables retain historical fit/cohort scope and must not be assumed matched automatically. Direct identifiers stripped/replaced. No new model scoring. Original fold2 plan weight0.1 is historical, not W2 weight1.0.

SHA256 for each payload file is in SHA256.json. ZIP part manifests are in delivery receipt. Combine extracted parts into one folder.
