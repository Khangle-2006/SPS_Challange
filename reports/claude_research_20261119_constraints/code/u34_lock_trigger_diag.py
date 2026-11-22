"""u34 (22 Nov, Claude): why did the integrated frontend switch native BTC videos to the lock0 path? Descriptive, numbers only (no
images leave the server). For each frame: MediaPipe FaceDetection full-range (model_selection=1, conf 0.5, the detector of x07/lock0)
on the frames decoded exactly as the team's frontend (x01_extract.BTCVideo), compared with the old native FaceMesh box (out/x01/native).
Reports frames with >= 2 detections, and for the extra detection: IoU with the native face box, area ratio, score, centre position,
persistence. Also the dev label vs the two reader_v3 predictions Codex reported. Usage: $AV code/u34_lock_trigger_diag.py <key> [<key>...]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import mediapipe as mp
import x01_extract as X
W = X.W
J = {j['key']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3]); i = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i; return i / u if u > 0 else 0.
for key in sys.argv[1:]:
    j = J[key]
    with np.load(W / 'out/x01/native' / (key + '.npz')) as z: nbox = z['box'].astype(float); nval = z['valid'].astype(bool); tref = z['t']
    with np.load(Path(j['crop']) / 'meta.npz') as z: ref_t = z['t']
    vid = X.BTCVideo(j['source'], ref_t); det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=.5)
    rows = []
    for k, fr in enumerate(vid):
        r = det.process(np.ascontiguousarray(fr)); H, Wd = fr.shape[:2]; D = []
        for d in (r.detections or []):
            bb = d.location_data.relative_bounding_box; D.append((bb.xmin * Wd, bb.ymin * H, (bb.xmin + bb.width) * Wd, (bb.ymin + bb.height) * H, float(d.score[0])))
        rows.append(D)
    det.close(); n = np.array([len(D) for D in rows]); multi = np.flatnonzero(n >= 2)
    print(f'== {key}: frames {len(rows)} | size {vid.w}x{vid.h} | detections per frame: {dict(zip(*np.unique(n, return_counts=True)))} | first frame with >=2: {multi[0] if len(multi) else None}')
    if len(multi):
        ex = []
        for k in multi:
            if k >= len(nbox) or not nval[k]: continue
            D = rows[k]; tb = nbox[k]; ious = [iou(tb, d[:4]) for d in D]; ti = int(np.argmax(ious))
            for m, d in enumerate(D):
                if m == ti: continue
                a = (d[2] - d[0]) * (d[3] - d[1]) / max(1., (tb[2] - tb[0]) * (tb[3] - tb[1]))
                ex.append((k, ious[m], iou(D[ti][:4], d[:4]), a, d[4], (d[0] + d[2]) / 2 / vid.w, (d[1] + d[3]) / 2 / vid.h, ious[ti]))
        if ex:
            e = np.array(ex); q = lambda c: f'{np.median(e[:, c]):.3f} [{e[:, c].min():.3f}; {e[:, c].max():.3f}]'
            runs = np.split(e[:, 0], np.flatnonzero(np.diff(e[:, 0]) > 1) + 1)
            print(f'   extra detections {len(e)} in {len(set(e[:, 0]))} frames | runs {len(runs)}, longest {max(len(r) for r in runs)} frames')
            print(f'   IoU(extra, native face box) {q(1)} | IoU(extra, detection on the target) {q(2)} | area / native face {q(3)} | score {q(4)}')
            print(f'   centre x/W {q(5)} | centre y/H {q(6)} | centre spread (sd) x {e[:, 5].std():.3f} y {e[:, 6].std():.3f} | IoU(target detection, native box) {q(7)}')
