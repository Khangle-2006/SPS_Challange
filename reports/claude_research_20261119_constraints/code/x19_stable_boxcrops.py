"""x19 - STABILISED face-box crops for the pretrained experts of reader_v4 (BTC dev, native). Motivation: open-rppg's own front end
(rppg/main.py: FaceDetector every 5 frames -> 1-D Kalman filter per box coordinate (process noise 0.01, measurement noise 0.5) ->
the box is only moved when its centre has shifted by more than 2 % (its own formula)) feeds the models a box that is FIXED most of the
time. x10 instead cuts the tight box of the team's oval anew in every frame, so the expert input jitters by 1-2 px per frame, i.e.
artificial motion. Here the team's per-frame native boxes (meta.npz) go through the same Kalman + hysteresis rule:
  stab    : stabilised team box                       stabsq : stabilised box made square (side = 1.1 x max(w, h)), same centre
  stabsqNN: square with side NN/10 x max(w, h) (e.g. stabsq14)      sqNN: the same square, per-frame (no stabilisation)
72x72 INTER_AREA crops, same layout as x10/x15 (crop72, t, valid; valid = the native flag), so y11 'run <model> <cond>' reads them.
Usage: nice -n 19 $AV code/x19_stable_boxcrops.py <workers> <variant,...> [keys|-]   -> out/x15/<variant>/<key>.npz, logs/x19.jsonl"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x01_extract as X
import json, time, traceback, resource
from pathlib import Path
import numpy as np, cv2
W = X.W; OUT = W / 'out/x15'
class K1:
    def __init__(self, x0): self.e, self.p = float(x0), 1.
    def update(self, m, dt, q=.01, r=.5):
        pe = self.p + q * (dt * 30.) ** 2; g = pe / (pe + r); self.e = self.e + g * (m - self.e); self.p = (1 - g) * pe; return self.e
def stabilise(boxes, valid, t):
    """boxes [n, 4] = x0, y0, x1, y1 -> stabilised boxes with open-rppg's Kalman + hysteresis rule (box held through invalid frames)."""
    out = np.zeros_like(boxes, dtype=float); kf = None; B = None; tl = None
    for i in range(len(boxes)):
        if valid[i]:
            b = boxes[i].astype(float); dt = 1 / 30. if tl is None else max(1e-3, t[i] - tl); tl = t[i]
            if kf is None: kf = [K1(v) for v in b]; kb = b.copy()
            else: kb = np.array([round(k.update(v, dt)) for k, v in zip(kf, b)], float)
            # open-rppg: box = ((y0, y1), (x0, x1)); centre shift / mean(box, axis=0)
            kb2 = np.array([[kb[1], kb[3]], [kb[0], kb[2]]]); 
            if B is None or np.max(np.abs(kb2.mean(1) - B.mean(1)) / B.mean(0)) > .02: B = kb2
        if B is not None: out[i] = [B[1, 0], B[0, 0], B[1, 1], B[0, 1]]
    return out
def run(a):
    j, variants = a; key = j['key']; t0 = time.time()
    if all((OUT / v / (key + '.npz')).exists() for v in variants): return dict(key=key, state='skip')
    held = False
    try:
        with np.load(Path(j['crop']) / 'meta.npz') as z: t, boxes, valid = z['t'], z['boxes'].astype(float), z['valid'].astype(bool)
        valid = valid & (boxes[:, 2] > boxes[:, 0])
        vid = X.BTCVideo(j['source'], t)
        if vid.w * vid.h > 4e6 or vid.hdr: held = X.heavy_slot(int(os.environ.get('X01_HEAVY', '2')))
        S = stabilise(boxes, valid, t); BX = {}
        for v in variants:
            b = S.copy() if v.startswith('stab') else boxes.copy()          # 'sqNN': per-frame (NOT stabilised) square box, for comparison
            if v.startswith('stabsq') or v.startswith('sq'):
                f_ = 1.1 if v == 'stabsq' else int(v[6:] if v.startswith('stabsq') else v[2:]) / 10.
                cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2; s = f_ * np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) / 2; b = np.stack([cx - s, cy - s, cx + s, cy + s], 1)
            BX[v] = b
        C = {v: np.zeros((len(t), 72, 72, 3), np.uint8) for v in variants}; n = 0; moves = 0
        for rgb in vid:
            H, Wd = rgb.shape[:2]
            for v in variants:
                x0, y0, x1, y1 = BX[v][n]; x0, y0 = max(0, int(round(x0))), max(0, int(round(y0))); x1, y1 = min(Wd, int(round(x1))), min(H, int(round(y1)))
                if valid[n] and x1 - x0 >= 8 and y1 - y0 >= 8: C[v][n] = cv2.resize(rgb[y0:y1, x0:x1], (72, 72), interpolation=cv2.INTER_AREA)
            n += 1
        assert n == len(t), f'{n} frames, {len(t)} time stamps'
        for v in variants:
            (OUT / v).mkdir(parents=True, exist_ok=True); tmp = OUT / v / (key + '.tmp.npz'); np.savez_compressed(tmp, crop72=C[v], t=t, valid=valid); tmp.replace(OUT / v / (key + '.npz'))
        d = np.abs(np.diff(S[valid], axis=0)).max(1) if valid.sum() > 2 else np.zeros(1); r = np.abs(np.diff(boxes[valid], axis=0)).max(1) if valid.sum() > 2 else np.zeros(1)
        return dict(key=key, state='done', frames=n, box_moves_share=round(float((d > 0).mean()), 4), raw_box_change_share=round(float((r > 0).mean()), 4), raw_median_change_px=float(np.median(r)), seconds=round(time.time() - t0, 1), maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
    except Exception as e:
        return dict(key=key, state='error', error=repr(e), traceback=traceback.format_exc()[-1500:])
    finally:
        if held: held.close()
if __name__ == '__main__':
    from multiprocessing import Pool
    workers = int(sys.argv[1]); variants = sys.argv[2].split(','); jobs = [j for j in json.loads((X.R / 'MANIFEST.json').read_text()) if j['dataset'] == 'BTC']
    if len(sys.argv) > 3 and sys.argv[3] != '-': jobs = [j for j in jobs if j['key'] in sys.argv[3].split(',')]
    log = (W / 'logs/x19.jsonl').open('a')
    with Pool(workers) as pool:
        for k, info in enumerate(pool.imap_unordered(run, [(j, variants) for j in jobs])):
            log.write(json.dumps(info) + '\n'); log.flush(); print(k + 1, len(jobs), {a: b for a, b in info.items() if a != 'traceback'}, flush=True)
