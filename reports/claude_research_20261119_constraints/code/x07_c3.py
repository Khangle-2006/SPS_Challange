"""x07 - C3 multi-face: a second person (another UBFC2 dev subject) is composited into the target's frame, then three ways of choosing
the face are compared on the SAME composite stream (PREREG.md):
  first   : the unchanged pipeline (FaceMesh max_num_faces=1 - whatever face it returns)
  largest : full-range face detector on the whole frame, largest box in every frame, FaceMesh on that face's region
  lock    : full-range face detector on the whole frame -> target_lock.TargetLock (Kalman + Hungarian assignment + appearance
            signature), target designated by a seed box = the target's own first valid face box -> FaceMesh runs only on the locked
            face's region (box enlarged, with hysteresis); frames are invalid while the target is coasting/lost or overlapped.
            (First version used multi-face FaceMesh for detection: its short-range detector missed the smaller face on the wide
            canvas of c3_big - 1.01 faces per frame on subject1 - so the lock never engaged; logs/x07_smoke.log.)
Layouts: c3_side (bystander on the left, same size) and c3_big (bystander 1.3x larger). Ground truth for 'wrong person': centre of the
chosen box lies in the bystander's part of the canvas. Output format as x01 (cond = <layout>_<method>), so x02/x05 score it unchanged.
Usage: nice -n 19 $AV code/x07_c3.py <workers> [keys]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x01_extract as X
import json, time, traceback, resource
from pathlib import Path
import numpy as np, cv2
from target_lock import TargetLock
co = X.co; W = X.W; OUT = W / 'out/x07'; CROPS = W / 'out/x07c'; LAY = {'c3_side': 1.0, 'c3_big': 1.3}
ONLY0 = os.environ.get('X07_ONLY') == 'lock0'      # 2nd pass (20 Nov): lock engaged at the first frame from the seed box (no 0.7 s vote), stored as <layout>_lock0
METHODS = ('lock0',) if ONLY0 else ('first', 'largest', 'lock'); LK = 'lock0' if ONLY0 else 'lock'
class Det:
    """MediaPipe FaceDetection, full-range model (faces up to ~5 m): boxes (x0, y0, x1, y1) in pixels for every face in the frame."""
    def __init__(self):
        import mediapipe as mp
        self.d = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=.5)
    def step(self, image):
        h, w = image.shape[:2]; r = self.d.process(image); out = []
        for d in (r.detections or []):
            b = d.location_data.relative_bounding_box; out.append(np.array([b.xmin * w, b.ymin * h, (b.xmin + b.width) * w, (b.ymin + b.height) * h]))
        return out
    def close(self): self.d.close()
def region_points(image, box, mesh, st):
    """FaceMesh on the region around one detection box (enlarged 50% sideways, 60% up, 40% down). The region is kept fixed until the box
    centre moves by >15% of its width or its size changes by >20%, so that the mesh tracker sees a stable image."""
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; r = st.get('reg')
    if r is None or abs(cx - st['c'][0]) > .15 * st['s'][0] or abs(cy - st['c'][1]) > .15 * st['s'][1] or abs(w / st['s'][0] - 1) > .2:
        H, W_ = image.shape[:2]; r = (int(max(0, x0 - .5 * w)), int(max(0, y0 - .6 * h)), int(min(W_, x1 + .5 * w)), int(min(H, y1 + .4 * h))); st.update(reg=r, c=(cx, cy), s=(w, h))
    if r[2] - r[0] < 16 or r[3] - r[1] < 16: return None
    pts = mesh.step(np.ascontiguousarray(image[r[1]:r[3], r[0]:r[2]]))
    return None if pts is None else pts + np.array([r[0], r[1]], np.float32)
def signature(gray, box):
    """12x12 mean-removed luminance thumbnail of the face box (cosine distance = 1 - correlation): a cheap appearance cue, not a face-ID model."""
    x0, y0, x1, y1 = [int(v) for v in box]; x0, y0 = max(x0, 0), max(y0, 0)
    if x1 - x0 < 8 or y1 - y0 < 8: return None
    th = cv2.resize(gray[y0:y1, x0:x1], (12, 12), interpolation=cv2.INTER_AREA).astype(np.float64).ravel(); th -= th.mean(); n = np.linalg.norm(th)
    return th / n if n > 0 else None
def cyc(j, n_frames):
    while True:
        k = 0
        for fr in X.frames_of(j, n_frames): k += 1; yield fr
        if k == 0: return
def compose(tg, by, scale):
    if scale == 1.: return np.concatenate([by, tg], 1), by.shape[1], 0
    bh, bw = int(round(by.shape[0] * scale / 2)) * 2, int(round(by.shape[1] * scale / 2)) * 2; c = np.zeros((bh, bw + tg.shape[1], 3), np.uint8)
    c[:, :bw] = co.c4_resize(by, (bw, bh)); y = (bh - tg.shape[0]) // 2; c[y:y + tg.shape[0], bw:] = tg; return c, bw, y
def run(job):
    j, jb = job; key = j['key']; conds = [f'{l}_{m}' for l in LAY for m in METHODS]; t0 = time.time()
    if all((OUT / c / (key + '.npz')).exists() for c in conds): return dict(key=key, state='skip')
    try:
        with np.load(j['meta']) as z: t, boxes = z['original_t'], z['boxes']
        with np.load(jb['meta']) as z: nb = len(z['original_t'])
        ok = np.flatnonzero(boxes[:, 2] > boxes[:, 0]); seed = boxes[ok[0]].astype(float); by = cyc(jb, nb)
        A = {c: dict(valid=[], blk=[], fill=[], box=[], crop=[], wrong=0) for c in conds}; M1 = {l: X.Mesh() for l in LAY} if 'first' in METHODS else {}; DT = {l: Det() for l in LAY}; ML = {(l, m): X.Mesh() for l in LAY for m in ('largest', 'lock') if m == 'lock' or 'largest' in METHODS}; RS = {k: {} for k in ML}; TL = {}; nfaces = {l: [] for l in LAY}; n = 0
        def put(c, image, pts, valid_flag, xoff):
            good, crop, box = X.fast_crop(image, pts); v = bool(good) and valid_flag; a = A[c]
            a['valid'].append(v); a['box'].append(box); a['crop'].append(crop); a['blk'].append(crop.reshape(6, 6, 6, 6, 3).mean((1, 3), dtype=np.float32)); a['fill'].append((crop.max(-1) > 0).reshape(6, 6, 6, 6).sum((1, 3)).astype(np.uint8))
            if v and (box[0] + box[2]) / 2 < xoff: a['wrong'] += 1
        for tg in X.frames_of(j, len(t)):
            bf = next(by)
            for l, sc in LAY.items():
                img, xoff, yoff = compose(tg, bf, sc); img = np.ascontiguousarray(img)
                if l not in TL: TL[l] = TargetLock(frame_size=(img.shape[1], img.shape[0]), seed_box=tuple(seed + np.array([xoff, yoff, xoff, yoff])), **(dict(warmup_s=0., min_hits=1) if ONLY0 else {}))
                if 'first' in METHODS: put(f'{l}_first', img, M1[l].step(img), True, xoff)
                bxs = DT[l].step(img); nfaces[l].append(len(bxs))
                if 'largest' in METHODS:
                    if bxs:
                        k = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in bxs])); put(f'{l}_largest', img, region_points(img, bxs[k], ML[(l, 'largest')], RS[(l, 'largest')]), True, xoff)
                    else: put(f'{l}_largest', img, None, False, xoff)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY); r = TL[l].step(float(t[n]), [tuple(b) for b in bxs], [signature(gray, b) for b in bxs])
                if r['box'] is not None and bxs:
                    iou = [X_iou(r['box'], b) for b in bxs]; k = int(np.argmax(iou)); hit = iou[k] > .3
                    put(f'{l}_{LK}', img, region_points(img, bxs[k], ML[(l, 'lock')], RS[(l, 'lock')]) if hit else None, bool(r['valid']) and hit, xoff)
                else: put(f'{l}_{LK}', img, None, False, xoff)
            n += 1
        for m in list(M1.values()) + list(DT.values()) + list(ML.values()): m.close()
        assert n == len(t); info = dict(key=key, bystander=jb['key'], frames=n)
        for c in conds:
            a = A[c]; (OUT / c).mkdir(parents=True, exist_ok=True); (CROPS / c).mkdir(parents=True, exist_ok=True)
            tmp = OUT / c / (key + '.tmp.npz'); np.savez_compressed(tmp, t=t, valid=np.array(a['valid']), blk=np.array(a['blk']), fill=np.array(a['fill']), box=np.array(a['box'], np.int16)); tmp.replace(OUT / c / (key + '.npz'))
            tmp = CROPS / c / (key + '.tmp.npz'); np.savez_compressed(tmp, crop=np.array(a['crop'])); tmp.replace(CROPS / c / (key + '.npz'))
            info['valid_' + c] = float(np.mean(a['valid'])); info['wrong_' + c] = int(a['wrong'])
        for l in LAY: info['faces_mean_' + l] = float(np.mean(nfaces[l])); info['events_' + l] = [(round(float(a), 2), b) for a, b in TL[l].id_events][:12]
        info.update(seconds=round(time.time() - t0, 1), maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, state='done'); return info
    except Exception as e:
        return dict(key=key, state='error', error=repr(e), traceback=traceback.format_exc()[-1500:])
def X_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3]); inter = max(0., x2 - x1) * max(0., y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter; return inter / ua if ua > 0 else 0.
if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = sorted([j for j in json.loads((X.R / 'MANIFEST.json').read_text()) if j['dataset'] == 'UBFC2'], key=lambda j: j['key']); pairs = [(jobs[i], jobs[(i + 1) % len(jobs)]) for i in range(len(jobs))]
    if len(sys.argv) > 2: pairs = [p for p in pairs if p[0]['key'] in sys.argv[2].split(',')]
    log = (W / ('logs/x07_c3_lock0.jsonl' if ONLY0 else 'logs/x07_c3.jsonl')).open('a'); done = 0
    with Pool(int(sys.argv[1]), maxtasksperchild=1) as pool:
        for info in pool.imap_unordered(run, pairs):
            done += 1; log.write(json.dumps(info) + '\n'); log.flush(); print(done, len(pairs), {k: v for k, v in info.items() if k != 'traceback'}, flush=True)
    (W / 'out/x07_DONE').write_text(time.strftime('%F %T'))
