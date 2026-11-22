"""x18 - front-end upgrade test for the conditions where the face detector is unstable (C1 strong noise, very low gain): CAUSAL
exponential smoothing of the oval landmarks before cropping (p_s(t) = a p(t) + (1 - a) p_s(t-1); a = 1 is the current front end).
Same regeneration as x01/x09 (same RNG streams; BTC noise in the enlarged native box, public sets on the whole frame); detection
as the 'aidtrack' front end of x09 (blurred, gain-normalised copy for FaceMesh + template hold). Writes the same per-frame traces as
x01/x09 (6x6 block means of the masked 36x36 crop, fill, valid), so x03 (BTC) and z04 (public) score them unchanged.
Also records frame-to-frame jitter of the oval centre (px, relative to the box width).
Usage: nice -n 19 $AV code/x18_roi_smooth.py <workers> <base,...> <alpha,...> [keys|UBFC2|MCD|BTC|-] [first_core]
       -> out/x18/<base>_a<alpha*100>/<key>.npz ; log logs/x18_roi_smooth.jsonl"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x01_extract as X
import x09_followup as F
import json, time, traceback, zlib, resource
from pathlib import Path
import numpy as np, cv2
co = X.co; W = X.W; OUT = W / 'out/x18'; C1 = F.C1
def name(b, a): return f'{b}_a{int(round(a * 100)):03d}'
def run(job):
    idx, j, bases, alphas = job; key, ds = j['key'], j['dataset']; names = [(b, a, name(b, a)) for b in bases for a in alphas]; t0 = time.time()
    if all((OUT / n / (key + '.npz')).exists() for _, _, n in names): return dict(key=key, state='skip')
    if ds == 'BTC' and os.environ.get('X01_SKIP_HEAVY', '1') == '1' and X.is_heavy(j): return dict(key=key, state='not_run_heavy')
    try:
        nb = None
        if ds == 'BTC':
            with np.load(Path(j['crop']) / 'meta.npz') as z: boxes, ref_t = z['boxes'], z['t']
            vid = X.BTCVideo(j['source'], ref_t); t = vid.t; src = iter(vid); ok = boxes[:, 2] > boxes[:, 0]
            if ok.any():
                near = np.flatnonzero(ok)[np.abs(np.arange(len(ok))[:, None] - np.flatnonzero(ok)[None]).argmin(1)]; b = boxes[near].astype(float); bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
                nb = np.stack([b[:, 0] - .5 * bw, b[:, 1] - .5 * bh, b[:, 2] + .5 * bw, b[:, 3] + .5 * bh], 1)
        else:
            with np.load(j['meta']) as z: t = z['original_t']
            src = X.frames_of(j, len(t))
        allc = X.conds(ds); seed = zlib.crc32(key.encode()); rng = {b: np.random.default_rng([20261119, seed, allc.index(b)]) for b in bases if b in C1}
        mesh = {b: X.Mesh() for b in bases}; last = {b: dict(pts=None, tpl=None) for b in bases}; ema = {n: None for _, _, n in names}
        A = {n: dict(valid=[], blk=[], fill=[], cen=[]) for _, _, n in names}; n_ = 0
        for rgb in src:
            rgb = np.ascontiguousarray(rgb); H, Wd = rgb.shape[:2]
            if nb is None: x0, y0, x1, y1 = 0, 0, Wd, H
            else: x0, y0 = max(0, int(nb[n_, 0])), max(0, int(nb[n_, 1])); x1, y1 = min(Wd, int(nb[n_, 2]) + 1), min(H, int(nb[n_, 3]) + 1)
            for b in bases:
                if b == 'native': f = rgb
                else:
                    _, g, kind, var = C1[b]; f = cv2.LUT(rgb, X.LUTS[g]); roi = f[y0:y1, x0:x1]; f[y0:y1, x0:x1] = X.pois(roi, rng[b]) if kind == 'p' else X.gauss(roi, var, rng[b])
                det = F.aid(f); p = mesh[b].step(det); S = last[b]
                if p is not None: S['pts'] = p; S['tpl'] = F.tpl_make(det, p)
                elif S.get('tpl') is not None:
                    d = F.tpl_find(det, S['tpl'])
                    if d is not None: p = S['pts'] + d
                pts = None if p is None else p * np.array([f.shape[1] / det.shape[1], f.shape[0] / det.shape[0]], np.float32)
                for bb, a, n in names:
                    if bb != b: continue
                    if pts is not None: ema[n] = pts.copy() if (ema[n] is None or a >= 1) else (a * pts + (1 - a) * ema[n]).astype(np.float32)
                    q = ema[n] if pts is not None else None
                    good, crop, box = X.fast_crop(f, q); v = bool(good) and (box[2] - box[0]) >= 8 and (box[3] - box[1]) >= 8; A[n]['valid'].append(v)
                    A[n]['blk'].append(crop.reshape(6, 6, 6, 6, 3).mean((1, 3), dtype=np.float32)); A[n]['fill'].append((crop.max(-1) > 0).reshape(6, 6, 6, 6).sum((1, 3)).astype(np.uint8))
                    A[n]['cen'].append(q.mean(0) / max(1., float(q[:, 0].max() - q[:, 0].min())) if q is not None else np.full(2, np.nan))
            n_ += 1
        for m in mesh.values(): m.close()
        assert n_ == len(t); info = dict(key=key, dataset=ds, frames=n_)
        for _, _, n in names:
            a = A[n]; (OUT / n).mkdir(parents=True, exist_ok=True); tmp = OUT / n / (key + '.tmp.npz'); v = np.array(a['valid']); c = np.array(a['cen'])
            np.savez_compressed(tmp, t=t, valid=v, blk=np.array(a['blk']), fill=np.array(a['fill'])); tmp.replace(OUT / n / (key + '.npz'))
            jit = np.nanmedian(np.linalg.norm(np.diff(c, axis=0), axis=1)) if len(c) > 2 else np.nan
            info['valid_' + n] = round(float(v.mean()), 4); info['jitter_' + n] = None if not np.isfinite(jit) else round(float(jit), 5)
        info.update(seconds=round(time.time() - t0, 1), maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, state='done'); return info
    except Exception as e:
        return dict(key=key, state='error', error=repr(e), traceback=traceback.format_exc()[-1500:])
if __name__ == '__main__':
    import multiprocessing as mp
    workers = int(sys.argv[1]); bases = sys.argv[2].split(','); alphas = [float(x) for x in sys.argv[3].split(',')]
    jobs = json.loads((X.R / 'MANIFEST.json').read_text()); have = {p.stem for p in (W / 'out/x01/c4_qvga').glob('*.npz')}
    jobs = [j for j in jobs if j['key'] in have]
    if len(sys.argv) > 4 and sys.argv[4] != '-': jobs = [j for j in jobs if j['key'] in sys.argv[4].split(',') or j['dataset'] in sys.argv[4].split(',')]
    first = int(sys.argv[5]) if len(sys.argv) > 5 else None
    print('jobs', len(jobs), flush=True); X.PT = X.poisson_table(); X.GT.update({v: X.gauss_table(v) for v in (.001, .01)}); log = (W / 'logs/x18_roi_smooth.jsonl').open('a'); done = 0
    order = np.random.default_rng(3).permutation(len(jobs))
    with (mp.Pool(workers, initializer=X.pin_worker, initargs=(first, workers)) if first is not None else mp.Pool(workers)) as pool:
        for info in pool.imap_unordered(run, [(int(i), jobs[i], bases, alphas) for i in order]):
            done += 1; log.write(json.dumps(info) + '\n'); log.flush(); print(done, len(jobs), {k: v for k, v in info.items() if k != 'traceback'}, flush=True)
