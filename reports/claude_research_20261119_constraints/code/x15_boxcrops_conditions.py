"""x15 - unmasked face-box crops (72x72) of BTC dev under the constraint conditions, for the pretrained third-party models (reader_v4).
Same regeneration as x01/x09 (same RNG streams, polygons, front ends); for every frame the tight box of the oval that the front end
found ON THE DEGRADED FRAME is cut from that frame and area-resized to 72x72 (x10 does the same for native with the stored oval).
A job is <fe>:<base> as in x09 (fe std|track|aid|aidtrack|hold|aidhold); extra bases here: c4_vga, c4_qvga (frame reduced first).
Check: the 36x36 block means of the regenerated crop are compared with the blocks stored by x09 / x01 for the same condition
(max_abs_diff_vs_stored, first 60 frames) - the box crops then belong to exactly the frames whose evidence was scored.
Usage: nice -n 19 $AV code/x15_boxcrops_conditions.py <workers> <fe:base,...> [keys|-] [first_core]  -> out/x15/<name>/<key>.npz (crop72, t, valid)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x01_extract as X
import x09_followup as F
import json, time, traceback, zlib, resource
from pathlib import Path
import numpy as np, cv2
co = X.co; W = X.W; OUT = W / 'out/x15'; C1 = F.C1; C7 = F.C7; C4 = dict(c4_vga=(640, 480), c4_qvga=(320, 240))
def stored(name, key):
    for tag in ('x09', 'x01'):
        p = W / 'out' / tag / name / (key + '.npz')
        if p.exists():
            with np.load(p) as z: return z['blk'][:60].astype(np.float32), z['valid'].astype(bool)
    return None, None
def run(job):
    idx, j, jobs = job; key = j['key']; names = [F.name_of(x) for x in jobs]; t0 = time.time()
    if all((OUT / c / (key + '.npz')).exists() for c in names): return dict(key=key, state='skip')
    if os.environ.get('X01_SKIP_HEAVY', '1') == '1' and X.is_heavy(j): return dict(key=key, state='not_run_heavy')
    held_sem = False
    try:
        with np.load(Path(j['crop']) / 'meta.npz') as z: boxes, ref_t = z['boxes'], z['t']
        vid = X.BTCVideo(j['source'], ref_t); t = vid.t; ok = boxes[:, 2] > boxes[:, 0]; nb = None
        if vid.w * vid.h > 4e6 or vid.hdr: held_sem = X.heavy_slot(int(os.environ.get('X01_HEAVY', '4')))
        if ok.any():
            near = np.flatnonzero(ok)[np.abs(np.arange(len(ok))[:, None] - np.flatnonzero(ok)[None]).argmin(1)]; b = boxes[near].astype(float); bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
            nb = np.stack([b[:, 0] - .5 * bw, b[:, 1] - .5 * bh, b[:, 2] + .5 * bw, b[:, 3] + .5 * bh], 1)
        allc = X.conds('BTC'); seed = zlib.crc32(key.encode()); bases = sorted({x.split(':')[1] for x in jobs}); c1b = [b for b in bases if b in C1]
        rng = {b: np.random.default_rng([20261119, seed, allc.index(b)]) for b in c1b}; need_lm = any(b in C7 or b == 'c2_vit' for b in bases)
        mesh = {n_: X.Mesh() for n_ in names}; nat = X.Mesh() if need_lm else None
        A = {c: dict(valid=[], crop=[], blk=[], held=0) for c in names}; patches = None; vr = np.random.default_rng([20261119, seed, 99]); poly = {}
        last = {c: dict(pts=None, t=-1e9, tpl=None) for c in names}; lm = None; n = 0; Z72 = np.zeros((72, 72, 3), np.uint8)
        for rgb in vid:
            rgb = np.ascontiguousarray(rgb); H, Wd = rgb.shape[:2]; tn = float(t[n])
            if nb is None: x0, y0, x1, y1 = 0, 0, Wd, H
            else: x0, y0 = max(0, int(nb[n, 0])), max(0, int(nb[n, 1])); x1, y1 = min(Wd, int(nb[n, 2]) + 1), min(H, int(nb[n, 3]) + 1)
            if need_lm:
                lm = nat.step(rgb, full=True)
                if lm is not None and patches is None: cen = vr.choice(468, 8, replace=False); patches = [np.argsort(((lm - lm[c]) ** 2).sum(1))[:25] for c in cen]
            fr = {}; usable = {}
            for b in bases:
                usable[b] = True
                if b == 'native': fr[b] = rgb
                elif b in C1:
                    _, g, kind, var = C1[b]; out = cv2.LUT(rgb, X.LUTS[g]); roi = out[y0:y1, x0:x1]; out[y0:y1, x0:x1] = X.pois(roi, rng[b]) if kind == 'p' else X.gauss(roi, var, rng[b]); fr[b] = out
                elif b in C4: fr[b] = np.ascontiguousarray(co.c4_resize(rgb, co.c4_size_keep_aspect(Wd, H, C4[b])))
                elif b == 'c6_us1': fr[b] = cv2.filter2D(rgb, -1, X.KUS, borderType=cv2.BORDER_REPLICATE)
                elif b in C7:
                    _, colour, fit = C7[b]
                    if lm is not None: poly[fit] = np.round(co.c7_polygon(lm, fit)).astype(np.int32)
                    out = rgb.copy(); usable[b] = fit in poly
                    if fit in poly: cv2.fillPoly(out, [poly[fit]], co.MASK_RGB[colour])
                    fr[b] = out
                elif b == 'c2_vit':
                    usable[b] = lm is not None and patches is not None; fr[b] = F.vit_apply(rgb, lm, patches) if usable[b] else rgb
                else: raise ValueError('unknown base ' + b)
            for job_, c in zip(jobs, names):
                fe, b = job_.split(':'); f = fr[b]; fh, fw = f.shape[:2]
                det = F.aid(f) if 'aid' in fe else f; p = mesh[c].step(det); S = last[c]
                if p is not None:
                    S['pts'], S['t'] = p, tn
                    if 'track' in fe: S['tpl'] = F.tpl_make(det, p)
                elif 'hold' in fe and S['pts'] is not None and tn - S['t'] <= F.HOLD_S: p = S['pts']; A[c]['held'] += 1
                elif 'track' in fe and S.get('tpl') is not None:
                    d = F.tpl_find(det, S['tpl'])
                    if d is not None: p = S['pts'] + d; A[c]['held'] += 1
                pts = None if p is None else p * np.array([fw / det.shape[1], fh / det.shape[0]], np.float32)
                good, crop, box = X.fast_crop(f, pts); v = bool(good) and usable[b] and (box[2] - box[0]) >= 8 and (box[3] - box[1]) >= 8; a_ = A[c]; a_['valid'].append(v)
                a_['crop'].append(cv2.resize(f[box[1]:box[3], box[0]:box[2]], (72, 72), interpolation=cv2.INTER_AREA) if v else Z72)
                if n < 60: a_['blk'].append(crop.reshape(6, 6, 6, 6, 3).mean((1, 3), dtype=np.float32))
            n += 1
        for m in list(mesh.values()) + ([nat] if nat else []): m.close()
        assert n == len(t); info = dict(key=key, frames=n)
        for c in names:
            a = A[c]; (OUT / c).mkdir(parents=True, exist_ok=True); tmp = OUT / c / (key + '.tmp.npz'); v = np.array(a['valid'])
            np.savez_compressed(tmp, crop72=np.array(a['crop'], np.uint8), t=t, valid=v); tmp.replace(OUT / c / (key + '.npz'))
            rb, rv = stored(c, key); info['valid_' + c] = round(float(v.mean()), 4); info['held_' + c] = int(a['held'])
            if rb is not None:
                k = min(len(rb), len(a['blk'])); info['max_abs_diff_vs_stored_' + c] = float(np.abs(np.array(a['blk'][:k]) - rb[:k]).max()) if k else None
                info['valid_mismatch_' + c] = int((rv != v).sum()) if len(rv) == len(v) else -1
        info.update(seconds=round(time.time() - t0, 1), maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, state='done'); return info
    except Exception as e:
        return dict(key=key, state='error', error=repr(e), traceback=traceback.format_exc()[-1500:])
    finally:
        if held_sem: held_sem.close()
if __name__ == '__main__':
    import multiprocessing as mp
    workers = int(sys.argv[1]); cs = sys.argv[2].split(','); jobs = [j for j in json.loads((X.R / 'MANIFEST.json').read_text()) if j['dataset'] == 'BTC']
    if len(sys.argv) > 3 and sys.argv[3] != '-': jobs = [j for j in jobs if j['key'] in sys.argv[3].split(',')]
    first = int(sys.argv[4]) if len(sys.argv) > 4 else None
    order = np.random.default_rng(2).permutation(len(jobs)); X.PT = X.poisson_table(); X.GT.update({v: X.gauss_table(v) for v in (.001, .01)}); log = (W / 'logs/x15_boxcrops.jsonl').open('a'); done = 0
    with (mp.Pool(workers, initializer=X.pin_worker, initargs=(first, workers)) if first is not None else mp.Pool(workers)) as pool:
        for info in pool.imap_unordered(run, [(int(i), jobs[i], cs) for i in order]):
            done += 1; log.write(json.dumps(info) + '\n'); log.flush(); print(done, len(jobs), {k: v for k, v in info.items() if k != 'traceback'}, flush=True)
