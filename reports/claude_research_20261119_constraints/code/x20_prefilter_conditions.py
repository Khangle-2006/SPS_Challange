"""x20 - 0.1 s pre-filter (x13 'nat_ma3') of the team CNN input UNDER THE PUBLIC CONSTRAINT CONDITIONS, so that the public reader P4
(y13: CNN, POS, CHROM + open-rppg experts, pooled fit) can be re-fitted with the pre-filtered CNN channel (z03, native only: experts +
pre-filter MCD 1.906 -> 1.629, UBFC2 0.754 -> 0.735).
Same regeneration as x16 (same RNG streams = index in X.conds(dataset); noise on the whole frame; same polygons, vitiligo patches and
front ends as x01 / x09). For every frame the masked 36x36 oval crop that the team CNN reads (X.fast_crop on the degraded frame) is kept,
then a centred moving average over k = ceil(fps/10) crops weighted by the condition's own valid flags (exactly x13 'nat_ma3'); block
means / fill from the filtered crops (x13 stats). -> out/x20/<name>/<key>.npz (t, valid, blk, fill, box) + out/x20c/<name>/<key>.npz
(crop float16), consumed (and the crops deleted) by: $PY code/x02_infer.py <slot> <nslots> x20
Check written to the log: UNFILTERED block means of the first 60 frames vs the blocks stored by x09 / x01, and the valid flags.
Usage: nice -n 19 $AV code/x20_prefilter_conditions.py <workers> <fe:base,...> [keys|UBFC2|MCD|-]   -> log logs/x20_prefilter.jsonl"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x01_extract as X
import x09_followup as F
import json, time, traceback, zlib, resource
from pathlib import Path
import numpy as np, cv2
co = X.co; W = X.W; OUT = W / 'out/x20'; CROPS = W / 'out/x20c'; C1 = F.C1; C7 = F.C7; C4 = dict(c4_qvga=(320, 240))
def stored(name, key):
    for tag in ('x09', 'x01'):
        p = W / 'out' / tag / name / (key + '.npz')
        if p.exists():
            with np.load(p) as z: return z['blk'][:60].astype(np.float32), z['valid'].astype(bool)
    return None, None
def ma3(crop, t, valid):
    """x13 'nat_ma3': centred moving average over k crops, invalid frames weight 0."""
    N = len(t); fps = (N - 1) / (t[-1] - t[0]); k = max(1, int(np.ceil(fps / 10 - 1e-3))); w = valid.astype(np.float32); pad = k // 2
    cs = np.cumsum(np.concatenate([np.zeros((1, 36, 36, 3), np.float64), crop.astype(np.float64) * w[:, None, None, None]]), 0); cw = np.concatenate([[0.], np.cumsum(w)])
    lo = np.clip(np.arange(N) - pad, 0, N); hi = np.clip(np.arange(N) - pad + k, 0, N)
    return ((cs[hi] - cs[lo]) / np.maximum(cw[hi] - cw[lo], 1)[:, None, None, None]).astype(np.float32), k
def stats(c):
    n = len(c); return c.reshape(n, 6, 6, 6, 6, 3).mean((2, 4), dtype=np.float32), (c.max(-1) > 0).reshape(n, 6, 6, 6, 6).sum((2, 4)).astype(np.uint8)
def run(job):
    idx, j, jobs = job; key, ds = j['key'], j['dataset']; names = [F.name_of(x) for x in jobs]; t0 = time.time()
    if all((OUT / c / (key + '.npz')).exists() for c in names): return dict(key=key, state='skip')
    try:
        with np.load(j['meta']) as z: t = z['original_t']
        src = X.frames_of(j, len(t))
        allc = X.conds(ds); seed = zlib.crc32(key.encode()); bases = sorted({x.split(':')[1] for x in jobs}); c1b = [b for b in bases if b in C1]
        rng = {b: np.random.default_rng([20261119, seed, allc.index(b)]) for b in c1b}; need_lm = any(b in C7 or b == 'c2_vit' for b in bases)
        mesh = {n_: X.Mesh() for n_ in names}; nat = X.Mesh() if need_lm else None
        A = {c: dict(valid=[], crop=[], blk=[], held=0) for c in names}; patches = None; vr = np.random.default_rng([20261119, seed, 99]); poly = {}
        last = {c: dict(pts=None, t=-1e9, tpl=None) for c in names}; lm = None; n = 0
        for rgb in src:
            rgb = np.ascontiguousarray(rgb); H, Wd = rgb.shape[:2]; tn = float(t[n])
            if need_lm:
                lm = nat.step(rgb, full=True)
                if lm is not None and patches is None: cen = vr.choice(468, 8, replace=False); patches = [np.argsort(((lm - lm[c]) ** 2).sum(1))[:25] for c in cen]
            fr = {}; usable = {}
            for b in bases:
                usable[b] = True
                if b == 'native': fr[b] = rgb
                elif b in C1:
                    _, g, kind, var = C1[b]; out = cv2.LUT(rgb, X.LUTS[g]); fr[b] = X.pois(out, rng[b]) if kind == 'p' else X.gauss(out, var, rng[b])
                elif b in C4: fr[b] = np.ascontiguousarray(co.c4_resize(rgb, co.c4_size_keep_aspect(Wd, H, C4[b])))
                elif b == 'c6_us1': fr[b] = cv2.filter2D(rgb, -1, X.KUS, borderType=cv2.BORDER_REPLICATE)
                elif b in C7:
                    _, colour, fit = C7[b]
                    if lm is not None: poly[fit] = np.round(co.c7_polygon(lm, fit)).astype(np.int32)
                    out = rgb.copy(); usable[b] = lm is not None
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
                good, crop, box = X.fast_crop(f, pts); a_ = A[c]; a_['valid'].append(bool(good) and usable[b]); a_['crop'].append(crop)
                if n < 60: a_['blk'].append(crop.reshape(6, 6, 6, 6, 3).mean((1, 3), dtype=np.float32))
            n += 1
        for m in list(mesh.values()) + ([nat] if nat else []): m.close()
        assert n == len(t), f'{n} frames, {len(t)} time stamps'; info = dict(key=key, dataset=ds, frames=n)
        for c in names:
            a = A[c]; v = np.array(a['valid']); cr, k = ma3(np.array(a['crop']), t, v); blk, fill = stats(cr); a['crop'] = None
            for dd in (OUT / c, CROPS / c): dd.mkdir(parents=True, exist_ok=True)
            tmp = CROPS / c / (key + '.tmp.npz'); np.savez_compressed(tmp, crop=cr.astype(np.float16)); tmp.replace(CROPS / c / (key + '.npz'))
            tmp = OUT / c / (key + '.tmp.npz'); np.savez_compressed(tmp, t=t, valid=v, blk=blk, fill=fill, box=np.zeros((n, 4), np.int16)); tmp.replace(OUT / c / (key + '.npz'))
            rb, rv = stored(c, key); info['valid_' + c] = round(float(v.mean()), 4); info['held_' + c] = int(a['held']); info['k'] = k
            if rb is not None:
                kk = min(len(rb), len(a['blk'])); info['max_abs_diff_vs_stored_' + c] = float(np.abs(np.array(a['blk'][:kk]) - rb[:kk]).max()) if kk else None
                info['valid_mismatch_' + c] = int((rv != v).sum()) if len(rv) == len(v) else -1
            else: info['no_stored_' + c] = True
        info.update(seconds=round(time.time() - t0, 1), maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, state='done'); return info
    except Exception as e:
        return dict(key=key, state='error', error=repr(e), traceback=traceback.format_exc()[-1500:])
if __name__ == '__main__':
    import multiprocessing as mp
    workers = int(sys.argv[1]); cs = sys.argv[2].split(',')
    have = {p.stem for p in (W / 'out/x01/c4_qvga').glob('*.npz')}
    jobs = [j for j in json.loads((X.R / 'MANIFEST.json').read_text()) if j['dataset'] in ('UBFC2', 'MCD') and j['key'] in have]
    if len(sys.argv) > 3 and sys.argv[3] != '-': jobs = [j for j in jobs if j['key'] in sys.argv[3].split(',') or j['dataset'] in sys.argv[3].split(',')]
    print('recordings', len(jobs), {d: sum(j['dataset'] == d for j in jobs) for d in ('UBFC2', 'MCD')}, 'conditions', [F.name_of(x) for x in cs], flush=True)
    # longest first (MCD) so that the pool ends together
    order = sorted(range(len(jobs)), key=lambda i: (jobs[i]['dataset'] != 'MCD', i)); X.PT = X.poisson_table(); X.GT.update({v: X.gauss_table(v) for v in (.001, .01)})
    log = (W / 'logs/x20_prefilter.jsonl').open('a'); done = 0
    with mp.Pool(workers, maxtasksperchild=1) as pool:
        for info in pool.imap_unordered(run, [(int(i), jobs[i], cs) for i in order]):
            done += 1; log.write(json.dumps(info) + '\n'); log.flush(); print(done, len(jobs), {k: v for k, v in info.items() if k != 'traceback'}, flush=True)
    (W / 'out/x20_DONE').write_text(time.strftime('%F %T'))
