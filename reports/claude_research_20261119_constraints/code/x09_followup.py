"""x09 - follow-up after x01: extra condition (vitiligo) and FRONT-END changes tested on regenerated x01 conditions.
Same output layout as x01 (out/x09/<name>/<key>.npz, crops for the public sets in out/x09c), so x02/x03/x04/x05 score it with tag x09.
A job is written  <fe>:<base>  and stored under the name <fe>_<base> (fe 'std' -> just <base>):
  base  native | any C1 / C6 / C7 condition of x01 (regenerated bit-exactly: same RNG streams, same polygons) |
        c2_vit : C2 'conditions such as vitiligo that alter skin reflectance' - eight depigmented patches glued to the face (each = convex
                 hull of a random landmark and its 24 nearest landmarks, fixed per video), reflectance raised inside (R x1.25, G x1.5,
                 B x1.6, saturating) with a soft edge.
  fe    std     : the team's front end, unchanged
        hold    : ROI hold - when FaceMesh returns no face, the last valid oval is reused for at most 0.5 s (the faces barely move; the
                  window rule of the pipeline otherwise discards a whole 10 s window for a single missed frame)
        track   : template-verified hold, no time limit - on a FaceMesh miss the upper 55% of the last valid face box (forehead/eyes,
                  grey, ~64 px wide) is searched by normalised cross-correlation within +-15% of the box size around where it was
                  taken; if the score is >= 0.6 the last oval is shifted by the displacement found, else the frame is invalid
                  (classic detect-then-track front end, cf. KLT tracking in Li et al., CVPR 2014; needed because FaceMesh loses
                  masked faces for long runs: logs/interim2.py)
        aid     : low-light detector aid - FaceMesh receives a copy reduced to <=640 px, Gaussian-blurred (sigma 1.5 px) and brightened
                  (99th percentile -> ~180); the 36x36 crop is still taken from the degraded frame itself
        aidhold / aidtrack : aid combined with hold / track (the template is then taken on the aided image)
Regeneration is checked on the first 30 frames of every C1 base against the blocks stored by x01 (regen_max_abs_diff in the log).
Usage: nice -n 19 $AV code/x09_followup.py <workers> <fe:base,fe:base,...> [keys|datasets|-] [first_core]   (env X01_HEAVY = max concurrent 4K/HDR decodes, default 8)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x01_extract as X
import json, time, traceback, zlib, resource
from pathlib import Path
import numpy as np, cv2
co = X.co; W = X.W; OUT = W / 'out/x09'; CROPS = W / 'out/x09c'; C1 = {c[0]: c for c in X.C1}; C7 = {c[0]: c for c in X.C7}; VGAIN = np.array([1.25, 1.5, 1.6], np.float32); HOLD_S = .5
def name_of(job): fe, base = job.split(':'); return base if fe == 'std' else fe + '_' + base
def aid(frame):
    H, Wd = frame.shape[:2]; s = min(1., 640 / max(H, Wd))
    small = frame if s == 1. else cv2.resize(frame, (int(round(Wd * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(small, (0, 0), 1.5); gain = min(max(180. / max(float(np.percentile(g[::4, ::4], 99)), 1.), 1.), 8.)
    return np.ascontiguousarray(g if gain == 1. else cv2.convertScaleAbs(g, alpha=gain))
def _grey(img, r, s):
    g = cv2.cvtColor(img[r[1]:r[3], r[0]:r[2]], cv2.COLOR_RGB2GRAY)
    return g if s >= 1 else cv2.resize(g, (max(8, int(round(g.shape[1] * s))), max(8, int(round(g.shape[0] * s)))), interpolation=cv2.INTER_AREA)
def tpl_make(img, pts):
    x0, y0 = pts.min(0); x1, y1 = pts.max(0); bw, bh = float(x1 - x0), float(y1 - y0)
    if bw < 12 or bh < 12: return None
    r = (int(max(0, x0)), int(max(0, y0)), int(min(img.shape[1], x1)), int(min(img.shape[0], y0 + .55 * bh)))
    if r[2] - r[0] < 8 or r[3] - r[1] < 8: return None
    s = min(1., 64. / bw); return dict(g=_grey(img, r, s), r=r, s=s, bw=bw, bh=bh)
def tpl_find(img, tp, thr=.6):
    """displacement (dx, dy) of the stored upper-face template in `img`, or None when the match is too weak."""
    r = tp['r']; mx, my = int(.15 * tp['bw']) + 1, int(.15 * tp['bh']) + 1; H, W_ = img.shape[:2]
    q = (max(0, r[0] - mx), max(0, r[1] - my), min(W_, r[2] + mx), min(H, r[3] + my)); g = _grey(img, q, tp['s'])
    if g.shape[0] < tp['g'].shape[0] or g.shape[1] < tp['g'].shape[1]: return None
    res = cv2.matchTemplate(g, tp['g'], cv2.TM_CCOEFF_NORMED); _, mxv, _, loc = cv2.minMaxLoc(res)
    if not np.isfinite(mxv) or mxv < thr: return None
    return np.array([q[0] + loc[0] / tp['s'] - r[0], q[1] + loc[1] / tp['s'] - r[1]], np.float32)
def vit_apply(rgb, lm, patches):
    pts = np.round(lm).astype(np.int32); allp = np.concatenate([pts[p] for p in patches]); h, w = rgb.shape[:2]
    x0, y0 = np.maximum(allp.min(0) - 12, 0); x1, y1 = np.minimum(allp.max(0) + 13, [w, h])
    if x1 <= x0 or y1 <= y0: return rgb
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for p in patches: cv2.fillConvexPoly(m, cv2.convexHull(pts[p] - np.array([x0, y0], np.int32)), 255)
    a = cv2.GaussianBlur(m, (0, 0), max(1., .012 * (pts[:, 0].max() - pts[:, 0].min()))).astype(np.float32)[..., None] / 255.
    out = rgb.copy(); out[y0:y1, x0:x1] = np.clip(rgb[y0:y1, x0:x1].astype(np.float32) * (1 + a * (VGAIN - 1)) + .5, 0, 255).astype(np.uint8); return out
def run(job):
    idx, j, jobs = job; key, ds = j['key'], j['dataset']; names = [name_of(x) for x in jobs]; t0 = time.time()
    if all((OUT / c / (key + '.npz')).exists() for c in names): return dict(key=key, state='skip')
    if os.environ.get('X01_SKIP_HEAVY') == '1' and X.is_heavy(j): return dict(key=key, state='not_run_heavy')      # 4K / HDR files: see logs/DEVIATIONS.txt
    core = None; held_sem = False
    try:
        if ds == 'BTC':
            with np.load(Path(j['crop']) / 'meta.npz') as z: boxes, ref_t = z['boxes'], z['t']
            vid = X.BTCVideo(j['source'], ref_t); t = vid.t; ok = boxes[:, 2] > boxes[:, 0]; nb = None
            if vid.w * vid.h > 4e6 or vid.hdr: held_sem = X.heavy_slot(int(os.environ.get('X01_HEAVY', '8')))
            src = iter(vid)
            if ok.any():
                near = np.flatnonzero(ok)[np.abs(np.arange(len(ok))[:, None] - np.flatnonzero(ok)[None]).argmin(1)]; b = boxes[near].astype(float); bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
                nb = np.stack([b[:, 0] - .5 * bw, b[:, 1] - .5 * bh, b[:, 2] + .5 * bw, b[:, 3] + .5 * bh], 1)
        else:
            with np.load(j['meta']) as z: t = z['original_t']
            src = X.frames_of(j, len(t)); nb = None
        allc = X.conds(ds); seed = zlib.crc32(key.encode()); bases = sorted({x.split(':')[1] for x in jobs}); c1b = [b for b in bases if b in C1]
        rng = {b: np.random.default_rng([20261119, seed, allc.index(b)]) for b in c1b}; need_lm = any(b in C7 or b == 'c2_vit' for b in bases)
        mesh = {n_: X.Mesh() for n_ in names}; chk = {b: X.Mesh() for b in c1b}; nat = X.Mesh() if need_lm else None; ref = {}
        for b in c1b:
            if (W / 'out/x01' / b / (key + '.npz')).exists():
                with np.load(W / 'out/x01' / b / (key + '.npz')) as z: ref[b] = z['blk'][:30].copy()
        A = {c: dict(valid=[], blk=[], fill=[], box=[], crop=[], held=0) for c in names}; regen = {b: 0. for b in c1b}; patches = None; vr = np.random.default_rng([20261119, seed, 99]); poly = {}
        last = {c: dict(pts=None, t=-1e9, tpl=None) for c in names}; lm = None; n = 0
        for rgb in src:
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
                elif b in C1:                                # identical draws to x01: one sampler call per frame and condition
                    _, g, kind, var = C1[b]; out = cv2.LUT(rgb, X.LUTS[g]); roi = out[y0:y1, x0:x1]; out[y0:y1, x0:x1] = X.pois(roi, rng[b]) if kind == 'p' else X.gauss(roi, var, rng[b]); fr[b] = out
                    if n < 30 and b in ref: good, crop, box = X.fast_crop(out, chk[b].step(out)); regen[b] = max(regen[b], float(np.abs(crop.reshape(6, 6, 6, 6, 3).mean((1, 3)) - ref[b][n]).max()))
                elif b == 'c6_us1': fr[b] = cv2.filter2D(rgb, -1, X.KUS, borderType=cv2.BORDER_REPLICATE)
                elif b in C7:
                    _, colour, fit = C7[b]
                    if lm is not None: poly[fit] = np.round(co.c7_polygon(lm, fit)).astype(np.int32)
                    out = rgb.copy(); usable[b] = fit in poly
                    if fit in poly: cv2.fillPoly(out, [poly[fit]], co.MASK_RGB[colour])
                    fr[b] = out
                elif b == 'c2_vit':
                    usable[b] = lm is not None and patches is not None; fr[b] = vit_apply(rgb, lm, patches) if usable[b] else rgb
                else: raise ValueError('unknown base ' + b)
            for job_, c in zip(jobs, names):
                fe, b = job_.split(':'); f = fr[b]
                det = aid(f) if 'aid' in fe else f; p = mesh[c].step(det); S = last[c]
                if p is not None:
                    S['pts'], S['t'] = p, tn
                    if 'track' in fe: S['tpl'] = tpl_make(det, p)
                elif 'hold' in fe and S['pts'] is not None and tn - S['t'] <= HOLD_S: p = S['pts']; A[c]['held'] += 1
                elif 'track' in fe and S.get('tpl') is not None:
                    d = tpl_find(det, S['tpl'])
                    if d is not None: p = S['pts'] + d; A[c]['held'] += 1
                pts = None if p is None else p * np.array([Wd / det.shape[1], H / det.shape[0]], np.float32)
                good, crop, box = X.fast_crop(f, pts); a_ = A[c]; a_['valid'].append(bool(good) and usable[b]); a_['box'].append(box)
                a_['blk'].append(crop.reshape(6, 6, 6, 6, 3).mean((1, 3), dtype=np.float32)); a_['fill'].append((crop.max(-1) > 0).reshape(6, 6, 6, 6).sum((1, 3)).astype(np.uint8))
                if ds != 'BTC': a_['crop'].append(crop)
            n += 1
        for m in list(mesh.values()) + list(chk.values()) + ([nat] if nat else []): m.close()
        assert n == len(t); info = dict(key=key, dataset=ds, frames=n, regen_max_abs_diff=regen)
        for c in names:
            a = A[c]; (OUT / c).mkdir(parents=True, exist_ok=True); tmp = OUT / c / (key + '.tmp.npz')
            np.savez_compressed(tmp, t=t, valid=np.array(a['valid']), blk=np.array(a['blk']), fill=np.array(a['fill']), box=np.array(a['box'], np.int16)); tmp.replace(OUT / c / (key + '.npz'))
            if ds != 'BTC':
                (CROPS / c).mkdir(parents=True, exist_ok=True); tmp = CROPS / c / (key + '.tmp.npz'); np.savez_compressed(tmp, crop=np.array(a['crop'])); tmp.replace(CROPS / c / (key + '.npz'))
            info['valid_' + c] = round(float(np.mean(a['valid'])), 4); info['held_' + c] = int(a['held'])
        info.update(seconds=round(time.time() - t0, 1), maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, state='done'); return info
    except Exception as e:
        return dict(key=key, state='error', error=repr(e), traceback=traceback.format_exc()[-1500:])
    finally:
        if held_sem: held_sem.close()
if __name__ == '__main__':
    import multiprocessing as mp
    workers = int(sys.argv[1]); cs = sys.argv[2].split(','); jobs = json.loads((X.R / 'MANIFEST.json').read_text())
    if len(sys.argv) > 3 and sys.argv[3] != '-': jobs = [j for j in jobs if j['key'] in sys.argv[3].split(',') or j['dataset'] in sys.argv[3].split(',')]
    first = int(sys.argv[4]) if len(sys.argv) > 4 else None
    order = np.random.default_rng(2).permutation(len(jobs)); order = sorted(order, key=lambda i: (jobs[i]['dataset'] != 'UBFC2', X.is_heavy(jobs[i]))); X.PT = X.poisson_table(); X.GT.update({v: X.gauss_table(v) for v in (.001, .01)}); log = (W / 'logs/x09_followup.jsonl').open('a'); done = 0
    with (mp.Pool(workers, initializer=X.pin_worker, initargs=(first, workers)) if first is not None else mp.Pool(workers)) as pool:
        for info in pool.imap_unordered(run, [(int(i), jobs[i], cs) for i in order]):
            done += 1; log.write(json.dumps(info) + '\n'); log.flush(); print(done, len(jobs), {k: v for k, v in info.items() if k != 'traceback'}, flush=True)
    (W / 'out/x09_DONE').write_text(time.strftime('%F %T'))
