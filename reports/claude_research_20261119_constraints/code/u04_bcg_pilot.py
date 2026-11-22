"""u04 (22 Nov, Claude) - SIMULATE FIRST: does a HEAD-MOTION pulse (ballistocardiography) survive the measured re-encode chain of the
BTC low-resolution class better than the colour pulse?
Method: Balakrishnan, Durand & Guttag, 'Detecting pulse from head motions in video', CVPR 2013: feature points on the head are tracked
(pyramidal Lucas-Kanade, forward-backward check), vertical trajectories are band-passed to the pulse band, PCA, the most periodic
component is the pulse. Here: per 10 s window (5 s hop) with the team's frozen spectrum operator, averaged over the windows.
Optional camera-motion removal: median trajectory of background points subtracted ('bgsub').
pilot: N camera originals AT REST (BTC dev, not 4K/HDR, random seed 7); per video three versions of the SAME frames: n848 (long side
       848 px, no compression), sim_l1 / sim_l2 (x12/x14 chains). Colour (CHROM+POS on the oval crop, x01 front end) on the same frames.
low  : the real low-resolution files (w*h < 1.8 MP), as decoded.
Decision rule (written before any result): the motion channel goes on to the real low files only if on sim_l1 AND sim_l2 its peak is
within 5 bpm of the label in >= 0.10 more of the pilot videos than the colour peak, and on n848 in >= 0.35 of them.
Usage: $AV code/u04_bcg_pilot.py run <workers> <n_videos> [pilot|low]  |  $AV code/u04_bcg_pilot.py score [pilot|low]"""
import os, sys, json, time, traceback, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np, cv2
from scipy import signal
import x01_extract as X
import x06_c5 as C5
import x12_btcsim as SIM
sys.path.insert(0, '/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118/code')
from lib_btc import avg_spectra, window_spectrum, supported, trend, GRID, zlog, videos
W = X.W; OUT = W / 'out/u04'; TMP = W / 'out/u04_tmp'; co = X.co
cv2.setNumThreads(1)
LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
SOS4 = signal.butter(4, [.7, 3.], btype='bandpass', fs=30, output='sos')
class Tracker:
    def __init__(self): self.g0 = None; self.P = None; self.hist = []; self.kind = np.zeros(0, int); self.alive = None
    def init(self, gray, box):
        x0, y0, x1, y1 = [int(v) for v in box]; w, h = x1 - x0, y1 - y0; Hh, Ww = gray.shape
        m = np.zeros_like(gray); xa, xb = x0 + int(.15 * w), x1 - int(.15 * w)
        m[max(y0, 0):y0 + int(.28 * h), xa:xb] = 255; m[y0 + int(.55 * h):min(y1, Hh), xa:xb] = 255
        pf = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.003, minDistance=3, mask=m, blockSize=5)
        mb = np.full_like(gray, 255); ex, ey = int(.35 * w), int(.35 * h)
        mb[max(0, y0 - ey):Hh, max(0, x0 - ex):min(Ww, x1 + ex)] = 0
        pb = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.003, minDistance=5, mask=mb, blockSize=5)
        pf = np.zeros((0, 1, 2), np.float32) if pf is None else pf; pb = np.zeros((0, 1, 2), np.float32) if pb is None else pb
        self.kind = np.r_[np.zeros(len(pf), int), np.ones(len(pb), int)]; self.P = np.concatenate([pf, pb]).astype(np.float32)
        self.alive = np.ones(len(self.P), bool); self.g0 = gray; self.hist.append(self.P[:, 0, :].copy())
    def step(self, gray):
        if self.P is None or not len(self.P): self.hist.append(None); return
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.g0, gray, self.P, None, **LK)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.g0, p1, None, **LK)
        fb = np.linalg.norm((self.P - pb)[:, 0], axis=1); self.alive &= (st[:, 0] == 1) & (st2[:, 0] == 1) & (fb < 1.0)
        self.P = p1; self.g0 = gray; q = p1[:, 0, :].copy(); q[~self.alive] = np.nan; self.hist.append(q)
def periodicity(p):
    i = int(np.argmax(p)); f = GRID[i]; return float((p[np.abs(GRID - f) <= 3].sum() + p[np.abs(GRID - 2 * f) <= 3].sum()) / p.sum())
def motion_spectra(t, hist, kind):
    T = len(hist); K = len(kind); Y = np.full((T, K), np.nan)
    for i, q in enumerate(hist):
        if q is not None and len(q) == K: Y[i] = q[:, 1]
    rows = np.flatnonzero(np.isfinite(Y).any(1)) if K else []
    if not len(rows): return dict(n_face=0, n_bg=0)
    i0 = int(rows[0]); face = np.flatnonzero(kind == 0); bg = np.flatnonzero(kind == 1)
    okf = [k for k in face if np.isfinite(Y[i0:, k]).all()]; okb = [k for k in bg if np.isfinite(Y[i0:, k]).all()]
    if len(okf) < 8: return dict(n_face=len(okf), n_bg=len(okb))
    Yf = Y[:, okf]; md = np.max(np.abs(np.diff(Yf[i0:], axis=0)), 0); Yf = Yf[:, md <= np.quantile(md, .75)]
    var = {'raw': Yf}
    if len(okb) >= 8: var['bgsub'] = Yf - np.median(Y[:, okb] - Y[i0, okb][None], 1)[:, None]
    valid = np.zeros(T, bool); valid[i0:] = True; out = dict(n_face=int(Yf.shape[1]), n_bg=len(okb), amp_px=float(np.median(np.std(signal.sosfiltfilt(SOS4, Yf[i0:] - Yf[i0:].mean(0), axis=0), 0))))
    for name, A in var.items():
        acc = {'best': [], 'pc1': []}
        for s in np.arange(0, max(0., t[-1] - 10 + 1e-8), 5.):
            if not supported(t, valid, s, s + 10): continue
            q = s + np.arange(300) / 30.; M = np.stack([np.interp(q, t[i0:], A[i0:, k]) for k in range(A.shape[1])], 1)
            Mf = signal.sosfiltfilt(SOS4, M - M.mean(0), axis=0); Mf -= Mf.mean(0); U, sv, _ = np.linalg.svd(Mf, full_matrices=False)
            comps = (U[:, :5] * sv[:5]).T; specs = [window_spectrum(c) for c in comps if c.std() > 1e-12]
            if not specs: continue
            sc = [periodicity(p) for p in specs]; acc['best'].append(specs[int(np.argmax(sc))]); acc['pc1'].append(specs[0])
        for v, L in acc.items():
            if L: out[f'{name}_{v}'] = np.mean(L, 0)
    return out
class Chain:
    def __init__(self): self.mesh = X.Mesh(); self.tr = Tracker(); self.rgb = []; self.valid = []; self.started = False
    def feed(self, frame):
        good, crop, box = X.fast_crop(frame, self.mesh.step(frame)); self.valid.append(bool(good)); self.rgb.append(crop.reshape(-1, 3).mean(0) if good else np.full(3, np.nan))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if not self.started:
            if good: self.tr.init(gray, box); self.started = True
            else: self.tr.hist.append(None)
        else: self.tr.step(gray)
    def finish(self, t, path):
        self.mesh.close(); t = np.asarray(t, float); valid = np.array(self.valid); rgb = np.array(self.rgb)
        if valid.any():
            for c in range(3):
                x = rgb[:, c]; good = np.isfinite(x); rgb[:, c] = np.interp(np.arange(len(x)), np.flatnonzero(good), x[good])
        res = dict(t=t, valid=valid, rgb=rgb.astype(np.float32))
        a = avg_spectra(t, valid, rgb) if valid.sum() > 10 else None
        if a is not None: res['C'], res['P'] = a[0], a[1]
        res.update(motion_spectra(t, self.tr.hist, self.tr.kind)); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp.npz'); np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in res.items()}); tmp.replace(path)
        return {k: (int(v) if isinstance(v, (int, np.integer)) else round(float(v), 3)) for k, v in res.items() if k in ('n_face', 'n_bg', 'amp_px')}
def run(job):
    j, mode = job; key = j['key']; t0 = time.time()
    names = ['n848', 'sim_l1', 'sim_l2'] if mode == 'pilot' else ['low']
    if all((OUT / c / (key + '.npz')).exists() for c in names): return dict(key=key, state='skip')
    try:
        with np.load(Path(j['crop']) / 'meta.npz') as z: ref_t = z['t']
        vid = X.BTCVideo(j['source'], ref_t); t = vid.t; n0 = len(t); fps = (n0 - 1) / (t[-1] - t[0]); info = dict(key=key, mode=mode, size=[vid.w, vid.h], fps=round(float(fps), 3))
        if mode == 'low':
            ch = Chain(); n = 0
            for rgb in vid: ch.feed(np.ascontiguousarray(rgb)); n += 1
            assert n == n0, f'decoded {n} of {n0}'; info['low'] = ch.finish(t, OUT / 'low' / (key + '.npz'))
        else:
            sc = 848. / max(vid.w, vid.h); Wd, H = C5.even(vid.w * sc), C5.even(vid.h * sc); sel = {}; tt = {}; fo = {}
            for c, cf in SIM.CH.items():
                if fps <= cf['fps'] * 1.02: sel[c] = np.arange(n0); tt[c] = t; fo[c] = fps
                else:
                    q = t[0] + np.arange(int((t[-1] - t[0]) * cf['fps']) + 1) / cf['fps']; i = np.searchsorted(t, q).clip(1, n0 - 1); i = np.where(q - t[i - 1] <= t[i] - q, i - 1, i); sel[c] = i; tt[c] = q; fo[c] = cf['fps']
            TMP.mkdir(parents=True, exist_ok=True); path = {c: TMP / f'{key}__{c}.mp4' for c in SIM.CH}; kb = {c: SIM.CH[c]['bps'] * (Wd * H / SIM.PX) * (fo[c] / SIM.CH[c]['fps']) / 1000 for c in SIM.CH}
            enc = {c: SIM.encoder(path[c], Wd, H, fo[c], kb[c], SIM.CH[c]['args']) for c in SIM.CH}; ptr = {c: 0 for c in SIM.CH}; ch = Chain(); n = 0
            for i, rgb in enumerate(vid):
                fr = np.ascontiguousarray(co.c4_resize(np.ascontiguousarray(rgb), (Wd, H))); ch.feed(fr); raw = fr.tobytes(); n += 1
                for c in SIM.CH:
                    while ptr[c] < len(sel[c]) and sel[c][ptr[c]] == i: enc[c].stdin.write(raw); ptr[c] += 1
            assert n == n0, f'decoded {n} of {n0}'; info['n848'] = ch.finish(t, OUT / 'n848' / (key + '.npz'))
            for c in SIM.CH:
                enc[c].stdin.close(); assert enc[c].wait() == 0, 'encoder failed'; m = 0; chc = Chain()
                info['kbps_' + c] = round(path[c].stat().st_size * 8 / (len(sel[c]) / fo[c]) / 1000, 1)
                for fr in C5.decode(path[c], Wd, H): chc.feed(np.ascontiguousarray(fr)); m += 1
                assert m == len(sel[c]), f'{c}: decoded {m} of {len(sel[c])}'; info[c] = chc.finish(tt[c], OUT / c / (key + '.npz')); path[c].unlink()
        info.update(seconds=round(time.time() - t0, 1), maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, state='done'); return info
    except Exception as e:
        return dict(key=key, state='error', error=repr(e), traceback=traceback.format_exc()[-1500:])
def media(j):
    import ast
    md = j['metadata'] if isinstance(j['metadata'], dict) else ast.literal_eval(j['metadata']); return md.get('media', {})
def jobs_for(mode, n):
    jobs = [j for j in json.loads((X.R / 'MANIFEST.json').read_text()) if j['dataset'] == 'BTC']
    px = lambda j: int(media(j).get('width', 0)) * int(media(j).get('height', 0))
    if mode == 'low': return [j for j in jobs if 0 < px(j) < 1800000]
    jobs = [j for j in jobs if px(j) >= 1800000 and not X.is_heavy(dict(j, metadata=dict(media=media(j)))) and 'Resting' in j['key']]
    return [jobs[i] for i in np.random.default_rng(7).permutation(len(jobs))[:n]]
def labels():
    INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
    J = {j['recording']: j for j in json.loads((X.R / 'MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
    return {J[INV.get(v['recording'], v['recording'])]['key']: float(v['pr_bpm']) for v in videos() if INV.get(v['recording'], v['recording']) in J}
if __name__ == '__main__':
    if sys.argv[1] == 'run':
        from multiprocessing import Pool
        mode = sys.argv[4] if len(sys.argv) > 4 else 'pilot'; js = jobs_for(mode, int(sys.argv[3])); LAB = labels(); js = [j for j in js if j['key'] in LAB]
        print('jobs', len(js), mode, flush=True); log = (W / f'logs/u04_{mode}.jsonl').open('a')
        with Pool(int(sys.argv[2]), maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(run, [(j, mode) for j in js]): log.write(json.dumps(r) + '\n'); log.flush(); print(r.get('key'), r.get('state'), r.get('seconds'), r.get('error', ''), flush=True)
    else:
        mode = sys.argv[2] if len(sys.argv) > 2 else 'pilot'; LAB = labels(); conds = ['n848', 'sim_l1', 'sim_l2'] if mode == 'pilot' else ['low']
        keys = sorted(set.intersection(*[{p.stem for p in (OUT / c).glob('*.npz') if not p.stem.endswith('.tmp')} for c in conds]))
        print(mode, 'videos with every version', len(keys))
        ref = {}
        for c in conds:
            E = {}; amp = []
            for k in keys:
                with np.load(OUT / c / (k + '.npz')) as z: d = {n: z[n] for n in z.files}
                if 'C' in d: E.setdefault('colour', {})[k] = GRID[((zlog(d['C']) + zlog(d['P'])) / 2).argmax()]
                for v in ('raw_best', 'raw_pc1', 'bgsub_best', 'bgsub_pc1'):
                    if v in d: E.setdefault(v, {})[k] = GRID[d[v].argmax()]
                if 'amp_px' in d: amp.append(float(d['amp_px']))
            if c in ('n848', 'low'): ref = E.get('colour', {})
            print(f'== {c}: median filtered face-track amplitude {np.median(amp) if amp else float("nan"):.3f} px')
            for v, D in E.items():
                e = np.array([abs(D[k] - LAB[k]) for k in D]); ag = [abs(D[k] - ref[k]) <= 5 for k in D if k in ref]
                print(f'   {v:11s} videos {len(D):3d} | peak within 5 bpm of label {np.mean(e <= 5):.3f} | median |err| {np.median(e):5.1f} | agrees with colour({conds[0]}) {np.mean(ag) if ag else float("nan"):.3f}')
        rng = np.random.default_rng(0); L = np.array([LAB[k] for k in keys]); sh = [np.mean(np.abs(rng.permutation(L) - L) <= 5) for _ in range(200)]
        print(f'chance (labels permuted between these videos): {np.mean(sh):.3f}')
