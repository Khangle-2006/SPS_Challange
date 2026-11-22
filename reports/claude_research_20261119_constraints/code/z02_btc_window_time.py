"""z02 - does WHEN in the video the evidence comes from matter for the BTC label (one PR per video, no time stamp)?
Per-window CHROM/POS spectra of the native trace (x01 native blocks, frozen operators of lib_btc, 10 s windows every 5 s) are saved
(out/z02_windows.npz). Then reader_v4 is re-run with the CLASSICAL evidence taken from a time-weighted subset of windows
(experts unchanged: video-level), either for Task videos only or for all videos:
   first k windows (k = 1, 2, 3) | last k windows | exponential weights exp(-s / tau) and exp(-(T - s) / tau), tau = 10, 20 s
and also chosen by inner CV inside the training folds ('nested'). Diagnostics: per-window peak HR vs time within Task videos.
Usage: $PY code/z02_btc_window_time.py <workers>"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
A = sys.argv[:]; sys.argv = ['x']
from lib_btc import supported, chrom_team, pos_team, window_spectrum, GRID, zlog, summarize, videos
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
def windows_of(key):
    p = W / 'out/x01/native' / (key + '.npz')
    if not p.exists(): return None
    with np.load(p) as z: t, valid, blk = z['t'], z['valid'].astype(bool), z['blk'].astype(float)
    rgb = blk.mean((1, 2)); C, P, S = [], [], []
    for s in np.arange(0, max(0., t[-1] - 10 + 1e-8), 5.):
        if not supported(t, valid, s, s + 10): continue
        q = s + np.arange(300) / 30.; x = np.stack([np.interp(q, t, rgb[:, c]) for c in range(3)], 1)
        if not np.isfinite(x).all() or x.mean(0).min() <= 0: continue
        wc, wp = chrom_team(x), pos_team(x)
        if not (np.isfinite(wc).all() and np.isfinite(wp).all()) or wc.std() < 1e-12 or wp.std() < 1e-12: continue
        C.append(window_spectrum(wc)); P.append(window_spectrum(wp)); S.append(s)
    return (np.array(C), np.array(P), np.array(S), float(t[-1])) if C else None
if __name__ == '__main__':
    import y06_reader_quality as Q
    import z01_btc_reader_ext as E
    jobs = {j['recording']: j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
    INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
    V = videos(); rec = [v['recording'] for v in V]; keys = [jobs[INV.get(r, r)]['key'] for r in rec]
    out = W / 'out/z02_windows.npz'
    if out.exists():
        z = np.load(out, allow_pickle=True); WIN = list(z['win'])
    else:
        with Pool(int(A[1])) as pool: WIN = pool.map(windows_of, keys, chunksize=4)
        np.savez_compressed(out, win=np.array(WIN, dtype=object), keys=np.array(keys))
    N = Q.N; lab, per, fold, low, task = Q.lab, Q.per, Q.fold, Q.low, Q.task; OK = E.OK.copy()
    # check: all windows reproduce Zn
    def Zc_from(wfun, who):
        Z = Q.Zn.copy()
        for i, w in enumerate(WIN):
            if w is None or not who[i]: continue
            C, P, S, T = w; a = wfun(S, T)
            if a.sum() <= 0: continue
            a = a / a.sum(); Z[i] = (zlog((a[:, None] * C).sum(0)) + zlog((a[:, None] * P).sum(0))) / 2
        return Z
    Zall = Zc_from(lambda S, T: np.ones(len(S)), np.ones(N, bool)); have = np.array([w is not None for w in WIN])
    print('videos with windows', int(have.sum()), '| max |Z(all windows) - Zn| on those:', float(np.abs(Zall[have & Q.okn] - Q.Zn[have & Q.okn]).max()), flush=True)
    # diagnostics: per-window peak vs time in Task / resting videos with >= 4 windows
    for nm, sel in (('Task', task), ('Resting/Walking', ~task)):
        sl = []; dfirst = []; dlast = []
        for i in np.flatnonzero(sel & have & (low == 0)):
            C, P, S, T = WIN[i]
            if len(S) < 4: continue
            pk = GRID[((zlog(C) + zlog(P)) / 2).argmax(1)]; sl.append(np.polyfit(S, pk, 1)[0] * 60); dfirst.append(abs(pk[0] - lab[i])); dlast.append(abs(pk[-1] - lab[i]))
        print(f'{nm:16s} original-class videos {len(sl)}: median slope of window peak {np.median(sl):+.1f} bpm/min | |first window peak - label| median {np.median(dfirst):.1f}, |last| {np.median(dlast):.1f}', flush=True)
    VAR = {'all': lambda S, T: np.ones(len(S))}
    for k in (1, 2, 3):
        VAR[f'first{k}'] = (lambda k: lambda S, T: (np.arange(len(S)) < k).astype(float))(k); VAR[f'last{k}'] = (lambda k: lambda S, T: (np.arange(len(S)) >= len(S) - k).astype(float))(k)
    for tau in (10., 20.):
        VAR[f'early{int(tau)}'] = (lambda tau: lambda S, T: np.exp(-S / tau))(tau); VAR[f'late{int(tau)}'] = (lambda tau: lambda S, T: np.exp(-(T - 10 - S) / tau))(tau)
    Zc = {}
    for nm, fn in VAR.items():
        for who, wsel in (('task', task), ('every', np.ones(N, bool))):
            if nm == 'all' and who == 'every': continue
            Zc[(nm, who)] = Zc_from(fn, wsel)
    def mix(Zcl): return (2 * Zcl + sum(E.ZM[m][0] for m in E.MODELS)) / 5
    base = E.cvC(mix(Q.Zn), lab, 'v4', OK); S_ = lambda a_, b_: summarize(a_[OK], lab[OK], per[OK], b_[OK])
    P = {}
    for k_, Zv in Zc.items():
        p = E.cvC(mix(Zv), lab, 'v4', OK); P[k_] = p; s = S_(p, base); st = summarize(p[OK & task], lab[OK & task], per[OK & task], base[OK & task])
        print(f"{k_[0]:8s} {k_[1]:5s} MAE {s['MAE']:.3f} RMSE {s['RMSE']:.2f} | vs v4 {s['delta']:+.3f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}] | Task videos {st['MAE']:.3f} vs v4 {st['delta']:+.3f} [{st['dCI'][0]:+.2f};{st['dCI'][1]:+.2f}]", flush=True)
    def inner_task(a):
        f, k_ = a; tr = np.flatnonzero((fold != f) & OK); return f, k_, E.inner(mix(Zc[k_]), lab, 'v4', tr)
    def inner_base(f): tr = np.flatnonzero((fold != f) & OK); return f, E.inner(mix(Q.Zn), lab, 'v4', tr)
    with Pool(int(A[1])) as pool:
        R1 = pool.map(inner_task, [(f, k_) for f in sorted(set(fold)) for k_ in Zc], chunksize=2); R0 = dict(pool.map(inner_base, sorted(set(fold))))
    p = base.copy(); ch = []
    for f in sorted(set(fold)):
        ls = {k_: l for ff, k_, l in R1 if ff == f}; k_ = min(ls, key=ls.get); te = np.flatnonzero((fold == f) & OK)
        if ls[k_] < R0[f] - .02: p[te] = P[k_][te]; ch.append(k_)
        else: ch.append('v4')
    s = S_(p, base); print(f"nested time weighting: chosen {ch} MAE {s['MAE']:.3f} | vs v4 {s['delta']:+.3f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}]", flush=True)
