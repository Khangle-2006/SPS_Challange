"""u15 (22 Nov, Claude): BACKGROUND-REFERENCED illumination noise cancellation for the classical channels of the BTC reader.
Root cause targeted (u13/u14): in the hard camera originals the in-band noise is common to the whole face (regions do not average it out),
and ~25 % of the band-passed face signal is explained by the background (illumination / camera processing). Literature: background as the
illumination reference (Li et al., CVPR 2014, NLMS; Cheng et al., IEEE JBHI 2017, joint BSS face/background).
Method (fixed before any score): per video, face RGB (oval mean) and background RGB (numeric_dev ROI 3) on a 30 Hz grid, normalised by their
means; the band-passed (0.7-3 Hz) background RGB at lags -3..3 frames is least-squares regressed out of the normalised face RGB over the whole
video (label-free); then the team's CHROM/POS and frozen window operator (10 s windows, 5 s hop, same supported windows).
Base = the same code without the regression (must reproduce y01 closely). Readers: v3 (classical only) and v4 (+3 experts), fits on
training folds. Accept if v4+bg beats v4 on all videos with paired CI < 0. Control: background taken from ANOTHER video (permuted)."""
import sys, pickle, json
from pathlib import Path
import numpy as np
from scipy import signal
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
NUM = W.parent / 'w2_btc_20261116/numeric_dev'; sys.path.insert(0, str(W.parent / 'claude_research_20261118/code'))
from lib_btc import pos_team, chrom_team, window_spectrum, supported
lab, per, low, task, fold, N, GRID = Y.lab, Y.per, Y.low, Y.task, Y.fold, Y.N, Y.GRID
BP = signal.butter(2, [.7, 3.], btype='bandpass', fs=30, output='sos')
def lagmat(A, L=3): return np.concatenate([np.roll(A, k, 0) for k in range(-L, L + 1)], 1)
def spectra(i, mode, bg_from=None):
    b = np.load(NUM / 'blocks' / (Y.rec[i] + '.npz')); t = b['t'].astype(float); v = b['valid'].astype(bool)
    q = np.arange(0, t[-1] + 1e-9, 1 / 30.); tv = t[v]
    F = np.stack([np.interp(q, tv, b['rgb_mean'][v, c]) for c in range(3)], 1)
    if mode != 'base':
        j = i if bg_from is None else bg_from; r = np.load(NUM / 'roi' / (Y.rec[j] + '.npz')); tr = r['t'].astype(float); vb = r['pixels'][:, 3] > 0
        Bq = np.stack([np.interp(q * (tr[-1] / t[-1] if bg_from is not None else 1.), tr[vb], r['rgb'][vb, 3, c]) for c in range(3)], 1) if vb.sum() > 10 else None
        if Bq is not None and np.isfinite(Bq).all() and Bq.mean(0).min() > 1:
            Fm = F.mean(0); Fn = F / Fm - 1; Bb = signal.sosfiltfilt(BP, Bq / Bq.mean(0) - 1, axis=0); X = lagmat(Bb)
            coef, *_ = np.linalg.lstsq(X, signal.sosfiltfilt(BP, Fn, axis=0), rcond=None); F = (Fn - X @ coef + 1) * Fm
    C, P = [], []
    for s in np.arange(0, max(0., t[-1] - 10 + 1e-8), 5.):
        if not supported(t, v, s, s + 10): continue
        a = int(round(s * 30)); x = F[a:a + 300]
        if len(x) < 300 or not np.isfinite(x).all() or x.mean(0).min() <= 0: continue
        wc, wp = chrom_team(x), pos_team(x)
        if wc.std() < 1e-12 or wp.std() < 1e-12: continue
        C.append(window_spectrum(wc)); P.append(window_spectrum(wp))
    return (np.mean(C, 0), np.mean(P, 0)) if C else None
if __name__ == '__main__':
    from multiprocessing import Pool
    rng = np.random.default_rng(0); perm = rng.permutation(N)
    with Pool(12) as p:
        base = p.starmap(spectra, [(i, 'base') for i in range(N)]); bg = p.starmap(spectra, [(i, 'bg') for i in range(N)]); bgp = p.starmap(spectra, [(i, 'bg', int(perm[i])) for i in range(N)])
    def Z(sp):
        Z0 = np.zeros((N, len(GRID))); okk = np.zeros(N, bool)
        for i, s in enumerate(sp):
            if s is not None: Z0[i] = (Y.zlog(s[0]) + Y.zlog(s[1])) / 2; okk[i] = True
        return Z0, okk
    Zb, ok0 = Z(base); Zg, ok1 = Z(bg); Zp, ok2 = Z(bgp)
    Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); Zy = np.zeros_like(Zb)
    for i, s in enumerate(Sp):
        if s['chrom']: Zy[i] = (Y.zlog(np.mean([w[0] for w in s['chrom']], 0)) + Y.zlog(np.mean([w[0] for w in s['pos']], 0))) / 2
    print('base reproduces y01: same CHROM+POS peak in', int(np.sum(GRID[Zb.argmax(1)] == GRID[Zy.argmax(1)])), 'of', N, 'videos')
    ZE = []; okE = np.ones(N, bool)
    for n in ['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']:
        with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
        h = np.isfinite(S).all(1); okE &= h; Zm = np.zeros_like(Zb); Zm[h] = Y.zlog(S[h]); ZE.append(Zm)
    sub = ok0 & ok1 & ok2 & okE
    def cv(Zx):
        p = np.full(N, np.nan)
        for f in sorted(set(fold)):
            tr = np.flatnonzero((fold != f) & sub); te = np.flatnonzero((fold == f) & sub)
            m = Y.fitG(Zx[tr], lab[tr], per[tr], low[tr], low[tr], task[tr]); p[te] = Y.predG(m, Zx[te], low[te], low[te], task[te])
        return p
    mix = lambda Z0: (2 * Z0 + sum(ZE)) / 5
    P = {'v3': cv(Zb), 'v3+bg': cv(Zg), 'v4': cv(mix(Zb)), 'v4+bg': cv(mix(Zg)), 'v4+bg(permuted bg)': cv(mix(Zp))}
    S_ = lambda a_, b_, m: Y.summarize(a_[m], lab[m], per[m], b_[m])
    pk = lambda Zx: GRID[Zx.argmax(1)]
    rest = ~task & (low == 0)
    for name, m in (('all', sub), ('camera originals, rest', sub & rest), ('camera originals, Task', sub & task & (low == 0)), ('low-resolution files', sub & (low == 1))):
        line = f"{name:24s} videos {int(m.sum()):3d} | classical peak within 5 bpm: base {np.mean(np.abs(pk(Zb)[m] - lab[m]) <= 5):.3f} bg {np.mean(np.abs(pk(Zg)[m] - lab[m]) <= 5):.3f} | "
        a = S_(P['v3+bg'], P['v3'], m); b = S_(P['v4+bg'], P['v4'], m); c = S_(P['v4+bg(permuted bg)'], P['v4'], m)
        line += f"v3 {S_(P['v3'], Y.const, m)['MAE']:5.2f} -> +bg {a['MAE']:5.2f} ({a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}]) | v4 {S_(P['v4'], Y.const, m)['MAE']:5.2f} -> +bg {b['MAE']:5.2f} ({b['delta']:+.2f} [{b['dCI'][0]:+.2f};{b['dCI'][1]:+.2f}]) | permuted bg {c['MAE']:5.2f} ({c['delta']:+.2f})"
        print(line)
    np.savez_compressed(W / 'out/u15_btc_bgref.npz', rec=Y.rec, Zbase=Zb, Zbg=Zg, Zperm=Zp, sub=sub, **{k.replace('+', '_').replace(' ', '_').replace('(', '').replace(')', ''): v for k, v in P.items()})
