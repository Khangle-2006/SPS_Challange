"""z07 - P-Hybrid (Ben Ahmed, arXiv 2606.04198, June 2026: patch-PCA residual added to CHROM; reported to help most under H.264/H.265,
i.e. the codec of BTC's 67 re-encoded files) adapted to the team's traces: CHROM of the oval mean (s_c) + 0.5 x the principal component
of the per-block GREEN residuals (blocks of the 6x6 grid that are >= 80 % inside the oval; each normalised by its mean, band-passed
0.7-3 Hz, s_c projected out) that correlates best with the mean residual. Window spectra with the frozen operator (lib_btc),
evidence added to reader_v4 three ways: replace CHROM, add as a third classical channel, alone. BTC dev native, person-disjoint CV.
Usage: $PY code/z07_btc_patch_hybrid.py <workers>"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy import signal
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
A = sys.argv[:]; sys.argv = ['x']
from lib_btc import supported, chrom_team, pos_team, window_spectrum, GRID, zlog, summarize, videos
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
SOS = signal.butter(2, [.7, 3.], btype='band', fs=30, output='sos')
def nb(x): x = np.asarray(x, float); x = x / x.mean() - 1; return signal.sosfiltfilt(SOS, x)
def unit(x): s = x.std(); return x / s if s > 1e-12 else x * 0
def one(key):
    p = W / 'out/x01/native' / (key + '.npz')
    if not p.exists(): return None
    with np.load(p) as z: t, valid, blk, fill = z['t'], z['valid'].astype(bool), z['blk'].astype(float), z['fill'].astype(float) / 36.
    rgb = blk.mean((1, 2)); inside = fill[valid].mean(0) >= .8 if valid.any() else np.zeros((6, 6), bool); idx = np.argwhere(inside)
    if len(idx) < 4: return None
    H, C, P = [], [], []
    for s in np.arange(0, max(0., t[-1] - 10 + 1e-8), 5.):
        if not supported(t, valid, s, s + 10): continue
        q = s + np.arange(300) / 30.; x = np.stack([np.interp(q, t, rgb[:, c]) for c in range(3)], 1)
        if not np.isfinite(x).all() or x.mean(0).min() <= 0: continue
        wc, wp = chrom_team(x), pos_team(x)
        if not (np.isfinite(wc).all() and np.isfinite(wp).all()) or wc.std() < 1e-12 or wp.std() < 1e-12: continue
        g = np.stack([np.interp(q, t, blk[:, i, j, 1]) for i, j in idx], 1)
        if (g.mean(0) <= 0).any(): continue
        sc = unit(signal.sosfiltfilt(SOS, wc - wc.mean())); G = np.stack([unit(nb(g[:, k])) for k in range(g.shape[1])], 1)
        Rm = G - np.outer(sc, sc @ G / (sc @ sc)); Rm -= Rm.mean(0); U, S_, Vt = np.linalg.svd(Rm, full_matrices=False); mr = Rm.mean(1)
        comps = U[:, :min(4, U.shape[1])] * S_[:min(4, U.shape[1])]; k = int(np.argmax([abs(np.corrcoef(c_, mr)[0, 1]) if c_.std() > 0 else 0 for c_ in comps.T]))
        sr = unit(comps[:, k] * np.sign(np.corrcoef(comps[:, k], mr)[0, 1] or 1)); h = sc + .5 * sr
        H.append(window_spectrum(h)); C.append(window_spectrum(wc)); P.append(window_spectrum(wp))
    return (np.mean(H, 0), np.mean(C, 0), np.mean(P, 0)) if H else None
if __name__ == '__main__':
    import y06_reader_quality as Q
    import z01_btc_reader_ext as E
    jobs = {j['recording']: j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
    INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
    keys = [jobs[INV.get(v['recording'], v['recording'])]['key'] for v in videos()]
    with Pool(int(A[1])) as pool: RES = pool.map(one, keys, chunksize=4)
    N = Q.N; ok = np.array([r is not None for r in RES]) & E.OK; Zh = np.zeros((N, len(GRID))); 
    for i, r in enumerate(RES):
        if r is not None: Zh[i] = zlog(r[0])
    print('videos with P-Hybrid evidence', int(ok.sum()), 'of', N, flush=True)
    zc = np.zeros((N, len(GRID))); zp = np.zeros((N, len(GRID)))
    for i, r in enumerate(RES):
        if r is not None: zc[i], zp[i] = zlog(r[1]), zlog(r[2])
    print('check CHROM/POS from this script vs Zn on ok videos: max diff', float(np.abs(((zc + zp) / 2)[ok] - Q.Zn[ok]).max()), flush=True)
    lab, per, low, task = Q.lab, Q.per, Q.low, Q.task
    EXP = sum(E.ZM[m][0] for m in E.MODELS)
    VAR = {'v4 (CHROM, POS)': (2 * Q.Zn + EXP) / 5, 'P-Hybrid replaces CHROM': (2 * (Zh + zp) / 2 + EXP) / 5,
           'P-Hybrid as 3rd classical': (2 * (zc + zp + Zh) / 3 + EXP) / 5, 'P-Hybrid + POS, no experts': (Zh + zp) / 2, 'CHROM + POS, no experts': Q.Zn}
    P_ = {k: E.cvC(Z, lab, 'v4', ok) for k, Z in VAR.items()}; base = P_['v4 (CHROM, POS)']
    for k, p in P_.items():
        s = summarize(p[ok], lab[ok], per[ok], base[ok]); sl = summarize(p[ok & (low == 1)], lab[ok & (low == 1)], per[ok & (low == 1)], base[ok & (low == 1)])
        print(f"{k:28s} MAE {s['MAE']:.3f} RMSE {s['RMSE']:.2f} | vs v4 {s['delta']:+.3f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}] | low class {sl['MAE']:.3f} vs v4 {sl['delta']:+.3f} [{sl['dCI'][0]:+.2f};{sl['dCI'][1]:+.2f}]", flush=True)
    for nm, sel in (('originals at rest', ok & (low == 0) & ~task), ('low class', ok & (low == 1))):
        for k, Z in (('CHROM', zc), ('POS', zp), ('P-Hybrid', Zh), ('CHROM+POS', Q.Zn)):
            print(f"   peak within 5 bpm, {nm:17s} (n={int(sel.sum())}): {k:9s} {np.mean(np.abs(GRID[Z[sel].argmax(1)] - lab[sel]) <= 5):.3f}", flush=True)
