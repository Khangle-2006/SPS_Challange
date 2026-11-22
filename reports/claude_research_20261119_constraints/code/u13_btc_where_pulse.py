"""u13 (22 Nov, Claude): WHERE on the face is the pulse in the hard videos?  Descriptive (uses the label only to measure, never to choose).
For each of the 6 block rows (forehead -> chin) and the 3 columns, POS on that region's mean trace, Welch PSD, SNR around the LABEL HR
(+ 2nd harmonic) vs the rest of 0.7-4 Hz; compared with the whole oval. Camera originals at rest, easy vs hard (u07 groups)."""
import sys, pickle, json
from pathlib import Path
import numpy as np
from scipy import signal
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
sys.path.insert(0, str(W.parent / 'claude_research_20261118/code'))
from lib_btc import pos_team
lab, low, task, N, GRID = Y.lab, Y.low, Y.task, Y.N, Y.GRID
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); pk = np.full((5, N), np.nan)
for i, s in enumerate(Sp):
    if s['chrom']: pk[0, i] = GRID[np.mean([w[0] for w in s['chrom']], 0).argmax()]; pk[1, i] = GRID[np.mean([w[0] for w in s['pos']], 0).argmax()]
for j, n in enumerate(['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']):
    with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
    h = np.isfinite(S).all(1); pk[2 + j, h] = GRID[S[h].argmax(1)]
ok = np.isfinite(pk).all(0); med = np.nanmedian(pk, 0); conf = ok & ((np.abs(pk - med) <= 3).sum(0) >= 4)
def snr(x, f0):
    f, P = signal.welch(pos_team(x), fs=30, nperseg=300, noverlap=150); band = (f >= .7) & (f <= 4.); sig = (np.abs(f - f0) <= .1) | (np.abs(f - 2 * f0) <= .2)
    return 10 * np.log10(P[band & sig].sum() / max(P[band & ~sig].sum(), 1e-30))
res = {'easy': [], 'hard': []}
for i in np.flatnonzero(~task & (low == 0) & ok):
    with np.load(W / 'out/x01/native' / (keys[i] + '.npz')) as z: t, valid, blk, fill = z['t'], z['valid'], z['blk'].astype(float), z['fill'].astype(float)
    t = t[valid]; blk = blk[valid]; fill = fill[valid]; q = np.arange(t[0], t[-1], 1 / 30.)
    def reg(mask):
        w = fill * mask[None]; rgb = (blk * w[..., None]).sum((1, 2)) / np.maximum(w.sum((1, 2)), 1)[:, None]
        return np.stack([np.interp(q, t, rgb[:, c]) for c in range(3)], 1)
    f0 = lab[i] / 60; r = [snr(reg(np.eye(6)[a][:, None] * np.ones(6)[None]), f0) for a in range(6)]
    cmask = [np.ones(6)[:, None] * (np.arange(6) // 2 == c)[None] for c in range(3)]; cc = [snr(reg(m.astype(float)), f0) for m in cmask]
    whole = snr(reg(np.ones((6, 6))), f0); top = snr(reg(np.r_[np.ones((3, 6)), np.zeros((3, 6))]), f0); bot = snr(reg(np.r_[np.zeros((3, 6)), np.ones((3, 6))]), f0)
    res['easy' if conf[i] else 'hard'].append(r + cc + [whole, top, bot])
for g, R in res.items():
    R = np.array(R); m = np.median(R, 0)
    print(f"{g:5s} n={len(R):2d} | SNR at the label (dB), rows forehead->chin: " + ' '.join(f'{x:6.2f}' for x in m[:6]) + ' | columns L/C/R: ' + ' '.join(f'{x:6.2f}' for x in m[6:9]) + f' | whole {m[9]:6.2f} upper half {m[10]:6.2f} lower half {m[11]:6.2f}')
    print(f"       share of videos where upper half beats the whole oval by > 1 dB: {np.mean(R[:, 10] > R[:, 9] + 1):.2f}; lower half beats whole by > 1 dB: {np.mean(R[:, 11] > R[:, 9] + 1):.2f}")
