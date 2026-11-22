"""u16 (22 Nov, Claude): hypothesis test - is the 'noise' of the hard camera originals an IRREGULAR PULSE (arrhythmia in patients)?
It would explain u12/u13: same band-passed amplitude, lower spectral-peak SNR (energy spread), spatially common (the pulse is coherent),
unstable window peaks. Descriptive; the label is used only to measure. Whole-video POS of the oval mean (numeric_dev), band-pass 0.7-3 Hz,
peaks >= 0.33 s apart with prominence >= 0.3 SD -> inter-beat intervals (IBI, kept 0.33-1.5 s).
  cv_ibi   : SD/mean of IBI (sinus rhythm at rest ~0.05-0.10; atrial fibrillation typically > 0.15-0.2)
  rmssd_rel: RMSSD of successive IBI / mean IBI
  hr_beats : 60 / median IBI;  hr_rate : 60 * (beats - 1) / span;  compared with the spectral consensus and the label."""
import sys, pickle, json
from pathlib import Path
import numpy as np
from scipy import signal
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
NUM = W.parent / 'w2_btc_20261116/numeric_dev'; sys.path.insert(0, str(W.parent / 'claude_research_20261118/code'))
from lib_btc import pos_team
lab, low, task, N, GRID = Y.lab, Y.low, Y.task, Y.N, Y.GRID
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); pk = np.full((5, N), np.nan)
for i, s in enumerate(Sp):
    if s['chrom']: pk[0, i] = GRID[np.mean([w[0] for w in s['chrom']], 0).argmax()]; pk[1, i] = GRID[np.mean([w[0] for w in s['pos']], 0).argmax()]
for j, n in enumerate(['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']):
    with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
    h = np.isfinite(S).all(1); pk[2 + j, h] = GRID[S[h].argmax(1)]
ok = np.isfinite(pk).all(0); med = np.nanmedian(pk, 0); conf = ok & ((np.abs(pk - med) <= 3).sum(0) >= 4)
BP = signal.butter(3, [.7, 3.], btype='bandpass', fs=30, output='sos')
R = {'easy': [], 'hard': [], 'low': []}
for i in range(N):
    g = 'low' if low[i] == 1 else ('easy' if conf[i] else ('hard' if ok[i] and not task[i] else None))
    if g is None: continue
    b = np.load(NUM / 'blocks' / (Y.rec[i] + '.npz')); t = b['t'].astype(float); v = b['valid'].astype(bool); q = np.arange(t[v][0], t[v][-1], 1 / 30.)
    X = np.stack([np.interp(q, t[v], b['rgb_mean'][v, c]) for c in range(3)], 1); h = signal.sosfiltfilt(BP, pos_team(X))
    p, _ = signal.find_peaks(h, distance=10, prominence=.3 * h.std()); ibi = np.diff(p) / 30.; ibi = ibi[(ibi >= .33) & (ibi <= 1.5)]
    if len(ibi) < 8: continue
    cv = ibi.std() / ibi.mean(); rm = np.sqrt(np.mean(np.diff(ibi) ** 2)) / ibi.mean(); hb = 60 / np.median(ibi); hr = 60 * (len(p) - 1) / ((p[-1] - p[0]) / 30.)
    R[g].append((cv, rm, abs(hb - lab[i]), abs(hr - lab[i]), abs(med[i] - lab[i]) if np.isfinite(med[i]) else abs(pk[1, i] - lab[i])))
for g, A in R.items():
    A = np.array(A); m = np.median(A, 0); mn = A.mean(0)
    print(f'{g:5s} n={len(A):3d} | IBI CV median {m[0]:.3f} [{np.quantile(A[:, 0], .25):.2f};{np.quantile(A[:, 0], .75):.2f}] | RMSSD/mean {m[1]:.3f} | share CV > 0.15 {np.mean(A[:, 0] > .15):.2f}'
          f' | |err| vs label (mean): beats-median {mn[2]:5.2f}, beat-rate {mn[3]:5.2f}, spectral consensus {mn[4]:5.2f}')
