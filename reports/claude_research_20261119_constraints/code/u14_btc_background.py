"""u14 (22 Nov, Claude): is the common-mode in-band noise of the hard camera originals shared with the BACKGROUND (illumination / camera
processing) - i.e. removable with a background reference?  Descriptive: the label is only used to MEASURE the SNR, never to choose.
Face = oval mean RGB (numeric_dev blocks rgb_mean), background = ROI 3 of numeric_dev roi (largest area, outside the face).
  corr     : correlation of the band-passed (0.7-3 Hz) normalised green of face and background (max |r| over lags -5..5 frames)
  r2       : share of the face band-passed green explained by background R,G,B at lags -3..3 (least squares over the video)
  snr      : POS SNR at the label HR before / after regressing the band-passed background RGB (lags -3..3) out of the face RGB."""
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
BP = signal.butter(2, [.7, 3.], btype='bandpass', fs=30, output='sos')
def snr(x, f0):
    f, P = signal.welch(pos_team(x), fs=30, nperseg=300, noverlap=150); band = (f >= .7) & (f <= 4.); sig = (np.abs(f - f0) <= .1) | (np.abs(f - 2 * f0) <= .2)
    return 10 * np.log10(P[band & sig].sum() / max(P[band & ~sig].sum(), 1e-30))
def lagmat(A, L=3):
    return np.concatenate([np.roll(A, k, 0) for k in range(-L, L + 1)], 1)
res = {'easy': [], 'hard': []}
for i in np.flatnonzero(~task & (low == 0) & ok):
    b = np.load(NUM / 'blocks' / (Y.rec[i] + '.npz')); r = np.load(NUM / 'roi' / (Y.rec[i] + '.npz'))
    t = b['t'].astype(float); v = b['valid'].astype(bool) & (r['pixels'][:, 3] > 0); t = t[v]; q = np.arange(t[0], t[-1], 1 / 30.)
    F = np.stack([np.interp(q, t, b['rgb_mean'][v, c]) for c in range(3)], 1); B = np.stack([np.interp(q, t, r['rgb'][v, 3, c]) for c in range(3)], 1)
    if not (np.isfinite(B).all() and np.isfinite(F).all() and B.mean(0).min() > 1): continue
    Fn = F / F.mean(0) - 1; Bn = B / B.mean(0) - 1; Fb = signal.sosfiltfilt(BP, Fn, axis=0); Bb = signal.sosfiltfilt(BP, Bn, axis=0)
    cor = max(abs(np.corrcoef(np.roll(Bb[:, 1], k), Fb[:, 1])[0, 1]) for k in range(-5, 6))
    X = lagmat(Bb); coef, *_ = np.linalg.lstsq(X, Fb, rcond=None); res_ = Fb - X @ coef; r2 = 1 - res_[:, 1].var() / Fb[:, 1].var()
    s0 = snr(F, lab[i] / 60); s1 = snr((res_ + 1) * F.mean(0), lab[i] / 60)
    res['easy' if conf[i] else 'hard'].append((cor, r2, s0, s1))
for g, R in res.items():
    R = np.array(R); m = np.median(R, 0)
    print(f'{g:5s} n={len(R):2d} | face-background in-band |corr| {m[0]:.3f} [{np.quantile(R[:, 0], .25):.2f};{np.quantile(R[:, 0], .75):.2f}] | R2 explained by background {m[1]:.3f} [{np.quantile(R[:, 1], .25):.2f};{np.quantile(R[:, 1], .75):.2f}]'
          f' | SNR at label {m[2]:6.2f} -> after background regression {m[3]:6.2f} (median change {np.median(R[:, 3] - R[:, 2]):+.2f} dB; videos improved > 1 dB {np.mean(R[:, 3] - R[:, 2] > 1):.2f})')
