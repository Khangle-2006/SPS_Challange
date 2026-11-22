"""u23 (22 Nov, Claude): a 'mean-rate' evidence channel for irregular pulses (literature review 22/11, Q2: in arrhythmia the spectral
peak splits and rPPG underestimates HR, Cramer et al. 2025; the oximeter reports an average rate). Estimator without beat detection:
FacePhys waveform (u17), 10 s windows, 2 s hop; coarse peak of the window periodogram (42-180 bpm); zero-phase band-pass +-0.25 Hz
around it on the full signal; Hilbert phase unwrapped over the central 8 s; rate = phase advance / (2 pi * 8 s). Per video the window
rates become a positive pseudo-spectrum on GRID (sum of Gaussians, sd 3 bpm), saved like out/y05 (keys, S) for u01.
Rule fixed before any result: the channel enters the reader only if the u01 nested forward selection picks it in >= 4/5 folds
(same rule as reader_v4). Usage: $PY code/u23_btc_phase_rate.py, then $PY code/u01_btc_nested_all.py out/u23/npz"""
import sys, json
from pathlib import Path
import numpy as np
from scipy import signal
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
N, GRID = Y.N, Y.GRID
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]
BP = signal.butter(3, [.7, 3.], btype='bandpass', fs=30, output='sos'); S = np.full((N, len(GRID)), np.nan); nw = np.zeros(N, int); rate_med = np.full(N, np.nan)
for i, k in enumerate(keys):
    p = W / 'out/u17/FacePhys.rlap' / (k + '.npz')
    if not p.exists(): continue
    with np.load(p) as z: bvp, ts = z['bvp'].astype(float), z['ts'].astype(float)
    if len(bvp) < 330: continue
    q = np.arange(ts[0], ts[-1], 1 / 30.); x = signal.sosfiltfilt(BP, np.interp(q, ts, bvp)); R = []
    f = np.fft.rfftfreq(4096, 1 / 30.)
    for s in range(0, len(x) - 300 + 1, 60):
        P = np.abs(np.fft.rfft(x[s:s + 300] * np.hanning(300), 4096)) ** 2; m = (f * 60 >= 42) & (f * 60 <= 180); f0 = f[m][P[m].argmax()]
        nb = signal.butter(2, [max(.5, f0 - .25), f0 + .25], btype='bandpass', fs=30, output='sos'); a0, a1 = max(0, s - 150), min(len(x), s + 450)
        y = signal.sosfiltfilt(nb, x[a0:a1]); ph = np.unwrap(np.angle(signal.hilbert(y))); c0 = s - a0 + 30; c1 = c0 + 240
        R.append((ph[c1] - ph[c0]) / (2 * np.pi * 8.) * 60)
    R = np.array(R); R = R[(R >= 40) & (R <= 185)]
    if len(R) < 2: continue
    S[i] = np.exp(-(GRID[None] - R[:, None]) ** 2 / (2 * 3. ** 2)).sum(0) + 1e-3; nw[i] = len(R); rate_med[i] = np.median(R)
ok = np.isfinite(S).all(1); print('videos with a phase-rate channel', int(ok.sum()), '| median windows', int(np.median(nw[ok])))
e = np.abs(rate_med - Y.lab); print(f'phase-rate median alone: |err| median {np.nanmedian(e):.2f}, mean {np.nanmean(e):.2f}, within 5 bpm {np.nanmean(e <= 5):.3f}')
(W / 'out/u23/npz').mkdir(parents=True, exist_ok=True); np.savez_compressed(W / 'out/u23/npz/PhaseRate.FacePhys.npz', keys=np.array(keys), S=S)
