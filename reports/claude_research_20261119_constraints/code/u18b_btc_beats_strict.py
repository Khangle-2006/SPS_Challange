"""u18b (22 Nov, Claude): u18 beat detection was too permissive (IBI CV 0.24 even in easy videos = spurious peaks). Refined, still label-free:
minimum peak distance = 0.6 x the period of the video's own FacePhys spectral peak, prominence >= 0.5 SD, band-pass 0.7-3 Hz.
Descriptive only: does the IBI irregularity differ between easy and hard camera originals at rest?"""
import sys, json
from pathlib import Path
import numpy as np
from scipy import signal
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
lab, low, task, N, GRID = Y.lab, Y.low, Y.task, Y.N, Y.GRID
d = np.load(W / 'out/u18_btc_beats.npz', allow_pickle=True); grp = d['grp']
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]
with np.load(W / 'out/y05/FacePhys.rlap.npz') as z: S = z['S']
BP = signal.butter(3, [.7, 3.], btype='bandpass', fs=30, output='sos'); out = {g: [] for g in ('easy', 'hard', 'low')}
for i, k in enumerate(keys):
    g = str(grp[i])
    if g not in out or not np.isfinite(S[i]).all(): continue
    with np.load(W / 'out/u17/FacePhys.rlap' / (k + '.npz')) as z: bvp, ts = z['bvp'].astype(float), z['ts'].astype(float)
    q = np.arange(ts[0], ts[-1], 1 / 30.); x = signal.sosfiltfilt(BP, np.interp(q, ts, bvp)); f0 = GRID[S[i].argmax()] / 60.
    p, _ = signal.find_peaks(x, distance=max(3, int(.6 * 30 / f0)), prominence=.5 * x.std()); ibi = np.diff(p) / 30.
    if len(ibi) < 8: continue
    cv = ibi.std() / ibi.mean(); out[g].append((cv, abs(60 / np.median(ibi) - lab[i]), abs(60 * f0 - lab[i]), np.mean(np.abs(np.diff(ibi)) > .2 * ibi.mean())))
for g, A in out.items():
    A = np.array(A); print(f'{g:5s} n={len(A):3d} | IBI CV median {np.median(A[:, 0]):.3f} [{np.quantile(A[:, 0], .25):.3f};{np.quantile(A[:, 0], .75):.3f}] | share CV > 0.15 {np.mean(A[:, 0] > .15):.2f}'
          f' | share of successive IBI changes > 20 % {np.median(A[:, 3]):.2f} | |err|: beat HR {A[:, 1].mean():5.2f}, FacePhys peak {A[:, 2].mean():5.2f}')
