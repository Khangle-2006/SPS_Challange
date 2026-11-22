"""u29 (22 Nov, Claude): why the optimal linear spatio-chromatic filter (MVDR) does not beat POS on BTC - the colour geometry.
Per video: 3x3 in-band noise covariance of relative face RGB (45-170 bpm, without +-6 bpm around the pulse frequency and its double),
the calibrated blood-volume signature p3 (u28 calib). Output SNR (pulse power per unit noise) of
  MVDR (optimal for the model):      p' S^-1 p
  best projection inside the plane orthogonal to [1,1,1] (what POS does geometrically): max over unit w in that plane of (w'p)^2/(w'Sw)
Gain of the optimum over the plane restriction, in dB; angle between p3 and the leading noise direction."""
import sys, pickle
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x', 'none']
import u28_sch_frontend as U
from scipy import signal, linalg
p = np.load(U.OUT / 'calib.npz')['p3']; med, conf = U.consensus(); one = np.ones(3) / np.sqrt(3)
B = np.linalg.svd(np.eye(3) - np.outer(one, one))[0][:, :2]       # orthonormal basis of the plane orthogonal to [1,1,1]
rows = []
for i in range(U.N):
    if U.Y.low[i]: g = 'low'
    elif U.Y.task[i]: g = 'Task'
    else: g = 'easy' if conf[i] else 'hard'
    x = U.traces(U.keys[i])
    if x is None: continue
    y = signal.sosfiltfilt(U.BPn, U.relative(x[:, :3]), axis=0); fp = med[i] if conf[i] else U.lab[i]
    Sn = U.csd_band(y, lambda f: (f >= 45) & (f <= 170) & (np.abs(f - fp) > 6) & (np.abs(f - 2 * fp) > 6))
    snr_opt = p @ np.linalg.solve(Sn, p)
    Sp_ = B.T @ Sn @ B; pp = B.T @ p; snr_plane = pp @ np.linalg.solve(Sp_, pp)   # optimum restricted to the plane = MVDR in 2-D
    ev, V = np.linalg.eigh(Sn); un = V[:, -1]
    rows.append(dict(g=g, gain=10 * np.log10(snr_opt / snr_plane), ang=np.degrees(np.arccos(min(1, abs(un @ p)))), ang1=np.degrees(np.arccos(min(1, abs(un @ one)))), share=ev[-1] / ev.sum(),
                     leak=10 * np.log10((pp @ np.linalg.solve(Sp_, pp)) / (p @ p) * np.trace(Sn) / 3)))
print('p3', np.round(p, 3), '| angle p3 to [1,1,1]', round(float(np.degrees(np.arccos(p @ one))), 1), 'deg')
q = lambda a: f'{np.median(a):6.2f} [{np.percentile(a, 25):6.2f};{np.percentile(a, 75):6.2f}]'
for g in ('easy', 'hard', 'Task', 'low'):
    G = [r for r in rows if r['g'] == g]
    print(f'{g:5s} n {len(G):3d} | gain of the optimal colour filter over the best intensity-free projection {q([r["gain"] for r in G])} dB | angle noise-pulse {q([r["ang"] for r in G])} deg | noise-[1,1,1] {q([r["ang1"] for r in G])} | noise rank-1 share {q([100 * r["share"] for r in G])}%')
