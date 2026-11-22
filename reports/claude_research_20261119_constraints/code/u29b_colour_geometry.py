"""u29b (22 Nov, Claude): u29 redone on the correct traces (u31 loader). Upper bound of any LINEAR colour / spatio-colour filter.
Per video, in-band noise covariance S (45-170 bpm without +-6 bpm around the pulse frequency and its double) of window-normalised channels;
pulse steering a (calibrated signature p3; 18 region channels: 1_6 (x) p3; background channels 0). Output SNR per unit pulse power:
  oracle 3-ch MVDR   a3' S3^-1 a3            oracle 21-ch MVDR (regions + background)   a21' S21^-1 a21
  best projection in the intensity-free plane (what POS does geometrically): MVDR restricted to the plane orthogonal to [1,1,1].
Gains in dB of the oracles over the plane restriction. These are optimistic (true S, true signature): a real filter can only do worse."""
import sys
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x', 'none']
import u31_sch_frontend_v2 as U
p = np.load(U.OUT / 'calib.npz')['p3']; med, conf = U.consensus(); one = np.ones(3) / np.sqrt(3)
B = np.linalg.svd(np.eye(3) - np.outer(one, one))[0][:, :2]; a21 = np.r_[np.kron(np.ones(6), p) / np.sqrt(6), np.zeros(3)]
BP = signal.butter(3, [.6, 3.2], btype='bandpass', fs=U.FS, output='sos'); rows = []
for i in range(U.N):
    g = 'low' if U.Y.low[i] else ('Task' if U.Y.task[i] else ('easy' if conf[i] else 'hard'))
    d = U.load(i)
    if d is None: continue
    t, v, X = d; q, Uu = U.uniform(t, v, X); y = Uu / uniform_filter1d(Uu, 60, axis=0, mode='nearest') - 1; y = signal.sosfiltfilt(BP, y, axis=0)
    fp = med[i] if conf[i] else U.lab[i]; f = signal.stft(y[:, 0], fs=U.FS, nperseg=U.L, noverlap=U.L // 2, nfft=2048)[0] * 60
    Z = np.stack([signal.stft(y[:, c], fs=U.FS, nperseg=U.L, noverlap=U.L // 2, nfft=2048)[2] for c in range(24)]); m = (f >= 45) & (f <= 170) & (np.abs(f - fp) > 6) & (np.abs(f - 2 * fp) > 6)
    Zm = Z[:, m].reshape(24, -1); S = np.real(Zm @ Zm.conj().T) / Zm.shape[1]
    S3 = S[:3, :3]; S21 = S[3:, 3:] + 1e-6 * np.trace(S[3:, 3:]) / 21 * np.eye(21)
    o3 = p @ np.linalg.solve(S3, p); pl = (B.T @ p) @ np.linalg.solve(B.T @ S3 @ B, B.T @ p); o21 = a21 @ np.linalg.solve(S21, a21)
    # scale: the 18 region channels carry the same pulse per channel as the face channels; a21 has unit norm over 6 regions -> pulse power
    # per region channel = 1/6 of a3's; multiply o21 by 6 to compare with one face channel triplet carrying the full pulse
    rows.append(dict(g=g, g3=10 * np.log10(o3 / pl), g21=10 * np.log10(6 * o21 / pl)))
q_ = lambda a: f'{np.median(a):6.2f} [{np.percentile(a, 25):6.2f};{np.percentile(a, 75):6.2f}]'
for g in ('easy', 'hard', 'Task', 'low'):
    G = [r for r in rows if r['g'] == g]
    print(f'{g:5s} n {len(G):3d} | oracle colour filter over the intensity-free projection {q_([r["g3"] for r in G])} dB | oracle 6 regions + background {q_([r["g21"] for r in G])} dB')
