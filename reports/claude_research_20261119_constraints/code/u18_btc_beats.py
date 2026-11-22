"""u18 (22 Nov, Claude): (1) test of the 'irregular pulse' hypothesis with the FacePhys BVP waveform of u17; (2) one fixed reader variant.
(1) Descriptive: inter-beat intervals (band-pass 0.7-3 Hz, peaks >= 0.33 s apart, prominence >= 0.3 SD, IBI kept 0.33-1.5 s):
    CV of IBI and RMSSD/mean per video, in the easy / hard camera originals at rest and the low files (u07 groups).
(2) Variant fixed before any result: a 'beat-rate' evidence curve, the log of a Gaussian (sd 4 bpm) centred on 60 / median IBI of the
    FacePhys waveform, standardised like the other channels, enters the reader_v4 mix as one more expert (weight 1). Reader fits on
    training folds. Accept only if it beats v4 on all videos with paired CI < 0.
Usage: $PY code/u18_btc_beats.py [model=FacePhys.rlap]"""
import sys, pickle, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import signal
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code'))
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'FacePhys.rlap'; sys.argv = ['x']
import y06_reader_quality as Y
lab, per, low, task, fold, N, GRID = Y.lab, Y.per, Y.low, Y.task, Y.fold, Y.N, Y.GRID
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); Z0 = np.zeros((N, len(GRID))); pk = np.full((5, N), np.nan)
for i, s in enumerate(Sp):
    if s['chrom']:
        c = np.mean([w[0] for w in s['chrom']], 0); p = np.mean([w[0] for w in s['pos']], 0); Z0[i] = (Y.zlog(c) + Y.zlog(p)) / 2; pk[0, i] = GRID[c.argmax()]; pk[1, i] = GRID[p.argmax()]
ZE = []; okE = np.ones(N, bool)
for j, n in enumerate(['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']):
    with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
    h = np.isfinite(S).all(1); okE &= h; Zm = np.zeros_like(Z0); Zm[h] = Y.zlog(S[h]); ZE.append(Zm); pk[2 + j, h] = GRID[S[h].argmax(1)]
med = np.nanmedian(pk, 0); conf = okE & ((np.abs(pk - med) <= 3).sum(0) >= 4)
BP = signal.butter(3, [.7, 3.], btype='bandpass', fs=30, output='sos')
cv_ = np.full(N, np.nan); rms = np.full(N, np.nan); hrb = np.full(N, np.nan)
for i, k in enumerate(keys):
    p = W / 'out/u17' / MODEL / (k + '.npz')
    if not p.exists(): continue
    with np.load(p) as z: bvp, ts = z['bvp'].astype(float), z['ts'].astype(float)
    if len(bvp) < 300: continue
    q = np.arange(ts[0], ts[-1], 1 / 30.); x = signal.sosfiltfilt(BP, np.interp(q, ts, bvp))
    pp, _ = signal.find_peaks(x, distance=10, prominence=.3 * x.std()); ibi = np.diff(pp) / 30.; ibi = ibi[(ibi >= .33) & (ibi <= 1.5)]
    if len(ibi) < 8: continue
    cv_[i] = ibi.std() / ibi.mean(); rms[i] = np.sqrt(np.mean(np.diff(ibi) ** 2)) / ibi.mean(); hrb[i] = 60 / np.median(ibi)
grp = np.where(low == 1, 'low', np.where(conf, 'easy', np.where(okE & ~task, 'hard', 'other')))
print(MODEL, 'videos with a waveform:', int(np.isfinite(cv_).sum()))
for g in ('easy', 'hard', 'low'):
    m = (grp == g) & np.isfinite(cv_)
    if not m.any(): continue
    print(f"{g:5s} n={int(m.sum()):3d} | IBI CV median {np.median(cv_[m]):.3f} [{np.quantile(cv_[m], .25):.2f};{np.quantile(cv_[m], .75):.2f}] | RMSSD/mean {np.median(rms[m]):.3f} | share CV > 0.15 {np.mean(cv_[m] > .15):.2f}"
          f" | |err| vs label: beat HR {np.mean(np.abs(hrb[m] - lab[m])):5.2f}, {MODEL} spectral peak {np.mean(np.abs(pk[2][m] - lab[m])):5.2f}, consensus {np.mean(np.abs(med[m] - lab[m])):5.2f}")
hb_ok = np.isfinite(hrb); Zb = np.zeros_like(Z0); Zb[hb_ok] = Y.zlog(np.exp(-(GRID[None] - hrb[hb_ok][:, None]) ** 2 / (2 * 4. ** 2)) + 1e-6)
sub = okE & hb_ok
def cvp(Z):
    p = np.full(N, np.nan)
    for f in sorted(set(fold)):
        tr = np.flatnonzero((fold != f) & sub); te = np.flatnonzero((fold == f) & sub)
        m = Y.fitG(Z[tr], lab[tr], per[tr], low[tr], low[tr], task[tr]); p[te] = Y.predG(m, Z[te], low[te], low[te], task[te])
    return p
v4 = cvp((2 * Z0 + sum(ZE)) / 5); vb = cvp((2 * Z0 + sum(ZE) + Zb) / 6)
S_ = lambda a_, b_, m: Y.summarize(a_[m], lab[m], per[m], b_[m])
for name, m in (('all', sub), ('camera originals, rest', sub & ~task & (low == 0)), ('  - hard', sub & (grp == 'hard')), ('camera originals, Task', sub & task & (low == 0)), ('low-resolution files', sub & (low == 1))):
    a = S_(vb, v4, m); print(f"{name:24s} videos {int(m.sum()):3d} people {a['people']:3d} | v4 {S_(v4, Y.const, m)['MAE']:6.2f} | v4 + beat-rate evidence {a['MAE']:6.2f} ({a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}])")
np.savez_compressed(W / 'out/u18_btc_beats.npz', rec=Y.rec, cv=cv_, rmssd=rms, hr_beats=hrb, v4=v4, v4_beats=vb, sub=sub, grp=grp)
