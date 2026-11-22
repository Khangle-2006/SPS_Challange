"""u20 (22 Nov, Claude): one fixed reader variant from u19 - weight each reader by its label-free TEMPORAL STABILITY.
w_r = 1 / (1 + SD_r / 10 bpm), SD_r = SD of the reader's window peaks within the video (CHROM, POS from y01 windows; FacePhys from the u17
waveform with the same windows/operator); ME-chunk / ME-flow keep weight 1 (no window spectra stored). Z = sum w_r z_r / sum w_r.
tau = 10 bpm fixed before any result. Reader fits on training folds; accept only if it beats v4 on all videos with paired CI < 0."""
import sys, pickle, json
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
sys.path.insert(0, str(W.parent / 'claude_research_20261118/code'))
from lib_btc import window_spectrum, supported
lab, per, low, task, fold, N, GRID = Y.lab, Y.per, Y.low, Y.task, Y.fold, Y.N, Y.GRID
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb'))
ZE = {}; okE = np.ones(N, bool)
for n in ['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']:
    with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
    h = np.isfinite(S).all(1); okE &= h; Zm = np.zeros((N, len(GRID))); Zm[h] = Y.zlog(S[h]); ZE[n] = Zm
zc = np.zeros((N, len(GRID))); zp = np.zeros_like(zc); sd = np.full((3, N), np.nan); okW = np.zeros(N, bool)
for i, k in enumerate(keys):
    if not Sp[i]['chrom']: continue
    C = [w[0] for w in Sp[i]['chrom']]; P = [w[0] for w in Sp[i]['pos']]; zc[i] = Y.zlog(np.mean(C, 0)); zp[i] = Y.zlog(np.mean(P, 0))
    sd[0, i] = np.std([GRID[x.argmax()] for x in C]); sd[1, i] = np.std([GRID[x.argmax()] for x in P])
    p = W / 'out/u17/FacePhys.rlap' / (k + '.npz')
    if not p.exists() or not okE[i]: continue
    with np.load(W / 'out/x10' / (k + '.npz')) as z: t, valid = z['t'].astype(float), z['valid'].astype(bool)
    with np.load(p) as z: bvp, ts = z['bvp'].astype(float), z['ts'].astype(float)
    ts = ts - ts[0]; F = []
    for s in np.arange(0, max(0., t[-1] - 10 + 1e-8), 5.):
        if not supported(t, valid, s, s + 10) or s + 10 > ts[-1] + .2: continue
        w = np.interp(s + np.arange(300) / 30., ts, bvp)
        if np.isfinite(w).all() and w.std() > 1e-12: F.append(window_spectrum(w))
    if len(F) >= 2: sd[2, i] = np.std([GRID[x.argmax()] for x in F]); okW[i] = True
sub = okE & okW & np.isfinite(sd).all(0)
wt = 1. / (1. + sd / 10.)
Zv4 = (zc + zp + ZE['FacePhys.rlap'] + ZE['ME-chunk.pure'] + ZE['ME-flow.pure']) / 5
Zst = (wt[0][:, None] * zc + wt[1][:, None] * zp + wt[2][:, None] * ZE['FacePhys.rlap'] + ZE['ME-chunk.pure'] + ZE['ME-flow.pure']) / (wt[0] + wt[1] + wt[2] + 2)[:, None]
def cv(Z):
    p = np.full(N, np.nan)
    for f in sorted(set(fold)):
        tr = np.flatnonzero((fold != f) & sub); te = np.flatnonzero((fold == f) & sub)
        m = Y.fitG(Z[tr], lab[tr], per[tr], low[tr], low[tr], task[tr]); p[te] = Y.predG(m, Z[te], low[te], low[te], task[te])
    return p
v4 = cv(Zv4); vs = cv(Zst)
rng = np.random.default_rng(0); wsh = wt.copy(); idx = np.flatnonzero(sub); wsh[:, idx] = wt[:, rng.permutation(idx)]
Zsh = (wsh[0][:, None] * zc + wsh[1][:, None] * zp + wsh[2][:, None] * ZE['FacePhys.rlap'] + ZE['ME-chunk.pure'] + ZE['ME-flow.pure']) / (wsh[0] + wsh[1] + wsh[2] + 2)[:, None]; vsh = cv(Zsh)
S_ = lambda a_, b_, m: Y.summarize(a_[m], lab[m], per[m], b_[m])
d = np.load(W / 'out/u18_btc_beats.npz', allow_pickle=True); grp = d['grp']
for name, m in (('all', sub), ('camera originals, rest', sub & ~task & (low == 0)), ('  - hard', sub & (grp == 'hard')), ('camera originals, Task', sub & task & (low == 0)), ('low-resolution files', sub & (low == 1))):
    a = S_(vs, v4, m); c = S_(vsh, v4, m); print(f"{name:24s} videos {int(m.sum()):3d} people {a['people']:3d} | v4 {S_(v4, Y.const, m)['MAE']:6.2f} | stability-weighted {a['MAE']:6.2f} ({a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}]) | weights shuffled {c['MAE']:6.2f} ({c['delta']:+.2f})")
np.savez_compressed(W / 'out/u20_btc_stability_reader.npz', rec=Y.rec, sub=sub, v4=v4, stab=vs, stab_shuffled=vsh, sd=sd)
