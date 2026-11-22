"""u10 (22 Nov, Claude): label-free AGREEMENT between the 5 readers as reader input, for the hard half of the BTC camera originals.
Fixed before any result (two variants, nothing else is tried):
  A  beta by agreement class: reader_v4 evidence (equal weights), but beta fitted separately for camera originals with >=4 / 3 / <=2 of
     the 5 reader peaks (CHROM, POS, FacePhys.rlap, ME-chunk.pure, ME-flow.pure) within 3 bpm of their median (label-free), plus one beta
     for the low-resolution files; offsets per (capture class, Task) cell as in reader_v3. Classes with < 12 training videos use the pooled beta.
  B  A + consensus weights: each reader's standardised log-spectrum weighted by 1 + (number of other readers within 3 bpm of its peak).
Reader fits on training folds only. Accept a variant only if it beats v4 on all 201 videos with paired CI < 0.
Control: agreement classes shuffled among the camera originals."""
import sys, pickle, json
from pathlib import Path
from collections import defaultdict
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
lab, per, low, task, fold, N, GRID, rm = Y.lab, Y.per, Y.low, Y.task, Y.fold, Y.N, Y.GRID, Y.rm
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); ZS = np.zeros((5, N, len(GRID))); ok = np.ones(N, bool)
for i, s in enumerate(Sp):
    if s['chrom']: ZS[0, i] = Y.zlog(np.mean([w[0] for w in s['chrom']], 0)); ZS[1, i] = Y.zlog(np.mean([w[0] for w in s['pos']], 0))
    else: ok[i] = False
for j, n in enumerate(['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']):
    with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
    h = np.isfinite(S).all(1); ok &= h; ZS[2 + j, h] = Y.zlog(S[h])
pk = GRID[ZS.argmax(2)]; med = np.median(pk, 0); agree = (np.abs(pk - med) <= 3).sum(0)
nag = np.stack([(np.abs(pk - pk[s]) <= 3).sum(0) - 1 for s in range(5)])
acls = np.where(low == 1, 3, np.where(agree >= 4, 0, np.where(agree == 3, 1, 2)))
Zeq = ZS.mean(0); wB = 1. + nag; Zcw = (wB[..., None] * ZS).sum(0) / wB.sum(0)[:, None]
def fitM(Z, L, pr, cb, co, tk):
    prior = rm.kde(L, rm.pw(pr)); gb = Y.fit_beta(Z, L, pr, prior); beta = defaultdict(lambda: gb)
    for c in np.unique(cb):
        i = np.flatnonzero(cb == c)
        if len(i) >= rm.MIN_N: beta[c] = Y.fit_beta(Z[i], L[i], pr[i], prior)
    m = dict(prior=prior, beta=dict(beta), gb=gb, off={k: 0. for k in range(4)}); r = L - predM(m, Z, cb, co, tk); w = rm.pw(pr); cell = co * 2 + tk.astype(int)
    for k in range(4):
        s = cell == k
        if s.sum() >= rm.MIN_N: m['off'][k] = float(np.sum(w[s] * r[s]) / np.sum(w[s]))
    return m
def predM(m, Z, cb, co, tk):
    p = np.zeros(len(Z))
    for c in np.unique(cb):
        i = np.flatnonzero(cb == c); p[i] = Y.pmedv(Z[i], m['prior'], m['beta'].get(c, m['gb']))
    return p + np.array([m['off'][k] for k in co * 2 + tk.astype(int)])
def cv(Z, cb):
    p = np.full(N, np.nan)
    for f in sorted(set(fold)):
        tr = np.flatnonzero((fold != f) & ok); te = np.flatnonzero((fold == f) & ok)
        m = fitM(Z[tr], lab[tr], per[tr], cb[tr], low[tr], task[tr]); p[te] = predM(m, Z[te], cb[te], low[te], task[te])
    return p
v4 = cv(Zeq, low.copy()); A = cv(Zeq, acls); B = cv(Zcw, acls)
rng = np.random.default_rng(0); sh = acls.copy(); cam = np.flatnonzero((low == 0) & ok); sh[cam] = acls[rng.permutation(cam)]; As = cv(Zeq, sh)
print('agreement classes on the 201 videos (camera originals >=4 / 3 / <=2, low files):', [int(((acls == c) & ok).sum()) for c in range(4)])
S_ = lambda a_, b_, m: Y.summarize(a_[m], lab[m], per[m], b_[m])
rest = ~task & (low == 0)
for name, m in (('all', ok), ('camera originals, rest', ok & rest), ('  - confident (>=4)', ok & rest & (acls == 0)), ('  - not confident (<=3)', ok & rest & (acls != 0)), ('camera originals, Task', ok & task & (low == 0)), ('low-resolution files', ok & (low == 1))):
    a = S_(A, v4, m); b = S_(B, v4, m); c = S_(As, v4, m); k = S_(v4, Y.const, m)
    print(f"{name:26s} videos {int(m.sum()):3d} people {a['people']:3d} | v4 {k['MAE']:6.2f} | A {a['MAE']:6.2f} ({a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}]) | B {b['MAE']:6.2f} ({b['delta']:+.2f} [{b['dCI'][0]:+.2f};{b['dCI'][1]:+.2f}]) | A shuffled {c['MAE']:6.2f} ({c['delta']:+.2f})")
np.savez_compressed(W / 'out/u10_btc_agreement.npz', rec=Y.rec, lab=lab, per=per, ok=ok, v4=v4, A=A, B=B, A_shuffled=As, acls=acls)
