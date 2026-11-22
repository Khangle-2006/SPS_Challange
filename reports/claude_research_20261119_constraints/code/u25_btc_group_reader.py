"""u25 (22 Nov, Claude): group-aware reader. The BTC release separates 'Normal' volunteers and 'Abnormal' patients (folder name, file
prefix S_N / S_AB), the same kind of file-name information the reader already uses for Task. The hard half of the resting camera
originals is 86% patients, so reliability of the evidence (beta) and the prior may differ by group (literature 22/11, Q1d/Q2).
Fixed before any result (two variants, nothing else is tried):
  G1  reader_v4 evidence (equal weights, as u10), beta per (capture class x group), offsets per (capture class, Task) as reader_v3.
  G2  G1 + prior per group (KDE of the training-fold labels of that group).
Reader fits on training folds only. Accept a variant only if it beats v4 on all 201 videos with paired CI < 0.
Control: group labels permuted between people (G2 with shuffled groups). Assumption to state in the report: the test release keeps
the Normal / Abnormal folders; otherwise the pooled reader is used."""
import sys, pickle, json
from pathlib import Path
from collections import defaultdict
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
lab, per, low, task, fold, N, GRID, rm = Y.lab, Y.per, Y.low, Y.task, Y.fold, Y.N, Y.GRID, Y.rm
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]; ab = np.array([k.split('_')[2].startswith('AB') for k in keys]).astype(int)
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); ZS = np.zeros((5, N, len(GRID))); ok = np.ones(N, bool)
for i, s in enumerate(Sp):
    if s['chrom']: ZS[0, i] = Y.zlog(np.mean([w[0] for w in s['chrom']], 0)); ZS[1, i] = Y.zlog(np.mean([w[0] for w in s['pos']], 0))
    else: ok[i] = False
for j, n in enumerate(['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']):
    with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
    h = np.isfinite(S).all(1); ok &= h; ZS[2 + j, h] = Y.zlog(S[h])
Zeq = ZS.mean(0)
def fitM(Z, L, pr, cb, co, tk, g, gprior):
    w = rm.pw(pr); P0 = rm.kde(L, w); pri = {0: P0, 1: P0}
    if gprior:
        for v in (0, 1):
            s = g == v
            if s.sum() >= rm.MIN_N: pri[v] = rm.kde(L[s], w[s])
    gb = Y.fit_beta(Z, L, pr, P0); beta = {}
    for c in np.unique(cb):
        i = np.flatnonzero(cb == c)
        if len(i) >= rm.MIN_N:
            gi = g[i]; vals = [Y.fit_beta(Z[i][gi == v], L[i][gi == v], pr[i][gi == v], pri[v]) if (gi == v).sum() >= rm.MIN_N else None for v in (0, 1)]
            # one beta per class: fitted with the prior each row will use
            best = None; bl = np.inf
            for b in rm.BETAS:
                e = 0.
                for v in (0, 1):
                    s = gi == v
                    if s.any(): e += np.sum(w[i][s] * np.abs(Y.pmedv(Z[i][s], pri[v], b) - L[i][s]))
                if e < bl: bl, best = e, b
            beta[c] = best
    m = dict(pri=pri, beta=beta, gb=gb, off={k: 0. for k in range(4)}); r = L - predM(m, Z, cb, co, tk, g); cell = co * 2 + tk.astype(int)
    for k in range(4):
        s = cell == k
        if s.sum() >= rm.MIN_N: m['off'][k] = float(np.sum(w[s] * r[s]) / np.sum(w[s]))
    return m
def predM(m, Z, cb, co, tk, g):
    p = np.zeros(len(Z))
    for c in np.unique(cb):
        for v in (0, 1):
            i = np.flatnonzero((cb == c) & (g == v))
            if len(i): p[i] = Y.pmedv(Z[i], m['pri'][v], m['beta'].get(c, m['gb']))
    return p + np.array([m['off'][k] for k in co * 2 + tk.astype(int)])
def cv(cb, g, gprior):
    p = np.full(N, np.nan)
    for f in sorted(set(fold)):
        tr = np.flatnonzero((fold != f) & ok); te = np.flatnonzero((fold == f) & ok)
        m = fitM(Zeq[tr], lab[tr], per[tr], cb[tr], low[tr], task[tr], g[tr], gprior); p[te] = predM(m, Zeq[te], cb[te], low[te], task[te], g[te])
    return p
zero = np.zeros(N, int)
v4 = cv(low.copy(), zero, False); G1 = cv(low * 2 + ab, ab, False); G2 = cv(low * 2 + ab, ab, True)
rng = np.random.default_rng(0); P = np.array(sorted(set(per))); pab = {p: ab[per == p][0] for p in P}; perm = dict(zip(P, rng.permutation([pab[p] for p in P]))); abs_ = np.array([perm[p] for p in per])
Gs = cv(low * 2 + abs_, abs_, True)
print('videos in the reader subset', int(ok.sum()), '| Abnormal', int((ok & (ab == 1)).sum()), '| Normal', int((ok & (ab == 0)).sum()))
S_ = lambda a_, b_, m: Y.summarize(a_[m], lab[m], per[m], b_[m])
rest = ~task & (low == 0)
for name, m in (('all', ok), ('camera originals, rest', ok & rest), ('  - Abnormal', ok & rest & (ab == 1)), ('  - Normal', ok & rest & (ab == 0)), ('camera originals, Task', ok & task & (low == 0)), ('low-resolution files', ok & (low == 1))):
    a = S_(G1, v4, m); b = S_(G2, v4, m); c = S_(Gs, v4, m); k = S_(v4, Y.const, m)
    print(f"{name:24s} videos {int(m.sum()):3d} people {a['people']:3d} | v4 {k['MAE']:6.2f} | G1 {a['MAE']:6.2f} ({a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}]) | G2 {b['MAE']:6.2f} ({b['delta']:+.2f} [{b['dCI'][0]:+.2f};{b['dCI'][1]:+.2f}]) | G2 shuffled groups {c['MAE']:6.2f} ({c['delta']:+.2f})")
np.savez_compressed(W / 'out/u25_btc_group_reader.npz', rec=Y.rec, ok=ok, ab=ab, v4=v4, G1=G1, G2=G2, G2_shuffled=Gs)
