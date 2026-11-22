"""z10 - BTC dev: does the 31 % LOW class (re-encoded files) gain from different evidence or from a beta per (class, Task) cell?
Observed (z07, peak within 5 bpm on the low class): CHROM 0.284, POS 0.224, CHROM+POS 0.194, random about 0.17-0.20; ME-chunk.pure 0.284
(y05 log). Variants, all with reader_v4 otherwise unchanged (fitted on the training folds):
   low class evidence: Z4 (current) | CHROM + experts | experts only | ME-chunk.pure only | CHROM only
   beta per (class, Task) cell instead of per class
and a nested choice by inner CV. Person-first MAE, paired person bootstrap.
Usage: $PY code/z10_btc_lowclass_evidence.py <workers>"""
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
A = sys.argv[:]; sys.argv = ['x']
import y06_reader_quality as Q
import z01_btc_reader_ext as E
from lib_btc import zlog, GRID, summarize
N = Q.N; lab, per, fold, low, task = Q.lab, Q.per, Q.fold, Q.low, Q.task; OK = E.OK
z = np.load(W / 'out/z02_windows.npz', allow_pickle=True); WIN = list(z['win'])
zc = np.zeros((N, len(GRID))); zp = np.zeros((N, len(GRID)))
for i, w in enumerate(WIN):
    if w is not None: C, P, S, T = w; zc[i], zp[i] = zlog(C.mean(0)), zlog(P.mean(0))
EX = {m: E.ZM[m][0] for m in E.MODELS}; Z4 = (2 * Q.Zn + sum(EX.values())) / 5
LOWZ = {'Z4 (current)': Z4, 'CHROM + experts': (2 * zc + sum(EX.values())) / 5, 'experts only': sum(EX.values()) / 3, 'ME-chunk.pure only': EX['ME-chunk.pure'], 'CHROM only': zc}
def zfor(k): return np.where((low == 1)[:, None], LOWZ[k], Z4)
def cv_beta4(Z, L, sub):
    """beta per (class, Task) cell: generic reader with class vector cl = 2*low + task (offsets still per (low, task))."""
    cl = 2 * low + task.astype(int); p = np.full(N, np.nan)
    for f in sorted(set(fold)):
        tr = np.flatnonzero((fold != f) & sub); te = np.flatnonzero((fold == f) & sub); m = fitB(Z[tr], L[tr], per[tr], cl[tr], low[tr] * 2 + task[tr].astype(int))
        p[te] = predB(m, Z[te], cl[te], low[te] * 2 + task[te].astype(int))
    return p
import reader_models as rm
def fitB(Z, L, pr, cl, cell):
    prior = rm.kde(L, rm.pw(pr)); w = rm.pw(pr); fb = lambda i: rm.BETAS[int(np.argmin([np.sum(w[i] * np.abs(Q.pmedv(Z[i], prior, b) - L[i])) for b in rm.BETAS]))]
    gb = fb(np.arange(len(L))); beta = {c: gb for c in range(4)}
    for c in range(4):
        i = np.flatnonzero(cl == c)
        if len(i) >= rm.MIN_N: beta[c] = fb(i)
    m = dict(prior=prior, beta=beta, off={}); p = predB(m, Z, cl, cell); r = L - p
    for c in range(4):
        s = cell == c; m['off'][c] = float(np.sum(w[s] * r[s]) / np.sum(w[s])) if s.sum() >= rm.MIN_N else 0.
    return m
def predB(m, Z, cl, cell):
    p = np.zeros(len(Z))
    for c in range(4):
        i = np.flatnonzero(cl == c)
        if len(i): p[i] = Q.pmedv(Z[i], m['prior'], m['beta'][c])
    return p + np.array([m['off'].get(c, 0.) for c in cell])
def inner(a):
    f, k = a; tr = np.flatnonzero((fold != f) & OK); return f, k, E.inner(zfor(k), lab, 'v4', tr)
if __name__ == '__main__':
    base = E.cvC(Z4, lab, 'v4', OK); P = {}
    for k in LOWZ:
        p = E.cvC(zfor(k), lab, 'v4', OK); P[k] = p; s = summarize(p[OK], lab[OK], per[OK], base[OK]); sl = summarize(p[OK & (low == 1)], lab[OK & (low == 1)], per[OK & (low == 1)], base[OK & (low == 1)])
        hit = np.mean(np.abs(GRID[LOWZ[k][OK & (low == 1)].argmax(1)] - lab[OK & (low == 1)]) <= 5)
        print(f"low-class evidence {k:20s} MAE {s['MAE']:.3f} | vs v4 {s['delta']:+.3f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}] | low class {sl['MAE']:.3f} vs v4 {sl['delta']:+.3f} [{sl['dCI'][0]:+.2f};{sl['dCI'][1]:+.2f}] | low-class peak hit {hit:.3f}", flush=True)
    with Pool(int(A[1])) as pool: R = pool.map(inner, [(f, k) for f in sorted(set(fold)) for k in LOWZ])
    p = base.copy(); ch = []
    for f in sorted(set(fold)):
        ls = {k: l for ff, k, l in R if ff == f}; k = min(ls, key=ls.get); k = k if ls[k] < ls['Z4 (current)'] - .02 else 'Z4 (current)'; ch.append(k); te = np.flatnonzero((fold == f) & OK); p[te] = P[k][te]
    s = summarize(p[OK], lab[OK], per[OK], base[OK]); print(f"nested low-class evidence: {ch} MAE {s['MAE']:.3f} | vs v4 {s['delta']:+.3f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}]", flush=True)
    pb = cv_beta4(Z4, lab, OK); s = summarize(pb[OK], lab[OK], per[OK], base[OK]); print(f"beta per (class, Task) cell: MAE {s['MAE']:.3f} RMSE {s['RMSE']:.2f} | vs v4 {s['delta']:+.3f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}]", flush=True)
