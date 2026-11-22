"""z01 - 21 Nov research round, BTC dev only (no holdout), no model inference: cheap extensions of reader_v4 tested by person-disjoint CV.
reader_v4 = reader_v3 on Z4 = (2*Z_classical + Z_FacePhys + Z_MEchunk + Z_MEflow) / 5 (y11 'mix'), fitted on native training folds.
  E1 weights : Z = (a*Zc + sum_m b_m Z_m) / (a + sum b_m), a, b_m in {0, .5, 1, 2}; chosen INSIDE the training folds (inner CV).
  E2 context : protocol metadata that BTC ships for every video (group Normal/Abnormal, task Resting/Task1-3/Walking):
               (i) offsets per (class, task type) instead of (class, Task yes/no); (ii) prior per group; (iii) both.
               Each variant is compared with reader_v4 on the same videos, and also chosen by inner CV ('nested').
Person-first MAE, paired person bootstrap (10,000, seed 0); control: labels shuffled between people.
Usage: $PY code/z01_btc_reader_ext.py <workers>  -> logs/z01.log (stdout), out/z01_pred.npz"""
import sys, itertools
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
A = sys.argv[:]; sys.argv = ['x']
import y06_reader_quality as Q
from lib_btc import GRID, zlog, summarize, videos
import reader_models as rm
V = videos(); N = Q.N; lab, per, fold, low, task = Q.lab, Q.per, Q.fold, Q.low, Q.task; Zn, okn = Q.Zn, Q.okn
grp = np.array([v['group'] for v in V]); ttype = np.array([v['task'] for v in V])
G4 = np.array([('T' if t.startswith('Task') else ('W' if t == 'Walking' else g[:1])) for g, t in zip(grp, ttype)])
MODELS = ['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']
def zmodel(S):
    ok = np.isfinite(S).all(1); Z = np.zeros((N, len(GRID))); Z[ok] = zlog(S[ok]); return Z, ok
ZM = {}
for m in MODELS:
    with np.load(W / 'out/y05' / (m + '.npz')) as z: S = z['S'].copy()
    with np.load(W / 'out/y11/heavy' / (m + '.npz')) as z: h = np.isfinite(z['S']).all(1); S[h] = z['S'][h]
    ZM[m] = zmodel(S)
OK = okn & np.all([ZM[m][1] for m in MODELS], 0)
def mixw(w): a = w[0]; b = w[1:]; return (a * Zn + sum(bi * ZM[m][0] for bi, m in zip(b, MODELS))) / (a + sum(b))
Z4 = mixw((2, 1, 1, 1))
# ---------- reader with generic context: prior per group (optional), beta per class, offsets per (class, cell) ----------
def kde_or(L, pr, gsel, prior_all):
    return rm.kde(L[gsel], rm.pw(pr[gsel])) if len(set(pr[gsel])) >= rm.MIN_N else prior_all
def fitC(Z, L, pr, cl, cell, gp):
    prior_all = rm.kde(L, rm.pw(pr)); priors = {g: (kde_or(L, pr, gp == g, prior_all) if gp is not None else prior_all) for g in set(gp.tolist())} if gp is not None else {None: prior_all}
    P = np.array([priors[g if gp is not None else None] for g in (gp if gp is not None else [None] * len(L))])
    def pmed(Zs, Ps, b):
        lp = np.log(Ps + 1e-12) + b * Zs; q = np.exp(lp - lp.max(1, keepdims=True)); q /= q.sum(1, keepdims=True); return GRID[np.minimum((np.cumsum(q, 1) < .5).sum(1), len(GRID) - 1)]
    w = rm.pw(pr); fb = lambda i: rm.BETAS[int(np.argmin([np.sum(w[i] * np.abs(pmed(Z[i], P[i], b) - L[i])) for b in rm.BETAS]))]
    gb = fb(np.arange(len(L))); beta = {0: gb, 1: gb}
    for c in (0, 1):
        i = np.flatnonzero(cl == c)
        if len(i) >= rm.MIN_N: beta[c] = fb(i)
    p = np.array([pmed(Z[i:i + 1], P[i:i + 1], beta[cl[i]])[0] for i in range(len(L))]); r = L - p; off = {}
    for k in set(zip(cl.tolist(), cell.tolist())):
        s = (cl == k[0]) & (cell == k[1])
        off[k] = float(np.sum(w[s] * r[s]) / np.sum(w[s])) if s.sum() >= rm.MIN_N else 0.
    return dict(priors=priors, beta=beta, off=off, pmed=pmed, usegp=gp is not None)
def predC(m, Z, cl, cell, gp):
    out = np.zeros(len(Z))
    for i in range(len(Z)):
        P = m['priors'].get(gp[i] if m['usegp'] else None, next(iter(m['priors'].values())))
        out[i] = m['pmed'](Z[i:i + 1], P[None], m['beta'][cl[i]])[0] + m['off'].get((cl[i], cell[i]), 0.)
    return out
VARS = {'v4': (task.astype(int), None), 'off_ttype': (np.array([{'Resting': 0, 'Task1': 1, 'Task2': 2, 'Task3': 3, 'Walking': 4}[t] for t in ttype]), None),
        'prior_g4': (task.astype(int), G4), 'off_ttype+prior_g4': (np.array([{'Resting': 0, 'Task1': 1, 'Task2': 2, 'Task3': 3, 'Walking': 4}[t] for t in ttype]), G4),
        'off_g4': (np.array([{'N': 0, 'A': 1, 'T': 2, 'W': 3}[g] for g in G4]), None)}
def cvC(Z, L, var, sub, rows=None):
    cell, gp = VARS[var]; p = np.full(N, np.nan)
    for f in sorted(set(fold)):
        tr = np.flatnonzero((fold != f) & sub); te = np.flatnonzero((fold == f) & sub)
        m = fitC(Z[tr], L[tr], per[tr], low[tr], cell[tr], None if gp is None else gp[tr]); p[te] = predC(m, Z[te], low[te], cell[te], None if gp is None else gp[te])
    return p
def inner(Z, L, var, tr):
    cell, gp = VARS[var]; p = np.zeros(len(tr)); fo = fold[tr]
    for g in sorted(set(fo)):
        a = tr[fo != g]; b = tr[fo == g]; m = fitC(Z[a], L[a], per[a], low[a], cell[a], None if gp is None else gp[a]); p[fo == g] = predC(m, Z[b], low[b], cell[b], None if gp is None else gp[b])
    w = rm.pw(per[tr]); return float(np.sum(w * np.abs(p - L[tr])) / w.sum())
WGRID = [w for w in itertools.product([0, .5, 1, 2], repeat=4) if w[0] + sum(w[1:]) > 0 and w[0] > 0]
def task_w(a):
    f, w, L = a; tr = np.flatnonzero((fold != f) & OK); return f, w, inner(mixw(w), L, 'v4', tr)
def task_v(a):
    f, v, L = a; tr = np.flatnonzero((fold != f) & OK); return f, v, inner(Z4, L, v, tr)
def run(L, pool, tag):
    S = lambda a_, b_: summarize(a_[OK], L[OK], per[OK], b_[OK])
    base = cvC(Z4, L, 'v4', OK); chk = Q.cv([('seen', 'seen')], OK, Z4, Q.qn, low, Z4, Q.qn, low, L, {})[0]
    print(f'[{tag}] self-check generic reader vs y06 reader on Z4: max |diff| {np.nanmax(np.abs(base[OK] - chk[OK])):.2e}', flush=True)
    b = S(base, Q.const); print(f"[{tag}] reader_v4 n={int(OK.sum())}: MAE {b['MAE']:.3f} RMSE {b['RMSE']:.2f} r {b['r']:+.3f} | vs const {b['delta']:+.2f} [{b['dCI'][0]:+.2f};{b['dCI'][1]:+.2f}]", flush=True)
    out = {'v4': base}
    for v in VARS:
        if v == 'v4': continue
        p = cvC(Z4, L, v, OK); s = S(p, base); out[v] = p
        print(f"[{tag}] {v:22s} MAE {s['MAE']:.3f} RMSE {s['RMSE']:.2f} r {s['r']:+.3f} | vs v4 {s['delta']:+.3f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}]", flush=True)
    RES = pool.map(task_v, [(f, v, L) for f in sorted(set(fold)) for v in VARS], chunksize=1); best = {}
    for f in sorted(set(fold)):
        ls = {v: l for ff, v, l in RES if ff == f}; k = min(ls, key=ls.get); best[f] = k if ls[k] < ls['v4'] - .02 else 'v4'
    p = np.full(N, np.nan)
    for f in sorted(set(fold)): te = np.flatnonzero((fold == f) & OK); p[te] = out[best[f]][te]
    s = S(p, base); out['nested_context'] = p; print(f"[{tag}] nested context choice {[best[f] for f in sorted(best)]} MAE {s['MAE']:.3f} | vs v4 {s['delta']:+.3f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}]", flush=True)
    RES = pool.map(task_w, [(f, w, L) for f in sorted(set(fold)) for w in WGRID], chunksize=4); bw = {}
    for f in sorted(set(fold)):
        ls = {w: l for ff, w, l in RES if ff == f}; bw[f] = min(ls, key=ls.get)
    p = np.full(N, np.nan)
    for f in sorted(set(fold)):
        tr = np.flatnonzero((fold != f) & OK); te = np.flatnonzero((fold == f) & OK); Zw = mixw(bw[f]); cell, gp = VARS['v4']
        p[te] = predC(fitC(Zw[tr], L[tr], per[tr], low[tr], cell[tr], None), Zw[te], low[te], cell[te], None)
    s = S(p, base); out['nested_weights'] = p; print(f"[{tag}] nested weights (a, FacePhys, MEchunk, MEflow) per fold {[bw[f] for f in sorted(bw)]} MAE {s['MAE']:.3f} RMSE {s['RMSE']:.2f} | vs v4 {s['delta']:+.3f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}]", flush=True)
    return out
if __name__ == '__main__':
    with Pool(int(A[1])) as pool:
        P = run(lab, pool, 'real')
        for g in ('N', 'A', 'T', 'W'):
            s = G4 == g
            for k in ('v4', 'off_ttype', 'prior_g4', 'nested_context', 'nested_weights'):
                sm = summarize(P[k][OK & s], lab[OK & s], per[OK & s])
                print(f"   group {g} n={int((OK & s).sum()):3d} {k:16s} MAE {sm['MAE']:.3f}", flush=True)
        rng = np.random.default_rng(1); ppl = np.array(sorted(set(per))); perm = dict(zip(ppl, rng.permutation(ppl))); Ls = lab.copy()
        for q in ppl:
            i = np.flatnonzero(per == q); Ls[i] = np.resize(lab[per == perm[q]], len(i))
        run(Ls, pool, 'shuffled')
    np.savez_compressed(W / 'out/z01_pred.npz', lab=lab, per=per, ok=OK, **P)
