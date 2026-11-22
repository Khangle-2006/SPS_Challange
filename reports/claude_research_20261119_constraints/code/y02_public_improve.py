"""y02 - public sets, native (UBFC2 dev, MCD): candidates (5) and (6) of PREREG.md section 4 on the 18 Nov estimator (fusion + Viterbi).
  base : fused emission E_t(f) = sum_k w_k z_k,t(f); Viterbi path over the windows of a recording (b04 / x05)                    [current]
  FB   : same HMM, forward-backward marginals instead of the single best path; point estimate = marginal median or mean
  Q    : window quality weight  E_t <- (peak_t / mean peak)^gamma * E_t, peak_t = max_f E_t(f), gamma in {0, 1, 2}
All of (sigma, tau, estimator, gamma) are fitted on the 4 training folds, per dataset; weights w are those of the base fit.
Input: spectra file with pad/pos/chrom per window (default: 18 Nov out/public_spectra.npz, cond == 'native'); any x02 tag also works.
Person-first MAE and window RMSE, paired bootstrap over people (10,000, seed 0).  Usage: $PY code/y02_public_improve.py <workers> [x02tag cond]"""
import sys, itertools
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code'))
import x05_public_score as X
BPM = X.BPM; DS = X.DS; KER = {s: np.exp(X.PEN[s]) for s in X.SIG}; GAM = [0., 1., 2.]
def fb(E, sigma, tau, est):
    """forward-backward on the same random-walk HMM as the Viterbi tracker; returns the marginal median/mean per window."""
    T, G = E.shape; e = np.exp(tau * (E - E.max(1, keepdims=True))); K = KER[sigma]; a = np.zeros((T, G)); b = np.ones((T, G)); a[0] = e[0] / e[0].sum()
    for t in range(1, T): v = (a[t - 1] @ K) * e[t]; a[t] = v / v.sum()
    for t in range(T - 2, -1, -1): v = K @ (b[t + 1] * e[t + 1]); b[t] = v / v.sum()
    g = a * b; g /= g.sum(1, keepdims=True)
    return g @ BPM if est == 'mean' else BPM[np.minimum((np.cumsum(g, 1) < .5).sum(1), G - 1)]
def qscale(E, gamma, ref):
    return E if gamma == 0 else E * ((E.max(1) / ref) ** gamma)[:, None]
def task(a):
    f, target = a; d = DATA; Z = d['Z']; w, cfg = X.fitp(d, Z, f); fus, trk = X.apply(d, Z, f, w, cfg); out = {'base': trk}; info = {'w': w, 'viterbi': cfg}
    for D in DS:
        tr = np.flatnonzero((d['ds'] == D) & (d['fold'] != f)); te = np.flatnonzero((d['ds'] == D) & (d['fold'] == f))
        if not len(tr) or not len(te): continue
        rtr = X.runs_of(d, tr); Etr = [np.tensordot(np.array(w), Z[:, r], 1) for r in rtr]; ref = float(np.mean(np.concatenate([E.max(1) for E in Etr])))
        loss = (lambda e, pp: X.pmae(np.abs(e), pp)[1].mean()) if target == 'MAE' else (lambda e, pp: float(np.sqrt(np.mean(np.square(e)))))
        for name, gams, ests in (('FB', [0.], ['median', 'mean']), ('FB+Q', GAM, ['median', 'mean']), ('Viterbi+Q', GAM, ['path'])):
            best = (1e9, None)
            for g in gams:
                Eg = [qscale(E, g, ref) for E in Etr]
                for s in X.SIG:
                    for ta in X.TAU:
                        for est in ests:
                            e = []; pp = []
                            for r, Er in zip(rtr, Eg): e.extend((X.viterbi(Er, s, ta) if est == 'path' else fb(Er, s, ta, est)) - d['truth'][r]); pp.extend(d['person'][r])
                            m = loss(np.array(e), pp)
                            if m < best[0] - 1e-9: best = (m, (g, s, ta, est))
            g, s, ta, est = best[1]; info[(name, D)] = best[1]; o = out.setdefault(name, {})
            for r in X.runs_of(d, te):
                Er = qscale(np.tensordot(np.array(w), Z[:, r], 1), g, ref); o.update(zip(r.tolist(), X.viterbi(Er, s, ta) if est == 'path' else fb(Er, s, ta, est)))
    return f, target, out, info
DATA = None
if __name__ == '__main__':
    if len(sys.argv) > 3:
        DATA = X.load(sys.argv[2], [sys.argv[3]])[sys.argv[3]]
    else:
        with np.load(W.parent / 'claude_research_20261118/out/public_spectra.npz') as z:
            m = z['cond'] == 'native'; DATA = dict(ds=z['dataset'][m], person=z['person'][m], fold=z['fold'][m], rec=z['recording'][m], start=z['start'][m], truth=z['truth'][m], Z=np.stack([X.zs(z[k][m]) for k in ('pad', 'pos', 'chrom')]))
    d = DATA
    import os
    if os.environ.get('SHUFFLE'):                       # control: every person receives the reference HR sequence of another person of the same set
        rng = np.random.default_rng(int(os.environ['SHUFFLE'])); tr = d['truth'].copy()
        for D in DS:
            ppl = sorted(set(d['person'][d['ds'] == D])); perm = dict(zip(ppl, rng.permutation(ppl)))
            for q in ppl:
                i = np.flatnonzero(d['person'] == q); src = d['truth'][d['person'] == perm[q]]; tr[i] = np.resize(src, len(i))
        d['truth'] = tr; print('CONTROL: reference HR shuffled between people, seed', os.environ['SHUFFLE'])
    with Pool(int(sys.argv[1])) as pool: RES = pool.map(task, [(f, t) for t in ('MAE', 'RMSE') for f in range(5)], chunksize=1)
    for target in ('MAE', 'RMSE'):
        P = defaultdict(dict); print(f'\n##### parameters chosen on the training folds by {target}')
        for f, t, out, info in RES:
            if t != target: continue
            for k, v in out.items(): P[k].update(v)
            print('  fold', f, {k: v for k, v in info.items() if k != 'w'})
        for D in DS:
            sel = np.flatnonzero(d['ds'] == D); t = d['truth'][sel]; per = d['person'][sel]; ppl = sorted(set(per)); ix = np.random.default_rng(0).integers(0, len(ppl), (10000, len(ppl)))
            base = np.array([P['base'][i] for i in sel.tolist()]); _, vb = X.pmae(np.abs(base - t), per)
            sq = lambda q: np.array([np.sum((q - t)[per == p] ** 2) for p in ppl]); cnt = np.array([np.sum(per == p) for p in ppl]); sb = sq(base)
            print(f'  {D}: {len(sel)} windows / {len(ppl)} people')
            for k in P:
                q = np.array([P[k][i] for i in sel.tolist()]); _, v = X.pmae(np.abs(q - t), per); dd = v - vb; ci = np.quantile(dd[ix].mean(1), [.025, .975]); s1 = sq(q)
                dr = np.sqrt(s1[ix].sum(1) / cnt[ix].sum(1)) - np.sqrt(sb[ix].sum(1) / cnt[ix].sum(1)); cr = np.quantile(dr, [.025, .975])
                print(f"     {k:10s} MAE {v.mean():6.3f} RMSE {np.sqrt(np.mean((q - t) ** 2)):6.3f} r {np.corrcoef(q, t)[0, 1]:.4f} | dMAE {dd.mean():+.3f} [{ci[0]:+.3f};{ci[1]:+.3f}] dRMSE {np.sqrt(s1.sum() / cnt.sum()) - np.sqrt(sb.sum() / cnt.sum()):+.3f} [{cr[0]:+.3f};{cr[1]:+.3f}]")
