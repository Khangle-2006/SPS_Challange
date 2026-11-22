"""u05 (22 Nov, Claude): LEARNED spectral fusion for the public sets, in place of the fixed weight grid of z13.
z13 fuses the standardised log-spectra of 4 channels (0.1 s pre-filtered team CNN, POS, CHROM, mean of the 3 open-rppg experts) with ONE
weight per channel. u05 scores every candidate frequency f of a window with a conditional-logit (softmax over the 277 frequency bins)
on 16 features: for each channel z(f), z(2f), z(f/2) (harmonic context) and z(f) x peak height of that channel in that window
(a channel that is peaky in this window is trusted more). Trained on the TRAINING folds only (both datasets, each dataset and each person
weighted equally), target = Gaussian label distribution around the reference HR (sigma 1.5 bpm; label-distribution learning as in
PhysFormer, Yu et al. CVPR 2022), L2 1e-3, L-BFGS. The window scores then go through the SAME forward-backward readout as z13
(gamma, sigma, tau, estimator fitted on the training folds).
  L4n : learned, fitted on native windows       (compare with N4m)
  L4p : learned, fitted on the pooled 7-condition set of z13 (compare with P4m, the frozen public reader candidate)
Rule written before any result: L4p replaces P4m only if MCD native is better with paired CI < 0, no condition x dataset row is worse
with CI > 0 (UBFC2 rows: a difference within +-0.05 bpm is treated as no change), and no gate is lost.
Usage: $PY code/u05_public_learned_fusion.py   -> logs/u05.log, out/u05_public_learned.npz"""
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code'))
import z13_public_p4_prefilter as Z
X, P7, Y2 = Z.X, Z.P7, Z.Y2
BPM = Z.BPM; ZERO = Z.ZERO; DS = Z.DS; F = len(BPM)
def at(z, fq):
    out = np.zeros((z.shape[0], len(fq)), np.float32); m = (fq >= BPM[0]) & (fq <= BPM[-1]); idx = np.interp(fq[m], BPM, np.arange(F))
    i0 = np.floor(idx).astype(int); a = (idx - i0).astype(np.float32); i1 = np.minimum(i0 + 1, F - 1); out[:, m] = (1 - a) * z[:, i0] + a * z[:, i1]; return out
def feats(B):
    L = []
    for c in range(B.shape[0]):
        z = B[c].astype(np.float32); q = z.max(1, keepdims=True); L += [z, at(z, 2 * BPM), at(z, BPM / 2), z * q]
    return np.stack(L, -1)
def weights(d, idx):
    w = np.zeros(len(idx))
    for D in DS:
        m = d['ds'][idx] == D; pp = d['person'][idx][m]
        if not m.any(): continue
        u, inv, cnt = np.unique(pp, return_inverse=True, return_counts=True); w[m] = 1. / (len(u) * cnt[inv])
    return w / w.sum()
def fit_theta(Phi, truth, w, lam=1e-3):
    Q = np.exp(-(BPM[None] - truth[:, None]) ** 2 / (2 * 1.5 ** 2)); Q /= Q.sum(1, keepdims=True)
    def fg(th):
        s = Phi @ th; s = s - s.max(1, keepdims=True); p = np.exp(s); p /= p.sum(1, keepdims=True)
        ll = -(w[:, None] * Q * np.log(p + 1e-300)).sum() + lam * th @ th
        g = np.einsum('nf,nfd->d', (w[:, None] * (p - Q)).astype(np.float32), Phi) + 2 * lam * th; return ll, g.astype(float)
    th0 = np.zeros(Phi.shape[-1]); th0[0::4] = 1.
    r = minimize(fg, th0, jac=True, method='L-BFGS-B', options=dict(maxiter=400)); return r.x
SRC = {'L4n': 'N', 'L4p': 'P'}
PHI = {}
def phi(src):
    if src not in PHI: PHI[src] = feats((Z.NAT if src == 'N' else Z.POOLD)['B'])
    return PHI[src]
def fitL(a):
    nm, f = a; src = SRC[nm]; d = Z.NAT if src == 'N' else Z.POOLD; Ph = phi(src); tr = np.flatnonzero(d['fold'] != f)
    th = fit_theta(Ph[tr], d['truth'][tr], weights(d, tr)); E = Ph @ th; out = dict(theta=th)
    for D in DS:
        trD = np.flatnonzero((d['ds'] == D) & (d['fold'] != f))
        if not len(trD): continue
        rtr = X.runs_of(d, trD); Etr = [E[r] for r in rtr]; ref = float(np.mean(np.concatenate([e.max(1) for e in Etr]))); tt = np.concatenate([d['truth'][r] for r in rtr]); pp = np.concatenate([d['person'][r] for r in rtr]); b2 = (1e9, None)
        for g in P7.GAM:
            Eg = [Y2.qscale(e, g, ref) for e in Etr]
            for s in X.SIG:
                for ta in X.TAU:
                    M_ = [P7.marg(e, s, ta, ZERO) for e in Eg]
                    for est in ('median', 'mean'):
                        l = X.pmae(np.abs(np.concatenate([P7.point(x, est) for x in M_]) - tt), pp)[1].mean()
                        if l < b2[0] - 1e-9: b2 = (l, (g, s, ta, est))
        out[D] = b2[1] + (ref,)
    return (nm, f), out
def applyL(FL, d, nm, Ph):
    pred = np.full(len(d['truth']), np.nan); raw = np.full(len(d['truth']), np.nan)
    for f in range(5):
        m = FL[(nm, f)]
        for D in DS:
            te = np.flatnonzero((d['ds'] == D) & (d['fold'] == f))
            if not len(te) or D not in m: continue
            E = Ph[te] @ m['theta']; g, s, ta, est, ref = m[D]; loc = {j: i for i, j in enumerate(te)}
            raw[te] = BPM[E.argmax(1)]
            for r in X.runs_of(d, te): pred[r] = P7.point(P7.marg(Y2.qscale(E[[loc[j] for j in r]], g, ref), s, ta, ZERO), est)
    return pred, raw
def applyZ(FZ, d, nm):
    ch = Z.READERS[nm][1]; pred = np.full(len(d['truth']), np.nan); M = d[ch]
    for f in range(5):
        m = FZ[(nm, f)]
        for D in DS:
            te = np.flatnonzero((d['ds'] == D) & (d['fold'] == f))
            if not len(te) or D not in m: continue
            g, s, ta, est, ref, cst = m[D]
            for r in X.runs_of(d, te): pred[r] = P7.point(P7.marg(Y2.qscale(np.tensordot(m['w'], M[:, r], 1), g, ref), s, ta, ZERO), est)
    return pred
if __name__ == '__main__':
    phi('N'); phi('P')
    with Pool(10) as pool:
        FL = dict(pool.map(fitL, [(nm, f) for nm in SRC for f in range(5)], chunksize=1))
        FZ = dict(pool.map(Z.fit, [(nm, f) for nm in ('N4m', 'P4m') for f in range(5)], chunksize=1))
    print('theta per fold (L4p; per channel cnn_pf/pos/chrom/experts: z, z(2f), z(f/2), z*peak):')
    for f in range(5): print('  ', np.round(FL[('L4p', f)]['theta'], 2).tolist())
    ix = {}
    def boot(v):
        if len(v) not in ix: ix[len(v)] = np.random.default_rng(0).integers(0, len(v), (10000, len(v)))
        return np.quantile(v[ix[len(v)]].mean(1), [.025, .975])
    save = {}
    NATp = {nm: applyL(FL, Z.NAT, nm, phi('N')) for nm in SRC}; NATz = {nm: applyZ(FZ, Z.NAT, nm) for nm in ('N4m', 'P4m')}
    for c, d in Z.D_.items():
        Ph = phi('N') if c == 'native' else feats(d['B'])
        PL = NATp if c == 'native' else {nm: applyL(FL, d, nm, Ph) for nm in SRC}; PZ = NATz if c == 'native' else {nm: applyZ(FZ, d, nm) for nm in ('N4m', 'P4m')}
        for D in DS:
            sel = np.flatnonzero(d['ds'] == D)
            if not len(sel): continue
            t = d['truth'][sel]; per = d['person'][sel]; V = {nm: X.pmae(np.abs(PL[nm][0][sel] - t), per)[1] for nm in SRC}; V.update({nm: X.pmae(np.abs(PZ[nm][sel] - t), per)[1] for nm in PZ})
            Vraw = X.pmae(np.abs(PL['L4p'][1][sel] - t), per)[1]; dp = boot(V['L4p'] - V['P4m']); dn = boot(V['L4n'] - V['N4m']); q = PL['L4p'][0][sel]; q0 = PZ['P4m'][sel]
            print(f"{c:20s} {D:5s} people {len(V['L4p']):2d} windows {len(sel):4d} | N4m {V['N4m'].mean():6.3f} L4n {V['L4n'].mean():6.3f} ({np.mean(V['L4n'] - V['N4m']):+.3f} [{dn[0]:+.2f};{dn[1]:+.2f}])"
                  f" | P4m {V['P4m'].mean():6.3f} L4p {V['L4p'].mean():6.3f} ({np.mean(V['L4p'] - V['P4m']):+.3f} [{dp[0]:+.2f};{dp[1]:+.2f}]) | L4p no-FB {Vraw.mean():6.3f}"
                  f" | RMSE L4p {np.sqrt(np.mean((q - t) ** 2)):5.2f} P4m {np.sqrt(np.mean((q0 - t) ** 2)):5.2f} | share >5 L4p {np.mean(np.abs(q - t) > 5):.3f} P4m {np.mean(np.abs(q0 - t) > 5):.3f}", flush=True)
        if c == 'native': save = dict(truth=d['truth'], ds=d['ds'], person=d['person'], rec=d['rec'].astype(str), start=d['start'], L4n=PL['L4n'][0], L4p=PL['L4p'][0], N4m=PZ['N4m'], P4m=PZ['P4m'])
    np.savez_compressed(W / 'out/u05_public_learned.npz', **save)
