"""y03 - BTC dev native: two further candidates (exploratory, NOT in the pre-registered list of PREREG.md section 4; same adoption rule):
  M1 motion regression : inside each 10 s window the detrended face-box trajectory (centre x/y, width, height; optionally their first
                         differences) is regressed out of the normalised RGB trace before CHROM/POS (cf. Li et al., CVPR 2014; de Haan &
                         Jeanne 2013 on motion-induced colour variation). Box = the tight oval box already stored by the front end.
  M2 calibration slope : prediction = a + b * posterior median per (class, Task) cell (weighted least squares on the training folds)
                         instead of median + offset.
Usage: $PY code/y03_btc_motion.py <workers>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import y01_btc_improve as Y
from multiprocessing import Pool
import numpy as np
from lib_btc import load_blocks, supported, pos_team, chrom_team, trend, zlog, GRID
rm = Y.rm; rec, lab, per, fold, low, task = Y.rec, Y.lab, Y.per, Y.fold, Y.low, Y.task
def one(r):
    b = load_blocks(r); t, valid, rgb, bx = b['t'], b['valid'], b['rgb_mean'], b['boxes'].astype(float)
    mot = np.stack([(bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2, bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]], 1); out = {k: [] for k in ('base', 'r4', 'r8')}; amp = []
    for s in np.arange(0, max(0., t[-1] - 10 + 1e-8), 5.):
        if not supported(t, valid, s, s + 10): continue
        q = s + np.arange(300) / 30.; x = np.stack([np.interp(q, t, rgb[:, c]) for c in range(3)], 1)
        if not np.isfinite(x).all() or x.mean(0).min() <= 0: continue
        wc, wp = chrom_team(x), pos_team(x)
        if not (np.isfinite(wc).all() and np.isfinite(wp).all()) or wc.std() < 1e-12 or wp.std() < 1e-12: continue
        m = np.stack([np.interp(q, t, mot[:, c]) for c in range(4)], 1); m = m - np.stack([trend(300)(m[:, c]) for c in range(4)], 1); sd = m.std(0)
        R4 = m[:, sd > 1e-9] / sd[sd > 1e-9]; d = np.diff(m, axis=0, prepend=m[:1]); sdd = d.std(0); R8 = np.concatenate([R4, d[:, sdd > 1e-9] / sdd[sdd > 1e-9]], 1)
        mu = x.mean(0); xn = x / mu; xd = xn - np.stack([trend(300)(xn[:, c]) for c in range(3)], 1); amp.append(float(np.hypot(m[:, 0], m[:, 1]).std() / max(np.median(mot[:, 2]), 1)))
        out['base'].append([Y.spec2(wc)[0], Y.spec2(wp)[0]])
        for k, R in (('r4', R4), ('r8', R8)):
            if R.shape[1] == 0: out[k].append(out['base'][-1]); continue
            x2 = (xn - R @ np.linalg.lstsq(R, xd, rcond=None)[0]) * mu; c2, p2 = chrom_team(x2), pos_team(x2)
            out[k].append([Y.spec2(c2)[0], Y.spec2(p2)[0]] if (np.isfinite(c2).all() and np.isfinite(p2).all() and c2.std() > 1e-12 and p2.std() > 1e-12) else out['base'][-1])
    return out, (float(np.mean(amp)) if amp else np.nan)
def slope_cv(Z):
    out = np.zeros(len(lab))
    for f in sorted(set(fold)):
        tr = np.flatnonzero(fold != f); te = np.flatnonzero(fold == f); m = rm.fit(Z[tr], lab[tr], per[tr], low[tr], task[tr], 'F'); w = rm.pw(per[tr])
        raw = lambda idx: np.array([rm.pmed(Z[i], m['prior'], m['beta'][low[i]]) for i in idx]); rt, re = raw(tr), raw(te); cell_tr = low[tr] * 2 + task[tr].astype(int); cell_te = low[te] * 2 + task[te].astype(int); p = re + np.array([m['off'][k] for k in cell_te])
        for k in range(4):
            s = cell_tr == k
            if s.sum() < 2 * rm.MIN_N or np.ptp(rt[s]) == 0: continue
            A = np.stack([np.ones(s.sum()), rt[s]], 1) * np.sqrt(w[s])[:, None]; a, b = np.linalg.lstsq(A, lab[tr][s] * np.sqrt(w[s]), rcond=None)[0]; p[cell_te == k] = a + b * re[cell_te == k]
        out[te] = p
    return out
if __name__ == '__main__':
    with Pool(int(sys.argv[1])) as pool: RES = pool.map(one, rec, chunksize=2)
    S = [r[0] for r in RES]; amp = np.array([r[1] for r in RES])
    def agg(k):
        Z = np.zeros((len(rec), len(GRID)))
        for i, s in enumerate(S):
            if s[k]: P = np.array(s[k]); Z[i] = (zlog(P[:, 0].mean(0)) + zlog(P[:, 1].mean(0))) / 2
        return Z
    Z0 = agg('base'); v3, _ = rm.crossval(Z0, lab, per, fold, low, task, 'F'); Y.show('v3 (re-computed here)', v3, v3)
    for k in ('r4', 'r8'):
        p, _ = rm.crossval(agg(k), lab, per, fold, low, task, 'F'); Y.show(f'M1 motion regression {k}', p, v3)
        hi = amp >= np.nanmedian(amp); Zm = Z0.copy(); Zm[hi] = agg(k)[hi]; p2, _ = rm.crossval(Zm, lab, per, fold, low, task, 'F'); Y.show(f'M1 {k} only in the more-moving half', p2, v3)
    Y.show('M2 calibration slope per cell', slope_cv(Z0), v3)
    print('motion amplitude (sd of box centre / face width): median %.4f, Task %.4f vs rest %.4f' % (np.nanmedian(amp), np.nanmedian(amp[task]), np.nanmedian(amp[~task])))
