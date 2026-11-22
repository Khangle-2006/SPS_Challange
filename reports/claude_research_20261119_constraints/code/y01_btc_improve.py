"""y01 - BTC dev native (218 videos / 158 people): candidates to lower MAE / RMSE of reader_v3, as listed in PREREG.md section 4.
  E1 point estimate : posterior median (v3) | posterior mean | blend a*mean+(1-a)*median, (beta, a) per class chosen on the training folds
  E2 harmonic comb  : evidence Z(f) + g*Z2(2f), g in {0,.25,.5,1} chosen on the training folds
  E3 more experts   : GREEN, PBV, LGI, OMIT added to {CHROM, POS} (fixed sets, no per-fold subset search)
  E5 window pooling : arithmetic mean of window spectra (v3) | mean of standardised log spectra | SNR-weighted mean
Everything is fitted on the 4 training folds; person-first MAE and RMSE, paired bootstrap over people (10,000, seed 0) for both.
Usage: $PY code/y01_btc_improve.py <workers>"""
import sys, math, pickle
from pathlib import Path
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
from scipy import signal
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
sys.path.insert(0, str(OLD / 'code'))
from lib_btc import videos, meta, load_blocks, supported, pos_team, chrom_team, trend, SOS1, GRID, zlog
import reader_models as rm
V = videos(); M = meta(); rec = [v['recording'] for v in V]; lab = np.array([v['pr_bpm'] for v in V]); fold = np.array([v['fold'] for v in V]); per = np.array([v['person'] for v in V])
task = np.char.startswith(np.array([v['task'] for v in V]), 'Task'); low = np.array([int(M[r]['width']) * int(M[r]['height']) < 1800000 for r in rec]).astype(int)
def green(x, fs=30.): return x[:, 1] / x[:, 1].mean() - 1
def pbv(x, fs=30.):
    c = x / x.mean(0); s = c.std(0); p = s / np.linalg.norm(s); q = c - 1; return q @ np.linalg.solve(q.T @ q + 1e-9 * np.eye(3), p)
def lgi(x, fs=30.):
    c = (x / x.mean(0) - 1).T; u = np.linalg.svd(c, full_matrices=False)[0][:, :1]; return ((np.eye(3) - u @ u.T) @ c)[1]
def omit(x, fs=30.):
    c = (x / x.mean(0) - 1).T; q = np.linalg.qr(c)[0][:, :1]; return ((np.eye(3) - q @ q.T) @ c)[1]
METH = dict(chrom=chrom_team, pos=pos_team, green=green, pbv=pbv, lgi=lgi, omit=omit)
def spec2(w):
    y = signal.sosfiltfilt(SOS1, w - trend(len(w))(w)); f, p = signal.periodogram(y, fs=30, window='hann', nfft=4096, detrend=False)
    p1 = np.interp(GRID, 60 * f, p); p2 = np.interp(2 * GRID, 60 * f, p); return p1 / p1.sum(), p2 / max(p2.sum(), 1e-300)
def one(r):
    b = load_blocks(r); t, valid, rgb = b['t'], b['valid'], b['rgb_mean']; out = {m: [] for m in METH}
    for s in np.arange(0, max(0., t[-1] - 10 + 1e-8), 5.):
        if not supported(t, valid, s, s + 10): continue
        q = s + np.arange(300) / 30.; x = np.stack([np.interp(q, t, rgb[:, c]) for c in range(3)], 1)
        if not np.isfinite(x).all() or x.mean(0).min() <= 0: continue
        ws = {m: f(x, 30.) for m, f in METH.items()}; good = lambda w: np.isfinite(w).all() and w.std() > 1e-12
        if not (good(ws['chrom']) and good(ws['pos'])): continue          # same acceptance rule as lib_btc.avg_spectra
        ws = {m: (w if good(w) else ws['pos']) for m, w in ws.items()}
        for m, w in ws.items(): out[m].append(spec2(w))
    return out
def boot_pair(pa, pb):
    """person-first MAE and RMSE of pa, and paired differences (pa - pb) with CI95."""
    d = defaultdict(list)
    for i, p in enumerate(per): d[p].append((abs(pa[i] - lab[i]), abs(pb[i] - lab[i]), pa[i], pb[i], lab[i]))
    v = np.array([np.mean(d[p], 0) for p in sorted(d)]); ix = np.random.default_rng(0).integers(0, len(v), (10000, len(v)))
    rm_ = lambda col: np.sqrt(np.mean((v[:, col] - v[:, 4]) ** 2)); rb = lambda col: np.sqrt(np.mean((v[ix, col] - v[ix, 4]) ** 2, 1))
    dm = (v[:, 0] - v[:, 1]); dr = rb(2) - rb(3); cr = lambda a, b: np.array([np.corrcoef(a[k], b[k])[0, 1] for k in ix[:2000]])
    r = float(np.corrcoef(v[:, 2], v[:, 4])[0, 1]) if np.ptp(v[:, 2]) > 0 else float('nan')
    return dict(MAE=float(v[:, 0].mean()), RMSE=float(rm_(2)), r=r, dMAE=float(dm.mean()), dMAE_ci=np.quantile(dm[ix].mean(1), [.025, .975]).tolist(), dRMSE=float(rm_(2) - rm_(3)), dRMSE_ci=np.quantile(dr, [.025, .975]).tolist())
def show(name, p, ref):
    a = boot_pair(p, ref); flag = ('MAE* ' if a['dMAE_ci'][1] < 0 else '') + ('RMSE* ' if a['dRMSE_ci'][1] < 0 else '') + ('worseMAE ' if a['dMAE_ci'][0] > 0 else '') + ('worseRMSE' if a['dRMSE_ci'][0] > 0 else '')
    print(f"{name:46s} MAE {a['MAE']:6.3f} RMSE {a['RMSE']:6.3f} r {a['r']:+.3f} | dMAE {a['dMAE']:+.3f} [{a['dMAE_ci'][0]:+.2f};{a['dMAE_ci'][1]:+.2f}] dRMSE {a['dRMSE']:+.3f} [{a['dRMSE_ci'][0]:+.2f};{a['dRMSE_ci'][1]:+.2f}] {flag}", flush=True); return a
# ---- generalised reader: estimator in {'med','mean','blend'}, optional harmonic evidence
A_GRID = [0, .25, .5, .75, 1.]; G_GRID = [0, .25, .5, 1.]
def post_stats(Z, prior, b):
    lp = np.log(prior + 1e-12)[None] + b * Z; q = np.exp(lp - lp.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    med = GRID[np.minimum((np.cumsum(q, 1) < .5).sum(1), len(GRID) - 1)]; return med, q @ GRID
def raw(Z, Zh, prior, b, a, g):
    med, mean = post_stats(Z if g == 0 else Z + g * Zh, prior, b); return a * mean + (1 - a) * med
def fitG(Z, Zh, lab_, per_, low_, task_, est, harm):
    w = rm.pw(per_); prior = rm.kde(lab_, w); m = dict(prior=prior, par={}, off={k: 0. for k in range(4)}, est=est)
    alist = {'med': [0.], 'mean': [1.], 'blend': A_GRID}[est]; glist = G_GRID if harm else [0.]
    def loss(idx, b, a, g):
        e = raw(Z[idx], Zh[idx], prior, b, a, g) - lab_[idx]; ww = rm.pw(per_[idx]); ww = ww / ww.sum()
        return {'med': np.sum(ww * np.abs(e)), 'mean': np.sqrt(np.sum(ww * e ** 2)), 'blend': .5 * np.sum(ww * np.abs(e)) + .5 * np.sqrt(np.sum(ww * e ** 2))}[est]
    def best(idx): return min(((loss(idx, b, a, g), i) for i, (b, a, g) in enumerate(COMB)), key=lambda x: x[0])[1]
    COMB = [(b, a, g) for b in rm.BETAS for a in alist for g in glist]; gbest = COMB[best(np.arange(len(lab_)))]
    for c in (0, 1):
        i = np.flatnonzero(low_ == c); m['par'][c] = COMB[best(i)] if len(i) >= rm.MIN_N else gbest
    p = np.zeros(len(lab_))
    for c in (0, 1):
        i = np.flatnonzero(low_ == c)
        if len(i): p[i] = raw(Z[i], Zh[i], prior, *m['par'][c])
    r = lab_ - p; cell = low_ * 2 + task_.astype(int)
    for k in range(4):
        s = cell == k
        if s.sum() >= rm.MIN_N: m['off'][k] = float(np.sum(w[s] * r[s]) / np.sum(w[s]))
    return m
def predG(m, Z, Zh, low_, task_):
    p = np.zeros(len(Z))
    for c in (0, 1):
        i = np.flatnonzero(low_ == c)
        if len(i): p[i] = raw(Z[i], Zh[i], m['prior'], *m['par'][c])
    return p + np.array([m['off'][k] for k in low_ * 2 + task_.astype(int)])
def cvG(Z, Zh, est='med', harm=False):
    out = np.zeros(len(lab)); pars = []
    for f in sorted(set(fold)):
        tr = np.flatnonzero(fold != f); te = np.flatnonzero(fold == f); m = fitG(Z[tr], Zh[tr], lab[tr], per[tr], low[tr], task[tr], est, harm)
        out[te] = predG(m, Z[te], Zh[te], low[te], task[te]); pars.append(m['par'])
    return out, pars
if __name__ == '__main__':
    cache = W / 'out/y01_spectra.pkl'
    if cache.exists(): S = pickle.load(cache.open('rb'))
    else:
        with Pool(int(sys.argv[1])) as pool: S = pool.map(one, rec, chunksize=2)
        pickle.dump(S, cache.open('wb'))
    G = len(GRID)
    def agg(meths, how='arith', harm=False):
        Z = np.zeros((len(rec), G)); k = 1 if harm else 0
        for i, s in enumerate(S):
            if not s['chrom']: continue
            z = []
            for m in meths:
                P = np.array([w[k] for w in s[m]])
                if how == 'arith': z.append(zlog(P.mean(0)))
                elif how == 'geo': z.append(zlog(np.exp(np.log(P + 1e-30).mean(0))))
                else:
                    snr = P.max(1) / np.median(P, 1); wt = snr / snr.sum(); z.append(zlog((wt[:, None] * P).sum(0)))
            Z[i] = np.mean(z, 0)
        return Z
    Z0 = agg(['chrom', 'pos']); Zh0 = agg(['chrom', 'pos'], harm=True); zero = np.zeros_like(Z0)
    v3, _ = rm.crossval(Z0, lab, per, fold, low, task, 'F'); chk, _ = cvG(Z0, zero, 'med')
    print('videos with evidence', int((np.abs(Z0).sum(1) > 0).sum()), '| generalised reader reproduces v3:', float(np.abs(chk - v3).max()))
    R = {'v3': v3}; show('v3 (posterior median, CHROM+POS)', v3, v3)
    print('--- E1 point estimate'); R['mean'], pm = cvG(Z0, zero, 'mean'); show('posterior mean (beta by RMSE)', R['mean'], v3); R['blend'], pb = cvG(Z0, zero, 'blend'); show('blend mean/median', R['blend'], v3); print('    blend params per fold (beta, a, g) by class:', pb)
    for a in (.25, .5, .75):
        p = np.zeros(len(lab))
        for f in sorted(set(fold)):
            tr = np.flatnonzero(fold != f); te = np.flatnonzero(fold == f); m0 = fitG(Z0[tr], zero[tr], lab[tr], per[tr], low[tr], task[tr], 'med', False); m1 = fitG(Z0[tr], zero[tr], lab[tr], per[tr], low[tr], task[tr], 'mean', False)
            p[te] = (1 - a) * predG(m0, Z0[te], zero[te], low[te], task[te]) + a * predG(m1, Z0[te], zero[te], low[te], task[te])
        R[f'avg{a}'] = p; show(f'fixed average {1 - a:.2f}*v3 + {a:.2f}*mean-reader', p, v3)
    print('--- E2 harmonic comb'); R['harm'], ph = cvG(Z0, Zh0, 'med', True); show('median + harmonic evidence', R['harm'], v3); print('    params per fold:', ph)
    for g in (.25, .5, 1.): R[f'harm{g}'], _ = rm.crossval(Z0 + g * Zh0, lab, per, fold, low, task, 'F'); show(f'median, fixed g={g}', R[f'harm{g}'], v3)
    print('--- E3 more experts (fixed sets)')
    for name, ms in [('+LGI', ['chrom', 'pos', 'lgi']), ('+OMIT', ['chrom', 'pos', 'omit']), ('+PBV', ['chrom', 'pos', 'pbv']), ('+GREEN', ['chrom', 'pos', 'green']), ('all six', list(METH)), ('POS only', ['pos']), ('CHROM only', ['chrom']), ('LGI only', ['lgi']), ('OMIT only', ['omit']), ('PBV only', ['pbv']), ('GREEN only', ['green'])]:
        R['E3' + name], _ = rm.crossval(agg(ms), lab, per, fold, low, task, 'F'); show('experts ' + name, R['E3' + name], v3)
    print('--- E5 window pooling')
    for how in ('geo', 'snr'): R['E5' + how], _ = rm.crossval(agg(['chrom', 'pos'], how), lab, per, fold, low, task, 'F'); show('pooling ' + how, R['E5' + how], v3)
    print('--- controls: labels shuffled between people (5 draws), mean-reader'); rng = np.random.default_rng(1); ppl = sorted(set(per)); keep = lab.copy()
    for k in range(5):
        perm = dict(zip(ppl, rng.permutation(ppl))); src = {p: lab[per == p] for p in ppl}; lab[:] = [float(np.mean(src[perm[p]])) for p in per]
        pc, _ = cvG(Z0, zero, 'mean'); pv, _ = rm.crossval(Z0, lab, per, fold, low, task, 'F'); a = boot_pair(pc, pv); print(f"   shuffle {k}: mean-reader MAE {a['MAE']:.3f} RMSE {a['RMSE']:.3f} | v3 MAE {boot_pair(pv, pv)['MAE']:.3f} RMSE {boot_pair(pv, pv)['RMSE']:.3f}"); lab[:] = keep
    np.savez_compressed(W / 'out/y01_pred.npz', rec=np.array(rec), lab=lab, per=per, fold=fold, **{k.replace('+', 'p').replace(' ', '_'): v for k, v in R.items()})
