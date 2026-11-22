"""y06 - step 2 of the 20 Nov list (BTC part): can reader_v3's capture class (file pixel count < 1.8 MP = 'low') be replaced by a class
read from the EVIDENCE itself, so that C4/C5 (where every file is 'low') stop losing ~0.8 bpm?   BTC dev only, no holdout.
reader_v3 uses the class twice: beta (how much the video is trusted) per class, and one offset per (class, Task) cell.
 1 Decomposition (diagnostic, uses the source class = not available to a real low-resolution input):
     seen     class the pipeline sees             source    class of the source file for beta AND offsets
     src_beta source class for beta only          src_off   source class for the offsets only
 2 Label-free quality classes, 'weak' if index < threshold (threshold = 33/50/67 % quantile of the TRAINING videos):
     peak   = max of Z (Z = mean of the standardised log CHROM and POS spectra)      margin = peak minus the highest value >10 bpm away
     agree  = -|CHROM peak - POS peak| in bpm
     q_<i>  class used for beta and offsets       qb_<i>    class used for beta only, offsets by the seen class
     nested = best of all the above (or 'seen' unless something is >= 0.02 bpm better), chosen by an INNER cross-validation
              inside the training folds (so the choice never sees the held-out fold).
 refit  : fitted on the training folds of the same condition     frozen : fitted on native, applied unchanged to the condition
Person-first MAE, paired bootstrap over people (10,000, seed 0). Control: labels shuffled between people.
Usage: $PY code/y06_reader_quality.py <workers>   -> out/y06_btc_quality.csv, out/y06_pred.npz"""
import sys, csv
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
sys.path.insert(0, str(OLD / 'code'))
from lib_btc import videos, meta, summarize, zlog, GRID, NUM
import reader_models as rm
V = videos(); M = meta(); rec = np.array([v['recording'] for v in V]); lab = np.array([v['pr_bpm'] for v in V]); fold = np.array([v['fold'] for v in V]); per = np.array([v['person'] for v in V])
task = np.char.startswith(np.array([v['task'] for v in V]), 'Task'); low = np.array([int(M[r]['width']) * int(M[r]['height']) < 1800000 for r in rec]).astype(int)
rows = list(csv.DictReader((NUM / 'tables/round1_bayes_20261108/VIDEO_PREDICTIONS.csv').open()))
cd = {r['recording']: float(r['prediction']) for r in rows if r['method'] == 'constant' and r['condition'] == 'native'}; const = np.array([cd[r] for r in rec])
N = len(rec); FOLDS = sorted(set(fold)); QQ = (.33, .5, .67); IDX = ('peak', 'margin', 'agree'); LOGP = None
def evidence(d, oa=False):
    C, P = (d['Co'], d['Po']) if oa else (d['C'], d['P']); ok = np.isfinite(C).all(1) & np.isfinite(P).all(1); zc = np.zeros((N, len(GRID))); zp = np.zeros_like(zc); zc[ok] = zlog(C[ok]); zp[ok] = zlog(P[ok]); Z = (zc + zp) / 2
    far = np.abs(GRID[None] - GRID[Z.argmax(1)][:, None]) > 10
    q = dict(peak=Z.max(1), margin=Z.max(1) - np.where(far, Z, -np.inf).max(1), agree=-np.abs(GRID[zc.argmax(1)] - GRID[zp.argmax(1)]))
    for v in q.values(): v[~ok] = -1e6                       # no evidence -> weakest class
    return Z, ok, q
def load(t, c): return dict(np.load(W / 'out' / (t + '_btc') / (c + '.npz')))
def pmedv(Z, prior, b):
    lp = np.log(prior + 1e-12)[None] + b * Z; q = np.exp(lp - lp.max(1, keepdims=True)); q /= q.sum(1, keepdims=True); return GRID[np.minimum((np.cumsum(q, 1) < .5).sum(1), len(GRID) - 1)]
def fit_beta(Z, L, pr, prior):
    w = rm.pw(pr); return rm.BETAS[int(np.argmin([np.sum(w * np.abs(pmedv(Z, prior, b) - L)) for b in rm.BETAS]))]
def fitG(Z, L, pr, cb, co, tk):
    """reader_v3 with separate class vectors for beta (cb) and for the offsets (co); cb == co == pixel class is reader_v3 itself."""
    prior = rm.kde(L, rm.pw(pr)); gb = fit_beta(Z, L, pr, prior); beta = {0: gb, 1: gb}
    for c in (0, 1):
        i = np.flatnonzero(cb == c)
        if len(i) >= rm.MIN_N: beta[c] = fit_beta(Z[i], L[i], pr[i], prior)
    m = dict(prior=prior, beta=beta, off={k: 0. for k in range(4)}); r = L - predG(m, Z, cb, co, tk); w = rm.pw(pr); cell = co * 2 + tk.astype(int)
    for k in range(4):
        s = cell == k
        if s.sum() >= rm.MIN_N: m['off'][k] = float(np.sum(w[s] * r[s]) / np.sum(w[s]))
    return m
def predG(m, Z, cb, co, tk):
    p = np.zeros(len(Z))
    for c in (0, 1):
        i = np.flatnonzero(cb == c)
        if len(i): p[i] = pmedv(Z[i], m['prior'], m['beta'][c])
    return p + np.array([m['off'][k] for k in co * 2 + tk.astype(int)])
def cls(spec, tr, qtr, qte, seen_tr, seen_te):
    """class vector on the training data and on the data to predict, for one specification; thresholds from the training rows only."""
    if spec == 'seen': return seen_tr, seen_te
    if spec == 'source': return low, low
    t = np.quantile(qtr[spec[0]][tr], spec[1]); return (qtr[spec[0]] < t).astype(int), (qte[spec[0]] < t).astype(int)
def inner_loss(s, tr, Z, q, seen, L):
    """person-weighted MAE of out-of-fold predictions INSIDE the training rows (inner folds = the remaining fold ids)."""
    p = np.zeros(len(tr)); fo = fold[tr]
    for g in sorted(set(fo)):
        a = tr[fo != g]; b = tr[fo == g]; cb = cls(s[0], a, q, q, seen, seen)[0]; co = cls(s[1], a, q, q, seen, seen)[0]
        p[fo == g] = predG(fitG(Z[a], L[a], per[a], cb[a], co[a], task[a]), Z[b], cb[b], co[b], task[b])
    w = rm.pw(per[tr]); return float(np.sum(w * np.abs(p - L[tr])) / w.sum())
def cv(cands, sub, Zte, qte, seen_te, Ztr, qtr, seen_tr, L, cache, margin=0.):
    p = np.full(N, np.nan); chosen = []
    for f in FOLDS:
        tr = np.flatnonzero((fold != f) & sub); te = np.flatnonzero((fold == f) & sub)
        if not len(te): continue
        s = cands[0]
        if len(cands) > 1:
            for c in cands:
                if (f, c) not in cache: cache[(f, c)] = inner_loss(c, tr, Ztr, qtr, seen_tr, L)
            ls = [cache[(f, c)] for c in cands]; k = int(np.argmin(ls)); s = cands[k] if ls[k] < ls[0] - margin else cands[0]
        cbt, cbe = cls(s[0], tr, qtr, qte, seen_tr, seen_te); cot, coe = cls(s[1], tr, qtr, qte, seen_tr, seen_te)
        p[te] = predG(fitG(Ztr[tr], L[tr], per[tr], cbt[tr], cot[tr], task[tr]), Zte[te], cbe[te], coe[te], task[te]); chosen.append(s)
    return p, chosen
def variants(differs):
    v = {'seen': [('seen', 'seen')]}
    if differs: v.update(source=[('source', 'source')], src_beta=[('source', 'seen')], src_off=[('seen', 'source')])
    for i in IDX: v['q_' + i] = [((i, q), (i, q)) for q in QQ]; v['qb_' + i] = [((i, q), 'seen') for q in QQ]
    v['nested'] = [('seen', 'seen')] + [c for k in v if k[0] == 'q' for c in v[k]]; return v
Dn = load('x01', 'native'); Zn, okn, qn = evidence(Dn)
def one(a):
    tag, c, oa = a; d = load(tag, c); sub = d['have'] & Dn['have'] & okn; n = int(sub.sum()); name = c + ('+oa' if oa else '')
    if n < 150: return name, n, [], {}
    Z, ok, q = evidence(d, oa); seen = np.ones(N, int) if (c.startswith('c4') or c in ('c5_vga', 'c5_face')) else low; var = variants(bool((seen != low).any())); out = []; P = {}
    S = lambda a_, b_: summarize(a_[sub], lab[sub], per[sub], b_[sub]); nat = cv([('seen', 'seen')], sub, Zn, qn, low, Zn, qn, low, lab, {})[0]
    for mode in ('refit', 'frozen'):
        cache = {}; ref = None
        for k, cands in var.items():
            if mode == 'frozen' and k in ('source', 'src_beta', 'src_off'): continue
            p, ch = cv(cands, sub, Z, q, seen, *((Z, q, seen) if mode == 'refit' else (Zn, qn, low)), lab, cache, .02 if k == 'nested' else 0.)
            if k == 'seen': ref = p
            s = S(p, ref); sc = S(p, const); sn = S(p, nat); P[f'{name}|{mode}|{k}'] = p
            out.append(dict(condition=name, videos=n, mode=mode, variant=k, MAE=s['MAE'], RMSE=s['RMSE'], r=s['r'], d_seen=s['delta'], ds_lo=s['dCI'][0], ds_hi=s['dCI'][1], d_const=sc['delta'], dc_lo=sc['dCI'][0], dc_hi=sc['dCI'][1],
                            d_native=sn['delta'], dn_lo=sn['dCI'][0], dn_hi=sn['dCI'][1], chosen=' '.join('seen' if x == ('seen', 'seen') else f"{x[0][0]}{int(x[0][1] * 100)}{'' if x[1] != 'seen' else 'b'}" for x in ch) if len(cands) > 1 else '',
                            weak_share=float(np.mean(q['peak'][sub] < np.median(qn['peak'][sub])))))
    return name, n, out, P
if __name__ == '__main__':
    p0 = cv([('seen', 'seen')], np.ones(N, bool), Zn, qn, low, Zn, qn, low, lab, {})[0]; p1 = rm.crossval(Zn, lab, per, fold, low, task, 'F')[0]
    print(f'self-check, vectorised reader vs reader_models on native (218 videos): max |difference| {np.abs(p0 - p1).max():.2e} bpm', flush=True)
    jobs = [('x01', c, False) for c in ('native', 'c1_g25p', 'c1_g25gv01', 'c4_vga', 'c4_qvga', 'c6_us1', 'c7_blue', 'c5_lit')] + [('x06', c, False) for c in ('c5_tuned', 'c5_vga', 'c5_face')] \
         + [('x09', c, o) for c, o in (('track_c7_blue', False), ('track_c7_blue', True), ('aidtrack_c1_g25gv01', False), ('c2_vit', False), ('track_c1_g25p', False))]
    with Pool(int(sys.argv[1])) as pool: RES = pool.map(one, jobs, chunksize=1)
    ALL = []; PRED = {}
    for name, n, out, P in RES:
        if not out: print(f'\n== {name}: only {n} videos - skipped'); continue
        print(f"\n== {name}  n={n}  share of videos whose peak index is below the native median: {out[0]['weak_share']:.2f}")
        for o in out:
            print(f"  {o['mode']:6s} {o['variant']:10s} MAE {o['MAE']:6.3f} RMSE {o['RMSE']:5.2f} r {o['r']:+.2f} | vs seen {o['d_seen']:+.2f} [{o['ds_lo']:+.2f};{o['ds_hi']:+.2f}] | vs const {o['d_const']:+.2f} [{o['dc_lo']:+.2f};{o['dc_hi']:+.2f}]"
                  f" | vs native {o['d_native']:+.2f} [{o['dn_lo']:+.2f};{o['dn_hi']:+.2f}] {o['chosen']}", flush=True)
        ALL += out; PRED.update(P)
    with (W / 'out/y06_btc_quality.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, list(ALL[0])); w.writeheader(); w.writerows(ALL)
    np.savez_compressed(W / 'out/y06_pred.npz', rec=rec, lab=lab, per=per, **PRED)
    sub = np.ones(N, bool); rng = np.random.default_rng(1); ppl = sorted(set(per)); var = variants(False)
    for k in range(3):
        perm = dict(zip(ppl, rng.permutation(ppl))); src = {p_: lab[per == p_] for p_ in ppl}; L2 = np.array([float(np.mean(src[perm[p_]])) for p_ in per])
        a = cv(var['seen'], sub, Zn, qn, low, Zn, qn, low, L2, {})[0]; b = cv(var['nested'], sub, Zn, qn, low, Zn, qn, low, L2, {}, .02)[0]
        print(f'control, labels shuffled between people #{k}: video-level MAE reader_v3 {np.abs(a - L2).mean():.2f} | nested quality class {np.abs(b - L2).mean():.2f}')
