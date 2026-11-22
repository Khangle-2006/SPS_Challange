"""z13 - public reader P4 (y13: CNN, POS, CHROM + mean standardised log-spectrum of the open-rppg experts, POOLED fit) with the team CNN fed
the 0.1 s pre-filtered crops: x13 'nat_ma3' on native, x20 under the conditions; POS / CHROM unchanged (= y07 'hybrid', z03 'both').
c5_split10 keeps its own CNN input (the <=10 fps binned split stream is already the pre-filter). Readers, all fitted on training folds
only (fusion weights {0,.5,1,2}^4 + forward-backward with quality weight, as y13), paired on the windows every variant has:
  P4  : y13 pooled reader (CNN without pre-filter)     P4m : same pooling, pre-filtered CNN channel
  N4  : y13 native-fit reader                          N4m : native-fit, pre-filtered CNN channel
Gates as PREREG for P4m (a coverage >= .9 of the condition's native windows, b not worse / c better than the train-fold constant);
paired person bootstrap 10,000x, seed 0.
Usage: $PY code/z13_public_p4_prefilter.py   -> out/z13_public_p4_prefilter.csv, log logs/z13.log"""
import sys, json, itertools, csv
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
import x05_public_score as X
import y02_public_improve as Y2
import y07_public_prior as P7
MODELS = ['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']
POOL = ['c4_qvga', 'c6_us1', 'aidtrack_c1_g25gv01', 'c2_vit', 'track_c1_g25p', 'track_c7_blue']
EVAL = ['c1_g50p', 'c1_g12p', 'c1_g25gv001', 'c7_white', 'c5_split10']
BPM = X.BPM; DS = X.DS; ZERO = np.zeros(len(BPM)); FIELDS = ('ds', 'person', 'fold', 'rec', 'start', 'truth')
jobs = [j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] in DS]; J = {j['recording']: j for j in jobs}
def tag_of(c): return 'x11' if c.startswith('c5_split') else ('x09' if (W / 'out/x09_spec' / c).exists() else 'x01')
def kk(d): return [(r, round(float(s), 3)) for r, s in zip(d['rec'], d['start'])]
def experts(d, cond):
    key = kk(d); ok = np.ones(len(key), bool); ZE = []
    for m in MODELS:
        tab = {}
        for rec in sorted(set(d['rec'].tolist())):
            p = (W / 'out/y12' / m / (J[rec]['key'] + '.npz')) if cond == 'native' else (W / 'out/y13' / cond / m / (J[rec]['key'] + '.npz'))
            if not p.exists(): continue
            with np.load(p) as z:
                for s, sp in zip(z['start'], z['spec']):
                    if np.isfinite(sp).all(): tab[(rec, round(float(s), 3))] = sp
        S = np.array([tab.get(k, np.full(len(BPM), np.nan)) for k in key]); h = np.isfinite(S).all(1); ok &= h; Zm = np.zeros((len(key), len(BPM))); Zm[h] = X.zs(S[h]); ZE.append(Zm)
    return np.mean(ZE, 0), ok
NAT_ALL = X.load_old_native()
def build(c):
    """windows of condition c with both channel sets: A = (CNN, POS, CHROM, experts), B = (pre-filtered CNN, POS, CHROM, experts)."""
    if c == 'native': o = NAT_ALL; m_ = X.load('x13', ['nat_ma3']).get('nat_ma3')
    else: o = X.load(tag_of(c), [c]).get(c); m_ = None if c.startswith('c5_split') else X.load('x20', [c]).get(c)
    if o is None or (m_ is None and not c.startswith('c5_split')): return None
    E, ok = experts(o, c); ko = kk(o)
    if m_ is None: zm = o['Z'][0]; okm = np.ones(len(ko), bool)
    else:
        km = {k: i for i, k in enumerate(kk(m_))}; ix = np.array([km.get(k, -1) for k in ko]); okm = ix >= 0; zm = np.zeros_like(o['Z'][0]); zm[okm] = m_['Z'][0][ix[okm]]
    keep = ok & okm; d = {k: np.asarray(o[k])[keep] for k in FIELDS}
    d['A'] = np.concatenate([o['Z'][:, keep], E[None, keep]]); d['B'] = np.concatenate([zm[None, keep], o['Z'][1:, keep], E[None, keep]])
    recs = sorted(set(d['rec'].tolist())); base = {D: int(((NAT_ALL['ds'] == D) & np.isin(NAT_ALL['rec'], recs)).sum()) for D in DS}
    d['cov'] = {D: (int((d['ds'] == D).sum()), base[D]) for D in DS}; d['lost_prefilter'] = int((ok & ~okm).sum()); return d
def sub(d, m): return {k: (v[m] if isinstance(v, np.ndarray) and v.ndim == 1 else (v[:, m] if isinstance(v, np.ndarray) else v)) for k, v in d.items()}
D_ = {}
for c in ['native'] + POOL + EVAL:
    d = build(c)
    if d is None: print(c, 'missing input - skipped', flush=True); continue
    D_[c] = d; print(c, 'windows', len(d['truth']), 'coverage', d['cov'], 'windows lost to missing pre-filtered CNN', d['lost_prefilter'], flush=True)
NAT = D_['native']; common = sorted(set.intersection(*[set(D_[c]['rec'].tolist()) for c in POOL])) if all(c in D_ for c in POOL) else []
assert common, 'pooled set needs every POOL condition'
parts = [sub(NAT, np.isin(NAT['rec'], common))] + [sub(D_[c], np.isin(D_[c]['rec'], common)) for c in POOL]
POOLD = {k: np.concatenate([p[k] for p in parts]) for k in ('ds', 'person', 'fold', 'start', 'truth')}
POOLD['rec'] = np.concatenate([np.char.add(n + '|', p['rec'].astype(str)) for n, p in zip(['native'] + POOL, parts)])
for ch in 'AB': POOLD[ch] = np.concatenate([p[ch] for p in parts], 1)
print('pooled', len(common), 'recordings', len(POOLD['truth']), 'windows', flush=True)
G4 = sorted({tuple(np.round(np.array(w) / max(w), 3)) for w in itertools.product([0, .5, 1, 2], repeat=4) if sum(w) > 0})
READERS = {'N4': ('N', 'A'), 'N4m': ('N', 'B'), 'P4': ('P', 'A'), 'P4m': ('P', 'B')}
def fit(a):
    nm, f = a; src, ch = READERS[nm]; d = NAT if src == 'N' else POOLD; Z = d[ch]; tr = {D: np.flatnonzero((d['ds'] == D) & (d['fold'] != f)) for D in DS}; best = (1e9, None)
    for w in G4:
        l = sum(X.pmae(np.abs(BPM[np.tensordot(np.array(w), Z[:, tr[D]], 1).argmax(1)] - d['truth'][tr[D]]), d['person'][tr[D]])[1].mean() for D in DS if len(tr[D]))
        if l < best[0] - 1e-9: best = (l, w)
    w = np.array(best[1]); out = dict(w=w)
    for D in DS:
        if not len(tr[D]): continue
        rtr = X.runs_of(d, tr[D]); Etr = [np.tensordot(w, Z[:, r], 1) for r in rtr]; ref = float(np.mean(np.concatenate([E.max(1) for E in Etr]))); tt = np.concatenate([d['truth'][r] for r in rtr]); pp = np.concatenate([d['person'][r] for r in rtr]); b2 = (1e9, None)
        for g in P7.GAM:
            Eg = [Y2.qscale(E, g, ref) for E in Etr]
            for s in X.SIG:
                for ta in X.TAU:
                    M_ = [P7.marg(E, s, ta, ZERO) for E in Eg]
                    for est in ('median', 'mean'):
                        l = X.pmae(np.abs(np.concatenate([P7.point(x, est) for x in M_]) - tt), pp)[1].mean()
                        if l < b2[0] - 1e-9: b2 = (l, (g, s, ta, est))
        out[D] = b2[1] + (ref, float(np.median(d['truth'][tr[D]])))
    return (nm, f), out
if __name__ == '__main__':
    names = list(READERS)
    with Pool(20) as pool: FITS = dict(pool.map(fit, [(nm, f) for nm in names for f in range(5)], chunksize=1))
    def apply(d, nm):
        ch = READERS[nm][1]; pred = np.full(len(d['truth']), np.nan); const = np.full(len(d['truth']), np.nan); Z = d[ch]
        for f in range(5):
            m = FITS[(nm, f)]
            for D in DS:
                te = np.flatnonzero((d['ds'] == D) & (d['fold'] == f))
                if not len(te) or D not in m: continue
                g, s, ta, est, ref, cst = m[D]; const[te] = cst
                for r in X.runs_of(d, te): pred[r] = P7.point(P7.marg(Y2.qscale(np.tensordot(m['w'], Z[:, r], 1), g, ref), s, ta, ZERO), est)
        return pred, const
    ix = {}
    def boot(v):
        if len(v) not in ix: ix[len(v)] = np.random.default_rng(0).integers(0, len(v), (10000, len(v)))
        return np.quantile(v[ix[len(v)]].mean(1), [.025, .975])
    NP = {nm: apply(NAT, nm) for nm in names}; nkey = {k: i for i, k in enumerate(kk(NAT))}; rows = []
    for c in D_:
        d = D_[c]; PR = NP if c == 'native' else {nm: apply(d, nm) for nm in names}; ni = np.array([nkey.get(k, -1) for k in kk(d)])
        for D in DS:
            sel = np.flatnonzero((d['ds'] == D) & (ni >= 0))
            if not len(sel): continue
            t = d['truth'][sel]; per = d['person'][sel]; V = {nm: X.pmae(np.abs(PR[nm][0][sel] - t), per)[1] for nm in names}
            V['P4m_nat'] = X.pmae(np.abs(NP['P4m'][0][ni[sel]] - t), per)[1]; V['const'] = X.pmae(np.abs(PR['P4'][1][sel] - t), per)[1]
            cov = 1.0 if c == 'native' else d['cov'][D][0] / max(d['cov'][D][1], 1)
            dc = boot(V['P4m'] - V['const']); dp = boot(V['P4m'] - V['P4']); dn = boot(V['P4m'] - V['P4m_nat']); ci = boot(V['P4m'])
            g_ = ('a' if cov >= .9 else '-') + ('b' if dc[0] <= 0 else '-') + ('c' if dc[1] < 0 else '-'); q = PR['P4m'][0][sel]; q0 = PR['P4'][0][sel]
            print(f"{c:20s} {D:5s} people {len(V['P4m']):2d} windows {len(sel):4d} cover {cov:.3f} | " + ' '.join(f'{nm} {V[nm].mean():6.3f}' for nm in names) + f" const {V['const'].mean():6.3f}"
                  f" | P4m [{ci[0]:.2f};{ci[1]:.2f}] RMSE {np.sqrt(np.mean((q - t) ** 2)):5.2f} (P4 {np.sqrt(np.mean((q0 - t) ** 2)):5.2f}) r {np.corrcoef(q, t)[0, 1]:+.3f} (P4 {np.corrcoef(q0, t)[0, 1]:+.3f})"
                  f" | P4m-P4 {np.mean(V['P4m'] - V['P4']):+.3f} [{dp[0]:+.2f};{dp[1]:+.2f}] | vs native {np.mean(V['P4m'] - V['P4m_nat']):+.3f} [{dn[0]:+.2f};{dn[1]:+.2f}] | vs const {np.mean(V['P4m'] - V['const']):+.3f} [{dc[0]:+.2f};{dc[1]:+.2f}] | gates {g_}", flush=True)
            rows.append(dict(condition=c, dataset=D, people=len(V['P4m']), windows=len(sel), coverage=cov, **{nm: float(V[nm].mean()) for nm in names}, constant=float(V['const'].mean()),
                             p4m_lo=ci[0], p4m_hi=ci[1], rmse_p4m=float(np.sqrt(np.mean((q - t) ** 2))), rmse_p4=float(np.sqrt(np.mean((q0 - t) ** 2))), r_p4m=float(np.corrcoef(q, t)[0, 1]), r_p4=float(np.corrcoef(q0, t)[0, 1]),
                             d_p4=float(np.mean(V['P4m'] - V['P4'])), dp_lo=dp[0], dp_hi=dp[1], d_native=float(np.mean(V['P4m'] - V['P4m_nat'])), dn_lo=dn[0], dn_hi=dn[1],
                             d_const=float(np.mean(V['P4m'] - V['const'])), dc_lo=dc[0], dc_hi=dc[1], gates=g_, pooled_in_training=c in ['native'] + POOL))
    print('weights (CNN, POS, CHROM, experts) per fold:', {nm: [tuple(float(x) for x in FITS[(nm, f)]['w']) for f in range(5)] for nm in names}, flush=True)
    print('readout (gamma, sigma, tau, estimator) per fold:', {nm: [{D: FITS[(nm, f)][D][:4] for D in DS if D in FITS[(nm, f)]} for f in range(5)] for nm in names}, flush=True)
    with (W / 'out/z13_public_p4_prefilter.csv').open('w', newline='') as fh:
        w_ = csv.DictWriter(fh, list(rows[0])); w_.writeheader(); w_.writerows(rows)
