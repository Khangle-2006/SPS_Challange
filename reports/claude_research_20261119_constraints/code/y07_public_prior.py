"""y07 - step 2 of the 20 Nov list (public part): a train-fold HR prior inside the pooled readout of the public estimator, so that
weak evidence falls back to the prior instead of returning a random spectral peak (x05: C1 with strong Gaussian noise is WORSE than the
no-video constant on UBFC2 and MCD). Same idea as the BTC reader (posterior = prior x evidence^beta), written for windows:
    potential of window t :  pi(f)^kappa * exp(tau * q_t^gamma * E_t(f))
       E_t   fused emission (weights of the base fit: standardised log-spectra of CNN, POS, CHROM)
       pi    person-weighted KDE (6 bpm) of the reference HR of the TRAINING windows of that set
       q_t   window quality = max_f E_t / mean training value of it (y02)
    random-walk HMM over the windows of a recording (sigma), forward-backward marginals, point estimate = marginal median or mean.
  base  : Viterbi path, no prior (x05, the 18 Nov estimator)      fbq : kappa = 0 (y02: FB + Q)      fbqp : kappa in {0, .25, .5, 1}
Everything (weights, sigma, tau, gamma, kappa, estimator, prior) is fitted on the 4 training folds.
  refit : fitted on the same condition (primary)        frozen : fitted on native, applied unchanged to the condition (secondary)
  pooled (POOL=1) : ONE reader fitted on the training folds of native and of every listed condition together (recordings common to
          all of them), applied unchanged to each condition - a single pipeline that is not told which condition it is looking at.
          EXTRA=tag:cond,... : conditions scored with that pooled reader although they were NOT in its training set.
Person-first MAE, paired bootstrap over people (10,000, seed 0). SHUFFLE=<seed>: control, reference HR shuffled between people of a set.
Usage: $PY code/y07_public_prior.py <workers> <tag:cond,tag:cond,...> [native: old|x01] [datasets]  -> out/y07_public_prior.csv"""
import sys, os, csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code'))
import x05_public_score as X
import y02_public_improve as Y2
BPM = X.BPM; DS = X.DS; KER = Y2.KER; GAM = [0., 1., 2., 4.]; KAP = [0., .25, .5, 1.]; BW = 6.
def prior(truth, person):
    c = defaultdict(int)
    for p in person: c[p] += 1
    w = np.array([1. / c[p] for p in person]); pr = (np.exp(-.5 * ((BPM[None] - np.asarray(truth)[:, None]) / BW) ** 2) * w[:, None]).sum(0); return np.log(pr / pr.sum() + 1e-12)
def marg(E, sigma, tau, lp):
    """forward-backward marginals of the random-walk HMM; lp = kappa * log prior, added to the log potential of every window."""
    L = tau * E + lp[None]; e = np.exp(L - L.max(1, keepdims=True)); K = KER[sigma]; T, G = e.shape; a = np.zeros((T, G)); b = np.ones((T, G)); a[0] = e[0] / e[0].sum()
    for t in range(1, T): v = (a[t - 1] @ K) * e[t]; a[t] = v / v.sum()
    for t in range(T - 2, -1, -1): v = K @ (b[t + 1] * e[t + 1]); b[t] = v / v.sum()
    g = a * b; return g / g.sum(1, keepdims=True)
def point(g, est): return g @ BPM if est == 'mean' else BPM[np.minimum((np.cumsum(g, 1) < .5).sum(1), len(BPM) - 1)]
def fit(d, Z, f):
    w, cfg = X.fitp(d, Z, f); m = dict(w=w, vit=cfg)
    for D in DS:
        tr = np.flatnonzero((d['ds'] == D) & (d['fold'] != f))
        if not len(tr): continue
        rtr = X.runs_of(d, tr); Etr = [np.tensordot(np.array(w), Z[:, r], 1) for r in rtr]; ref = float(np.mean(np.concatenate([E.max(1) for E in Etr]))); lp = prior(d['truth'][tr], d['person'][tr])
        tt = np.concatenate([d['truth'][r] for r in rtr]); pp = np.concatenate([d['person'][r] for r in rtr]); best = {0: (1e9, None), 1: (1e9, None)}
        for g in GAM:
            Eg = [Y2.qscale(E, g, ref) for E in Etr]
            for s in X.SIG:
                for ta in X.TAU:
                    for k in KAP:
                        G_ = [marg(E, s, ta, k * lp) for E in Eg]
                        for est in ('median', 'mean'):
                            loss = X.pmae(np.abs(np.concatenate([point(x, est) for x in G_]) - tt), pp)[1].mean()
                            for wp in (0, 1):
                                if (wp or k == 0) and loss < best[wp][0] - 1e-9: best[wp] = (loss, (g, s, ta, k, est))
        m[D] = dict(ref=ref, lp=lp, fbq=best[0][1], fbqp=best[1][1], const=float(np.median(d['truth'][tr])), loss=best[0][0])
    return m
def apply(d, Z, f, m):
    out = defaultdict(dict)
    for D in DS:
        te = np.flatnonzero((d['ds'] == D) & (d['fold'] == f))
        if not len(te) or D not in m: continue
        for r in X.runs_of(d, te):
            E = np.tensordot(np.array(m['w']), Z[:, r], 1); out['base'].update(zip(r.tolist(), BPM[E.argmax(1)] if m['vit'][D] is None else X.viterbi(E, *m['vit'][D])))
            for name in ('fbq', 'fbqp'):
                g, s, ta, k, est = m[D][name]; out[name].update(zip(r.tolist(), point(marg(Y2.qscale(E, g, m[D]['ref']), s, ta, k * m[D]['lp']), est)))
            out['const'].update({i: m[D]['const'] for i in r.tolist()})
    return out
def natfit(f): return fit(DATA['native'], DATA['native']['Z'], f)
def poolfit(f): return fit(POOLD, POOLD['Z'], f)
def pool_data(names, keep):
    """one data set made of the listed conditions restricted to the recordings `keep` (recording ids prefixed by the condition, so runs stay separate)."""
    ds = [X.restrict(DATA[n], keep) for n in names]; out = {k: np.concatenate([d[k] for d in ds]) for k in ('ds', 'person', 'fold', 'start', 'truth')}
    out['rec'] = np.concatenate([np.char.add(n + '|', d['rec'].astype(str)) for n, d in zip(names, ds)]); out['Z'] = np.concatenate([d['Z'] for d in ds], 1); return out
def hybrid(a, b):
    """CNN evidence of data set a with the POS / CHROM evidence of data set b, on the windows both have."""
    kb = {(r, round(float(s_), 3)): i for i, (r, s_) in enumerate(zip(b['rec'], b['start']))}; ia = [i for i, (r, s_) in enumerate(zip(a['rec'], a['start'])) if (r, round(float(s_), 3)) in kb]
    ib = [kb[(a['rec'][i], round(float(a['start'][i]), 3))] for i in ia]; out = {k: a[k][ia] for k in ('ds', 'person', 'fold', 'rec', 'start', 'truth')}
    out['Z'] = np.stack([a['Z'][0][ia], b['Z'][1][ib], b['Z'][2][ib]]); out['Zo'] = out['Z']; out['keys'] = set(a['keys']) & set(b['keys']); return out
def task(a):
    name, f = a; d = DATA[name]; m = NATFIT[f] if name == 'native' else fit(d, d['Z'], f); o = apply(d, d['Z'], f, m); z = apply(d, d['Z'], f, NATFIT[f])
    info = {D: (m[D]['fbq'], m[D]['fbqp'], m[D].get('loss')) for D in DS if D in m}; res = dict(o); res.update({'fz_' + k: v for k, v in z.items() if k != 'const'})
    if POOLFIT: res.update({'pl_' + k: v for k, v in apply(d, d['Z'], f, POOLFIT[f]).items() if k != 'const'})
    return name, f, res, info
def shuffle(d, seed):
    rng = np.random.default_rng(seed); tr = d['truth'].copy()
    for D in DS:
        ppl = sorted(set(d['person'][d['ds'] == D])); perm = dict(zip(ppl, rng.permutation(ppl)))
        for q in ppl:
            i = np.flatnonzero(d['person'] == q); tr[i] = np.resize(d['truth'][d['person'] == perm[q]], len(i))
    d['truth'] = tr
DATA = {}; NATFIT = None; POOLFIT = None; POOLD = None
if __name__ == '__main__':
    spec = [s.split(':') for s in sys.argv[2].split(',')]; ntag = sys.argv[3] if len(sys.argv) > 3 else 'old'
    if len(sys.argv) > 4: DS[:] = sys.argv[4].split(',')
    for tag in sorted({t for t, _ in spec}): DATA.update(X.load(tag, [c for t, c in spec if t == tag and c != 'native']))
    DATA['native'] = X.load_old_native() if ntag == 'old' else X.load(ntag, ['native'])['native']; nk = DATA['native']['keys']; POOLN = list(DATA)
    extra = [s.split(':') for s in os.environ.get('EXTRA', '').split(',') if s]          # evaluated with the pooled reader but NOT part of its training set
    for tag in sorted({t for t, _ in extra}): DATA.update(X.load(tag, [c for t, c in extra if t == tag]))
    DATA = {c: (d if c == 'native' else X.restrict(d, nk)) for c, d in DATA.items()}
    for h in filter(None, os.environ.get('HYBRID', '').split(',')):                 # name=cnn_condition+classical_condition
        name, src = h.split('='); a_, b_ = src.split('+'); DATA[name] = hybrid(DATA[a_], DATA[b_])
    if os.environ.get('SHUFFLE'):
        for d in DATA.values(): shuffle(d, int(os.environ['SHUFFLE']))
        print('CONTROL: reference HR shuffled between people, seed', os.environ['SHUFFLE'])
    print('recordings per condition:', {c: len(d['keys']) for c, d in DATA.items()}, flush=True)
    import pickle
    cache = W / f"out/y07_natfit_{ntag}_g{len(GAM)}_{'-'.join(DS)}{'_shuffle' + os.environ['SHUFFLE'] if os.environ.get('SHUFFLE') else ''}.pkl"
    if cache.exists(): NATFIT = pickle.load(cache.open('rb'))
    else:
        with Pool(5) as pool: NATFIT = dict(zip(range(5), pool.map(natfit, range(5))))
        pickle.dump(NATFIT, cache.open('wb'))
    if os.environ.get('POOL'):
        import hashlib
        common = set.intersection(*[set(DATA[c]['keys']) for c in POOLN]); pn = POOLN; POOLD = pool_data(pn, common)
        print(f'pooled reader: {len(pn)} conditions {pn} x {len(common)} common recordings = {len(POOLD["truth"])} windows | evaluated but not pooled: {[c for c in DATA if c not in pn]}', flush=True)
        pc = W / f"out/y07_poolfit_{hashlib.sha1(repr((sorted(pn), sorted(common), list(DS), GAM, KAP, os.environ.get('SHUFFLE'))).encode()).hexdigest()[:10]}.pkl"
        if pc.exists(): POOLFIT = pickle.load(pc.open('rb'))
        else:
            with Pool(5) as pool: POOLFIT = dict(zip(range(5), pool.map(poolfit, range(5))))
            pickle.dump(POOLFIT, pc.open('wb'))
        for D in DS: print(f'   pooled {D}: weights {[tuple(float(x) for x in POOLFIT[f]["w"]) for f in range(5)]} | fbqp (gamma, sigma, tau, kappa, est) per fold {[POOLFIT[f][D]["fbqp"] for f in range(5) if D in POOLFIT[f]]}', flush=True)
    names = ['native'] + [c for c in DATA if c != 'native']
    with Pool(int(sys.argv[1])) as pool: RES = pool.map(task, [(n, f) for n in names for f in range(5)], chunksize=1)
    P = defaultdict(lambda: defaultdict(dict)); INFO = defaultdict(dict)
    for name, f, res, info in RES:
        INFO[name][f] = info
        for k, v in res.items(): P[name][k].update(v)
    ix_of = {}; rows = []; dn = DATA['native']; natkey = {(dn['rec'][i], round(float(dn['start'][i]), 3)): i for i in range(len(dn['truth']))}; SAVE = {}
    def boot(v):
        if len(v) not in ix_of: ix_of[len(v)] = np.random.default_rng(0).integers(0, len(v), (10000, len(v)))
        return np.quantile(v[ix_of[len(v)]].mean(1), [.025, .975])
    for name in names:
        d = DATA[name]; dn = DATA['native']
        for D in DS:
            sel = np.flatnonzero(d['ds'] == D)
            if not len(sel): continue
            t = d['truth'][sel]; per = d['person'][sel]; cov = len(sel) / max(((dn['ds'] == D) & np.isin(dn['rec'], sorted(d['keys']))).sum(), 1); arr = lambda k: np.array([P[name][k][i] for i in sel.tolist()])
            ppl, vc = X.pmae(np.abs(arr('const') - t), per); print(f"\n=== {name} {D}: {len(sel)} windows / {len(ppl)} people, coverage {cov:.3f} | constant {vc.mean():.3f} | (gamma, sigma, tau, kappa, est) per fold, fbqp: {[INFO[name][f][D][1] for f in range(5) if D in INFO[name][f]]}")
            V = {}
            for k in ('base', 'fbq', 'fbqp', 'fz_base', 'fz_fbq', 'fz_fbqp') + (('pl_base', 'pl_fbq', 'pl_fbqp') if POOLFIT else ()):
                q = arr(k); _, v = X.pmae(np.abs(q - t), per); V[k] = v; ref = V['base' if k.startswith('pl') else ('fz_base' if k.startswith('fz') else 'base')]; ci = boot(v); c1 = boot(v - vc); c2 = boot(v - ref)
                r = float(np.corrcoef(q, t)[0, 1]) if np.ptp(q) > 0 else float('nan'); rmse = float(np.sqrt(np.mean((q - t) ** 2)))
                # same method on the native recording, same windows (paired)
                cm = [(n_, natkey[(d['rec'][i], round(float(d['start'][i]), 3))]) for n_, i in enumerate(sel.tolist()) if (d['rec'][i], round(float(d['start'][i]), 3)) in natkey]; dnat = (float('nan'),) * 3
                if cm and k in P['native']:
                    a_, b_ = map(list, zip(*cm)); _, v1 = X.pmae(np.abs(q[a_] - t[a_]), per[a_]); _, v0 = X.pmae(np.abs(np.array([P['native'][k][i] for i in b_]) - dn['truth'][b_]), per[a_]); c3 = boot(v1 - v0); dnat = (float(np.mean(v1 - v0)), float(c3[0]), float(c3[1]))
                print(f"   {k:8s} MAE {v.mean():6.3f} [{ci[0]:.2f};{ci[1]:.2f}] RMSE {rmse:6.3f} r {r:+.3f} | vs const {np.mean(v - vc):+.3f} [{c1[0]:+.2f};{c1[1]:+.2f}] | vs {'frozen ' if k.startswith('fz') else ('refit ' if k.startswith('pl') else '')}base {np.mean(v - ref):+.3f} [{c2[0]:+.2f};{c2[1]:+.2f}] | vs native, same method {dnat[0]:+.3f} [{dnat[1]:+.2f};{dnat[2]:+.2f}]", flush=True)
                rows.append(dict(condition=name, dataset=D, windows=len(sel), people=len(ppl), coverage=cov, method=k, MAE=float(v.mean()), lo=float(ci[0]), hi=float(ci[1]), RMSE=rmse, r=r, constant=float(vc.mean()),
                                 d_const=float(np.mean(v - vc)), dc_lo=float(c1[0]), dc_hi=float(c1[1]), d_base=float(np.mean(v - ref)), db_lo=float(c2[0]), db_hi=float(c2[1]), d_native=dnat[0], dn_lo=dnat[1], dn_hi=dnat[2], pooled_in_training=name in (POOLN if POOLFIT else [])))
                SAVE[f'{name}|{D}|{k}'] = q
            SAVE[f'{name}|{D}|truth'] = t; SAVE[f'{name}|{D}|person'] = per; SAVE[f'{name}|{D}|rec'] = d['rec'][sel]; SAVE[f'{name}|{D}|start'] = d['start'][sel]
    if os.environ.get('NESTED'):                                                    # per fold, the variant with the lowest TRAINING loss (fbq); scored on the native windows
        vs = os.environ['NESTED'].split(','); print(f'\n=== nested choice among {vs} (lowest training-fold loss of fbq; windows of native)')
        for D in DS:
            seln = np.flatnonzero(dn['ds'] == D); keyn = [(dn['rec'][i], round(float(dn['start'][i]), 3)) for i in seln]; t = dn['truth'][seln]; per = dn['person'][seln]; fo = dn['fold'][seln]
            maps = {v: {(DATA[v]['rec'][i], round(float(DATA[v]['start'][i]), 3)): P[v]['fbq'][i] for i in np.flatnonzero(DATA[v]['ds'] == D).tolist()} for v in vs}
            choice = {f: min(vs, key=lambda v: INFO[v][f][D][2]) for f in range(5)}; ok = np.array([all(k in maps[v] for v in vs) for k in keyn])
            q = np.array([maps[choice[int(f_)]].get(k, np.nan) for k, f_ in zip(keyn, fo)]); qn = np.array([P['native']['fbq'][i] for i in seln.tolist()]); qb = np.array([P['native']['base'][i] for i in seln.tolist()])
            _, v1 = X.pmae(np.abs(q - t)[ok], per[ok]); _, v0 = X.pmae(np.abs(qn - t)[ok], per[ok]); _, vb = X.pmae(np.abs(qb - t)[ok], per[ok]); c0 = boot(v1 - v0); cb = boot(v1 - vb)
            print(f"   {D}: windows {int(ok.sum())}/{len(ok)} | chosen per fold {[choice[f] for f in range(5)]} | nested MAE {v1.mean():.3f} RMSE {np.sqrt(np.mean((q - t)[ok] ** 2)):.3f} | vs native fbq {np.mean(v1 - v0):+.3f} [{c0[0]:+.2f};{c0[1]:+.2f}] | vs native base {np.mean(v1 - vb):+.3f} [{cb[0]:+.2f};{cb[1]:+.2f}]"
                  f" | training losses fold 0: { {v: round(INFO[v][0][D][2], 3) for v in vs} }", flush=True)
    if not os.environ.get('SHUFFLE'):
        ot = os.environ.get('OUTTAG', 'pool' if POOLFIT else 'refit')
        with (W / f"out/y07_public_prior_{ot}.csv").open('w', newline='') as fh:
            w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
        np.savez_compressed(W / f'out/y07_pred_{ot}.npz', **SAVE)
