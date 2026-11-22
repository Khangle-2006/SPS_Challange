"""z03 - public sets, native: do the two accepted public upgrades ADD UP?  (a) 0.1 s moving average of the crops before the team CNN
(x13 'nat_ma3', POS/CHROM unchanged = y07 'hyb_ma3'), (b) the reader_v4 experts as a 4th channel (y12). Same windows for all
(2,152 windows where every expert has an output), same fitting as y12 (fusion weights from {0,.5,1,2}^k and forward-backward with
quality weight, both on the training folds only). Paired person bootstrap (10,000, seed 0).
Usage: $PY code/z03_public_prefilter_experts.py"""
import sys, itertools, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
import x05_public_score as X
import y02_public_improve as Y2
import y07_public_prior as P7
MODELS = ['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']; BPM = X.BPM; DS = X.DS; ZERO = np.zeros(len(BPM))
jobs = [j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] in DS]; J = {j['recording']: j for j in jobs}
nat = X.load_old_native(); ma3 = X.load('x13', ['nat_ma3'])['nat_ma3']; hyb = P7.hybrid(ma3, nat)
key = lambda d: [(r, round(float(s), 3)) for r, s in zip(d['rec'], d['start'])]
kn = {k: i for i, k in enumerate(key(nat))}; kh = {k: i for i, k in enumerate(key(hyb))}
ZE = []; okE = np.ones(len(kn), bool)
for m in MODELS:
    tab = {}
    for rec in sorted(nat['keys']):
        p = W / 'out/y12' / m / (J[rec]['key'] + '.npz')
        with np.load(p) as z:
            for s, sp in zip(z['start'], z['spec']):
                if np.isfinite(sp).all(): tab[(rec, round(float(s), 3))] = sp
    S = np.array([tab.get(k, np.full(len(BPM), np.nan)) for k in key(nat)]); h = np.isfinite(S).all(1); okE &= h; Zm = np.zeros((len(S), len(BPM))); Zm[h] = X.zs(S[h]); ZE.append(Zm)
ZE = np.mean(ZE, 0); keep = np.array([okE[i] and (k in kh) for k, i in kn.items()]); ii = np.flatnonzero(keep); ih = np.array([kh[k] for k in np.array(key(nat), dtype=object)[ii].tolist()]) if False else np.array([kh[key(nat)[i]] for i in ii])
d = {k: nat[k][ii] for k in ('ds', 'person', 'fold', 'rec', 'start', 'truth')}
CH = dict(cnn=nat['Z'][0][ii], cnn_ma3=hyb['Z'][0][ih], pos=nat['Z'][1][ii], chrom=nat['Z'][2][ii], experts=ZE[ii])
print('windows', len(ii), {D: int((d['ds'] == D).sum()) for D in DS}, flush=True)
SETS = {'base (cnn, pos, chrom)': ['cnn', 'pos', 'chrom'], 'prefilter (cnn_ma3, pos, chrom)': ['cnn_ma3', 'pos', 'chrom'],
        'experts (cnn, pos, chrom, experts)': ['cnn', 'pos', 'chrom', 'experts'], 'both (cnn_ma3, pos, chrom, experts)': ['cnn_ma3', 'pos', 'chrom', 'experts']}
def fit(a):
    name, f = a; chs = SETS[name]; Z = np.stack([CH[c] for c in chs]); tr = {D: np.flatnonzero((d['ds'] == D) & (d['fold'] != f)) for D in DS}; best = (1e9, None)
    for w in sorted({tuple(np.round(np.array(w) / max(w), 3)) for w in itertools.product([0, .5, 1, 2], repeat=len(chs)) if sum(w) > 0}):
        l = sum(X.pmae(np.abs(BPM[np.tensordot(np.array(w), Z[:, tr[D]], 1).argmax(1)] - d['truth'][tr[D]]), d['person'][tr[D]])[1].mean() for D in DS)
        if l < best[0] - 1e-9: best = (l, w)
    w = np.array(best[1]); pred = {}
    for D in DS:
        rtr = X.runs_of(d, tr[D]); Etr = [np.tensordot(w, Z[:, r], 1) for r in rtr]; ref = float(np.mean(np.concatenate([E.max(1) for E in Etr]))); tt = np.concatenate([d['truth'][r] for r in rtr]); pp = np.concatenate([d['person'][r] for r in rtr]); b2 = (1e9, None)
        for g in P7.GAM:
            Eg = [Y2.qscale(E, g, ref) for E in Etr]
            for s in X.SIG:
                for ta in X.TAU:
                    M_ = [P7.marg(E, s, ta, ZERO) for E in Eg]
                    for est in ('median', 'mean'):
                        l = X.pmae(np.abs(np.concatenate([P7.point(x, est) for x in M_]) - tt), pp)[1].mean()
                        if l < b2[0] - 1e-9: b2 = (l, (g, s, ta, est))
        g, s, ta, est = b2[1]; te = np.flatnonzero((d['ds'] == D) & (d['fold'] == f))
        for r in X.runs_of(d, te): pred.update(zip(r.tolist(), P7.point(P7.marg(Y2.qscale(np.tensordot(w, Z[:, r], 1), g, ref), s, ta, ZERO), est)))
    return name, f, tuple(w), pred
if __name__ == '__main__':
    with Pool(10) as pool: RES = pool.map(fit, [(nm, f) for nm in SETS for f in range(5)], chunksize=1)
    P = {nm: {} for nm in SETS}; Wt = {nm: [] for nm in SETS}
    for nm, f, w, pr in RES: P[nm].update(pr); Wt[nm].append(w)
    ix = {}
    def boot(v):
        if len(v) not in ix: ix[len(v)] = np.random.default_rng(0).integers(0, len(v), (10000, len(v)))
        return np.quantile(v[ix[len(v)]].mean(1), [.025, .975])
    for D in DS:
        sel = np.flatnonzero(d['ds'] == D); t = d['truth'][sel]; per = d['person'][sel]; V = {}
        Q_ = {nm: np.array([P[nm][i] for i in sel]) for nm in SETS}; V = {nm: X.pmae(np.abs(Q_[nm] - t), per)[1] for nm in SETS}
        for nm in SETS:
            q = Q_[nm]; ci = boot(V[nm]); ref = V['base (cnn, pos, chrom)']; c2 = boot(V[nm] - ref); c3 = boot(V[nm] - V['experts (cnn, pos, chrom, experts)'])
            print(f"{D:5s} {nm:38s} MAE {V[nm].mean():6.3f} [{ci[0]:.2f};{ci[1]:.2f}] RMSE {np.sqrt(np.mean((q - t) ** 2)):5.2f} r {np.corrcoef(q, t)[0, 1]:+.3f} | vs base {np.mean(V[nm] - ref):+.3f} [{c2[0]:+.2f};{c2[1]:+.2f}] | vs experts {np.mean(V[nm] - V['experts (cnn, pos, chrom, experts)']):+.3f} [{c3[0]:+.2f};{c3[1]:+.2f}] | weights {Wt[nm]}", flush=True)
