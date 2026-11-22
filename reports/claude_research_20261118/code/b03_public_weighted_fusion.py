"""Public sets: one set of fusion weights per condition, fitted on the other four person-folds (UBFC2 and MCD pooled, each dataset
weighing the same), applied to the held-out fold. score(f) = sum_k w_k * standardised log-spectrum_k(f), k in {CNN, POS, CHROM}; argmax.
Compared with the padding-fixed CNN alone (the team's reported model), the best single method and the unweighted three-way fusion.
Person-first MAE, paired bootstrap over people (10,000, seed 0). Input: out/public_spectra.npz."""
import itertools
import numpy as np
from collections import defaultdict
from pathlib import Path
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118')
z = np.load(W / 'out/public_spectra.npz'); BPM = z['bpm']; ds, cond, person, fold, truth = z['dataset'], z['cond'], z['person'], z['fold'], z['truth']
def zs(S):
    l = np.log(S.astype(np.float64) + 1e-30); return (l - l.mean(1, keepdims=True)) / l.std(1, keepdims=True)
Z = np.stack([zs(z[k]) for k in ['pad', 'pos', 'chrom']])          # [3, n, F]
GRIDW = [w for w in itertools.product([0, .5, 1, 2], repeat=3) if sum(w) > 0 and (max(w) == 1 or (max(w) == 2 and min(x for x in w if x > 0) >= 1) or sum(x > 0 for x in w) == 1)]
GRIDW = sorted({tuple(np.round(np.array(w) / max(w), 3)) for w in GRIDW})
def pmae(err, per):
    d = defaultdict(list)
    for e, p in zip(err, per): d[p].append(e)
    k = sorted(d); return k, np.array([np.mean(d[p]) for p in k])
def predict(idx, w): return BPM[np.tensordot(np.array(w), Z[:, idx], 1).argmax(1)]
rng = np.random.default_rng(0); total = {}
for Cn in ['native', 'c5prime_bf0', 'c5prime_default', 'c5']:
    pred = np.full(len(truth), np.nan); chosen = []
    for f in range(5):
        best = (1e9, None)
        for w in GRIDW:
            m = 0.
            for D in ['UBFC2', 'MCD']:
                tr = np.flatnonzero((cond == Cn) & (ds == D) & (fold != f)); m += pmae(np.abs(predict(tr, w) - truth[tr]), person[tr])[1].mean()
            if m < best[0] - 1e-9: best = (m, w)
        chosen.append(best[1]); te = np.flatnonzero((cond == Cn) & (fold == f)); pred[te] = predict(te, best[1])
    print(f"\n=== {Cn}: weights (CNN, POS, CHROM) chosen per fold: {chosen}")
    for D in ['UBFC2', 'MCD']:
        sel = np.flatnonzero((cond == Cn) & (ds == D)); t = truth[sel]; p = person[sel]
        cand = {'CNN (padding-fixed)': predict(sel, (1, 0, 0)), 'POS': predict(sel, (0, 1, 0)), 'CHROM': predict(sel, (0, 0, 1)), 'equal 3-way': predict(sel, (1, 1, 1)), 'fitted weights (nested)': pred[sel]}
        ppl, base = pmae(np.abs(cand['CNN (padding-fixed)'] - t), p); ix = rng.integers(0, len(ppl), (10000, len(ppl)))
        singles = {k: pmae(np.abs(cand[k] - t), p)[1] for k in list(cand)[:3]}; bs = min(singles, key=lambda k: singles[k].mean())
        print(f"  {D}: {len(sel)} windows / {len(ppl)} people; best single: {bs}")
        for k, q in cand.items():
            _, v = pmae(np.abs(q - t), p); ci = np.quantile(v[ix].mean(1), [.025, .975]); d = v - base; c1 = np.quantile(d[ix].mean(1), [.025, .975]); d2 = v - singles[bs]; c2 = np.quantile(d2[ix].mean(1), [.025, .975])
            rmse = np.sqrt(np.mean((q - t) ** 2)); r = np.corrcoef(q, t)[0, 1]
            print(f"     {k:24s} MAE {v.mean():6.3f} [{ci[0]:.2f};{ci[1]:.2f}] | vs CNN {d.mean():+.3f} [{c1[0]:+.2f};{c1[1]:+.2f}] | vs best single {d2.mean():+.3f} [{c2[0]:+.2f};{c2[1]:+.2f}] | window RMSE {rmse:5.2f} r {r:.3f} | off >10 bpm {100*np.mean(np.abs(q-t)>10):4.1f}%")
            total[(Cn, D, k)] = v.mean()
np.save(W / 'out/b03_summary.npy', np.array([(a, b, c, v) for (a, b, c), v in total.items()], dtype=object), allow_pickle=True)
