"""Public sets: does fusing the saved CNN / POS / CHROM spectra per window, and tracking the rate across the windows of a recording, lower the MAE?
Input: out/public_spectra.npz (b01). Person-first MAE (mean over a person's windows, then over people); paired bootstrap over people, 10,000 draws, seed 0.
Fusion has no fitted parameter: sum of standardised log-spectra. Tracking (Viterbi over consecutive 10 s windows) has two numbers, the transition
width sigma (bpm per 10 s step) and the emission weight tau; both are fitted on the other four person-folds of the same dataset and condition."""
import numpy as np
from collections import defaultdict
from pathlib import Path
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118')
z = np.load(W / 'out/public_spectra.npz'); BPM = z['bpm']
ds, cond, person, fold, recid, start, truth = z['dataset'], z['cond'], z['person'], z['fold'], z['recording'], z['start'], z['truth']
def zs(S):
    l = np.log(S.astype(np.float64) + 1e-30); return (l - l.mean(1, keepdims=True)) / l.std(1, keepdims=True)
Z = {k: zs(z[k]) for k in ['pad', 'pos', 'chrom']}
EM = {'pad': Z['pad'], 'pos': Z['pos'], 'chrom': Z['chrom'], 'pos+chrom': Z['pos'] + Z['chrom'], 'pad+pos': Z['pad'] + Z['pos'], 'pad+chrom': Z['pad'] + Z['chrom'], 'pad+pos+chrom': Z['pad'] + Z['pos'] + Z['chrom']}
def person_mae(err, per):
    d = defaultdict(list)
    for e, p in zip(err, per): d[p].append(e)
    ppl = sorted(d); return ppl, np.array([np.mean(d[p]) for p in ppl])
def viterbi(E, sigma, tau):
    """E: [T, F] emission scores of consecutive windows; returns the best path in bpm."""
    T, F = E.shape; pen = -((BPM[:, None] - BPM[None]) ** 2) / (2 * sigma ** 2); sc = tau * E[0].copy(); back = np.zeros((T, F), int)
    for t in range(1, T):
        m = sc[:, None] + pen; back[t] = m.argmax(0); sc = m.max(0) + tau * E[t]
    path = np.zeros(T, int); path[-1] = sc.argmax()
    for t in range(T - 1, 0, -1): path[t - 1] = back[t][path[t]]
    return BPM[path]
def sequences(idx):
    """split the windows of one recording into runs of consecutive 10 s windows."""
    idx = idx[np.argsort(start[idx])]; runs = [[idx[0]]]
    for a, b in zip(idx[:-1], idx[1:]):
        if abs(start[b] - start[a] - 10) < .6: runs[-1].append(b)
        else: runs.append([b])
    return [np.array(r) for r in runs]
SIG = [2., 4., 6., 10.]; TAU = [.05, .1, .2, .4, .8]
rng = np.random.default_rng(0)
for D in ['UBFC2', 'MCD']:
    for Cn in ['native', 'c5prime_bf0', 'c5prime_default', 'c5']:
        sel = np.flatnonzero((ds == D) & (cond == Cn))
        if len(sel) == 0: continue
        recs = defaultdict(list)
        for i in sel: recs[recid[i]].append(i)
        runs = [r for v in recs.values() for r in sequences(np.array(v))]
        pred = {}
        for k, E in EM.items(): pred[k] = BPM[E[sel].argmax(1)]
        # tracking on selected emissions, nested by person fold
        for k in ['pad', 'pos', 'pad+pos+chrom', 'pos+chrom']:
            out = np.full(len(z['truth']), np.nan); chosen = []
            for f in range(5):
                tr = [r for r in runs if fold[r[0]] != f]; te = [r for r in runs if fold[r[0]] == f]
                best = (1e9, None)
                for s in SIG:
                    for ta in TAU:
                        e = []; pp = []
                        for r in tr:
                            e.extend(np.abs(viterbi(EM[k][r], s, ta) - truth[r])); pp.extend(person[r])
                        m = person_mae(e, pp)[1].mean()
                        if m < best[0] - 1e-9: best = (m, (s, ta))
                chosen.append(best[1])
                for r in te: out[r] = viterbi(EM[k][r], *best[1])
            pred['track(' + k + ')'] = out[sel]; pred['_cfg_' + k] = chosen
        ppl, base = person_mae(np.abs(pred['pad'] - truth[sel]), person[sel]); ix = rng.integers(0, len(ppl), (10000, len(ppl)))
        singles = {k: person_mae(np.abs(pred[k] - truth[sel]), person[sel])[1] for k in ['pad', 'pos', 'chrom']}; bs = min(singles, key=lambda k: singles[k].mean())
        print(f"\n=== {D} {Cn}: {len(sel)} windows, {len(recs)} recordings, {len(ppl)} people | best single = {bs} ({singles[bs].mean():.3f})")
        for k in pred:
            if k.startswith('_'): continue
            _, v = person_mae(np.abs(pred[k] - truth[sel]), person[sel]); ci = np.quantile(v[ix].mean(1), [.025, .975]); d1 = v - base; c1 = np.quantile(d1[ix].mean(1), [.025, .975]); d2 = v - singles[bs]; c2 = np.quantile(d2[ix].mean(1), [.025, .975])
            big = np.mean(np.abs(pred[k] - truth[sel]) > 10)
            cfg = pred.get('_cfg_' + k[6:-1]) if k.startswith('track(') else None
            print(f"  {k:22s} MAE {v.mean():6.3f} [{ci[0]:.2f};{ci[1]:.2f}] | vs CNN {d1.mean():+.3f} [{c1[0]:+.2f};{c1[1]:+.2f}] | vs best single {d2.mean():+.3f} [{c2[0]:+.2f};{c2[1]:+.2f}] | windows off >10 bpm: {100*big:4.1f}%" + (f" | (sigma,tau) per fold {cfg}" if cfg else ''))
