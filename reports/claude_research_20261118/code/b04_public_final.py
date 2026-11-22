"""Public sets, proposed readout = condition-aware spectral fusion + tracking across the windows of a recording. No retraining, no new inference:
everything is computed from the saved waveforms (out/public_spectra.npz, built by b01 and checked against the saved readouts).
  step 1  emission E_t(f) = sum_k w_k * standardised log-spectrum_k,t(f), k in {padding-fixed CNN, POS, CHROM}; one weight triple per condition,
          fitted on the other four person-folds (UBFC2 and MCD pooled, each dataset weighing the same).
  step 2  Viterbi path over the consecutive 10 s windows of a recording: sum_t tau*E_t(f_t) - (f_t - f_{t-1})^2 / (2 sigma^2); (sigma, tau) fitted on
          the other four person-folds of the same dataset; 'no tracking' is one of the options.
Reported per dataset and condition: CNN alone (the team's reported model), best single method, fusion only (window by window), fusion + tracking.
Person-first MAE, paired bootstrap over people (10,000, seed 0). Tracking uses the whole recording, so it is an offline (non-causal) estimate."""
import itertools, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118')
with np.load(W / 'out/public_spectra.npz') as z:
    BPM = z['bpm']; ds, cond, person, fold, recid, start, truth = z['dataset'], z['cond'], z['person'], z['fold'], z['recording'], z['start'], z['truth']
    def zs(S):
        l = np.log(S.astype(np.float64) + 1e-30); return (l - l.mean(1, keepdims=True)) / l.std(1, keepdims=True)
    Z = np.stack([zs(z[k]) for k in ['pad', 'pos', 'chrom']])
GW = sorted({tuple(np.round(np.array(w) / max(w), 3)) for w in itertools.product([0, .5, 1, 2], repeat=3) if sum(w) > 0})
SIG = [2., 4., 6., 10., 15., 25.]; TAU = [.05, .1, .2, .4, .8, 1.6]
PEN = {s: -((BPM[:, None] - BPM[None]) ** 2) / (2 * s ** 2) for s in SIG}
def pmae(err, per):
    d = defaultdict(list)
    for e, p in zip(err, per): d[p].append(e)
    k = sorted(d); return k, np.array([np.mean(d[p]) for p in k])
def emis(idx, w): return np.tensordot(np.array(w), Z[:, idx], 1)
def viterbi(E, sigma, tau):
    T, F = E.shape; pen = PEN[sigma]; sc = tau * E[0].copy(); back = np.zeros((T, F), int)
    for t in range(1, T):
        m = sc[:, None] + pen; back[t] = m.argmax(0); sc = m.max(0) + tau * E[t]
    path = np.zeros(T, int); path[-1] = sc.argmax()
    for t in range(T - 1, 0, -1): path[t - 1] = back[t][path[t]]
    return BPM[path]
def runs_of(sel):
    by = defaultdict(list)
    for i in sel: by[recid[i]].append(i)
    out = []
    for v in by.values():
        v = np.array(v)[np.argsort(start[v])]; cur = [v[0]]
        for a, b in zip(v[:-1], v[1:]):
            if abs(start[b] - start[a] - 10) < .6: cur.append(b)
            else: out.append(np.array(cur)); cur = [b]
        out.append(np.array(cur))
    return out
def task(args):
    Cn, f = args; best = (1e9, None)
    for w in GW:
        m = 0.
        for D in ['UBFC2', 'MCD']:
            tr = np.flatnonzero((cond == Cn) & (ds == D) & (fold != f)); m += pmae(np.abs(BPM[emis(tr, w).argmax(1)] - truth[tr]), person[tr])[1].mean()
        if m < best[0] - 1e-9: best = (m, w)
    w = best[1]; res = {'w': w, 'cfg': {}, 'fus': {}, 'trk': {}}
    for D in ['UBFC2', 'MCD']:
        tr = np.flatnonzero((cond == Cn) & (ds == D) & (fold != f)); te = np.flatnonzero((cond == Cn) & (ds == D) & (fold == f)); rtr = runs_of(tr); Etr = {r[0]: emis(r, w) for r in rtr}
        bestc = (pmae(np.abs(BPM[emis(tr, w).argmax(1)] - truth[tr]), person[tr])[1].mean(), None)          # no tracking
        for s in SIG:
            for ta in TAU:
                e = []; pp = []
                for r in rtr: e.extend(np.abs(viterbi(Etr[r[0]], s, ta) - truth[r])); pp.extend(person[r])
                m = pmae(e, pp)[1].mean()
                if m < bestc[0] - 1e-9: bestc = (m, (s, ta))
        res['cfg'][D] = bestc[1]; res['fus'][D] = (te, BPM[emis(te, w).argmax(1)])
        out = np.zeros(len(te)); pos_in = {i: k for k, i in enumerate(te)}
        for r in runs_of(te):
            p = BPM[emis(r, w).argmax(1)] if bestc[1] is None else viterbi(emis(r, w), *bestc[1])
            for i, v in zip(r, p): out[pos_in[i]] = v
        res['trk'][D] = (te, out)
    return Cn, f, res
if __name__ == '__main__':
    CONDS = ['native', 'c5prime_bf0', 'c5prime_default', 'c5']; jobs = [(c, f) for c in CONDS for f in range(5)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 20) as pool: R = pool.map(task, jobs)
    fus = np.full(len(truth), np.nan); trk = fus.copy(); info = defaultdict(dict)
    for Cn, f, res in R:
        info[Cn][f] = (res['w'], res['cfg'])
        for D in ['UBFC2', 'MCD']: fus[res['fus'][D][0]] = res['fus'][D][1]; trk[res['trk'][D][0]] = res['trk'][D][1]
    rng = np.random.default_rng(0); table = []
    for Cn in CONDS:
        print(f"\n=== {Cn} | weights (CNN, POS, CHROM) per fold: {[info[Cn][f][0] for f in range(5)]}")
        for D in ['UBFC2', 'MCD']:
            sel = np.flatnonzero((cond == Cn) & (ds == D)); t = truth[sel]; p = person[sel]
            cand = {'CNN alone (reported so far)': BPM[Z[0][sel].argmax(1)], 'POS': BPM[Z[1][sel].argmax(1)], 'CHROM': BPM[Z[2][sel].argmax(1)], 'fusion (window by window)': fus[sel], 'fusion + tracking': trk[sel]}
            ppl, base = pmae(np.abs(cand['CNN alone (reported so far)'] - t), p); ix = rng.integers(0, len(ppl), (10000, len(ppl)))
            print(f"  {D}: {len(sel)} windows / {len(ppl)} people | (sigma, tau) per fold: {[info[Cn][f][1][D] for f in range(5)]}")
            for k, q in cand.items():
                _, v = pmae(np.abs(q - t), p); ci = np.quantile(v[ix].mean(1), [.025, .975]); d = v - base; c1 = np.quantile(d[ix].mean(1), [.025, .975])
                print(f"     {k:28s} MAE {v.mean():6.3f} [{ci[0]:.2f};{ci[1]:.2f}] | vs CNN {d.mean():+.3f} [{c1[0]:+.2f};{c1[1]:+.2f}] | window RMSE {np.sqrt(np.mean((q-t)**2)):5.2f}  r {np.corrcoef(q,t)[0,1]:.3f} | off >10 bpm {100*np.mean(np.abs(q-t)>10):4.1f}%")
                table.append((Cn, D, k, float(v.mean()), float(ci[0]), float(ci[1]), float(d.mean()), float(c1[0]), float(c1[1])))
    import csv
    with (W / 'out/b04_public_final.csv').open('w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['condition', 'dataset', 'method', 'MAE', 'ci_lo', 'ci_hi', 'delta_vs_cnn', 'd_lo', 'd_hi']); w.writerows(table)
    np.savez_compressed(W / 'out/b04_public_predictions.npz', fusion=fus, tracked=trk, truth=truth, dataset=ds, cond=cond, person=person, recording=recid, start=start)
