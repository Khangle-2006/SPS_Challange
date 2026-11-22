"""Public sets scored the way BTC is scored: one rate per 30 s segment (three consecutive 10 s windows), reference = mean of the three window references.
Estimators per segment:
  cnn_med3   median of the three window arg-maxima of the padding-fixed CNN (the team's video-level rule)
  *_avg      arg-max of the window-averaged spectrum (CNN alone / fused)
  fused_med3 median of the three window arg-maxima of the fused spectrum
  fused_trk  mean of the three values of the Viterbi path through the fused spectra of the whole recording (sigma 10, tau 1.6; offline)
Fusion weights (CNN, POS, CHROM) per condition as selected in every fold of b04: native 1/1/0.5, bf0 1/0.25/0.5, default 1/0/0, c5 1/0/0.5.
Person-first MAE over segments, paired bootstrap over people (10,000, seed 0). Nothing is fitted in this script."""
import sys, numpy as np
sys.path.insert(0, '/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118/code')
import b04_public_final as B
WTS = {'native': (1, 1, .5), 'c5prime_bf0': (1, .25, .5), 'c5prime_default': (1, 0, 0), 'c5': (1, 0, .5)}
rng = np.random.default_rng(0)
print(f"{'dataset cond':22s} segments people | cnn_med3 | cnn_avg | pos_med3 | fused_avg | fused_med3 | fused_trk | fused_med3 vs cnn_med3 [CI] | fused_trk vs cnn_med3 [CI]")
for D in ['UBFC2', 'MCD']:
    for Cn in ['native', 'c5prime_bf0', 'c5prime_default', 'c5']:
        sel = np.flatnonzero((B.cond == Cn) & (B.ds == D)); segs = []; trk = {}
        for r in B.runs_of(sel):
            p = B.viterbi(B.emis(r, WTS[Cn]), 10., 1.6)
            for i, v in zip(r, p): trk[i] = v
            for k in range(0, len(r) - 2, 3): segs.append(r[k:k + 3])
        t = np.array([B.truth[s].mean() for s in segs]); pp = [B.person[s[0]] for s in segs]
        def avg(w): return np.array([B.BPM[B.emis(s, w).mean(0).argmax()] for s in segs])
        def med(w): return np.array([np.median(B.BPM[B.emis(s, w).argmax(1)]) for s in segs])
        est = {'cnn_med3': med((1, 0, 0)), 'cnn_avg': avg((1, 0, 0)), 'pos_med3': med((0, 1, 0)), 'fused_avg': avg(WTS[Cn]), 'fused_med3': med(WTS[Cn]), 'fused_trk': np.array([np.mean([trk[i] for i in s]) for s in segs])}
        ppl, base = B.pmae(np.abs(est['cnn_med3'] - t), pp); ix = rng.integers(0, len(ppl), (10000, len(ppl))); v = {k: B.pmae(np.abs(e - t), pp)[1] for k, e in est.items()}
        d = v['fused_med3'] - base; ci = np.quantile(d[ix].mean(1), [.025, .975]); d2 = v['fused_trk'] - base; c2 = np.quantile(d2[ix].mean(1), [.025, .975])
        print(f"{D:6s} {Cn:15s} {len(segs):6d} {len(ppl):5d}   | " + ' | '.join(f"{v[k].mean():6.3f}" for k in est) + f" | {d.mean():+.3f} [{ci[0]:+.2f};{ci[1]:+.2f}] | {d2.mean():+.3f} [{c2[0]:+.2f};{c2[1]:+.2f}]")
