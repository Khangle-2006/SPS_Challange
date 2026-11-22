"""Negative control for the tracking step: if the gain comes from the continuity of the pulse rate inside a recording, it must vanish when the
windows of each recording are put in random order before tracking (predictions are mapped back to their own windows). MCD and UBFC2, native and
c5prime_bf0, fixed settings (weights 1, 1, 0.5; sigma 10; tau 1.6) so that nothing is fitted here."""
import sys, numpy as np
sys.path.insert(0, '/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118/code')
import b04_public_final as B
rng = np.random.default_rng(0)
for D in ['MCD', 'UBFC2']:
    for Cn in ['native', 'c5prime_bf0', 'c5']:
        sel = np.flatnonzero((B.cond == Cn) & (B.ds == D)); w = (1, 1, .5) if Cn == 'native' else (1, 0, 0); runs = B.runs_of(sel)
        e0, e1, e2, pp = [], [], [], []
        for r in runs:
            E = B.emis(r, w); t = B.truth[r]; pp.extend(B.person[r]); e0.extend(np.abs(B.BPM[E.argmax(1)] - t)); e1.extend(np.abs(B.viterbi(E, 10., 1.6) - t))
            acc = np.zeros(len(r))
            for _ in range(5):
                o = rng.permutation(len(r)); p = B.viterbi(E[o], 10., 1.6); q = np.zeros(len(r)); q[o] = p; acc += np.abs(q - t) / 5
            e2.extend(acc)
        f = lambda e: B.pmae(e, pp)[1].mean()
        print(f"{D:6s} {Cn:12s} windows {len(sel):5d} runs {len(runs):4d} | no tracking {f(e0):6.3f} | tracking, true order {f(e1):6.3f} | tracking, shuffled order (mean of 5) {f(e2):6.3f}")
