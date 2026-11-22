"""u03 (22 Nov, Claude): error anatomy of the best public readers on NATIVE windows (no new model, no new fit rule):
N4m (native-fit, CNN pre-filtered 0.1 s + POS + CHROM + expert mean) and P4m (same channels, pooled fit) of z13.
Dumps per-window predictions -> out/u03_native_windows.npz and prints where the MCD/UBFC2 MAE comes from (tails, harmonics,
before/after sessions, recordings). Fits use training folds only (z13.fit)."""
import sys
from pathlib import Path
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code'))
import z13_public_p4_prefilter as Z
X, P7, Y2 = Z.X, Z.P7, Z.Y2
NAT = Z.NAT; BPM = Z.BPM; ZERO = Z.ZERO
def apply(FITS, d, nm):
    ch = Z.READERS[nm][1]; pred = np.full(len(d['truth']), np.nan); M = d[ch]
    for f in range(5):
        m = FITS[(nm, f)]
        for D in Z.DS:
            te = np.flatnonzero((d['ds'] == D) & (d['fold'] == f))
            if not len(te) or D not in m: continue
            g, s, ta, est, ref, cst = m[D]
            for r in X.runs_of(d, te): pred[r] = P7.point(P7.marg(Y2.qscale(np.tensordot(m['w'], M[:, r], 1), g, ref), s, ta, ZERO), est)
    return pred
if __name__ == '__main__':
    names = ['N4m', 'P4m']
    with Pool(10) as pool: FITS = dict(pool.map(Z.fit, [(nm, f) for nm in names for f in range(5)], chunksize=1))
    P = {nm: apply(FITS, NAT, nm) for nm in names}
    t = NAT['truth']; ds = NAT['ds']; per = NAT['person']; rec = NAT['rec'].astype(str)
    chan = {nm: BPM[NAT['B'][i].argmax(1)] for i, nm in enumerate(['cnn_pf', 'pos', 'chrom', 'experts'])}
    np.savez_compressed(W / 'out/u03_native_windows.npz', truth=t, ds=ds, person=per, rec=rec, start=NAT['start'], fold=NAT['fold'], **{'pred_' + k: v for k, v in P.items()}, **{'peak_' + k: v for k, v in chan.items()})
    for D in Z.DS:
        m = ds == D
        for nm in names:
            e = np.abs(P[nm][m] - t[m]); pm = X.pmae(e, per[m])[1].mean()
            print(f'{D:5s} {nm}: person-MAE {pm:.3f} | window-MAE {e.mean():.3f} median {np.median(e):.2f} | share of windows |err|>3 {np.mean(e > 3):.3f}, >5 {np.mean(e > 5):.3f}, >10 {np.mean(e > 10):.3f}'
                  f' | MAE if windows >5 were perfect {np.where(e > 5, 0, e).mean():.3f} | sum-share of >5 windows {e[e > 5].sum() / e.sum():.2f}')
        e = np.abs(P['P4m'][m] - t[m]); big = e > 5; tt = t[m]; pp = P['P4m'][m]; rr = rec[m]
        ratio = pp[big] / tt[big]
        print(f'   P4m windows >5 bpm: {int(big.sum())} | pred/truth ratio: <0.6 {np.mean(ratio < .6):.2f}, 0.6-0.9 {np.mean((ratio >= .6) & (ratio < .9)):.2f}, 0.9-1.1 {np.mean((ratio >= .9) & (ratio <= 1.1)):.2f}, 1.1-1.6 {np.mean((ratio > 1.1) & (ratio <= 1.6)):.2f}, >1.6 {np.mean(ratio > 1.6):.2f}'
              f' | truth HR of big-error windows median {np.median(tt[big]) if big.any() else np.nan:.0f} vs all {np.median(tt):.0f}')
        for k, v in chan.items():
            ev = np.abs(v[m] - tt); print(f'   channel {k:8s} raw-peak window-MAE {ev.mean():6.2f} | within 3 bpm {np.mean(ev <= 3):.3f}')
        anyok = np.min(np.stack([np.abs(v[m] - tt) for v in chan.values()]), 0) <= 3
        print(f'   big-error windows where SOME channel peak is within 3 bpm of truth: {np.mean(anyok[big]) if big.any() else np.nan:.2f} (oracle channel choice)')
        if D == 'MCD':
            sess = np.array(['after' if '_after' in x else ('before' if '_before' in x else 'other') for x in rr])
            for s_ in ('before', 'after'):
                q = sess == s_; ee = e[q]; print(f'   session {s_:6s}: windows {int(q.sum())} window-MAE {ee.mean():.3f} share >5 {np.mean(ee > 5):.3f} truth HR median {np.median(tt[q]):.0f}')
            byrec = defaultdict(list)
            for x, v in zip(rr, e): byrec[x].append(v)
            worst = sorted(byrec.items(), key=lambda kv: -np.mean(kv[1]))[:12]
            print('   worst recordings (mean |err|, windows, share >5):', [(k[-40:], round(float(np.mean(v)), 2), len(v), round(float(np.mean(np.array(v) > 5)), 2)) for k, v in worst])
            tot = e.sum(); top = sum(np.sum(v) for _, v in worst[:10]); print(f'   share of total |err| from the 10 worst recordings: {top / tot:.2f} (of {len(byrec)})')
            dt = np.abs(np.diff(tt)); same = rr[1:] == rr[:-1]; print(f'   truth jumps between consecutive windows of one recording: median {np.median(dt[same]):.2f}, >8 bpm {np.mean(dt[same] > 8):.3f}')
