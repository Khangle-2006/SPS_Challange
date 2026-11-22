"""y09 - BTC dev: ONE frozen pipeline under every constraint (no per-condition refit; the reader is never told the condition).
  reader      : reader_v3 fitted on the NATIVE training folds only (5 person-disjoint folds), applied unchanged to every condition.
  class rule  : 'low' = the input FILE is a low-bitrate re-encode (< 4 Mb/s; on dev exactly the 67 files of the old pixel rule: low <= 2.6,
                camera originals >= 6 Mb/s). Frames changed in memory (C1, C2, C4, C6, C7 as specified by the organisers: no re-encoding)
                keep the class of their file; a C5 re-encode at <= 500 kbps is 'low' by the same rule; the C5 split stream is not a
                video re-encode (36x36 crops of the device's own capture) and keeps the class of the capture.
  front end   : the 20 Nov one (template-tracked hold; detector aid in low light) where it was extracted (x09), else the standard one.
'refit' (reader fitted on the same condition) is printed as a secondary column. Person-first MAE, paired bootstrap over people (10,000, seed 0).
Gates (PREREG.md): a = evidence for >= 90% of the videos that have it natively, b = not worse than the no-video constant, c = better than it.
Usage: $PY code/y09_btc_single.py <workers>  -> out/y09_btc_single.csv, out/y09_btc_pred.npz"""
import sys, csv
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code'))
A = sys.argv[:]; sys.argv = ['x']
import y06_reader_quality as Y
import y08_fps_sim as F
lab, per, low, N, M = Y.lab, Y.per, Y.low, Y.N, Y.M
br = np.array([float(M[r]['bit_rate']) / 1e6 for r in Y.rec]); assert ((br < 4) == (low == 1)).all(), 'bitrate rule and pixel rule differ on dev'
ALLLOW = np.ones(N, int)
#        label                              tag    condition               occlusion-aware  class seen by the reader
CONDS = [('native (new front end)',         'x09', 'track_native',         False, low),
         ('C1 gain 0.25 + Poisson',         'x09', 'track_c1_g25p',        False, low),
         ('C1 gain 0.25 + Gauss var .01',   'x09', 'aidtrack_c1_g25gv01',  False, low),
         ('C2 vitiligo-like patches',       'x09', 'c2_vit',               False, low),
         ('C4 VGA',                         'x01', 'c4_vga',               False, low),
         ('C4 QVGA',                        'x01', 'c4_qvga',              False, low),
         ('C5 literal re-encode (team)',    'x01', 'c5_lit',               False, ALLLOW),
         ('C5 re-encode, tuned encoder',    'x06', 'c5_tuned',             False, ALLLOW),
         ('C5 split stream, 10 fps skip',   'sim', 'skip10',               False, low),
         ('C5 split stream, 10 fps bin',    'sim', 'bin10',                False, low),
         ('C6 unsharp (wrinkle contrast)',  'x01', 'c6_us1',               False, low),
         ('C7 blue mask',                   'x09', 'track_c7_blue',        False, low),
         ('C7 blue mask, occlusion-aware',  'x09', 'track_c7_blue',        True,  low)]
def sim(key): return F.one(key)
if __name__ == '__main__':
    with Pool(int(A[1])) as pool: RES = pool.map(sim, F.KEYS, chunksize=4)
    SIM = {m: dict(C=np.array([r[m][0] for r in RES]), P=np.array([r[m][1] for r in RES]), have=np.array([np.isfinite(r['native'][0]).all() for r in RES])) for m in ('skip10', 'bin10')}
    Zn, okn, qn = Y.Zn, Y.okn, Y.qn; rows = []; PRED = dict(rec=Y.rec, lab=lab, per=per, const=Y.const); one = [('seen', 'seen')]
    print(f"{'condition':32s}   n  cover | frozen MAE [CI95]        RMSE    r  | vs native [CI95]        | vs constant [CI95]      | gates | refit MAE (vs constant)")
    for name, tag, cond, oa, cls in CONDS:
        d = SIM[cond] if tag == 'sim' else Y.load(tag, cond); Z, ok, q = Y.evidence(d, oa); sub = d['have'] & Y.Dn['have'] & okn; n = int(sub.sum()); S = lambda a_, b_: Y.summarize(a_[sub], lab[sub], per[sub], b_[sub])
        nat = Y.cv(one, sub, Zn, qn, low, Zn, qn, low, lab, {})[0]; pf = Y.cv(one, sub, Z, q, cls, Zn, qn, low, lab, {})[0]; pr = Y.cv(one, sub, Z, q, cls, Z, q, cls, lab, {})[0]
        a = S(pf, nat); c = S(pf, Y.const); r = S(pr, Y.const); cov = float(ok[sub].mean()); g = ('a' if cov >= .9 else '-') + ('b' if c['dCI'][0] <= 0 else '-') + ('c' if c['dCI'][1] < 0 else '-')
        print(f"{name:32s} {n:3d} {cov:6.3f} | {a['MAE']:6.3f} [{a['CI'][0]:.2f};{a['CI'][1]:.2f}] {a['RMSE']:6.2f} {a['r']:+.2f} | {a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}] | {c['delta']:+.2f} [{c['dCI'][0]:+.2f};{c['dCI'][1]:+.2f}] | {g}   | {r['MAE']:6.3f} ({r['delta']:+.2f} [{r['dCI'][0]:+.2f};{r['dCI'][1]:+.2f}])", flush=True)
        rows.append(dict(condition=name, tag=tag, cond=cond + ('+oa' if oa else ''), videos=n, people=a['people'], coverage=cov, frozen_MAE=a['MAE'], lo=a['CI'][0], hi=a['CI'][1], RMSE=a['RMSE'], r=a['r'], d_native=a['delta'], dn_lo=a['dCI'][0], dn_hi=a['dCI'][1],
                         d_const=c['delta'], dc_lo=c['dCI'][0], dc_hi=c['dCI'][1], gates=g, refit_MAE=r['MAE'], refit_d_const=r['delta'], rc_lo=r['dCI'][0], rc_hi=r['dCI'][1], constant_MAE=S(Y.const, Y.const)['MAE'], native_MAE=S(nat, nat)['MAE']))
        PRED[cond + ('+oa' if oa else '') + '|frozen'] = pf; PRED[cond + ('+oa' if oa else '') + '|native_same_videos'] = nat
    with (W / 'out/y09_btc_single.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(W / 'out/y09_btc_pred.npz', **PRED)
