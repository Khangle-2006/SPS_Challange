"""z06 - reader_v4 with the experts run on an ALTERNATIVE native crop set (e.g. x19 'stab' = stabilised box) instead of x10 crops.
This is a different front end for the experts (not a degraded condition), so the reader is cross-validated on the variant itself.
Paired with reader_v4 on the x10 crops (same videos). Also each expert alone (reader_v3 machinery) and its peak hit rate.
Usage: $PY code/z06_btc_crop_variants_eval.py <variant> [models]   (expert spectra in out/y11/<variant>/<model>.npz)"""
import sys
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
A = sys.argv[:]; sys.argv = ['x']
import y06_reader_quality as Q
from lib_btc import GRID, zlog, summarize
var = A[1]; models = (A[2] if len(A) > 2 else 'FacePhys.rlap,ME-chunk.pure,ME-flow.pure').split(','); N = Q.N; lab, per, low = Q.lab, Q.per, Q.low
def zmodel(S):
    ok = np.isfinite(S).all(1); Z = np.zeros((N, len(GRID))); Z[ok] = zlog(S[ok]); return Z, ok
def base_models():
    out = {}
    for m in models:
        with np.load(W / 'out/y05' / (m + '.npz')) as z: S = z['S'].copy()
        p = W / 'out/y11/heavy' / (m + '.npz')
        if p.exists():
            with np.load(p) as z: h = np.isfinite(z['S']).all(1); S[h] = z['S'][h]
        out[m] = zmodel(S)
    return out
def var_models():
    out = {}
    for m in models:
        with np.load(W / 'out/y11' / var / (m + '.npz')) as z: out[m] = zmodel(z['S'])
    return out
def mix(ZM): return (2 * Q.Zn + sum(ZM[m][0] for m in models)) / (2 + len(models))
B = base_models(); V = var_models(); ok = Q.okn & np.all([B[m][1] & V[m][1] for m in models], 0)
fr = lambda Z: Q.cv([('seen', 'seen')], ok, Z, Q.qn, low, Z, Q.qn, low, lab, {})[0]
pb, pv = fr(mix(B)), fr(mix(V)); s = summarize(pv[ok], lab[ok], per[ok], pb[ok]); sb = summarize(pb[ok], lab[ok], per[ok])
print(f"videos {int(ok.sum())} | reader_v4 x10 crops MAE {sb['MAE']:.3f} RMSE {sb['RMSE']:.2f} r {sb['r']:+.3f} | {var}: MAE {s['MAE']:.3f} RMSE {s['RMSE']:.2f} r {s['r']:+.3f} | vs x10 {s['delta']:+.3f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}]")
hi = ok & (low == 0) & ~Q.task
for m in models:
    a, b = fr((2 * Q.Zn + B[m][0]) / 3), fr((2 * Q.Zn + V[m][0]) / 3); s = summarize(b[ok], lab[ok], per[ok], a[ok])
    hb = np.mean(np.abs(GRID[B[m][0][hi].argmax(1)] - lab[hi]) <= 5); hv = np.mean(np.abs(GRID[V[m][0][hi].argmax(1)] - lab[hi]) <= 5)
    print(f"   {m:15s} CHROM+POS+model: x10 {s['MAE'] - s['delta']:.3f} -> {var} {s['MAE']:.3f} ({s['delta']:+.3f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}]) | peak hit, originals at rest (n={int(hi.sum())}): {hb:.3f} -> {hv:.3f}")
