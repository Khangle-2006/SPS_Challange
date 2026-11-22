"""z11 - BTC pilot of x18 (causal landmark smoothing) under C1 strong Gaussian noise (gain 0.25, var 0.01): classical evidence of the
30 pilot videos (originals at rest) for native, the stored aidtrack front end (x09), and x18 with alpha = 1 (same as x09) / 0.3.
Peak within 5 bpm of the label (CHROM+POS evidence) and reader_v3 fitted on native training folds, applied unchanged.
Usage: $PY code/z11_btc_roi_smooth_eval.py"""
import sys
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
sys.argv = ['x']
import y06_reader_quality as Q
import y05_openrppg as Y5
from lib_btc import avg_spectra, zlog, GRID, summarize
keys = (W / 'scratch/x19_pilot_keys.txt').read_text().strip().split(','); _, allk = Y5.btc_keys(); idx = np.array([allk.index(k) for k in keys])
SETS = {'native': 'x01/native', 'aidtrack (x09)': 'x09/aidtrack_c1_g25gv01', 'x18 alpha 1': 'x18/c1_g25gv01_a100', 'x18 alpha 0.3': 'x18/c1_g25gv01_a030'}
Z = {}; HAVE = {}
for nm, s in SETS.items():
    Zs = Q.Zn.copy(); have = np.zeros(Q.N, bool)
    for k, i in zip(keys, idx):
        p = W / 'out' / s / (k + '.npz')
        if not p.exists(): continue
        with np.load(p) as z: t, valid, blk = z['t'], z['valid'].astype(bool), z['blk'].astype(float)
        a = avg_spectra(t, valid, blk.mean((1, 2)))
        if a is None: continue
        Zs[i] = (zlog(a[0]) + zlog(a[1])) / 2; have[i] = True
    Z[nm] = Zs; HAVE[nm] = have
sub = np.all(list(HAVE.values()), 0); print('pilot videos with evidence in every set:', int(sub.sum()))
base = None
for nm in SETS:
    p = Q.cv([('seen', 'seen')], sub, Z[nm], Q.qn, Q.low, Q.Zn, Q.qn, Q.low, Q.lab, {})[0]
    if base is None: base = p
    s = summarize(p[sub], Q.lab[sub], Q.per[sub], base[sub]); c = summarize(p[sub], Q.lab[sub], Q.per[sub], Q.const[sub])
    hit = np.mean(np.abs(GRID[Z[nm][sub].argmax(1)] - Q.lab[sub]) <= 5)
    print(f"{nm:16s} peak within 5 bpm {hit:.3f} | reader_v3 (frozen) MAE {s['MAE']:.2f} | vs native {s['delta']:+.2f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}] | vs constant {c['delta']:+.2f} [{c['dCI'][0]:+.2f};{c['dCI'][1]:+.2f}]")
