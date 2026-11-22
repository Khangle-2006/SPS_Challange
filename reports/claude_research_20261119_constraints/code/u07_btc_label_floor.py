"""u07 (22 Nov, Claude): how far are the BTC labels from the pulse that is actually in the video?  Re-estimate of the 16 Nov 'label floor'
with the stronger evidence now available. A video counts as 'confident' when at least 4 of 5 independent readers (CHROM, POS, FacePhys.rlap,
ME-chunk.pure, ME-flow.pure; spectral peak of each) agree within 3 bpm of their median - decided without the label. On those videos the
median peak is taken as the HR in the video; |video HR - label| measures how far the label is from it (label timing / device), i.e. the
error a perfect pulse reader would still make. Person-first MAE, bootstrap over people (10,000, seed 0). Dev only."""
import sys, pickle, json
from pathlib import Path
from collections import defaultdict
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
lab, per, low, task, N, GRID = Y.lab, Y.per, Y.low, Y.task, Y.N, Y.GRID
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); pk = np.full((5, N), np.nan)
for i, s in enumerate(Sp):
    if s['chrom']: pk[0, i] = GRID[np.mean([w[0] for w in s['chrom']], 0).argmax()]; pk[1, i] = GRID[np.mean([w[0] for w in s['pos']], 0).argmax()]
for j, n in enumerate(['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']):
    with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
    h = np.isfinite(S).all(1); pk[2 + j, h] = GRID[S[h].argmax(1)]
ok = np.isfinite(pk).all(0); med = np.nanmedian(pk, 0); agree = (np.abs(pk - med) <= 3).sum(0); conf = ok & (agree >= 4)
v4 = np.load(W / 'out/u01_btc_nested_all.npz')['v4']
def pmae(e, m):
    d = defaultdict(list)
    for i in np.flatnonzero(m): d[per[i]].append(e[i])
    v = np.array([np.mean(x) for x in d.values()]); rng = np.random.default_rng(0); b = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
    return v.mean(), np.quantile(b, [.025, .975]), len(v)
err = np.abs(med - lab); sgn = med - lab
for name, m in (('camera originals, rest', ~task & (low == 0)), ('camera originals, Task', task & (low == 0)), ('low-resolution files', low == 1)):
    c = conf & m; a, ci, n = pmae(err, c)
    print(f'{name:24s} videos {int(m.sum()):3d} | confident {int(c.sum()):3d} ({c.sum() / max(m.sum(), 1):.0%}), people {n:3d} | |video HR - label| person-MAE {a:5.2f} [{ci[0]:.2f};{ci[1]:.2f}]'
          f' | median {np.median(err[c]) if c.any() else np.nan:4.1f} | >10 bpm {np.mean(err[c] > 10) if c.any() else np.nan:.2f} | mean signed {np.mean(sgn[c]) if c.any() else np.nan:+.1f}'
          + (f' | v4 on the same videos {pmae(np.abs(v4 - lab), c & np.isfinite(v4))[0]:5.2f}' if c.any() else ''))
print('--- where does the error of the camera originals at rest come from?')
ab = np.array(['_AB' in k for k in keys]); rest = ~task & (low == 0)
for name, m in (('confident', rest & conf), ('not confident', rest & ok & ~conf), ('no expert output', rest & ~ok)):
    if not m.any(): continue
    a, ci, n = pmae(np.abs(v4 - lab), m & np.isfinite(v4)); hit = (np.abs(pk[:, m] - lab[m]) <= 5).any(0)
    print(f'  {name:16s} videos {int(m.sum()):3d} people {n:3d} | abnormal (patients) {ab[m].mean():.0%} | v4 person-MAE {a:5.2f} [{ci[0]:.2f};{ci[1]:.2f}] | some reader within 5 bpm of the label {hit.mean():.0%} | agreeing readers median {np.median(agree[m]):.0f}/5')
