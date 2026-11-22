"""u24 (22 Nov, Claude): head-room of BTC dev under the label floor. Person-first MAE if a perfect pulse reader brought the errors of
a group down to the measured label floor (3.6 bpm, u07), the other groups unchanged. Arithmetic on the cross-validated v4 errors."""
import sys
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
U = np.load(W / 'out/u18_btc_beats.npz', allow_pickle=True); v4, sub, grp = U['v4'], U['sub'], U['grp']
e = np.abs(v4 - Y.lab); per = Y.per
def pf(err, m):
    P = sorted(set(per[m])); return np.mean([err[m & (per == p)].mean() for p in P]), len(P)
rest = sub & ~Y.task & (Y.low == 0); task = sub & Y.task & (Y.low == 0); low = sub & (Y.low == 1); hard = rest & (grp == 'hard'); easy = rest & (grp == 'easy')
for n, m in (('all', sub), ('rest originals', rest), ('  easy (confident)', easy), ('  hard', hard), ('  other rest', rest & ~hard & ~easy), ('Task originals', task), ('low files', low)):
    a, k = pf(e, m); print(f'{n:20s} videos {int(m.sum()):3d} people {k:3d} | v4 person-first MAE {a:5.2f}')
F = 3.6
for n, groups in (('hard rest -> floor', [hard]), ('all rest originals -> floor', [rest]), ('all rest + Task originals -> floor', [rest, task])):
    e2 = e.copy()
    for g in groups: e2[g] = F
    print(f'{n:36s} overall {pf(e2, sub)[0]:5.2f}')
