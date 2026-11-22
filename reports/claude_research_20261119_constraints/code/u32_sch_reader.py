"""u32 (22 Nov, Claude): real-data scoring of the u31 front ends (rules fixed before this run).
PRIMARY (chosen by the u31 simulation, written to out/u31/PRIMARY.txt before any real scoring): the primary channel REPLACES the
classical channel (CHROM+POS, weight 2) of reader_v4. Accept only if the paired CI (people, 10000, seed 0) of the MAE change vs v4 lies
below 0 on all reader videos. Sanity: 'CP' (our re-implementation of the classical channel) must give ~0 change.
Secondary (descriptive, no acceptance): SCbg, SC, PBVfix as replacements; u01 nested selection with {SC, SCbg, PBVfix, PRIMARY} as candidates."""
import sys, pickle, shutil
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
lab, per, low, task, fold, N, GRID = Y.lab, Y.per, Y.low, Y.task, Y.fold, Y.N, Y.GRID
U = np.load(W / 'out/u18_btc_beats.npz', allow_pickle=True); grp, sub = U['grp'], U['sub']
PRIM = (W / 'out/u31/PRIMARY.txt').read_text().strip()
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); Z0 = np.zeros((N, len(GRID)))
for i, s in enumerate(Sp):
    if s['chrom']: Z0[i] = (Y.zlog(np.mean([w[0] for w in s['chrom']], 0)) + Y.zlog(np.mean([w[0] for w in s['pos']], 0))) / 2
ZE = []
for n in ['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']:
    with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
    h = np.isfinite(S).all(1); Zm = np.zeros_like(Z0); Zm[h] = Y.zlog(S[h]); ZE.append(Zm)
NEW = {}
for m in ['CP', 'POS', PRIM, 'SCbg', 'SC', 'PBVfix']:
    with np.load(W / 'out/u31/npz' / (m + '.npz')) as z: S = z['S']
    ok = np.isfinite(S).all(1); Zm = np.zeros_like(Z0); Zm[ok] = Y.zlog(S[ok]); NEW[m] = (Zm, ok, S)
use = sub & np.all([v[1] for v in NEW.values()], 0); print('PRIMARY', PRIM, '| reader videos', int(use.sum()), 'people', len(set(per[use])))
gname = np.where(low == 1, 'low', np.where(task, 'Task', grp))
print('evidence peak within 5 bpm of the label:')
for g in ('easy', 'hard', 'Task', 'low'):
    m = use & (gname == g); line = f'  {g:5s} n {int(m.sum()):3d} | y01 classical {np.mean(np.abs(GRID[Z0[m].argmax(1)] - lab[m]) <= 5):.2f}'
    for k, (Zm, ok, S) in NEW.items(): line += f' | {k} {np.mean(np.abs(GRID[S[m].argmax(1)] - lab[m]) <= 5):.2f}'
    print(line)
def cvp(Z):
    p = np.full(N, np.nan)
    for f in sorted(set(fold)):
        tr = np.flatnonzero((fold != f) & use); te = np.flatnonzero((fold == f) & use)
        m = Y.fitG(Z[tr], lab[tr], per[tr], low[tr], low[tr], task[tr]); p[te] = Y.predG(m, Z[te], low[te], low[te], task[te])
    return p
v4 = cvp((2 * Z0 + sum(ZE)) / 5); P = {k: cvp((2 * NEW[k][0] + sum(ZE)) / 5) for k in NEW}
S_ = lambda a_, b_, m: Y.summarize(a_[m], lab[m], per[m], b_[m])
for name, m in (('all', use), ('camera originals, rest', use & ~task & (low == 0)), ('  - hard', use & (gname == 'hard')), ('camera originals, Task', use & task & (low == 0)), ('low-resolution files', use & (low == 1))):
    line = f"{name:24s} n {int(m.sum()):3d} | v4 {S_(v4, Y.const, m)['MAE']:6.2f}"
    for k in NEW: a = S_(P[k], v4, m); line += f" | {k} {a['MAE']:6.2f} ({a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}])"
    print(line)
a = S_(P[PRIM], v4, use); print('PRIMARY decision:', 'ACCEPT' if a['dCI'][1] < 0 else 'REJECT', f"({a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}])")
d = W / 'out/u31/npz_sel'; d.mkdir(exist_ok=True)
for m in ['SC', 'SCbg', 'PBVfix', PRIM]: shutil.copy(W / 'out/u31/npz' / (m + '.npz'), d / ('u31_' + m + '.npz'))
np.savez_compressed(W / 'out/u32_sch_reader.npz', rec=Y.rec, use=use, v4=v4, **{k: v for k, v in P.items()})
