"""u30 (22 Nov, Claude): real-data scoring of the u28 front ends (after the simulation; rule fixed before this run).
Primary (pre-registered): reader_v4 with its classical channel (CHROM+POS of y01, weight 2) REPLACED by SC_T1 (spatio-chromatic MVDR +
track-before-detect). Accept only if the paired CI (people, 10000, seed 0) of the MAE change vs v4 lies below 0 on all reader videos.
Secondary: POS_T1 (our POS + TBD) as the replacement; and u01 nested selection with the new channels as extra candidates (>= 4/5 rule).
Descriptive: evidence peak within 5 bpm of the label by group, for every front end."""
import sys, pickle, json
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
lab, per, low, task, fold, N, GRID = Y.lab, Y.per, Y.low, Y.task, Y.fold, Y.N, Y.GRID
U = np.load(W / 'out/u18_btc_beats.npz', allow_pickle=True); grp, sub = U['grp'], U['sub']
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); Z0 = np.zeros((N, len(GRID)))
for i, s in enumerate(Sp):
    if s['chrom']: Z0[i] = (Y.zlog(np.mean([w[0] for w in s['chrom']], 0)) + Y.zlog(np.mean([w[0] for w in s['pos']], 0))) / 2
ZE = []
for n in ['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']:
    with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
    h = np.isfinite(S).all(1); Zm = np.zeros_like(Z0); Zm[h] = Y.zlog(S[h]); ZE.append(Zm)
NEW = {}
for f in sorted((W / 'out/u28/npz').glob('*.npz')):
    with np.load(f) as z: S = z['S']
    ok = np.isfinite(S).all(1); Zm = np.zeros_like(Z0); Zm[ok] = Y.zlog(S[ok]); NEW[f.stem] = (Zm, ok, S)
okN = np.all([v[1] for v in NEW.values()], 0); use = sub & okN
print('reader videos', int(use.sum()), 'people', len(set(per[use])))
gname = np.where(low == 1, 'low', np.where(task, 'Task', grp))
print('evidence peak within 5 bpm of the label:')
for g in ('easy', 'hard', 'Task', 'low'):
    m = use & (gname == g); line = f'  {g:5s} n {int(m.sum()):3d} | y01 CHROM+POS {np.mean(np.abs(GRID[Z0[m].argmax(1)] - lab[m]) <= 5):.2f}'
    for k, (Zm, ok, S) in NEW.items(): line += f' | {k} {np.mean(np.abs(GRID[S[m].argmax(1)] - lab[m]) <= 5):.2f}'
    print(line)
def cvp(Z):
    p = np.full(N, np.nan)
    for f in sorted(set(fold)):
        tr = np.flatnonzero((fold != f) & use); te = np.flatnonzero((fold == f) & use)
        m = Y.fitG(Z[tr], lab[tr], per[tr], low[tr], low[tr], task[tr]); p[te] = Y.predG(m, Z[te], low[te], low[te], task[te])
    return p
v4 = cvp((2 * Z0 + sum(ZE)) / 5); A = cvp((2 * NEW['SC_T1'][0] + sum(ZE)) / 5); B = cvp((2 * NEW['POS_T1'][0] + sum(ZE)) / 5)
S_ = lambda a_, b_, m: Y.summarize(a_[m], lab[m], per[m], b_[m])
for name, m in (('all', use), ('camera originals, rest', use & ~task & (low == 0)), ('  - hard', use & (gname == 'hard')), ('camera originals, Task', use & task & (low == 0)), ('low-resolution files', use & (low == 1))):
    a = S_(A, v4, m); b = S_(B, v4, m)
    print(f"{name:24s} videos {int(m.sum()):3d} people {a['people']:3d} | v4 {S_(v4, Y.const, m)['MAE']:6.2f} | A: SC_T1 replaces classical {a['MAE']:6.2f} ({a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}]) | B: POS_T1 replaces {b['MAE']:6.2f} ({b['delta']:+.2f} [{b['dCI'][0]:+.2f};{b['dCI'][1]:+.2f}])")
np.savez_compressed(W / 'out/u30_sch_reader.npz', rec=Y.rec, use=use, v4=v4, A=A, B=B)
