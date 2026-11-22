"""How much of reader_v3 comes from the video and how much from knowing the capture class and the Task flag?
Same folds and metrics as a11. 'cells only' = no video at all: person-weighted training median of the (class, Task) cell (pooled median if the cell
has fewer than 12 training videos). Also Bland-Altman numbers by person for the candidates."""
import csv, sys, numpy as np
sys.path.insert(0, '/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118/code')
from lib_btc import *
import reader_models as rm
rows = list(csv.DictReader((W / 'reader_v3/OOF_PREDICTIONS.csv').open()))
lab = np.array([float(r['truth_bpm']) for r in rows]); per = np.array([r['person'] for r in rows]); fold = np.array([int(r['fold']) for r in rows]); low = np.array([int(r['low_class']) for r in rows])
task = np.array([r['protocol'].startswith('Task') for r in rows]); P = {k: np.array([float(r[k]) for r in rows]) for k in ['constant', 'calibrated', 'reader_v1', 'reader_v2', 'reader_v3']}
def wmedian(x, w):
    o = np.argsort(x); c = np.cumsum(w[o]); return float(x[o][np.searchsorted(c, c[-1] / 2)])
cells = np.zeros(len(lab)); cell = low * 2 + task.astype(int)
for f in range(5):
    tr = np.flatnonzero(fold != f); te = np.flatnonzero(fold == f); w = rm.pw(per[tr]); g = wmedian(lab[tr], w)
    for k in range(4):
        m = cell[tr] == k; cells[te[cell[te] == k]] = wmedian(lab[tr][m], w[m]) if m.sum() >= 12 else g
P['cells only (no video)'] = cells
print(f"{'method':26s} MAE [CI]            RMSE    r     | vs constant           | Bland-Altman by person: bias, limits of agreement")
for k, p in P.items():
    s = summarize(p, lab, per, P['constant']); d = {}
    for i, q in enumerate(per): d.setdefault(q, []).append(p[i] - lab[i])
    e = np.array([np.mean(v) for v in d.values()]); loa = (e.mean() - 1.96 * e.std(ddof=1), e.mean() + 1.96 * e.std(ddof=1))
    print(f"{k:26s} {s['MAE']:6.3f} [{s['CI'][0]:.2f};{s['CI'][1]:.2f}]  {s['RMSE']:5.2f}  {s['r']:+.2f} | {s['delta']:+.3f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}] | {e.mean():+.2f}, [{loa[0]:+.1f}; {loa[1]:+.1f}]")
s = summarize(P['reader_v3'], lab, per, cells); print(f"reader_v3 vs cells only: {s['delta']:+.3f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}]  <- contribution of the video evidence once class and Task are known")
