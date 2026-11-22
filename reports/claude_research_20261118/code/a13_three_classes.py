"""Exploratory: does a third capture class help? Classes read from the file: 0 = high-resolution SDR, 1 = low resolution (< 1.8 MP), 2 = high-resolution HDR (HLG transfer, 10-bit).
Same reader as F but with any number of classes; cells = class x Task; classes/cells with fewer than MIN_N training videos fall back to the pooled beta / zero offset.
Also a variant with one offset per protocol level (Resting, Task1, Task2, Task3, Walking) inside the high class. Controls: class flag shuffled x20.
This is a small-sample exploration (16 HLG dev videos): report only, not a frozen candidate."""
import csv, sys, numpy as np
sys.path.insert(0, '/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118/code')
from lib_btc import *
import reader_models as rm
V = videos(); M = meta(); rec = [v['recording'] for v in V]
lab = np.array([v['pr_bpm'] for v in V]); fold = np.array([v['fold'] for v in V]); per = np.array([v['person'] for v in V]); prot = np.array([v['task'] for v in V]); task = np.char.startswith(prot, 'Task')
low = np.array([int(M[r]['width']) * int(M[r]['height']) < 1800000 for r in rec]); hlg = np.array([M[r]['color_transfer'] == 'arib-std-b67' for r in rec])
c2 = low.astype(int); c3 = np.where(low, 1, np.where(hlg, 2, 0))
Z = np.zeros((len(rec), len(GRID)))
for i, r in enumerate(rec):
    b = load_blocks(r); a = avg_spectra(b['t'], b['valid'], b['rgb_mean'])
    if a is not None: Z[i] = (zlog(a[0]) + zlog(a[1])) / 2
def cv(cls, cellof, prior_by_class=False):
    out = np.zeros(len(lab)); fits = []
    for f in range(5):
        tr = np.flatnonzero(fold != f); te = np.flatnonzero(fold == f); w = rm.pw(per[tr]); gprior = rm.kde(lab[tr], w); gb = rm.fit_beta(Z[tr], lab[tr], per[tr], gprior); B = {}; PR = {}
        for c in np.unique(cls):
            i = tr[cls[tr] == c]; PR[c] = rm.kde(lab[i], rm.pw(per[i])) if (prior_by_class and len(i) >= rm.MIN_N) else gprior
            B[c] = rm.fit_beta(Z[i], lab[i], per[i], PR[c]) if len(i) >= rm.MIN_N else gb
        p = np.array([rm.pmed(Z[i], PR[cls[i]], B[cls[i]]) for i in range(len(lab))]); r = lab - p; O = {}
        for k in np.unique(cellof):
            m = cellof[tr] == k; O[k] = float(np.sum(w[m] * r[tr][m]) / np.sum(w[m])) if m.sum() >= rm.MIN_N else 0.
        out[te] = p[te] + np.array([O[k] for k in cellof[te]]); fits.append({int(c): B[c] for c in B})
    return out, fits
base, _ = cv(c2, c2 * 2 + task)
def rep(name, p, fits=None):
    s = summarize(p, lab, per, base); print(f"{name:58s} MAE {s['MAE']:6.3f} [{s['CI'][0]:.2f};{s['CI'][1]:.2f}] RMSE {s['RMSE']:.2f} r {s['r']:.2f} | vs reader_v3 {s['delta']:+.3f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}]" + (f" | betas {fits}" if fits else ''), flush=True)
rep('reader_v3 (2 classes)', base)
p3, f3 = cv(c3, c3 * 2 + task); rep('3 classes (SDR / low / HLG), global prior', p3, f3)
p3p, f3p = cv(c3, c3 * 2 + task, True); rep('3 classes, prior per class', p3p, f3p)
lev = np.array([{'Resting': 0, 'Task1': 1, 'Task2': 2, 'Task3': 3, 'Walking': 4}[x] for x in prot])
p5, f5 = cv(c2, c2 * 10 + lev); rep('2 classes, offset per protocol level', p5, f5)
p35, f35 = cv(c3, c3 * 10 + lev); rep('3 classes, offset per protocol level', p35, f35)
rng = np.random.default_rng(3); sh = []
for _ in range(20):
    c = c3.copy(); hi = np.flatnonzero(~low); c[hi] = rng.permutation(c3[hi]); sh.append(summarize(cv(c, c * 2 + task)[0], lab, per)['MAE'])
print(f"control: HLG flag shuffled among the high-resolution videos x20 -> MAE min {min(sh):.3f} mean {np.mean(sh):.3f} (3 classes true: {summarize(p3, lab, per)['MAE']:.3f})")
print('per-video MAE on the 16 HLG dev videos: reader_v3', round(float(np.abs(base - lab)[hlg].mean()), 2), '| 3 classes', round(float(np.abs(p3 - lab)[hlg].mean()), 2), '| 3 classes + class prior', round(float(np.abs(p3p - lab)[hlg].mean()), 2))
