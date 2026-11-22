"""Posterior readers on a per-video evidence curve Z(f) (mean of the standardised log CHROM and POS spectra, window-averaged).
  v1 : posterior median of prior(f) * exp(beta * Z(f)); prior = person-weighted KDE (6 bpm) of train labels; one beta.            (frozen 16 Nov)
  v2 : v1 + two constants (Task / not Task) fitted on train residuals.                                                          (frozen 16 Nov)
  F  : v1 with one beta per capture class (class read from the video file: below ~1080p or not) and one constant per (class, Task) cell.
Every fitted quantity comes from the training folds passed in; cells or classes with fewer than MIN_N training videos fall back to the pooled fit."""
import numpy as np
from collections import Counter
GRID = np.arange(42., 180.01, .5); BETAS = [0, .25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8]; MIN_N = 12
def pw(per):
    c = Counter(per); return np.array([1. / c[p] for p in per])
def kde(l, w, bw=6.):
    p = (np.exp(-.5 * ((GRID[None] - np.asarray(l)[:, None]) / bw) ** 2) * w[:, None]).sum(0); return p / p.sum()
def pmed(zrow, prior, b):
    lp = np.log(prior + 1e-12) + b * zrow; q = np.exp(lp - lp.max()); q /= q.sum(); return GRID[min(int((np.cumsum(q) < .5).sum()), len(GRID) - 1)]
def fit_beta(Z, lab, per, prior):
    w = pw(per); L = [np.sum(w * np.abs(np.array([pmed(z, prior, b) for z in Z]) - lab)) for b in BETAS]; return BETAS[int(np.argmin(L))]
def fit(Z, lab, per, low, task, mode):
    """returns a dict that `predict` understands. Z [n, grid], lab [n], per [n] person ids, low [n] 0/1, task [n] bool."""
    lab = np.asarray(lab, float); per = np.asarray(per); low = np.asarray(low, int); task = np.asarray(task, bool)
    prior = kde(lab, pw(per)); gb = fit_beta(Z, lab, per, prior); m = dict(mode=mode, prior=prior, beta={0: gb, 1: gb}, off={k: 0. for k in range(4)})
    if mode == 'v1': return m
    if mode == 'F':
        for c in (0, 1):
            i = np.flatnonzero(low == c)
            if len(i) >= MIN_N: m['beta'][c] = fit_beta(Z[i], lab[i], per[i], prior)
    p = np.array([pmed(Z[i], prior, m['beta'][low[i]]) for i in range(len(lab))]); r = lab - p; w = pw(per)
    if mode == 'v2':
        for t in (0, 1):
            s = task == bool(t); o = float(np.sum(w[s] * r[s]) / np.sum(w[s])) if s.any() else 0.
            m['off'][t] = o; m['off'][2 + t] = o
    else:
        cell = low * 2 + task.astype(int)
        for k in range(4):
            s = cell == k
            if s.sum() >= MIN_N: m['off'][k] = float(np.sum(w[s] * r[s]) / np.sum(w[s]))
    return m
def predict(m, Z, low, task):
    low = np.asarray(low, int); task = np.asarray(task, bool)
    return np.array([pmed(Z[i], m['prior'], m['beta'][low[i]]) + m['off'][low[i] * 2 + int(task[i])] for i in range(len(Z))])
def crossval(Z, lab, per, fold, low, task, mode):
    out = np.zeros(len(lab)); fits = []
    for f in sorted(set(fold)):
        tr = np.flatnonzero(fold != f); te = np.flatnonzero(fold == f); m = fit(Z[tr], lab[tr], per[tr], low[tr], task[tr], mode)
        out[te] = predict(m, Z[te], low[te], task[te]); fits.append((m['beta'], {k: round(v, 1) for k, v in m['off'].items()}))
    return out, fits
