"""BTC pulse-rate reader: prior x spectrum posterior on the whole-face (oval) RGB trace.

Reference implementation for the 16 Nov 2026 analysis. NumPy/SciPy only, no neural network.

Input  : the BTC_DEV_NUMERIC export folder (VIDEOS.json, blocks/<recording>.npz with
         t, valid, rgb_mean; optional tables/ for the saved baselines).
Method : 1. For every supported 10 s window starting at 0, 5, 10, ... s: resample rgb_mean to
            30 Hz (300 samples), run the team's POS and CHROM, then the team's frozen readout
            operator (smoothness-prior detrend lambda=100, Butterworth order 1 0.7-3 Hz, Hann
            periodogram nfft=4096). Each window spectrum is normalised to unit sum in 42-180 bpm.
         2. Average the window spectra per video and per method; take the standardised
            log-spectrum z(f) of each; Z(f) = (z_CHROM + z_POS) / 2.
         3. Posterior over the label: p(f) ~ prior(f) * exp(beta * Z(f)); prediction = posterior
            median. prior = person-weighted Gaussian KDE (6 bpm) of the training labels;
            beta = the grid value with the lowest person-weighted training MAE.
         A video without any supported window falls back to the prior median.
Protocol: five person-disjoint folds from VIDEOS.json; per-person MAE; paired bootstrap over
         people (10,000 draws, seed 0), exactly as in the team's reports.

Usage  : python btc_posterior_reader.py --data /path/to/export --out oof_predictions.csv
         add --model reader_model.json to also save the estimator fitted on all dev videos
         (prior + beta) for a single later holdout evaluation.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import factorized

GRID = np.arange(42.0, 180.01, 0.5)                 # bpm
BETAS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]
PRIOR_BW = 6.0                                      # bpm
HOP_S = 5.0


# ---------------------------------------------------------------- team-equivalent operators
def supported(t, valid, start, end, gap=0.2):
    """Same contract as t8_t1_loader.supported: no extrapolation, no invalid frame, no gap > 0.2 s."""
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        return False
    if start < t[0] - 1e-9 or end > t[-1] + 1e-9:
        return False
    for i in np.flatnonzero(~valid):
        a, b = t[max(0, i - 1)], t[min(len(t) - 1, i + 1)]
        if a < end - 1e-10 and b > start + 1e-10:
            return False
    for i in np.flatnonzero(np.diff(t) > gap + 1e-12):
        if t[i] < end - 1e-10 and t[i + 1] > start + 1e-10:
            return False
    return not np.any((~valid) & (t >= start - 1e-10) & (t <= end + 1e-10))


def pos(rgb, fs=30.0):
    length = math.ceil(1.6 * fs)
    out = np.zeros(len(rgb))
    for end in range(length, len(rgb) + 1):
        x = rgb[end - length:end]
        n = x / np.maximum(x.mean(axis=0), 1e-8)
        s = np.array([[0, 1, -1], [-2, 1, 1]]) @ n.T
        h = s[0] + np.std(s[0]) / max(np.std(s[1]), 1e-8) * s[1]
        out[end - length:end] += h - h.mean()
    return out


def chrom(rgb, fs=30.0):
    n = len(rgb)
    w = math.ceil(1.6 * fs)
    w += w % 2
    b, a = signal.butter(3, [0.7, 3], fs=fs, btype="bandpass")
    starts = list(range(0, n - w + 1, w // 2))
    if starts[-1] != n - w:
        starts.append(n - w)
    out = np.zeros(n)
    for start in starts:
        v = rgb[start:start + w]
        r, g, bl = (v / v.mean(axis=0)).T
        x = signal.filtfilt(b, a, 3 * r - 2 * g)
        y = signal.filtfilt(b, a, 1.5 * r + g - 1.5 * bl)
        out[start:start + w] += (x - np.std(x) / np.std(y) * y) * signal.windows.hann(w)
    return out


_N = 300
_D = sparse.diags([np.ones(_N - 2), -2 * np.ones(_N - 2), np.ones(_N - 2)], [0, 1, 2], shape=(_N - 2, _N), format="csc")
_TREND = factorized(sparse.eye(_N, format="csc") + 100 ** 2 * (_D.T @ _D))
_SOS = signal.butter(1, [0.7, 3.0], btype="bandpass", fs=30, output="sos")


def window_spectrum(wave):
    """Frozen readout operator of score_round1.reference_hr, returned on GRID and normalised."""
    y = signal.sosfiltfilt(_SOS, wave - _TREND(wave))
    f, p = signal.periodogram(y, fs=30, window="hann", nfft=4096, detrend=False)
    p = np.interp(GRID, 60 * f, p)
    return p / p.sum()


def video_scores(t, valid, rgb_mean):
    """Standardised log of the window-averaged CHROM and POS spectra; None when no window is supported."""
    acc_c, acc_p = [], []
    for start in np.arange(0, max(0.0, t[-1] - 10 + 1e-8), HOP_S):
        if not supported(t, valid, start, start + 10):
            continue
        q = start + np.arange(300) / 30.0
        x = np.stack([np.interp(q, t, rgb_mean[:, c]) for c in range(3)], axis=1)
        if not np.isfinite(x).all() or x.mean(axis=0).min() <= 0:
            continue
        wc, wp = chrom(x), pos(x)
        if not (np.isfinite(wc).all() and np.isfinite(wp).all()) or wc.std() < 1e-12 or wp.std() < 1e-12:
            continue
        acc_c.append(window_spectrum(wc))
        acc_p.append(window_spectrum(wp))
    if not acc_c:
        return None

    def z(s):
        l = np.log(np.mean(s, axis=0) + 1e-30)
        return (l - l.mean()) / l.std()

    return (z(acc_c) + z(acc_p)) / 2


# ---------------------------------------------------------------- reader
def person_weights(persons):
    count = defaultdict(int)
    for p in persons:
        count[p] += 1
    return np.array([1.0 / count[p] for p in persons])


def kde_prior(labels, persons):
    w = person_weights(persons)
    p = (np.exp(-0.5 * ((GRID[None] - np.asarray(labels)[:, None]) / PRIOR_BW) ** 2) * w[:, None]).sum(axis=0)
    return p / p.sum()


def posterior_median(z, prior, beta):
    if z is None:
        z = np.zeros(len(GRID))
    lp = np.log(prior + 1e-12) + beta * z
    q = np.exp(lp - lp.max())
    q /= q.sum()
    return float(GRID[min(int((np.cumsum(q) < 0.5).sum()), len(GRID) - 1)])


def fit(zs, labels, persons):
    prior = kde_prior(labels, persons)
    w = person_weights(persons)
    losses = [float(np.sum(w * np.abs(np.array([posterior_median(z, prior, b) for z in zs]) - labels))) for b in BETAS]
    return prior, BETAS[int(np.argmin(losses))]


# ---------------------------------------------------------------- metrics
def person_table(pred, truth, persons, other=None):
    per = defaultdict(list)
    for i, p in enumerate(persons):
        row = [abs(pred[i] - truth[i]), pred[i] - truth[i]]
        if other is not None:
            row.append(abs(other[i] - truth[i]))
        per[p].append(row)
    return np.array([np.mean(per[p], axis=0) for p in sorted(per)])


def report(name, pred, truth, persons, other=None):
    v = person_table(pred, truth, persons, other)
    ix = np.random.default_rng(0).integers(0, len(v), (10000, len(v)))
    ci = np.quantile(v[ix, 0].mean(axis=1), [0.025, 0.975])
    text = f"{name:28s} MAE {v[:, 0].mean():6.3f} [{ci[0]:.2f}; {ci[1]:.2f}]  bias {v[:, 1].mean():+.2f}"
    if other is not None:
        d = v[:, 0] - v[:, 2]
        dci = np.quantile(d[ix].mean(axis=1), [0.025, 0.975])
        text += f"  | delta {d.mean():+.3f} [{dci[0]:+.2f}; {dci[1]:+.2f}]"
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--model", type=Path)
    a = ap.parse_args()

    videos = json.loads((a.data / "VIDEOS.json").read_text())
    zs = []
    for v in videos:
        with np.load(a.data / "blocks" / (v["recording"] + ".npz"), allow_pickle=False) as f:
            zs.append(video_scores(f["t"], f["valid"], f["rgb_mean"]))
    truth = np.array([v["pr_bpm"] for v in videos], float)
    fold = np.array([v["fold"] for v in videos])
    persons = [v["person"] for v in videos]

    pred = np.zeros(len(videos))
    const = np.zeros(len(videos))
    betas = []
    for k in sorted(set(fold)):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        prior, beta = fit([zs[i] for i in tr], truth[tr], [persons[i] for i in tr])
        betas.append(beta)
        for i in te:
            pred[i] = posterior_median(zs[i], prior, beta)
        const[te] = np.median(truth[tr])

    print(f"{len(videos)} videos / {len(set(persons))} people; windows missing for {sum(z is None for z in zs)} videos; beta per fold {betas}")
    report("constant (this cohort)", const, truth, persons)
    report("posterior reader", pred, truth, persons)
    report("reader vs constant", pred, truth, persons, const)

    saved = a.data / "tables/round1_bayes_20261108/VIDEO_PREDICTIONS.csv"
    if saved.exists():
        rows = list(csv.DictReader(saved.open()))
        for method, label in [("constant", "saved constant"), ("calibrated_median_POS_CHROM_V3", "saved calibrated")]:
            look = {r["recording"]: float(r["prediction"]) for r in rows if r["method"] == method and r["condition"] == "native"}
            other = np.array([look[v["recording"]] for v in videos])
            report(f"reader vs {label}", pred, truth, persons, other)

    if a.out:
        with a.out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["recording", "person", "fold", "group", "task", "truth_bpm", "reader_bpm", "constant_bpm"])
            for v, p, c in zip(videos, pred, const):
                w.writerow([v["recording"], v["person"], v["fold"], v["group"], v["task"], v["pr_bpm"], p, c])
    if a.model:
        prior, beta = fit(zs, truth, persons)
        a.model.write_text(json.dumps(dict(grid_bpm=GRID.tolist(), prior=prior.tolist(), beta=beta, prior_bandwidth_bpm=PRIOR_BW,
                                           hop_s=HOP_S, fitted_on="all dev videos in VIDEOS.json", videos=len(videos)), indent=1))
        print("model fitted on all dev videos: beta =", beta)


if __name__ == "__main__":
    main()
