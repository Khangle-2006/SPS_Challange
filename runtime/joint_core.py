"""One-video inference for the BTC source router and low-bitrate fallback."""
import csv
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

from btc_spectral_math import GRID, zlog
from common import ROOT, save, sha
from joint_fallback import capture_features, trace_features, local_pulse_features, predict as residual_prediction
from joint_spectra import windows
from reader_v4 import MODELS, evidence

SOURCES = ("five_source", "CHROM", "POS", *MODELS)


def load_fit(fold):
    directory = ROOT / "models/btc_joint"
    manifest = json.loads((directory / "manifest.json").read_bytes())
    suffix = "deploy" if fold is None else str(fold)
    models, hashes = {}, {}
    for kind in ("router", "fallback"):
        path = directory / f"{kind}_{suffix}.json"
        hashes[kind] = sha(path)
        if hashes[kind] != manifest[path.name]:
            raise ValueError(f"Invalid model hash for {path.name}")
        models[kind] = json.loads(path.read_bytes())
    return models, hashes


def median(p):
    return GRID[min(int(np.sum(np.cumsum(p) < 0.5)), len(GRID) - 1)]


def read_spectra(models, chrom, pos, expert, trace, front, task):
    low = int(front["source_bitrate"] < 4_000_000)
    cell = 2 * low + int(task)
    combined = evidence(chrom, pos, expert)
    anchor = models["fallback"]["anchor"]
    if low and anchor["beta"][1] == 0:
        prior = np.asarray(anchor["prior"])[cell]
        base = median(softmax(np.log(prior + 1e-12))) + anchor["offset"][cell]
        if "source_fps" in front:
            fps = front["source_fps"]
        else:
            try:
                fps = round(float(Fraction(front["source_avg_frame_rate"])), 4)
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                raise ValueError("Original average frame rate required for low-bitrate fallback") from exc
        cap = capture_features(front["source_bitrate"], front["source_width"], front["source_height"],
                               fps, front["source_codec"])
        features = trace_features(trace["t"], trace["valid"], trace["rgb_mean"], trace["boxes"])
        pulse = local_pulse_features(base, chrom, pos, expert)
        pr = residual_prediction(models["fallback"]["fallback"]["signal"], base, [True], 0, cap, features[None, :], pulse)
        return float(pr[0]), "low_bitrate_fallback"

    spectra = np.stack([chrom, pos, *(expert[name] for name in MODELS)])
    candidates = []
    for fit, signal in zip(models["router"]["readers"], [combined, *zlog(spectra)], strict=True):
        prior = np.asarray(fit["priors"])[int(task)]
        logits = np.log(prior + 1e-12) + fit["beta"][low] * signal
        candidates.append(median(np.exp(logits - logsumexp(logits))) + fit["offsets"][cell])
    peaks = GRID[np.argmax(spectra, axis=1)]
    costs = [0.]
    for i, (mean, scale, intercept, coefficients) in enumerate(models["router"]["router"]):
        power = spectra[i]
        features = np.array([
            power[np.abs(GRID - peaks[i]) <= 3].sum(),
            -np.sum(power * np.log(power + 1e-30)) / np.log(len(GRID)),
            np.median(np.abs(peaks[i] - np.delete(peaks, i))),
            abs(candidates[i + 1] - candidates[0]), low, int(task),
        ])
        costs.append((features - mean) @ (np.asarray(coefficients) / scale) + intercept)
    index = int(np.argmin(costs))
    return float(candidates[index]), SOURCES[index]


def main(out):
    req = json.loads((out / "request.json").read_bytes())
    front = json.loads((out / "frontend.json").read_bytes())
    experts = json.loads((out / "experts.json").read_bytes())
    models, hashes = load_fit(req["fold"])
    reason = experts.get("reason") if experts.get("abstain") else None
    pr, branch, count = None, None, 0
    if not experts.get("abstain"):
        with np.load(out / "trace.npz") as trace, np.load(out / "experts.npz") as expert:
            measured = windows(trace["t"], trace["valid"], trace["rgb_mean"], 10, 2.5)
            if not any(start % 5 == 0 for start, _ in measured):
                reason = "no_valid_classical_spectrum"
            elif not all(np.isfinite(expert[k]).all() and expert[k].shape == GRID.shape for k in MODELS):
                reason = "invalid_expert_spectrum"
            else:
                count = len(measured)
                chrom, pos = np.mean([pair for _, pair in measured], axis=0)
                pr, branch = read_spectra(models, chrom, pos, expert, trace, front, req["activity"] == "Task")
                if not np.isfinite(pr):
                    pr, branch, reason = None, None, "invalid_spectrum"
    result = dict(method="BTC joint", pr_bpm=pr, fallback_prior_bpm=None, abstain=pr is None, reason=reason, branch=branch,
                  activity=req["activity"], frontend=req["frontend"], model_use="deployment" if req["fold"] is None
                  else f"dev_oof_fold{req['fold']}", reader_hashes=hashes, runtime_sha256={p.name: sha(p) for p in sorted((ROOT / "runtime").glob("*.py"))},
                  expert_weight_sha256=experts["weight_sha256"],
                  video_sha256=front["video_sha256"], capture_class="low" if front["capture_class_low"] else "high",
                  valid_frame_fraction=front["valid_frames"] / front["frames"], classical_windows=count,
                  expert_windows=experts.get("windows", {}))
    save(out / "result.json", result)
    fields = ["method", "pr_bpm", "fallback_prior_bpm", "abstain", "reason", "activity", "capture_class", "frontend", "branch", "model_use"]
    with (out / "result.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: result[k] for k in fields})
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
