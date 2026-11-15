"""UBFC1 diagnostics only: no synchronization inference, model fit, or HR score."""
import argparse
import csv
import json
from itertools import zip_longest
from pathlib import Path

import av
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_adapters import VideoFrames
from video_roi_frontend import Tracker


def mean_zoh(t, y, start, duration):
    """Time-weighted sensor label mean, zero-order hold; never extrapolate."""
    t, y = np.asarray(t), np.asarray(y)
    if len(t) != len(y) or len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("Strictly increasing sensor clock and matching labels required")
    end = start + duration
    if duration <= 0 or start < t[0] or end > t[-1]:
        raise ValueError("Window outside measured sensor support")
    knots = np.r_[start, t[(t > start) & (t < end)], end]
    indices = np.searchsorted(t, knots[:-1], side="right")-1
    return float(np.dot(np.diff(knots), y[indices])/duration)


def longest_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    lengths = np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1)
    return int(lengths.max()) if len(lengths) else 0


def baseline_metrics(y, predicted, ids):
    errors = np.abs(np.asarray(y)-predicted)
    return {"mae_percentage_points": float(errors.mean()),
            "recording_macro_mae_percentage_points": float(np.mean(
                [errors[np.array(ids) == name].mean() for name in sorted(set(ids))])),
            "rmse_percentage_points": float(np.sqrt(np.mean(errors**2)))}


def labels(root, out):
    rows, histograms, sensitivity = [], {}, []
    figure, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    for ax, folder in zip(axes.flat, sorted(root.iterdir())):
        if not (folder/"gtdump.xmp").exists():
            raise ValueError(f"Unexpected input directory {folder}")
        raw = np.loadtxt(folder/"gtdump.xmp", delimiter=",")
        t = (raw[:, 0]-raw[0, 0])/1000
        ppg, spo2 = raw[:, 3], raw[:, 2]
        assert np.all(ppg == np.round(ppg))
        values, counts = np.unique(ppg, return_counts=True)
        n127, n126 = int(np.sum(ppg == 127)), int(np.sum(ppg == 126))
        histograms[folder.name] = {
            "values": values.tolist(), "counts": counts.tolist(),
            "samples": len(ppg), "count_127": n127, "count_126": n126,
            "fraction_127": n127/len(ppg), "longest_127_run_samples": longest_run(ppg == 127),
            "count_0": int(np.sum(ppg == 0)),
            "longest_0_run_samples": longest_run(ppg == 0),
            "interpretation": "Boundary accumulation is a clipping warning, not proof of its cause."}
        ax.bar(values, counts, width=1)
        ax.axvline(127, color="red", ls="--", lw=1)
        ax.set(title=f"{folder.name}: 127 = {n127}/{len(ppg)}",
               xlabel="Recorded contact PPG value", ylabel="Sample count", xlim=(-1, 129))
        # These are SENSOR-clock windows only, not paired RGB windows.
        # 0.4 s margin makes all tested shifts fully observed, without padding.
        for start in np.arange(.4, t[-1]-30-.4+1e-9, 30):
            averages = [mean_zoh(t, spo2, start+shift, 30) for shift in [-.4, 0, .4]]
            row = {"recording": folder.name, "sensor_start_s": float(start),
                   "duration_s": 30, "spo2_mean": averages[1],
                   "minus_0p4s_mean": averages[0], "plus_0p4s_mean": averages[2],
                   "max_shift_change_pp": max(abs(averages[0]-averages[1]),
                                              abs(averages[2]-averages[1]))}
            rows.append(row)
        sensitivity.append({"recording": folder.name,
                            "windows": sum(r["recording"] == folder.name for r in rows)})
    figure.savefig(out/"ppg_histograms.png", dpi=160)
    plt.close(figure)
    y = np.array([r["spo2_mean"] for r in rows])
    ids = [r["recording"] for r in rows]
    fitted_mean = float(y.mean())
    # No split of UBFC1. This fit and score reuse the same labels intentionally
    # to expose label concentration, and MUST NOT be called held-out accuracy.
    result = {"window_count": len(rows), "window_duration_s": 30,
              "clock": "sensor only, zero-origin; no RGB/reference pairing",
              "shift_assumption_s": [-.4, .4],
              "offset_bound_verified": False,
              "tolerance_statement": "Sensitivity scenario only; actual offset/drift unknown.",
              "fitted_mean_in_sample": {"prediction_percent": fitted_mean,
                                       "evaluation": "same-label descriptive oracle; NOT held-out",
                                       **baseline_metrics(y, fitted_mean, ids)},
              "fixed_97_illustration": {"prediction_percent": 97,
                                       "evaluation": "chosen after seeing label range; NOT preregistered",
                                       **baseline_metrics(y, 97, ids)},
              "max_shift_change_pp": max(r["max_shift_change_pp"] for r in rows),
              "per_recording_window_counts": sensitivity,
              "future_evaluation_rule": "Fit constant on independent training data only, freeze it, "
                                       "and report alongside every model on identical windows. "
                                       "Keep all UBFC1 recordings in one group."}
    with (out/"spo2_sensor_windows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (out/"ppg_histograms.json").write_text(json.dumps(histograms, indent=2))
    (out/"spo2_baseline.json").write_text(json.dumps(result, indent=2))
    return result


def motion(root, out, name, fps):
    """Full-frame sequential RGB differences: a screening proxy, not dropped-frame labels."""
    path = root/name/"vid.avi"
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    rows, previous = [], None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        if previous is not None:
            delta = frame-previous
            rows.append([len(rows)+1, (len(rows)+1)/fps,
                         float(np.abs(delta).mean()), float(np.square(delta).mean()),
                         float(delta.mean()), bool(np.array_equal(frame, previous))])
        previous = frame
    cap.release()
    array = np.array(rows)
    sad = array[:, 2]
    median = float(np.median(sad))
    mad = float(np.median(np.abs(sad-median)))
    threshold = median+8*1.4826*mad
    flagged = sad > threshold
    with (out/f"motion_{name}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "container_time_s", "mean_absolute_difference",
                         "mean_squared_difference", "signed_mean_change", "exact_duplicate"])
        writer.writerows(rows)
    top = np.argsort(sad)[-10:][::-1]
    return {"recording": name, "transitions": len(rows), "median_sad": median,
            "mad_sad": mad, "threshold": threshold, "criterion": "median + 8*1.4826*MAD",
            "flagged_transitions": int(flagged.sum()), "exact_duplicates": int(array[:, 5].sum()),
            "top10": [dict(zip(["frame", "container_time_s", "sad", "mse", "mean_change",
                               "exact_duplicate"], array[i].tolist())) for i in top],
            "limit": "Motion/lighting/scene changes can create peaks; a missing frame in a still "
                     "scene can create none. Cannot identify drops or distinguish clock-scale error.",
            "_array": array}


def compare_pixels(root, out, name, count, fps):
    """PyAV RGB vs adapter OpenCV RGB; compare legacy BGR crop order too."""
    path = root/name/"vid.avi"
    tracker = Tracker(fps)
    max_difference = raw_mismatch = crop_mismatch = crops_checked = 0
    legacy_order_mismatch = 0
    absolute_sum = pixels = n = 0
    sentinel = object()
    with av.open(str(path)) as container:
        for a, b in zip_longest(container.decode(video=0), VideoFrames(path, count),
                               fillvalue=sentinel):
            assert a is not sentinel and b is not sentinel, "Different decoder frame counts"
            a = a.to_ndarray(format="rgb24")
            assert a.shape == b.shape
            delta = np.abs(a.astype(np.int16)-b.astype(np.int16))
            max_difference = max(max_difference, int(delta.max()))
            raw_mismatch += int(np.any(delta))
            absolute_sum += int(delta.sum())
            pixels += delta.size
            h, w = b.shape[:2]
            gray = (cv2.cvtColor(cv2.resize(b, (w//2, h//2)), cv2.COLOR_RGB2GRAY)
                    if tracker.needs_detection(n) else None)
            valid, box, _ = tracker.step(n, w, h, gray)
            if valid:
                x1, y1, x2, y2 = box
                av_crop = cv2.resize(a[y1:y2, x1:x2], (36, 36), interpolation=cv2.INTER_AREA)
                adapter_crop = cv2.resize(b[y1:y2, x1:x2], (36, 36), interpolation=cv2.INTER_AREA)
                # Matches prepare_efficientphys.py resize(BGR)->convert(RGB).
                bgr = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)
                legacy_crop = cv2.cvtColor(cv2.resize(bgr[y1:y2, x1:x2], (36, 36),
                                                        interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_BGR2RGB)
                crop_mismatch += int(not np.array_equal(av_crop, adapter_crop))
                legacy_order_mismatch += int(not np.array_equal(adapter_crop, legacy_crop))
                crops_checked += 1
            n += 1
    assert n == count
    result = {"recording": name, "frames_compared": n, "raw_mismatch_frames": raw_mismatch,
              "max_absolute_channel_difference": max_difference,
              "mean_absolute_channel_difference": absolute_sum/pixels,
              "crops_checked_with_identical_boxes": crops_checked,
              "pyav_opencv_crop_mismatch_frames": crop_mismatch,
              "legacy_bgr_vs_adapter_rgb_crop_order_mismatch_frames": legacy_order_mismatch,
              "scope": "One raw AVI, shared boxes. Does not prove parity for every codec or detector.",
              "actual_adapter_pixel_decoder": "OpenCV, same library used by training preprocessing",
              "pyav_usage_in_adapter": "PTS extraction fallback only"}
    (out/"pixel_parity.json").write_text(json.dumps(result, indent=2))
    return result


def run(root, audit, out):
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    records = json.loads(audit.read_text())["records"]
    by_id = {r["recording_id"]: r for r in records}
    grid = []
    for r in records:
        fps, n = r["nominal_fps"], r["decoded_frames"]
        intervals = r["video_clock"]["interval_s"]
        grid.append({"recording": r["recording_id"],
                     "span_residual_s": r["video_clock"]["span_s"]-(n-1)/fps,
                     "max_interval_error_s": max(abs(intervals[x]-1/fps) for x in ["min", "max"]),
                     "relative_span_delta": r["sensor_minus_video_span_s"]/r["video_clock"]["span_s"],
                     "provenance": "Uniform container PTS; acquisition jitter is not observable here."})
    (out/"pts_grid.json").write_text(json.dumps(grid, indent=2))
    print(json.dumps({"spo2": labels(root, out)}), flush=True)
    motions = [motion(root, out, name, by_id[name]["nominal_fps"]) for name in ["8-gt", "5-gt"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    for row, item in zip(axes, motions):
        array = item.pop("_array")
        for ax, col, label in zip(row, [2, 3], ["Mean absolute RGB difference", "Mean squared RGB difference"]):
            ax.plot(array[:, 1], array[:, col], lw=.7)
            if col == 2:
                ax.axhline(item["threshold"], color="red", ls="--", label="8 MAD screening threshold")
                ax.legend()
            ax.set(title=f"{item['recording']}: {label}", xlabel="Container time (s)", ylabel=label)
    fig.savefig(out/"motion_diagnostics.png", dpi=160)
    plt.close(fig)
    (out/"motion_summary.json").write_text(json.dumps(motions, indent=2))
    print(json.dumps({"motion": motions}), flush=True)
    print(json.dumps({"pixels": compare_pixels(root, out, "10-gt", by_id["10-gt"]["decoded_frames"],
                                               by_id["10-gt"]["nominal_fps"])}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.audit, a.out)
