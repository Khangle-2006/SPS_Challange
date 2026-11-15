"""Read-only full UBFC1 decode and adapter audit. Never infer clock alignment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from data_adapters import UBFC1
from prepare_recording import prepare
from video_roi_frontend import Tracker, clip_tensor


def stats(v):
    return {"min": float(np.min(v)), "max": float(np.max(v)),
            "median": float(np.median(v)), "unique": int(len(np.unique(v))),
            "zero_count": int(np.sum(v == 0)), "nonfinite": int(np.sum(~np.isfinite(v)))}


def clock(t):
    dt = np.diff(t)
    positive = dt[dt > 0]
    return {"samples": len(t), "start_s": float(t[0]), "last_s": float(t[-1]),
            "span_s": float(t[-1] - t[0]), "duplicate_count": int(np.sum(dt == 0)),
            "backward_count": int(np.sum(dt < 0)), "interval_s": stats(dt),
            "effective_hz": float((len(t)-1)/(t[-1]-t[0])),
            "gaps_over_2x_median": int(np.sum(dt > 2*np.median(positive)))}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def smoke(root, out, name):
    """Exercise RGB -> Haar -> model input only; no aligned reference or HR score."""
    r = UBFC1(root).load(name)
    tracker = Tracker((len(r.timestamps)-1)/(r.timestamps[-1]-r.timestamps[0]))
    crops, valid = [], []
    cv2.setNumThreads(1)
    for i, rgb in enumerate(r.frames):
        h, w = rgb.shape[:2]
        gray = (cv2.cvtColor(cv2.resize(rgb, (w//2, h//2)), cv2.COLOR_RGB2GRAY)
                if tracker.needs_detection(i) else None)
        ok, box, _ = tracker.step(i, w, h, gray)
        crop = np.zeros((36, 36, 3), dtype=np.uint8)
        if ok:
            x1, y1, x2, y2 = box
            crop = cv2.resize(rgb[y1:y2, x1:x2], (36, 36), interpolation=cv2.INTER_AREA)
        crops.append(crop)
        valid.append(ok)
    crops, valid = np.asarray(crops), np.asarray(valid)
    assert np.all(np.diff(r.timestamps) > 0)
    accepted, rejected = [], []
    for start in np.arange(0, r.timestamps[-1]-10, 10):
        tensor, bad = clip_tensor(crops, r.timestamps, valid, start)
        if tensor is None:
            rejected.append({"start_s": float(start), "invalid_support_samples": bad})
        else:
            assert tensor.shape == (301, 3, 36, 36) and np.isfinite(tensor).all()
            accepted.append(float(start))
    report = {"recording": name, "decoded_frames": len(crops),
              "valid_roi_frames": int(valid.sum()), "crop_shape": list(crops.shape),
              "accepted_window_starts_s": accepted, "rejected_windows": rejected,
              "model_input_shape": [301, 3, 36, 36], "reference_alignment_used": False,
              "model_inference_run": False, "hr_metrics_computed": False,
              "frontend": "existing Haar, not landmarks"}
    (out/"frontend_smoke.json").write_text(json.dumps(report, indent=2))
    return report


def audit(root, out):
    out.mkdir(parents=True, exist_ok=False)
    backend = UBFC1(root)
    records = []
    for name in backend.recordings():
        started = time.monotonic()
        folder = root/name
        # Independent CSV parser checks column mapping and units.
        with (folder/"gtdump.xmp").open() as f:
            raw = np.array([[float(x) for x in row] for row in csv.reader(f)])
        r = backend.load(name)
        assert raw.shape[1] == 4 and np.isfinite(raw).all()
        for column, label in [(1, "hr"), (2, "spo2"), (3, "ppg")]:
            np.testing.assert_array_equal(r.labels[label].values, raw[:, column])
            np.testing.assert_allclose(r.labels[label].timestamps,
                                       (raw[:, 0]-raw[0, 0])/1000, rtol=0, atol=1e-12)
        assert r.subject_id == f"ubfc1:{name}" and len(r.as_tuple()) == 4
        cap = cv2.VideoCapture(str(folder/"vid.avi"))
        nominal_fps = cap.get(cv2.CAP_PROP_FPS)
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first_bgr = cap.read()
        cap.release()
        assert ok
        count = 0
        shape = first_bgr.shape
        for rgb in r.frames:
            assert rgb.shape == shape and rgb.dtype == np.uint8
            if count == 0:
                np.testing.assert_array_equal(rgb, first_bgr[:, :, ::-1])
            count += 1
        assert count == len(r.timestamps) == reported_frames
        # Re-iteration must start at the first frame, not exhaust the stream.
        stream = iter(r.frames)
        np.testing.assert_array_equal(next(stream), first_bgr[:, :, ::-1])
        stream.close()
        blocked = False
        try:
            r.require_synchronized_ppg()
        except ValueError:
            blocked = True
        assert blocked
        # Verify end-to-end cache preparation refuses unsupported window labels.
        try:
            prepare(r, out/f"must_not_exist_{name}")
        except ValueError as exc:
            assert "Synchronized PPG unavailable" in str(exc)
        else:
            raise AssertionError("Unverified alignment reached cache preparation")
        assert not (out/f"must_not_exist_{name}").exists()
        video_clock, sensor_clock = clock(r.timestamps), clock(r.labels["ppg"].timestamps)
        row = {"recording_id": name, "adapter_passed": True,
               "decoded_frames": count, "rgb_shape": list(shape),
               "nominal_fps": nominal_fps, "reported_frames": reported_frames,
               "video_clock": video_clock, "sensor_clock": sensor_clock,
               "raw_sensor_start_s": float(raw[0, 0]/1000),
               "raw_sensor_end_s": float(raw[-1, 0]/1000),
               "sensor_minus_video_span_s": sensor_clock["span_s"]-video_clock["span_s"],
               "labels": {key: stats(label.values) for key, label in r.labels.items()},
               "synchronization_verified": False, "window_evaluation_blocked": blocked,
               "subject_identity_verified": False,
               "files": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                         for p in [folder/"vid.avi", folder/"gtdump.xmp"]},
               "elapsed_s": time.monotonic()-started}
        records.append(row)
        (out/"audit.json").write_text(json.dumps({"records": records}, indent=2))
        print(json.dumps({k: row[k] for k in ["recording_id", "decoded_frames",
                                             "sensor_minus_video_span_s", "adapter_passed"]}), flush=True)
    summary = {"records": records, "recording_count": len(records),
               "decoded_frames": sum(r["decoded_frames"] for r in records),
               "all_adapter_checks_passed": all(r["adapter_passed"] for r in records),
               "window_evaluation_ready": False,
               "limitations": ["No independently established sensor-to-video offset or drift mapping.",
                               "Folder IDs are provisional recording identities, not verified people.",
                               "SpO2 units/columns verified; physiological accuracy not calibrated."]}
    (out/"audit.json").write_text(json.dumps(summary, indent=2))
    with (out/"summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["recording", "frames", "video_span_s", "sensor_samples", "sensor_span_s",
                         "span_delta_s", "hr_min", "hr_max", "spo2_min", "spo2_max"])
        for r in records:
            writer.writerow([r["recording_id"], r["decoded_frames"], r["video_clock"]["span_s"],
                             r["sensor_clock"]["samples"], r["sensor_clock"]["span_s"],
                             r["sensor_minus_video_span_s"],
                             r["labels"]["hr"]["min"], r["labels"]["hr"]["max"],
                             r["labels"]["spo2"]["min"], r["labels"]["spo2"]["max"]])
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run RGB frontend check in an existing audit directory.")
    args = parser.parse_args()
    if args.smoke_only:
        print(json.dumps(smoke(args.root, args.out, "10-gt")), flush=True)
    else:
        audit(args.root, args.out)
