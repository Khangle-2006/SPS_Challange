"""CPU-only replay of saved waveforms. Never rewrite historical artifacts."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from hr_decoder import spectrum, estimate_hr, BAND_HZ, LEGACY_BAND_HZ

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
SEEDS = ["2027", "2028", "2029"]


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sources():
    base = REPORTS / "cv_5fold_20261020/results/cv_5fold_20261020"
    results = read(base / "fold_results.json")
    for p in sorted((base / "evaluation").glob("*.npz")):
        fold, family, condition = p.stem.split("_", 2)
        yield p, "waves", {"single": results[fold[4:]][family][condition]}
    for name in ["matched_detector_roi_20261019", "matched_spectral_20261019",
                 "spectral_ensemble_20261019", "end_to_end_detection_20261019"]:
        base = REPORTS / name / "results" / name
        results = read(base / "results.json")
        for p in sorted(base.glob("evaluation/*.npz")):
            arm, condition = p.stem.removesuffix("_waves").split("_", 1)
            yield p, "waves", results[arm][condition]
        for p in sorted(base.glob("waveforms/*.npz")):
            if name.startswith("spectral_ensemble"):
                family, partition, condition = p.stem.split("_", 2)
                yield p, "waves", results["groups"][family][partition][condition]
            else:
                family, condition = p.stem.split("_", 1)
                group = results["groups"][family][condition]
                yield p, "waves", group["detected_ROI"]
                yield p, "native_waves", group["native_ROI"]


def decode(waves, band):
    waves = np.asarray(waves)
    if waves.ndim == 2:
        waves = waves[None]
    ns, nw, _ = waves.shape
    out = {k: [None]*nw for k in (["single"] if ns == 1 else SEEDS+["hr_median", "psd_mean"])}
    for i in range(nw):
        powers, hrs = [], []
        for j in range(ns):
            try:
                f, p = spectrum(waves[j, i], band=band)
                hr = float(f[np.argmax(p)])
            except ValueError:
                p, hr = None, None
            out["single" if ns == 1 else SEEDS[j]][i] = hr
            hrs.append(hr)
            powers.append(p)
        if ns == 3 and all(h is not None for h in hrs):
            out["hr_median"][i] = float(np.median(hrs))
            out["psd_mean"][i] = float(f[np.argmax(np.mean(powers, axis=0))])
    return out


def metrics(rows, key):
    errors = np.array([abs(r[key]-r["gt_hr"]) if r[key] is not None else 135. for r in rows])
    by_subject = defaultdict(list)
    for r, e in zip(rows, errors):
        by_subject[r["subject"]].append(e)
    return {"n": len(rows), "subject_macro_MAE": float(np.mean([np.mean(e) for e in by_subject.values()])),
            "window_MAE": float(errors.mean()), "P95": float(np.percentile(errors, 95)),
            "errors_over20": int((errors > 20).sum()),
            "invalid": sum(r[key] is None for r in rows)}


def main(out=None):
    out = Path(out) if out else REPORTS / "e0_band_20261020"
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"old_band_hz": LEGACY_BAND_HZ, "new_band_hz": BAND_HZ,
                "primary_reference": "Historical GT HR frozen to isolate decoder-only change",
                "secondary_reference": "Recompute contact PPG with new band; report separately",
                "no_training": True, "no_checkpoint_selection": True,
                "test_re_evaluated": False, "invalid_penalty_bpm": 135,
                "code_sha256": {p.name: sha(p) for p in [Path(__file__), HERE/"hr_decoder.py"]}}
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2))
    allrows, summaries, inventory = [], [], []
    for path, key, reports in sources():
        with np.load(path) as z:
            waves = z[key]
        old, new = decode(waves, LEGACY_BAND_HZ), decode(waves, BAND_HZ)
        source_id = str(path.relative_to(REPORTS)).replace("\\", "/") + ":" + key
        inventory.append({"source": source_id, "sha256": sha(path), "shape": list(waves.shape)})
        for method, predictions in old.items():
            reference = reports[method]["records"]
            if len(reference) != len(predictions):
                raise ValueError("Waveform/record count mismatch")
            rows = []
            for i, (r, h) in enumerate(zip(reference, predictions)):
                expected = r["pred_hr"]
                if (h is None) != (expected is None) or (h is not None and abs(h-expected) > 1e-7):
                    raise ValueError(f"Legacy prediction not reproduced: {source_id}/{method}/{i}: {h}/{expected}")
                rows.append({**r, "source": source_id, "method": method, "old_hr": h, "new_hr": new[method][i]})
            a, b = metrics(rows, "old_hr"), metrics(rows, "new_hr")
            summaries.append({"source": source_id, "method": method, "old": a, "new": b,
                              "delta_MAE": b["subject_macro_MAE"]-a["subject_macro_MAE"],
                              "changed": sum(r["old_hr"] != r["new_hr"] for r in rows)})
            allrows.extend(rows)
    # PPG clocks are already audited and stored for the 35 development subjects.
    ref_cache = {}
    for row in allrows:
        k = (row["subject"], row["start_s"])
        if k not in ref_cache:
            p = REPORTS/"cv_5fold_20261020/inputs/meta/clean"/f"{k[0]}_meta.npz"
            z = np.load(p)
            q = k[1] + np.arange(300)/30
            if q[0] < z["t"][0] or q[-1] > z["t"][-1]:
                raise ValueError("Reference interpolation outside support")
            w = np.interp(q, z["t"], z["ppg"])
            ref_cache[k] = (estimate_hr(w, band=LEGACY_BAND_HZ), estimate_hr(w))
        row["gt_hr_reconstructed_legacy"], row["gt_hr_new_band"] = ref_cache[k]
    pooled = {}
    for family in ["mse", "spectral"]:
        pooled[family] = {}
        for condition in ["clean", "h264_500k", "h264_250k", "h264_1000k"]:
            rows = [r for r in allrows if "cv_5fold" in r["source"]
                    and f"_{family}_{condition}.npz" in r["source"]]
            if len(rows) != 203:
                raise ValueError("Incomplete CV OOF coverage")
            both = [{**r, "gt_hr": r["gt_hr_new_band"]} for r in rows]
            pooled[family][condition] = {"old": metrics(rows, "old_hr"),
                                        "decoder_only": metrics(rows, "new_hr"),
                                        "decoder_and_reference": metrics(both, "new_hr")}
    coverage = {"replayed_arrays": len(inventory), "prediction_rows_including_ensembles": len(allrows),
                "legacy_predictions_verified": True,
                "not_replayed": ["Original clean test: no stored raw output waveforms in these packages",
                                "JPEG/stress/ROI-EMA and early control checkpoints without saved waveforms",
                                "POS/GREEN require original signal packages and their separate sampling clocks"],
                "complete_all_historical_results": False}
    result = {"coverage": coverage, "cv": pooled, "groups": summaries, "inventory": inventory,
              "reference_max_legacy_reconstruction_delta": max(abs(r["gt_hr"]-r["gt_hr_reconstructed_legacy"]) for r in allrows)}
    (out/"results.json").write_text(json.dumps(result, indent=2))
    columns = ["source", "method", "subject", "window", "start_s", "gt_hr", "gt_hr_reconstructed_legacy",
               "gt_hr_new_band", "old_hr", "new_hr"]
    with (out/"windows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(allrows)
    print(json.dumps({"coverage": coverage, "cv": pooled}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    main(args.out)
