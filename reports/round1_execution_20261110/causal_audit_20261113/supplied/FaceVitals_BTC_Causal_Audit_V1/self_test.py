"""Numerical/export-contract tests ONLY. No FaceVitals model or patient data."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from compare_stage_exports import compare, InvalidComparison


def main():
    tests = []
    with tempfile.TemporaryDirectory(prefix="btc_export_contract_") as td:
        root = Path(td)
        t = np.arange(301, dtype=np.float64) / 30
        rgb = np.ones((301, 4, 4, 3), dtype=np.float32)
        rgb[..., 0] *= 80
        rgb[..., 1] *= 110
        rgb[..., 2] *= 160
        rgb += np.sin(2 * np.pi * 1.5 * t)[:, None, None, None].astype(np.float32)
        d = np.diff(np.sin(2 * np.pi * 1.5 * t)).astype(np.float32)
        arrays = {"pts": t, "roi_rgb": rgb, "model_input": rgb / 255,
                  "d_hat": d, "valid": np.ones(301, dtype=bool)}
        manifest = {
            "schema_version": 1, "comparison_kind": "route_equivalence", "case_id": "SYNTHETIC_FIXTURE_ONLY",
            "common_input_sha256": hashlib.sha256(t.tobytes() + rgb.tobytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(b"no_checkpoint_synthetic_fixture").hexdigest(),
            "pipeline_contract_id": "synthetic-contract-v1", "model_mode": "eval_frozen",
            "run_context": {"evidence_type": "synthetic self-test; no model inference"},
            "stages": [
                {"name": "pts", "path": "stages.npz", "key": "pts", "axis_order": "T", "units": "seconds", "atol": 0.0, "rtol": 0.0},
                {"name": "roi_rgb", "path": "stages.npz", "key": "roi_rgb", "axis_order": "THWC", "units": "RGB_0_255", "atol": 1e-7, "rtol": 1e-5},
                {"name": "model_input", "path": "stages.npz", "key": "model_input", "axis_order": "THWC", "units": "RGB_div255", "atol": 1e-7, "rtol": 1e-5},
                {"name": "d_hat", "path": "stages.npz", "key": "d_hat", "axis_order": "T_minus_1", "units": "synthetic_difference", "atol": 1e-7, "rtol": 1e-5},
            ],
        }
        def write(side, a, m):
            p = root / side
            p.mkdir(exist_ok=True)
            np.savez_compressed(p / "stages.npz", **a)
            (p / "manifest.json").write_text(json.dumps(m))
            return p / "manifest.json"
        left = write("left", arrays, manifest)
        def run(name, right_arrays, right_manifest, first=None, invalid=False, extra=None):
            rp = write("right", right_arrays, right_manifest)
            try:
                result = compare(left, rp)
                ok = (not invalid and result["first_recorded_divergence"] == first)
                if extra:
                    ok = ok and extra(result)
                detail = result["status"]
            except InvalidComparison as exc:
                ok, detail = invalid, str(exc)
            tests.append({"test": name, "passed": bool(ok), "observed": detail})
            assert ok, tests[-1]

        run("identical_exports", arrays, manifest)
        changed = {k: v.copy() for k, v in arrays.items()}
        changed["roi_rgb"] = changed["roi_rgb"][..., ::-1]
        run("channel_order_divergence", changed, manifest, first="roi_rgb")
        changed = {k: v.copy() for k, v in arrays.items()}
        changed["pts"] *= 1000
        run("time_unit_divergence", changed, manifest, first="pts")
        changed = {k: v.copy() for k, v in arrays.items()}
        changed["d_hat"] += np.float32(0.1)
        run("changed_prediction_with_same_input", changed, manifest, first="d_hat")
        wrong = copy.deepcopy(manifest)
        wrong["checkpoint_sha256"] = "f" * 64
        run("different_checkpoint_is_invalid_comparison", arrays, wrong, invalid=True)
        wrong = copy.deepcopy(manifest)
        wrong["stages"][0]["path"] = "../outside.npz"
        run("outside_directory_rejected", arrays, wrong, invalid=True)
        changed = {k: v.copy() for k, v in arrays.items()}
        changed["d_hat"][0] = np.nan
        run("nonfinite_valid_sample_rejected", changed, manifest, invalid=True)

        # Explicit missing support must match; masked nonfinite samples are allowed.
        masked = {k: v.copy() for k, v in arrays.items()}
        masked["valid"][0] = False
        masked["roi_rgb"][0] = np.nan
        mm = copy.deepcopy(manifest)
        mm["stages"][1]["valid_mask"] = {"path": "stages.npz", "key": "valid"}
        left = write("left", masked, mm)
        run("declared_missing_support_is_preserved", masked, mm)
        changed = {k: v.copy() for k, v in masked.items()}
        changed["valid"][1] = False
        run("support_mask_divergence_is_not_hidden", changed, mm, first="roi_rgb")

        ia = {"ticks": np.array([10**18, 10**18 + 10], dtype=np.int64)}
        im = copy.deepcopy(manifest)
        im["stages"] = [{"name": "ticks", "path": "stages.npz", "key": "ticks", "axis_order": "T", "units": "integer_ticks", "atol": 0, "rtol": 0}]
        left = write("left", ia, im)
        ib = {"ticks": ia["ticks"] + 1}
        run("integer_timestamp_precision", ib, im, first="ticks",
            extra=lambda r: r["stages"][0]["max_abs_difference"] == 1.0)
        if np.dtype(np.longdouble).itemsize > 8:
            xa = {"ticks": np.array([1], dtype=np.longdouble)}
            xb = {"ticks": np.nextafter(xa["ticks"], np.longdouble(2))}
            left = write("left", xa, im)
            run("extended_precision_is_not_silently_rounded", xb, im, invalid=True)
    result = {"evidence_type": "synthetic export-comparator self-tests; not BTC experiments",
              "tests_passed": sum(x["passed"] for x in tests), "tests_total": len(tests), "tests": tests}
    output = Path(__file__).with_name("SELF_TEST_RESULTS.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
