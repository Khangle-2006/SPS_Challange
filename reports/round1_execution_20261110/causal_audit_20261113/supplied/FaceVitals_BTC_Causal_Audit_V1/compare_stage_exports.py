#!/usr/bin/env python3
"""Compare declared-equivalent production exports without changing their values.

This tool locates the first RECORDED divergence. It does not establish a
physiological root cause or a contribution to BTC MAE. No model is loaded.
Requires Python 3.10+ and NumPy. No files outside each export directory are read.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


class InvalidComparison(ValueError):
    pass


CONTRACT_FIELDS = (
    "schema_version", "comparison_kind", "case_id", "common_input_sha256",
    "checkpoint_sha256", "pipeline_contract_id", "model_mode",
)
KINDS = {"route_equivalence", "batch_invariance", "cache_replay"}


def read_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in CONTRACT_FIELDS:
        if key not in data:
            raise InvalidComparison(f"Missing contract field {key}: {path.name}")
    if data["schema_version"] != 1 or data["comparison_kind"] not in KINDS:
        raise InvalidComparison("Unsupported schema_version or comparison_kind")
    if data["model_mode"] != "eval_frozen":
        raise InvalidComparison("This contract requires eval_frozen; model state must be reset consistently")
    for key in ("common_input_sha256", "checkpoint_sha256"):
        value = data[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise InvalidComparison(f"{key} must contain a lowercase SHA256 digest")
    stages = data.get("stages", [])
    if not stages or len({s.get("name") for s in stages}) != len(stages):
        raise InvalidComparison("Stages must be nonempty and have unique names")
    for stage in stages:
        for key in ("name", "path", "axis_order", "units", "atol", "rtol"):
            if key not in stage:
                raise InvalidComparison(f"Missing {key} in a stage")
        for key in ("atol", "rtol"):
            value = stage[key]
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise InvalidComparison(f"Invalid {key} at {stage['name']}")
    return path, data


def local_array(base, spec, max_bytes):
    relative = Path(spec["path"])
    if relative.is_absolute():
        raise InvalidComparison("Array paths must be relative to the export directory")
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise InvalidComparison("Array path leaves the export directory")
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise InvalidComparison(f"Missing or oversized array file: {relative}")
    if path.suffix == ".npz":
        import zipfile
        key = spec.get("key")
        if not isinstance(key, str):
            raise InvalidComparison("NPZ arrays require a key")
        with zipfile.ZipFile(path) as archive:
            try:
                size = archive.getinfo(key + ".npy").file_size
            except KeyError as exc:
                raise InvalidComparison(f"Missing NPZ key {key}") from exc
            if size > max_bytes:
                raise InvalidComparison(f"Uncompressed array exceeds cap: {key}")
        with np.load(path, allow_pickle=False) as archive:
            array = np.asarray(archive[key]).copy()
    elif path.suffix == ".npy":
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    else:
        raise InvalidComparison("Only .npy and .npz are supported")
    if array.nbytes > max_bytes or array.dtype.kind not in "biufc":
        raise InvalidComparison("Array must be numeric and within the memory cap")
    if (array.dtype.kind == "f" and array.dtype.itemsize > 8) or (array.dtype.kind == "c" and array.dtype.itemsize > 16):
        raise InvalidComparison("Extended-precision floating arrays are unsupported; refusing lossy comparison")
    if array.size == 0:
        raise InvalidComparison("Empty arrays cannot establish equivalence")
    return np.asarray(array)


def valid_mask(base, spec, array, max_bytes):
    mask_spec = spec.get("valid_mask")
    if mask_spec is None:
        return np.ones(array.shape, dtype=bool)
    mask = local_array(base, mask_spec, max_bytes)
    if mask.dtype.kind != "b":
        raise InvalidComparison("valid_mask must have boolean dtype")
    # A T or T,H,W mask applies to all remaining channel dimensions.
    if mask.ndim < array.ndim and mask.shape == array.shape[:mask.ndim]:
        mask = mask.reshape(mask.shape + (1,) * (array.ndim - mask.ndim))
    try:
        return np.broadcast_to(mask, array.shape)
    except ValueError as exc:
        raise InvalidComparison("valid_mask is not broadcastable to the array") from exc


def compare(left_path, right_path, max_bytes=256 * 1024 * 1024):
    lp, left = read_manifest(left_path)
    rp, right = read_manifest(right_path)
    mismatches = [key for key in CONTRACT_FIELDS if left[key] != right[key]]
    if mismatches:
        raise InvalidComparison("Different comparison contracts: " + ", ".join(mismatches))
    ls, rs = left["stages"], right["stages"]
    if [s["name"] for s in ls] != [s["name"] for s in rs]:
        raise InvalidComparison("Stage names/order differ; missing stages must not be silently skipped")
    records = []
    for a_spec, b_spec in zip(ls, rs):
        name = a_spec["name"]
        for key in ("axis_order", "units", "atol", "rtol"):
            if a_spec[key] != b_spec[key]:
                raise InvalidComparison(f"Different {key} at {name}")
        a = local_array(lp.parent, a_spec, max_bytes)
        b = local_array(rp.parent, b_spec, max_bytes)
        record = {"stage": name, "left_shape": list(a.shape), "right_shape": list(b.shape),
                  "left_dtype": str(a.dtype), "right_dtype": str(b.dtype),
                  "units": a_spec["units"], "atol": a_spec["atol"], "rtol": a_spec["rtol"]}
        if a.shape != b.shape:
            record.update(equivalent=False, reason="shape_mismatch")
            records.append(record)
            continue
        am = valid_mask(lp.parent, a_spec, a, max_bytes)
        bm = valid_mask(rp.parent, b_spec, b, max_bytes)
        if not np.array_equal(am, bm):
            record.update(equivalent=False, reason="support_mask_mismatch")
            records.append(record)
            continue
        if not np.any(am):
            raise InvalidComparison(f"No valid support at {name}")
        av, bv = a[am], b[bm]
        if not np.all(np.isfinite(av)) or not np.all(np.isfinite(bv)):
            raise InvalidComparison(f"Nonfinite values on declared valid support at {name}")
        integer_pair = a.dtype.kind in "biu" and b.dtype.kind in "biu"
        dtype = np.longdouble if integer_pair else (
            np.complex128 if (np.iscomplexobj(av) or np.iscomplexobj(bv)) else np.float64)
        avf, bvf = av.astype(dtype), bv.astype(dtype)
        delta = np.abs(avf - bvf)
        if integer_pair:
            close = av == bv
        else:
            # Symmetric tolerance; no rescaling, lag search or sign alignment.
            threshold = a_spec["atol"] + a_spec["rtol"] * np.maximum(np.abs(avf), np.abs(bvf))
            close = delta <= threshold
        norm = max(float(np.linalg.norm(avf)), float(np.linalg.norm(bvf)), np.finfo(float).tiny)
        same_dtype = a.dtype == b.dtype
        record.update(
            equivalent=bool(np.all(close) and same_dtype),
            reason="within_declared_tolerance" if np.all(close) and same_dtype else
                   ("dtype_mismatch" if not same_dtype else "value_mismatch"),
            valid_elements=int(av.size), invalid_elements=int(a.size - av.size),
            max_abs_difference=float(np.max(delta)),
            rms_difference=float(np.sqrt(np.mean(delta ** 2))),
            relative_l2_difference=float(np.linalg.norm(avf - bvf) / norm),
            elements_outside_tolerance=int(np.count_nonzero(~close)),
        )
        records.append(record)
    first = next((r["stage"] for r in records if not r["equivalent"]), None)
    return {
        "status": "DIVERGENCE_REQUIRES_EXPLANATION" if first else "COMPATIBLE_WITH_DECLARED_TOLERANCES",
        "case_id": left["case_id"], "comparison_kind": left["comparison_kind"],
        "first_recorded_divergence": first,
        "manifest_sha256": {"left": hashlib.sha256(lp.read_bytes()).hexdigest(),
                            "right": hashlib.sha256(rp.read_bytes()).hexdigest()},
        "stages": records,
        "limitations": [
            "Only compares exported observations; hashes and instrumentation must be independently verified.",
            "First recorded divergence need not be the root cause; add instrumentation between adjacent stages.",
            "Passing does not validate shared bugs or demonstrate physiological information.",
            "A numerical divergence requires repeated-run and intended-configuration checks before calling it a defect.",
            "No BTC score, training, calibration or holdout evaluation is performed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_manifest")
    parser.add_argument("right_manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-array-mib", type=int, default=256)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output already exists; use a fresh run path to preserve evidence")
    if args.max_array_mib <= 0:
        parser.error("max-array-mib must be positive")
    try:
        result = compare(args.left_manifest, args.right_manifest, args.max_array_mib * 1024 * 1024)
        code = 1 if result["first_recorded_divergence"] else 0
    except (InvalidComparison, OSError, ValueError, KeyError, TypeError) as exc:
        result = {"status": "INVALID_COMPARISON", "error": str(exc), "not_a_root_cause_result": True}
        code = 2
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "first_recorded_divergence": result.get("first_recorded_divergence")}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
