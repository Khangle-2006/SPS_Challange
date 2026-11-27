"""Replay all BTC joint-reader dev folds from authorized traces and frozen expert spectra."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]
from joint_core import MODELS, load_fit, read_spectra, windows
from training.btc_joint.assets import verify


def replay(reports, assets_manifest):
    verify(reports, assets_manifest, 'replay')
    bundle = reports / "w2_btc_20261116"
    videos = json.loads((bundle / "export_extracted/VIDEOS.json").read_bytes())
    with (bundle / "delivery/claude_server_20261118/dev_video_meta.csv").open() as f:
        capture = {r["recording"]: r for r in csv.DictReader(f)}
    with np.load(reports / "matlab_native_20261124/fixtures/btc_native.npz") as f:
        arrays = dict(f)
    with np.load(reports / "post_s1_gradient_20261124/headroom/DEV_PRIVATE.npz") as f:
        references = f["nested_router"]
    with np.load(reports / "post_s1_round2/fallback/DEV_OOF.npz") as f:
        fallback = f["signal"]
    models = [load_fit(i)[0] for i in range(5)]
    people = np.array([v["person"] for v in videos])
    for field in ("person", "fold", "truth"):
        expected = [v["pr_bpm" if field == "truth" else field] for v in videos]
        np.testing.assert_array_equal(arrays[field], expected)
    errors = []
    differences = []
    for i, v in enumerate(videos):
        key = v["recording"]
        meta = capture[key]
        front = dict(source_bitrate=int(meta["bit_rate"]), source_height=int(meta["height"]),
                     source_width=int(meta["width"]), source_fps=float(meta["avg_fps"]), source_codec=meta["codec"])
        fit = models[v["fold"]]
        expert = {name: arrays[name][i] for name in MODELS}
        with np.load(bundle / "export_extracted/blocks" / (key + ".npz")) as trace:
            spectra = windows(trace["t"], trace["valid"], trace["rgb_mean"], 10, 2.5)
            assert any(start % 5 == 0 for start, _ in spectra)
            chrom, pos = np.mean([s for _, s in spectra], axis=0)
            value, branch = read_spectra(fit, chrom, pos, expert, trace, front, v["task"].startswith("Task"))
        use_fallback = arrays["low"][i] and fit["fallback"]["anchor"]["beta"][1] == 0
        assert (branch == "low_bitrate_fallback") == bool(use_fallback)
        differences.append(value - (fallback[i] if use_fallback else references[i]))
        errors.append(abs(value-v["pr_bpm"]))
    np.testing.assert_allclose(differences, 0, atol=1e-9, rtol=0)
    mae = np.mean([np.asarray(errors)[people == p].mean() for p in np.unique(people)])
    assert len(videos) == 218 and len(set(people)) == 158
    np.testing.assert_allclose(mae, 8.020099853710796, atol=1e-9, rtol=0)
    print(f"{len(videos)} videos, {len(set(people))} people, MAE {mae:.9f} bpm, max prediction difference {max(map(abs, differences)):.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True, help="FaceVitals project reports directory")
    parser.add_argument('--assets-manifest', type=Path, required=True, help='Verified dev bundle; create with tools/btc_joint_assets.py')
    args = parser.parse_args()
    try:
        replay(args.reports, args.assets_manifest)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'BLOCKED: {exc}\nSee docs/BTC_JOINT_REPRODUCING.md.\n')
