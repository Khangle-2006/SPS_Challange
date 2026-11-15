"""Decode a development recording through the adapter; no training/test access."""
import argparse
import json
import numpy as np
from data_adapters import UBFC2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--subject", default="subject1")
    args = p.parse_args()
    r = UBFC2(args.root).load(args.subject)
    count = 0
    shape = None
    for rgb in r.frames:
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3
        if shape is None:
            shape = rgb.shape
        assert rgb.shape == shape
        count += 1
    assert count == len(r.timestamps) == len(r.require_synchronized_ppg().values)
    print(json.dumps({"passed": True, "subject_id": r.subject_id, "decoded_frames": count,
                      "shape": shape, "duration_s": float(r.timestamps[-1]),
                      "duplicate_timestamps": r.metadata["duplicate_timestamps"]}))
