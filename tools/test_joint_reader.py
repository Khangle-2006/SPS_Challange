"""Check the packaged BTC deployment and fold readers without datasets or network."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from joint_core import GRID, MODELS, load_fit, main, read_spectra


class JointReaderTests(unittest.TestCase):
    def test_all_frozen_fits_load(self):
        for fold in [None, *range(5)]:
            models, _ = load_fit(fold)
            self.assertEqual(len(models["router"]["readers"]), 6)
            self.assertEqual(len(models["router"]["router"]), 5)

    def test_abstains_on_static_frontend(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name, data in {
                "request": dict(fold=None, activity="Task", frontend="reference"),
                "frontend": dict(frames=301, valid_frames=301, capture_class_low=False, video_sha256="test"),
                "experts": dict(abstain=True, reason="static_input", weight_sha256={}),
            }.items():
                (out / f"{name}.json").write_text(json.dumps(data))
            with redirect_stdout(StringIO()):
                main(out)
            result = json.loads((out / "result.json").read_bytes())
            self.assertTrue(result["abstain"])
            self.assertEqual(result["reason"], "static_input")
            self.assertIsNone(result["pr_bpm"])

    def test_deploy_crosses_low_bitrate_threshold(self):
        power = np.exp(-.5*((GRID-90)/3)**2) + 1e-6
        power /= power.sum()
        expert = dict.fromkeys(MODELS, power)
        trace = dict(t=np.arange(301)/30, valid=np.ones(301, bool), boxes=np.tile([0,0,36,36], (301,1)),
                     rgb_mean=np.ones((301,3))*100)
        front = dict(source_avg_frame_rate="30/1", source_codec="h264", source_width=640, source_height=480)
        models, _ = load_fit(None)
        for rate in [3999999, 4000000]:
            value, source = read_spectra(models, power, power, expert, trace, dict(front, source_bitrate=rate), False)
            self.assertTrue(np.isfinite(value))
            self.assertEqual(source == "low_bitrate_fallback", rate < 4000000)


if __name__ == "__main__":
    unittest.main()
