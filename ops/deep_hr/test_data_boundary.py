"""Clock/identity/label-contract and decoder regression tests (CPU only)."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from data_adapters import UBFC2, UBFC1, AiPEX, PURE, Pilot, Label
from hr_decoder import estimate_hr
from spectral_ensemble import normalized_psd
from scipy import signal, linalg


class DataBoundary(unittest.TestCase):
    def test_ubfc2_preserves_duplicate_times_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"subject1"
            p.mkdir()
            np.savetxt(p/"ground_truth.txt", [[1, 2, 3], [70, 71, 72], [10, 10, 10.1]])
            r = UBFC2(d).load("subject1")
            np.testing.assert_allclose(r.timestamps, [0, 0, .1])
            self.assertEqual(r.subject_id, "ubfc2:subject1")
            self.assertEqual(r.metadata["duplicate_timestamps"], 1)
            self.assertEqual(len(r.as_tuple()), 4)

    def test_ubfc1_keeps_sensor_clock_and_requires_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"s1"
            p.mkdir()
            (p/"vid.avi").touch()
            np.savetxt(p/"gtdump.xmp", [[1000, 70, 98, 1], [1020, 71, 99, 2]], delimiter=",")
            r = UBFC1(d).load("s1", frame_timestamps=[0, .03, .06])
            self.assertEqual(len(r.labels["ppg"].values), 2)
            self.assertEqual(r.metadata["split_group_id"], "ubfc1:all")
            self.assertFalse(r.metadata["allow_within_dataset_split"])
            np.testing.assert_allclose(r.labels["ppg"].timestamps, [0, .02])
            with self.assertRaises(ValueError):
                r.require_synchronized_ppg()
            aligned = UBFC1(d).load("s1", [0, .03, .06], sensor_offset_s=.01)
            np.testing.assert_allclose(aligned.require_synchronized_ppg().timestamps, [.01, .03])

    def test_aipex_scalar_is_not_a_waveform(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"manifest.json"
            p.write_text(json.dumps([{"recording_id": "a_left", "subject_id": "a",
                                     "video": "left.avi", "mean_hr_bpm": 72}]))
            r = AiPEX(d, p).load("a_left", [0, .1])
            self.assertEqual(r.labels["hr"].scope, "recording")
            self.assertNotIn("ppg", r.labels)
            with self.assertRaises(ValueError):
                r.require_synchronized_ppg()

    def test_pure_integer_clock_subtraction_before_float_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"01-01"
            images = p/"01-01"
            images.mkdir(parents=True)
            origin = 1392643993642815000
            for ns in [origin, origin+33333333]:
                (images/f"Image{ns}.png").touch()
            (p/"01-01.json").write_text(json.dumps({"/FullPackage": [
                {"Timestamp": origin-10000000, "Value": {"waveform": 1}},
                {"Timestamp": origin+40000000, "Value": {"waveform": 2}}]}))
            r = PURE(d).load("01-01")
            self.assertAlmostEqual(r.timestamps[1], .033333333, places=12)
            self.assertEqual(r.subject_id, "pure:01")
            np.testing.assert_allclose(r.labels["ppg"].timestamps, [-.01, .04])

    def test_reject_bad_clocks_and_missing_pilot(self):
        with self.assertRaises(ValueError):
            Label([1, 2], [1, 0], "a.u.")
        with self.assertRaises(NotImplementedError):
            Pilot().load("one")

    def test_new_lower_band_and_invalid_signals(self):
        t = np.arange(900)/30
        for bpm in [42.5, 44, 72, 120, 168]:
            self.assertLess(abs(estimate_hr(np.sin(2*np.pi*bpm/60*t))-bpm), .5)
        with self.assertRaises(ValueError):
            estimate_hr(np.ones(300))

    def test_fft_paths_match_independent_dense_detrending(self):
        rng = np.random.default_rng(2027)
        t = np.arange(300)/30
        w = np.sin(2*np.pi*1.3*t) + .5*np.sin(2*np.pi*.72*t) + rng.normal(0, .1, len(t))
        eye = np.eye(len(w))
        d = np.diff(eye, n=2, axis=0)
        detrended = w-linalg.solve(eye+100**2*(d.T@d), w, assume_a="pos")
        filt = signal.sosfiltfilt(signal.butter(1, [.7, 3], fs=30, btype="bandpass", output="sos"), detrended)
        f, p = signal.periodogram(filt, fs=30, window="hann", nfft=4096, detrend=False)
        selected = (f >= .7) & (f <= 3)
        expected = f[selected][np.argmax(p[selected])]*60
        self.assertEqual(estimate_hr(w), expected)
        freq, psd = normalized_psd(w)
        self.assertEqual(freq[np.argmax(psd)], expected)
        np.testing.assert_allclose(psd, p[selected]/p[selected].sum(), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
