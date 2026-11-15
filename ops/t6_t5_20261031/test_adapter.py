"""Independent clocks, heldout protection, exact ROI semantics and cap classification."""
import copy
import json
import unittest
from pathlib import Path
import numpy as np
from data_adapters import Label, Recording, VideoFrames
from window_support import recording_windows
from run_t5_qa import references, approved_ids, mask_for, classify, merge
from score_round1 import reference_hr
from t8_t1_loader import supported


class AdapterTests(unittest.TestCase):
    def test_approved_cohort_excludes_heldout_people(self):
        code = Path(__file__).parent
        folds = json.loads((code/'folds_frozen.json').read_text())
        mcd = json.loads((code/'mcd_cohort.json').read_text())
        ids = approved_ids('UBFC2', folds, mcd)
        self.assertEqual(len(ids), 35)
        self.assertTrue(set(ids).isdisjoint(folds['original_ubfc_test_subjects']))
        ids = approved_ids('MCD', folds, mcd)
        self.assertEqual(len(ids), 119)
        self.assertFalse(any(i.startswith('4952') or i == '7412_FullHDwebcam_after' for i in ids))
        broken = copy.deepcopy(folds)
        broken['folds'][0]['ubfc_train_subjects'][0] = folds['original_ubfc_test_subjects'][0]
        with self.assertRaises(AssertionError):
            approved_ids('UBFC2', broken, mcd)

    def test_reference_uses_raw_ppg_clock_and_preserves_duplicate_invalid_endpoint(self):
        vt = np.arange(751)/30
        pt = .13+np.arange(2501)/100
        pp = np.sin(2*np.pi*1.2*pt)
        rec = Recording(VideoFrames(Path('never_opened'), len(vt)), vt,
            {'ppg': Label(pp, pt, 'a.u.')}, 'fixture', 'record', {'synchronization_verified': True})
        canonical = recording_windows(rec)
        rows = references(rec, canonical)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]['window_start_s'], .13)
        self.assertAlmostEqual(reference_hr(rows[0]), 72., delta=.5)
        # Duplicate firstclock endpoint with an invalid label must stay invalid.
        pt = np.r_[pt[0], pt]
        pp = np.r_[np.nan, pp]
        rec.labels['ppg'] = Label(pp, pt, 'a.u.')
        canonical = recording_windows(rec)
        rows = references(rec, canonical)
        self.assertEqual(len(rows), 2)
        self.assertIn('ppg_invalid_sample', rows[0]['source_reasons'])
        with self.assertRaises(ValueError):
            reference_hr(rows[0])

    def test_saved_rectangles_have_exclusive_right_bottom(self):
        mask = mask_for((20, 30), [2, 3, 8, 10], True)
        self.assertEqual(int(mask.sum()), 6*7)
        self.assertEqual(mask[10, 8], 0)
        self.assertEqual(mask_for((20, 30), [0, 0, 0, 0], False).sum(), 0)
        with self.assertRaises(ValueError):
            mask_for((20, 30), [-1, 3, 8, 10], True)

    def test_duplicate_validity_and_large_gap_are_not_hidden(self):
        t = np.array([0., .1, .1, .2, .6])
        x = np.arange(15).reshape(5, 3)
        valid = np.array([True, True, False, True, True])
        u, y, good = merge(t, x, valid)
        np.testing.assert_array_equal(good, [True, False, True, True])
        np.testing.assert_allclose(y[1], (x[1]+x[2])/2)
        self.assertFalse(supported(u, good, 0., .2))
        self.assertFalse(supported(u, np.ones(4, bool), .2, .6))

    def test_c5_does_not_accept_30fps_or_overcap_even_if_flag_true(self):
        m = dict(kind='c5', fps_requested=10, mux_fps=10, bitrate_limit_kbps=500,
                 bitrate_pass=True, actual={'whole_file_kbps': 499.9})
        self.assertTrue(classify(m))
        m['mux_fps'] = 30
        self.assertFalse(classify(m))
        m['mux_fps'] = 10
        m['actual']['whole_file_kbps'] = 500.01
        self.assertFalse(classify(m))


if __name__ == '__main__':
    unittest.main(verbosity=2)
