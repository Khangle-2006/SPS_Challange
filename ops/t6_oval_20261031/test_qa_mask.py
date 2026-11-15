import unittest
import numpy as np
from qa_mask import QALite


class MaskTests(unittest.TestCase):
    def test_first_and_all_stale_frames_preserve_unweighted_color(self):
        rng = np.random.default_rng(20261031)
        image = rng.integers(10, 200, size=(32, 48, 3), dtype=np.uint8)
        mask = np.ones(image.shape[:2], bool)
        q = QALite()
        b, y, s = q.step(image, [mask])
        np.testing.assert_array_equal(b, y)
        self.assertFalse(s['gate'])
        b, y, s = q.step(image, [mask])
        np.testing.assert_allclose(b, y, atol=1e-12)
        self.assertTrue(s['gate'])
        self.assertEqual(s['unchanged_fraction'], 1.)
        self.assertEqual(s['mean_roi_weight'], .25)

    def test_gate_is_strict_and_uniform_update_is_unchanged(self):
        image = np.full((32, 32, 3), 100, np.uint8)
        mask = np.ones((32, 32), bool)
        for stale in (False, True):
            q = QALite()
            q.step(image, [mask])
            update = image+5
            if stale:
                update[:16, :16] = image[:16, :16]
            b, y, s = q.step(update, [mask])
            self.assertFalse(s['gate'])  # Exactly25% does not exceed threshold.
            np.testing.assert_allclose(b, y, atol=1e-12)

    def test_only_current_roi_area_opens_gate(self):
        image = np.full((32, 64, 3), 100, np.uint8)
        update = image.copy()
        update[:, :16] += 5
        roi = np.zeros((32, 64), bool)
        roi[:, :16] = True
        q = QALite()
        q.step(image, [roi])
        _, _, s = q.step(update, [roi])
        self.assertFalse(s['gate'])  # Static background outsideface is irrelevant.
        self.assertEqual(s['unchanged_fraction'], 0.)

    def test_refresh_weights_favor_changed_block_without_uint8_wrap(self):
        image = np.full((32, 32, 3), 100, np.uint8)
        update = image.copy()
        update[:16, :16] = 90
        q = QALite()
        mask = np.ones((32, 32), bool)
        q.step(image, [mask])
        b, y, s = q.step(update, [mask])
        self.assertTrue(s['gate'])
        self.assertEqual(s['unchanged_fraction'], .75)
        self.assertTrue(np.all(y < b))
        self.assertTrue(np.all(y >= 90))
        np.testing.assert_allclose(y, (90+3*.25*100)/(1+3*.25))

    def test_incomplete_edge_blocks_are_not_classified(self):
        image = np.full((19, 19, 3), 100, np.uint8)
        mask = np.zeros((19, 19), bool)
        mask[16:, :] = True
        q = QALite()
        q.step(image, [mask])
        b, y, s = q.step(image, [mask])
        self.assertEqual(s['full_block_roi_pixels'], 0)
        self.assertFalse(s['gate'])
        np.testing.assert_array_equal(b, y)

    def test_missing_roi_does_not_reuse_older_frame(self):
        image = np.full((32, 32, 3), 100, np.uint8)
        empty, full = np.zeros((32, 32), bool), np.ones((32, 32), bool)
        q = QALite()
        q.step(image, [full])
        b, y, _ = q.step(image+10, [empty])
        self.assertTrue(np.isnan(b).all() and np.isnan(y).all())
        _, _, s = q.step(image+10, [full])
        self.assertEqual(s['unchanged_fraction'], 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
