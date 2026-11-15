import unittest
import numpy as np
from ubfc1_diagnostics import mean_zoh, longest_run, baseline_metrics


class DiagnosticTests(unittest.TestCase):
    def test_time_weighted_mean_and_boundary(self):
        # Irregular samples: arithmetic averaging would give the wrong answer.
        self.assertAlmostEqual(mean_zoh([0, 1, 4], [90, 99, 100], 0, 4), 96.75)
        self.assertAlmostEqual(mean_zoh([0, 1, 4], [90, 99, 100], .5, 1), 94.5)
        with self.assertRaises(ValueError):
            mean_zoh([0, 1, 4], [90, 99, 100], -.1, 1)
        with self.assertRaises(ValueError):
            mean_zoh([0, 1, 4], [90, 99, 100], 3, 2)

    def test_boundary_runs(self):
        self.assertEqual(longest_run(np.array([True, True, False, True])), 2)
        self.assertEqual(longest_run(np.zeros(4, bool)), 0)
        self.assertEqual(longest_run(np.ones(4, bool)), 4)

    def test_recording_macro_is_not_pooled(self):
        result = baseline_metrics([96, 96, 96, 100], 96, ["a", "a", "a", "b"])
        self.assertEqual(result["mae_percentage_points"], 1)
        self.assertEqual(result["recording_macro_mae_percentage_points"], 2)


if __name__ == "__main__":
    unittest.main()
