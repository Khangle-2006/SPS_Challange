import unittest
from finalize_dev import threshold_gate
class Gate(unittest.TestCase):
    def test_boundaries(self):
        self.assertTrue(threshold_gate(5,8))
        self.assertFalse(threshold_gate(4.9,6.9))
        self.assertTrue(threshold_gate(4.9,6.89))
if __name__=='__main__':unittest.main()
