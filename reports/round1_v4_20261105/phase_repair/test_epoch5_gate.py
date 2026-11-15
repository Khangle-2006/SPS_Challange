import math
import unittest
from pilot_gate import check_epoch5


class PilotLossGate(unittest.TestCase):
    def history(self, losses):
        return [dict(epoch=i, train_loss=value) for i,value in enumerate(losses,1)]

    def test_inclusive_ten_percent_and_no_best_epoch_selection(self):
        self.assertTrue(check_epoch5(self.history([1,.5,.5,.5,.9]))["passed"])
        self.assertFalse(check_epoch5(self.history([1,.5,.5,.5,math.nextafter(.9,1)]))["passed"])
        self.assertFalse(check_epoch5(self.history([1,.5,.5,.5,.99]))["passed"])

    def test_reject_incomplete_or_invalid_history(self):
        for losses in ([1,1,1,1], [0,0,0,0,0], [1,1,float("nan"),1,.1]):
            with self.assertRaises(ValueError):
                check_epoch5(self.history(losses))

if __name__=="__main__":
    unittest.main()
