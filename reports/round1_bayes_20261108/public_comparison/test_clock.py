import unittest
import numpy as np
from clock_contract import align,window
class ClockTests(unittest.TestCase):
    def test_duplicate_and_declared_tail(self):
        t=np.array([0.,.1,.1,.2]);x=np.arange(5*12).reshape(5,4,3).astype(float)
        q,y=align(t,[0,.1,.2],x,1)
        np.testing.assert_array_equal(y[1],(x[1]+x[2])/2)
        with self.assertRaises(AssertionError):align(t,[0,.1,.2],x,0)
    def test_no_extrapolation_or_gap_bridge(self):
        t=np.arange(0,11,.1);x=np.ones((len(t),4,3));q,y=window(t,x,.05,10)
        self.assertEqual(q[0],.05);self.assertEqual(q[-1],10.05)
        with self.assertRaises(AssertionError):window(t,x,-.1)
        t=np.array([0,.4,10.4]);x=np.ones((3,4,3))
        with self.assertRaises(AssertionError):window(t,x,.1)
if __name__=='__main__':unittest.main()
