import unittest
import numpy as np
from pos_corrected import pos_wave, decode_wave, estimate_window, FRONTEND

class Contract(unittest.TestCase):
    def test_static_abstains(self):
        self.assertIsNone(decode_wave(pos_wave(np.tile([130.,100.,80.], (300,1))))['hr'])
    def test_known_frequency(self):
        t=np.arange(300)/30
        rgb=np.array([130.,100.,80.])[None,:]*(1+np.sin(2*np.pi*1.2*t)[:,None]*np.array([.001,.004,.002])[None,:])
        self.assertLess(abs(decode_wave(pos_wave(rgb))['hr']-72),.5)
    def test_wrong_frontend_rejected(self):
        with self.assertRaises(ValueError):estimate_window(np.arange(301)/30,np.ones((301,3)),np.ones(301,bool),0,frontend='tiny_skin_roi')
    def test_gap_rejected(self):
        t=np.r_[np.arange(150)/30,np.arange(160,301)/30]
        with self.assertRaises(ValueError):estimate_window(t,np.ones((len(t),3)),np.ones(len(t),bool),0,frontend=FRONTEND)
    def test_no_extrapolation(self):
        with self.assertRaises(ValueError):estimate_window(np.arange(300)/30,np.ones((300,3)),np.ones(300,bool),0,frontend=FRONTEND)

if __name__=='__main__':unittest.main()
