import unittest,json
import numpy as np
from infer_dev import image_clip,decode_wave,decode_model,candidate_windows,classical_functions
from t8_t1_loader import clip_arrays
from t8_evaluate import decode_difference

class InferenceContract(unittest.TestCase):
    def test_image_parity_independent_target(self):
        rng=np.random.default_rng(0);t=np.arange(1001)/29.99
        rgb=rng.integers(0,256,(len(t),36,36,3),dtype=np.uint8);v=np.ones(len(t),bool)
        pt=np.arange(4000)/100;p=np.sin(2*np.pi*1.2*pt)
        x,*_=clip_arrays(rgb,t,v,p,pt,np.ones(len(pt),bool),1.)
        np.testing.assert_array_equal(image_clip(rgb,t,v,1.,10),x)
        v[50]=False
        with self.assertRaises(ValueError):image_clip(rgb,t,v,1.,10)
    def test_decoder_exact10_and30_frequency(self):
        y=np.diff(np.sin(2*np.pi*1.2*np.arange(301)/30))
        self.assertEqual(decode_model(y),decode_difference(y))
        for n in [300,900]:self.assertLess(abs(decode_wave(np.sin(2*np.pi*1.2*np.arange(n)/30))-72),1)
        with self.assertRaises(ValueError):decode_model(np.ones(900))
    def test_windows_no_padding(self):
        self.assertEqual(candidate_windows(9.99),[])
        w=candidate_windows(30.)
        self.assertEqual(sum(s=='10s' for s,_,_ in w),3)
        self.assertEqual(sum(s=='10s_stride5' for s,_,_ in w),5)
        self.assertEqual(sum(s=='30s' for s,_,_ in w),1)
    def test_classical_frozen_functions(self):
        ns,_=classical_functions();t=np.arange(900)/30
        rgb=np.array([120.,90.,70.])[None,:]*(1+np.sin(2*np.pi*1.2*t)[:,None]*np.array([.003,.01,.002])+.005*np.sin(2*np.pi*.2*t)[:,None])
        for name in ['pos','chrom']:
            for n in [300,900]:self.assertLess(abs(decode_wave(ns[name](rgb[:n],30))-72),1)
if __name__=='__main__':unittest.main()
