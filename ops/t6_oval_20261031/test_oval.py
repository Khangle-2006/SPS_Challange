import unittest
import numpy as np
from oval_frontend import oval_mask,verify_control,OVAL
from qa_mask import QALite

class OvalTests(unittest.TestCase):
    def test_exact_vertices(self):
        self.assertEqual(len(OVAL),36);self.assertEqual(len(set(OVAL)),36)
    def test_missing_no_fallback(self):
        m=oval_mask((64,64),np.full((36,2),np.nan));self.assertEqual(m.sum(),0)
        image=np.zeros((64,64,3),np.uint8);b,w,s=QALite().step(image,[m]);verify_control(image,m,b[0],w[0],s)
    def test_clipped_hull_and_independent_mean(self):
        angle=np.arange(36)*2*np.pi/36
        points=np.c_[32+40*np.cos(angle),32+20*np.sin(angle)]
        mask=oval_mask((64,64),points);self.assertGreater(mask.sum(),1000)
        image=np.random.default_rng(23).integers(0,256,(64,64,3),dtype=np.uint8)
        b,w,s=QALite().step(image,[mask]);self.assertLess(verify_control(image,mask,b[0],w[0],s),1e-9)
    def test_active_gate_same_oval_control(self):
        theta=np.arange(36)*2*np.pi/36;mask=oval_mask((64,64),np.c_[32+28*np.cos(theta),32+28*np.sin(theta)])
        image=np.full((64,64,3),80,np.uint8);qa=QALite();qa.step(image,[mask]);image[:32,:32]=100
        b,w,s=qa.step(image,[mask]);self.assertTrue(s['gate']);verify_control(image,mask,b[0],w[0],s)
        self.assertFalse(np.allclose(b,w))

if __name__=='__main__':unittest.main()
