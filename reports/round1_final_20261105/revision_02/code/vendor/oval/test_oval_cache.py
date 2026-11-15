import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile,json
import numpy as np
import cv2
from oval_crop import crop_from_points,OVAL
from oval_frontend import oval_mask

class OvalCacheTests(unittest.TestCase):
    def test_fixed_anatomy_and_bbox(self):
        self.assertEqual(len(OVAL),36)
        t=np.arange(36)*2*np.pi/36;points=np.c_[50+30*np.cos(t),40+20*np.sin(t)]
        image=np.full((80,100,3),100,np.uint8)
        ok,crop,box=crop_from_points(image,points)
        self.assertTrue(ok);self.assertEqual(box.tolist(),[20,20,81,61]);self.assertEqual(crop.shape,(36,36,3))
        self.assertEqual(crop.dtype,np.uint8);self.assertEqual(int(crop[0,0].sum()),0)
    def test_background_outside_oval_cannot_enter(self):
        t=np.arange(36)*2*np.pi/36;points=np.c_[40+25*np.cos(t),40+30*np.sin(t)]
        image=np.full((80,80,3),101,np.uint8);other=image.copy();other[oval_mask((80,80),points)==0]=255
        np.testing.assert_array_equal(crop_from_points(image,points)[1],crop_from_points(other,points)[1])
    def test_missing_face_no_haar_fallback(self):
        ok,crop,box=crop_from_points(np.ones((80,80,3),np.uint8),np.full((36,2),np.nan))
        self.assertFalse(ok);self.assertFalse(crop.any());self.assertFalse(box.any())
    def test_prepare_keeps_separate_clock_and_duplicates(self):
        from data_adapters import Recording,Label
        from prepare_recording import prepare
        from oval_crop import OvalCrop
        ts=np.array([0.,.1,.1,.2]);pt=np.arange(31)/100
        images=[np.full((60,60,3),i*10,np.uint8) for i in range(4)]
        r=Recording(images,ts,{'ppg':Label(np.sin(pt*8),pt,'a.u.')},'toy:1','toy',{'operational_alignment_accepted':True})
        def step(self,image):return True,np.full((36,36,3),image[0,0],np.uint8),np.array([0,0,60,60]),np.zeros((36,2),np.float32)
        with tempfile.TemporaryDirectory() as td,patch.object(OvalCrop,'__init__',lambda self:None),patch.object(OvalCrop,'step',step),patch.object(OvalCrop,'close',lambda self:None):
            folder=Path(td)/'crop';prepare(r,folder);z=np.load(folder/'meta.npz')
            np.testing.assert_array_equal(z['ppg_t'],pt);np.testing.assert_array_equal(z['original_t'],ts)
            self.assertEqual(z['oval_points'].shape,(4,36,2));self.assertEqual(np.load(folder/'rgb.npy')[1,0,0,0],15)

if __name__=='__main__':unittest.main()
