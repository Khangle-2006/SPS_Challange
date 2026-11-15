import unittest
import tempfile
from pathlib import Path
import numpy as np
from video_landmark_frontend import (canonical_template,shrink_polygon,polygons_from_landmarks,
 rasterize_means,least_squares_affine,AlignedWriter,ROI_INDICES,FACE_OVAL)


class LandmarkFrontendTests(unittest.TestCase):
    def test_shrink_keeps_centroid(self):
        p=np.array([[0,0],[10,0],[10,10],[0,10]],float)
        q=shrink_polygon(p)
        np.testing.assert_allclose(q.mean(0),p.mean(0))
        np.testing.assert_allclose(q-p.mean(0),.88*(p-p.mean(0)))

    def test_canonical_four_regions_nonempty_and_rgb(self):
        template=canonical_template()*6
        ps,small=polygons_from_landmarks(template);self.assertFalse(small)
        rgb=np.zeros((450,450,3),np.uint8);rgb[:]=[31,83,191]
        means,valid,areas=rasterize_means(rgb,ps)
        self.assertTrue(valid.all());self.assertTrue((areas>=16).all())
        np.testing.assert_allclose(means,np.tile([31,83,191],(4,1)))

    def test_affine_recovers_known_transform(self):
        t=canonical_template();source=t@np.array([[2,.1],[.2,1.4]])+np.array([55,21])
        m=least_squares_affine(source,t)
        np.testing.assert_allclose(np.c_[source,np.ones(468)]@m.T,t,atol=1e-4)

    def test_missing_roi_not_filled(self):
        m,v,a=rasterize_means(np.ones((40,40,3),np.uint8),[None]*4)
        self.assertFalse(v.any());self.assertTrue(np.isnan(m).all());self.assertFalse(a.any())

    def test_small_face_only_slot0(self):
        ps,small=polygons_from_landmarks(canonical_template()*.4)
        self.assertTrue(small);self.assertEqual(len(ps[0]),len(FACE_OVAL));self.assertTrue(all(p is None for p in ps[1:]))

    def test_cropped_mask_equals_full_mask(self):
        import cv2
        rng=np.random.default_rng(20261030);rgb=rng.integers(0,256,(80,100,3),dtype=np.uint8)
        p=np.array([[-4,12],[38,4],[57,51],[1,55]],float)
        mean,v,_=rasterize_means(rgb,[p,None,None,None]);mask=np.zeros((80,100),np.uint8)
        cv2.fillPoly(mask,[p.astype(np.int32)],255)
        np.testing.assert_allclose(mean[0],cv2.mean(rgb,mask=mask)[:3],atol=1e-5)

    def test_ffv1_rgb_roundtrip(self):
        import av
        rng=np.random.default_rng(20261030);frames=[rng.integers(0,256,(72,72,3),dtype=np.uint8) for _ in range(3)]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.avi';w=AlignedWriter(path,29.87)
            for f in frames:w.write(f)
            w.close()
            with av.open(str(path)) as c:decoded=[f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
            self.assertEqual(len(decoded),len(frames))
            for a,b in zip(frames,decoded):np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
