"""Same large-oval vertex set as T7b; detect on the current compressed video."""
import numpy as np
import cv2

OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]

def oval_mask(shape,points):
    mask=np.zeros(shape,np.uint8);points=np.asarray(points)
    if points.shape!=(36,2) or not np.isfinite(points).all():return mask
    hull=cv2.convexHull(np.rint(points).astype(np.int32))
    cv2.fillPoly(mask,[hull],1)
    return mask

class OvalFrontend:
    def __init__(self):
        import mediapipe as mp
        self.mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=False,min_detection_confidence=.5,min_tracking_confidence=.5)
    def step(self,image):
        h,w=image.shape[:2];r=self.mesh.process(image)
        points=np.full((36,2),np.nan,np.float32)
        if r.multi_face_landmarks:
            lm=r.multi_face_landmarks[0].landmark
            points=np.asarray([(lm[i].x*w,lm[i].y*h) for i in OVAL],np.float32)
        return oval_mask((h,w),points),points
    def close(self):self.mesh.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def verify_control(image,mask,baseline,weighted,stats):
    if np.count_nonzero(mask)<16:
        assert np.isnan(baseline).all() and np.isnan(weighted).all()
        return 0.
    independent=np.asarray(cv2.mean(image,mask=mask)[:3]);err=float(np.max(abs(independent-baseline)))
    assert err<1e-9,'Mean RGB differs from independent masked accumulator'
    if not stats['gate']:np.testing.assert_allclose(baseline,weighted,rtol=0,atol=1e-9)
    return err
