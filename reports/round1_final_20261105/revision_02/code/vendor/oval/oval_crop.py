"""Dataset-agnostic T8 frontend for both preparation and inference."""
import numpy as np
import cv2
from oval_frontend import OvalFrontend,oval_mask,OVAL

def crop_from_points(image,points):
    mask=oval_mask(image.shape[:2],points)
    empty=np.zeros((36,36,3),np.uint8);zero=np.zeros(4,np.int32)
    if mask.sum()<16:return False,empty,zero
    pp=np.rint(points).astype(np.int32);h,w=image.shape[:2]
    x0,y0=np.maximum(pp.min(0),0);x1,y1=np.minimum(pp.max(0)+1,[w,h])
    if x1<=x0 or y1<=y0:return False,empty,zero
    region=image[y0:y1,x0:x1].copy();region[mask[y0:y1,x0:x1]==0]=0
    crop=cv2.resize(region,(36,36),interpolation=cv2.INTER_AREA)
    return True,crop,np.array([x0,y0,x1,y1],np.int32)

class OvalCrop:
    def __init__(self):self.frontend=OvalFrontend()
    def step(self,image):
        _,points=self.frontend.step(image)
        valid,crop,box=crop_from_points(image,points)
        return valid,crop,box,points
    def close(self):self.frontend.close()
