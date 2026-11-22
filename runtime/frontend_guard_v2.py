"""22 Nov identity guard: retain full-frame native landmarks when they match T."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['__EGL_VENDOR_LIBRARY_DIRS']='/nonexistent'
import numpy as np
import cv2
from primitives import Mesh, Det, fast_crop, aid, tpl_make, tpl_find, region_points
from target_lock import TargetLock, Track, to_z, to_box, iou

class Frontend:
    def __init__(self, mode='track', lock_multiple=True, seed_box=None):
        if mode not in ('track','aidtrack','lock0'):raise ValueError('Unknown frontend mode')
        self.mode=mode;self.mesh=Mesh();self.det=Det() if lock_multiple or mode=='lock0' else None
        self.lock_mesh=None;self.region={};self.last_points=None;self.template=None
        self.seed_box=None if seed_box is None else np.asarray(seed_box,float)
        self.cfg=TargetLock(seed_box=seed_box,warmup_s=0.,min_hits=1)
        self.target=None;self.prev_time=None

    def _start(self,box,t):
        z=to_z(box);s=np.sqrt(max(z[2]*z[3],1.))
        self.target=Track(0,np.r_[z,0.,0.],np.diag([(self.cfg.r_meas*s)**2]*4+[(2*s)**2]*2),t)

    def step(self,rgb,timestamp):
        rgb=np.ascontiguousarray(rgb);h,w=rgb.shape[:2]
        image=aid(rgb) if self.mode=='aidtrack' else rgb;dh,dw=image.shape[:2]
        native=self.mesh.step(image)
        boxes=self.det.step(image) if self.det is not None else []
        mb=None if native is None or not np.isfinite(native).all() else np.r_[native.min(0),native.max(0)].astype(float)
        t=float(timestamp)
        if self.target is None:
            seed=self.seed_box if self.seed_box is not None else mb
            if seed is not None and (self.seed_box is not None or fast_crop(image,native)[0]):self._start(seed,t)
        elif self.prev_time is not None:
            self.target.predict(max(1e-3,t-self.prev_time),self.cfg.q_acc,self.cfg.q_size)
        self.prev_time=t
        predicted=None if self.target is None else to_box(self.target.x).copy()
        miou=None if mb is None or predicted is None else float(iou(mb,predicted))
        det_ious=[] if predicted is None else [float(iou(b,predicted)) for b in boxes]
        other=[(b,v) for b,v in zip(boxes,det_ious) if v<.3]
        overlap=any(v>.15 for _,v in other)
        points=None;branch='invalid_lost';measurement=None
        if mb is not None and miou is not None and miou>=.3:
            points=native;measurement=mb;branch='native'
        elif predicted is not None and det_ious and max(det_ious)>=.3:
            k=int(np.argmax(det_ious))
            if self.lock_mesh is None:self.lock_mesh=Mesh()
            points=region_points(image,boxes[k],self.lock_mesh,self.region)
            if points is not None:measurement=boxes[k];branch='switch'
        if measurement is not None:
            self.target.update(to_z(measurement),self.cfg.r_meas)
            self.target.t_update=t
        if points is None and not overlap and self.template is not None:
            delta=tpl_find(image,self.template)
            if delta is not None:points=self.last_points+delta;branch='held'
        if overlap:branch='invalid_overlap'
        # Rejected bystanders cannot become the template. Held coordinates retain
        # the original template origin, exactly as x09's fallback.
        if points is not None and branch in ('native','switch'):
            self.last_points=points;self.template=tpl_make(image,points)
        source_points=None if points is None else points*np.array([w/dw,h/dh],np.float32)
        good,oval,box=fast_crop(rgb,source_points)
        valid=bool(good) and branch in ('native','switch','held')
        if not good and branch!='invalid_overlap':branch='invalid_lost'
        face=np.zeros((72,72,3),np.uint8);x0,y0,x1,y1=map(int,box)
        if valid and x1-x0>=8 and y1-y0>=8:face=cv2.resize(rgb[y0:y1,x0:x1],(72,72),interpolation=cv2.INTER_AREA)
        tr=dict(n_det=len(boxes),branch=branch,iou_mesh_target=miou,
                predicted_box=None if predicted is None else predicted.tolist(),
                mesh_box=None if mb is None else mb.tolist(),det_boxes=[np.asarray(b).tolist() for b in boxes],
                det_iou_target=det_ious,image_size=[dw,dh])
        return dict(valid=valid,oval36=oval,crop72=face,box=box,
            blk=oval.reshape(6,6,6,6,3).mean((1,3),dtype=np.float32),
            fill=(oval.max(-1)>0).reshape(6,6,6,6).sum((1,3)).astype(np.uint8),
            rgb_mean=oval.astype(float).mean((0,1)),held=branch=='held',trace=tr)

    def close(self):
        self.mesh.close()
        if self.det is not None:self.det.close()
        if self.lock_mesh is not None:self.lock_mesh.close()
