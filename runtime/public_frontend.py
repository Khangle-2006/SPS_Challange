"""Decode raw public video, redetect landmarks, retain acquisition clock when supplied; never load cached ROIs or labels."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from common import save, sha
from primitives import BTCVideo, Mesh, fast_crop, frames_of, tpl_make, tpl_find


def main(out):
    req = json.loads((out/'request.json').read_text()); source = Path(req['video'])
    source_sha = sha(source)
    if req['clock']:
        with np.load(req['clock'], allow_pickle=False) as z:
            if set(z.files) != {'t', 'source_valid', 'window_start', 'window_eligible', 'source_sha256'}: raise ValueError('Unexpected acquisition-clock fields; no labels/ROIs permitted')
            if str(z['source_sha256']) != source_sha: raise ValueError('Clock/source hash mismatch')
            t=z['t'].copy();sv=z['source_valid'].copy();starts=z['window_start'].copy();eligible=z['window_eligible'].copy()
        frames = frames_of(dict(dataset=req['dataset'], source=str(source)), len(t))
        clock_kind = 'source-bound acquisition clock'
    else:
        video=BTCVideo(source);t=video.t;frames=iter(video);sv=np.ones(len(np.unique(t)),bool)
        starts=np.arange(t[0], t[-1]-10+1e-8, 10.);eligible=np.ones(len(starts),bool);clock_kind='display PTS'
    if len(t)<2 or not np.isfinite(t).all() or (np.diff(t)<0).any(): raise ValueError('Invalid acquisition clock')
    n=len(t)
    if n*(72*72*3+36*36*3)>500_000_000: raise ValueError('Single-run crop allocation exceeds 500 MB')
    box=np.lib.format.open_memmap(out/'crop72.npy',mode='w+',dtype=np.uint8,shape=(n,72,72,3))
    oval=np.lib.format.open_memmap(out/'oval36.npy',mode='w+',dtype=np.uint8,shape=(n,36,36,3))
    valid=np.zeros(n,bool);boxes=np.zeros((n,4),np.int32);points=np.zeros((n,36,2),np.float32)
    mesh=Mesh();last=None;template=None;previous=None;static=True;count=0;held=0
    try:
        for i,frame in enumerate(frames):
            if i>=n: raise ValueError('More frames than source clock')
            if previous is not None and not np.array_equal(previous,frame): static=False
            previous=frame.copy();lm=mesh.step(frame)
            if lm is not None: last=lm;template=tpl_make(frame,lm)
            elif template is not None:
                delta=tpl_find(frame,template)
                if delta is not None: lm=last+delta;held+=1
            good,crop,b=fast_crop(frame,lm);valid[i]=good;oval[i]=crop;boxes[i]=b;box[i]=0
            if lm is not None: points[i]=lm
            x0,y0,x1,y1=map(int,b)
            if good and x1-x0>=8 and y1-y0>=8: box[i]=cv2.resize(frame[y0:y1,x0:x1],(72,72),interpolation=cv2.INTER_AREA)
            count+=1
            if count%600==0: print('frames',count,n,flush=True)
    finally: mesh.close()
    if count!=n: raise ValueError(f'Decoded {count} frames, expected {n}')
    box.flush();oval.flush()
    if len(sv)!=len(np.unique(t)): raise ValueError('Source-valid clock mismatch')
    np.savez_compressed(out/'trace.npz',t=t,valid=valid,source_valid=sv,window_start=starts,window_eligible=eligible,boxes=boxes,oval_points=points)
    save(out/'frontend.json',dict(frames=count,static_input=static,source_sha256=source_sha,clock=clock_kind,valid_frames=int(valid.sum()),template_held_frames=held,frontend='full-frame FaceMesh with original template fallback',size=list(previous.shape[:2])))


if __name__=='__main__': main(Path(sys.argv[1]))
