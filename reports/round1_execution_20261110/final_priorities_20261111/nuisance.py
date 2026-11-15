"""Preregistered, label-free schedule and face-motion regression; diagnostic only."""
import numpy as np
def schedule_design(types,boxes=None):
    types=np.asarray(types);n=len(types);age=[];count=0
    for t in types:
        count=count+1 if t=='B' else 0;age.append(count)
    age=np.asarray(age)
    columns=[(types==s).astype(float) for s in sorted(set(types))]
    columns += [(age==i).astype(float) for i in sorted(set(age))]
    if boxes is not None:
        boxes=np.asarray(boxes,float);assert boxes.shape==(n,4)
        # Changes in tracked face rectangle, independent of RGB waveform or labels.
        scale=np.maximum(np.nanmedian(abs(boxes),axis=0),1)
        motion=np.diff(boxes/scale,axis=0,prepend=(boxes/scale)[:1])
        columns.extend(motion[:,i] for i in range(4))
    X=np.stack(columns,axis=1);X-=np.mean(X,axis=0)
    return X
def regress(rgb,types,boxes=None,valid=None):
    rgb=np.asarray(rgb,float);X=schedule_design(types,boxes)
    good=np.isfinite(rgb).all(1)&np.isfinite(X).all(1)
    if valid is not None:good&=np.asarray(valid,bool)
    assert len(rgb)==len(X) and good.sum()>X.shape[1]+2,'insufficient regression support'
    mean=rgb[good].mean(0);center=X[good].mean(0);X=X-center
    coef=np.linalg.lstsq(X[good],rgb[good]-mean,rcond=None)[0]
    cleaned=rgb-X@coef
    return cleaned,dict(valid_fit_frames=int(good.sum()),features=X.shape[1],rank=int(np.linalg.matrix_rank(X[good])),fit='within-video unsupervised, no PR used',motion_included=boxes is not None)
def omit(rgb):
    X=np.asarray(rgb,float).T;Q,_=np.linalg.qr(X);v=Q[:,0]
    return ((np.eye(3)-np.outer(v,v))@X)[1]
