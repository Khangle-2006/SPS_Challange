import numpy as np,cv2
from mcd_lag_runner import regional_means,grid_signals,POLYS
rng=np.random.default_rng(17)
rgb=rng.integers(0,256,(80,90,3),dtype=np.uint8)
for _ in range(30):
    lm=rng.uniform([-20,-20],[110,100],(468,2)).astype(np.float32)
    got,areas=regional_means(rgb,lm)
    for j,ids in enumerate(POLYS.values()):
        mask=np.zeros(rgb.shape[:2],np.uint8);cv2.fillPoly(mask,[np.rint(lm[ids]).astype(np.int32)],1)
        assert int(mask.sum())==areas[j],('area',j)
        if areas[j]>=16:assert np.array_equal(got[j],rgb[mask.astype(bool)].mean(axis=0).astype(np.float32))
t=np.array([0,.05,.1,.15,.4,.45]);rg=np.tile(t[:,None],(1,3));p=t.copy();v=np.ones(6,bool)
g,a,b,ok=grid_signals(t,rg,p,v,20)
assert np.allclose(a[:,0],g) and np.allclose(b,g)
assert ok[3] and not ok[4] and not ok[5] and not ok[6] and not ok[7] and ok[8]
v[1]=False;g,a,b,ok=grid_signals(t,rg,p,v,40)
assert not ok[1] and not ok[2] and not ok[3] and ok[4]
try:grid_signals(np.array([0,0]),np.ones((2,3)),np.ones(2),np.ones(2,bool),30);raise AssertionError('duplicate accepted')
except ValueError:pass
print('PASS trace ROI pixels; interpolation endpoints, longgap, invalidface, nonmonotonicclock')
