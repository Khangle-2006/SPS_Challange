"""Implementation checks on synthetic signals/video; no dataset acceptance audit."""
import tempfile
from pathlib import Path
from fractions import Fraction
import numpy as np
import av
from mcd_e0_metrics import analyze_windows, summarize, ARMS, sample_grid
from mcd_e0_runner import cross_means, native_frames, encode
from mcd_lag_runner import regional_means
from mcd_development_gate_v3 import sha

t=np.arange(606)/30
s=np.sin(2*np.pi*1.2*t)
rgb=np.stack([100+.1*s,100+s,100+.2*s],axis=1)
trace=np.broadcast_to(rgb[:,None,None,:],(len(t),4,3,3)).copy()
windows=analyze_windows(t,trace,s,np.ones(len(t),bool))
assert len(windows)==2
assert all(abs(w['reference_hr']-72)<1 and all(abs(w['predictions'][a]-72)<1 for a in ARMS) for w in windows)
trace[450,3]=np.nan
windows=analyze_windows(t,trace,s,np.ones(len(t),bool))
assert windows[1]['predictions']['P1R1'] is None
assert windows[1]['predictions']['P0R0'] is not None
summary=summarize([dict(video='synthetic',subject='s',step='before',windows=windows,candidate_windows=2)],expected=1)
assert summary['matched_windows']==1 and summary['reference_windows']==2
assert summary['per_cell_available']['P0R0']['windows']==2
assert summary['per_cell_available']['P1R1']['windows']==1
assert abs(summary['paired_effects']['diagonal_total']['mean_bpm'])<1e-8
_,ok=sample_grid(np.array([0.,.3]),np.ones(2),np.ones(2,bool),np.array([.15]))
assert not ok[0]

rng=np.random.default_rng(23)
images=[rng.integers(0,255,(64,64,3),dtype=np.uint8) for _ in range(2)]
lm=[rng.uniform(10,40,(468,2)).astype(np.float32),rng.uniform(20,55,(468,2)).astype(np.float32)]
cross=cross_means(images,lm)
for p in range(2):
    for r in range(2):
        np.testing.assert_equal(cross[p*2+r],regional_means(images[p],lm[r])[0])
missing=cross_means(images,[lm[0],None])
assert np.isnan(missing[1]).all() and np.isnan(missing[3]).all()
assert np.isfinite(missing[0]).all() and np.isfinite(missing[2]).all()

with tempfile.TemporaryDirectory() as td:
    folder=Path(td);source=folder/'synthetic.avi'
    with av.open(str(source),'w') as c:
        st=c.add_stream('ffv1',rate=30);st.width=64;st.height=64;st.pix_fmt='bgr0'
        for color in [(220,20,20),(20,220,20),(20,20,220),(200,200,20)]:
            frame=av.VideoFrame.from_ndarray(np.full((64,64,3),color,dtype=np.uint8),format='rgb24')
            for pkt in st.encode(frame):c.mux(pkt)
        for pkt in st.encode():c.mux(pkt)
    with av.open(str(source)) as c:
        pts=[str(f.pts*f.time_base) for f in c.decode(video=0)]
    gate={'gate':{'candidate_prefix_frames':3},'container_pts_rational':pts,'source_sha256':sha(source)}
    frames=list(native_frames(source,gate));assert len(frames)==3
    encoded,receipt=encode(source,gate,folder)
    with av.open(str(encoded)) as c:
        decoded=[f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
    assert len(decoded)==3 and receipt['frames']==3
    assert [int(f.mean(axis=(0,1)).argmax()) for f in decoded]==[0,1,2]
    assert all(f.shape==(64,64,3) for f in decoded)
print('PASS E0: known HR, missing-window denominators, four ROI cells, frame count/order, no tail inclusion')
