import numpy as np
from mcd_figure5_metrics import lag_metrics,drift_metrics
rng=np.random.default_rng(20261030)
for fs in [24,30]:
    ppg=rng.normal(size=fs*30)
    for shift in [-20,-7,0,5,20]:
        pos=np.roll(ppg,shift)
        r=lag_metrics(pos,ppg,fs)
        assert r['lag_frames']==shift,(fs,shift,r['lag_frames'])
        assert r['search_boundary']==(abs(shift)==20)
    inv=lag_metrics(-ppg,ppg,fs)
    assert inv['pearson_zero']<-.999 and inv['lag_frames']!=0 # maximize signed Pearson, not abs
    try:lag_metrics(np.zeros(fs*30),ppg,fs);raise AssertionError('constant accepted')
    except ValueError:pass
r=drift_metrics([15,45,75],[0,.05,.1]);assert abs(r['slope_ms_per_min']-100)<1e-10
print('PASS Figure5 kernels:24/30Hz,+/-lag,boundaries,signedPearson,constant rejection,drift')
