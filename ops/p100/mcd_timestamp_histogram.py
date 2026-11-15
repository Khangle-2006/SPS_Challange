"""Three fixed recordings; descriptive test of the proposed 15.625 ms lattice."""
import json,hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('reports/mcd_mapping_v3_20261030')
fig,axes=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
results=[]
for k,cam in enumerate(['FullHDwebcam','USBVideo','IriunWebcam']):
    p=OUT/f'1020_{cam}_after.txt'
    times=[datetime.fromisoformat(x.split(maxsplit=1)[1]) for x in p.read_text().splitlines() if x.strip()]
    dt=np.array([(b-a)//__import__('datetime').timedelta(microseconds=1) for a,b in zip(times,times[1:])])
    ms=dt/1000
    residual=ms-15.625*np.rint(ms/15.625)
    hist,bins=np.histogram(ms,bins=np.arange(0,max(100,np.ceil(ms.max())+1)+.25,.25))
    peaks=np.argsort(hist)[-6:][::-1]
    r={'camera':cam,'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
       'intervals':len(dt),'fps_elapsed':(len(times)-1)/(times[-1]-times[0]).total_seconds(),
       'interval_ms_quantiles':np.quantile(ms,[0,.01,.5,.95,.99,1]).tolist(),
       'top_0p25ms_bins':[{'low':float(bins[i]),'count':int(hist[i])} for i in peaks],
       'within_0p5ms_of_15p625_multiple':float(np.mean(abs(residual)<=.5)),
       'within_1ms_of_15p625_multiple':float(np.mean(abs(residual)<=1)),
       'lattice_abs_residual_ms_quantiles':np.quantile(abs(residual),[.5,.9,.95,.99]).tolist()}
    results.append(r)
    axes[k,0].hist(ms,bins=bins,color='#25758b')
    for v in np.arange(15.625,100,15.625):axes[k,0].axvline(v,color='#c04c36',ls='--',alpha=.6)
    axes[k,0].set(xlim=(0,max(80,float(ms.max())+1)),xlabel='Timestamp interval (ms)',ylabel='Count',title=f'{cam}: all {len(dt)} intervals')
    axes[k,1].hist(residual,bins=np.linspace(-7.8125,7.8125,126),color='#25758b')
    axes[k,1].axvline(0,color='#c04c36',ls='--')
    axes[k,1].set(xlabel='Residual to nearest 15.625 ms multiple (ms)',ylabel='Count',title=f'Within +/-1 ms: {r["within_1ms_of_15p625_multiple"]:.1%}')
fig.suptitle('MCD subject1020, after: acquisition-log intervals (no frame-loss inference)')
fig.savefig(OUT/'timestamp_histograms.png',dpi=170)
fig.savefig(OUT/'timestamp_histograms.pdf')
(OUT/'timestamp_histograms.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
