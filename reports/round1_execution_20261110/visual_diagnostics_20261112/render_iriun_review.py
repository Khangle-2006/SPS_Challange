"""Display-only repair: fixed waveform colors, legend and full 30s axis."""
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
P=Path(__file__).resolve().parent
sos=signal.butter(3,[.7,3],fs=30,btype='bandpass',output='sos')
for r in json.loads((P/'SELECTION.json').read_text()):
    if r['group']!='MCD Iriun':continue
    o=np.load(P/'model'/r['key']/'outputs.npz');p=np.load(P/'signal'/r['key']/'signal.npz')['pos']
    fig,ax=plt.subplots(3,1,figsize=(12,8),layout='constrained');labeled=False
    for j,start in enumerate(o['starts']):
        pred=signal.sosfiltfilt(sos,signal.detrend(np.cumsum(o['difference'][j])))
        for k,y in enumerate([pred,p[j]]):
            ff,tt,ss=signal.spectrogram(y,fs=30,window='hann',nperseg=150,noverlap=135,nfft=1024,mode='psd');band=(ff>=.7)&(ff<=3)
            z=10*np.log10(np.maximum(ss[band]/max(ss[band].max(),1e-30),1e-8));im=ax[k].pcolormesh(start+tt,ff[band]*60,z,vmin=-35,vmax=0,cmap='magma',shading='auto')
            if np.isfinite(o['pr'][j]):ax[k].plot([start,start+10],[o['pr'][j]]*2,'c--')
            ax[k].set_ylim(40,182);ax[k].set_ylabel('bpm')
        if np.isfinite(o['ppg'][j]).all():
            gt=signal.sosfiltfilt(sos,signal.detrend(o['ppg'][j]));t=start+np.arange(300)/30
            ax[2].plot(t,pred/max(pred.std(),1e-12),color='#1764ab',lw=.8,label='Model, cumsum + filter' if not labeled else None)
            ax[2].plot(t,gt/max(gt.std(),1e-12),color='#e47716',lw=.8,label='Canonical contact PPG' if not labeled else None);labeled=True
    for a in ax:
        a.set_xlim(0,30.2)
        for x in [10,20]:a.axvline(x,color='gray',lw=.5)
    for a,title in zip(ax,['Padding-fixed output; cyan = reference PR','POS; relative power scaled within each 10 s','Waveforms: no lag shift; blank = unavailable paired PPG']):a.set_title(title)
    ax[2].legend(loc='upper right');ax[2].set_ylabel('Standard deviations');ax[2].set_xlabel('Time (s)')
    fig.colorbar(im,ax=list(ax[:2]),label='dB relative to own-window maximum');fig.suptitle(r['recording']);fig.savefig(P/'signal'/r['key']/'temporal_review.png',dpi=140);plt.close(fig)
