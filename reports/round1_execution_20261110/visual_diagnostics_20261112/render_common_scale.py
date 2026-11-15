"""Display-only common column scales; no inference or statistic changes."""
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;rows=json.loads((P/'SELECTION.json').read_text());groups=sorted({r['group'] for r in rows});arrays=[]
for g in groups:
    grad=[];occ=[];maps=[]
    for r in [r for r in rows if r['group']==g]:
        a=np.load(P/'model'/r['key']/'outputs.npz');h=a['gradient'];grad.append(h/h.sum());h=abs(a['occlusion_signed']);occ.append(h/h.sum());maps.append(np.nanmean(np.load(P/'signal'/r['key']/'signal.npz')['pulse_maps'],axis=0))
    arrays.append([np.mean(grad,axis=0),np.mean(occ,axis=0),np.nanmean(maps,axis=0)[0],np.nanmean(maps,axis=0)[1]])
limits=[max(a[k].max() for a in arrays) for k in range(2)]+[1,1]
fig,ax=plt.subplots(5,4,figsize=(13,15),layout='constrained')
for j,g in enumerate(groups):
    for k,a in enumerate(arrays[j]):
        im=ax[j,k].imshow(a,vmin=0,vmax=limits[k],cmap='viridis');fig.colorbar(im,ax=ax[j,k],shrink=.6)
        ax[j,k].set_title(g+' n=20: '+['gradient','occlusion','G PR-band ratio','POS PR-band ratio'][k]);ax[j,k].set_xticks([]);ax[j,k].set_yticks([])
fig.suptitle('Same color scale within each column; white = undefined / zero energy')
fig.savefig(P/'SPATIAL_SUMMARY_COMMON.png',dpi=140);plt.close(fig)
fig,ax=plt.subplots(5,2,figsize=(8,15),layout='constrained')
for j,g in enumerate(groups):
    for k in range(2):
        im=ax[j,k].imshow(arrays[j][k+2],vmin=0,vmax=1,cmap='viridis');fig.colorbar(im,ax=ax[j,k],shrink=.6)
        ax[j,k].set_title(g+' — '+['G','POS'][k]);ax[j,k].set_xticks([]);ax[j,k].set_yticks([])
fig.suptitle('Power within reference PR +/-3 bpm / 42–180 bpm power\nLabel-guided diagnostic; white = undefined')
fig.savefig(P/'PULSE_SUMMARY.png',dpi=140);plt.close(fig)
(P/'DISPLAY_SCALES.json').write_text(json.dumps(dict(column_max=limits,inference_changed=False),indent=2))
