"""Render measured cached frames/PSD locally; no generated or altered data."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent/'loader'
summary=json.loads((HERE/'SUMMARY.json').read_text())
for file in HERE.glob('*_frames.npz'):
    name=file.stem.removesuffix('_frames')
    with np.load(file) as a:
        fig,axes=plt.subplots(2,4,figsize=(10,6),facecolor='white')
        for ax,rgb,i,t,valid in zip(axes.flat,a['rgb'],a['indices'],a['t'],a['valid']):
            ax.imshow(rgb,interpolation='nearest');ax.set_title(f'frame {i} | {t:.3f}s\nvalid={bool(valid)}',fontsize=10);ax.axis('off')
        fig.suptitle(name+' / actual cached oval36',fontsize=13);fig.tight_layout();fig.savefig(HERE/(name+'_montage.png'),dpi=150);plt.close(fig)
for q in summary['qc']:
    rec,cond=q['recording'],q['condition'];name=rec+'_'+cond
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),facecolor='white')
    for ax,start in zip(axes,[0,10,20]):
        s=next((s for s in summary['spectra'] if s['recording']==rec and s['condition']==cond and s['start_s']==start),None)
        if s is None:ax.set_title('Invalid support');continue
        with (HERE/f'{name}_{start}_PSD.csv').open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
        ax.plot([float(r['bpm']) for r in rr],[float(r['power']) for r in rr],color='#174a7e')
        ax.axvline(s['PR'],color='#c3312a',ls='--',label=f'PR label {s["PR"]:g}');ax.axvline(s['POS_HR'],color='#277b34',ls=':',label=f'POS {s["POS_HR"]:.2f}')
        ax.set_xlim(30,200);ax.set_xlabel('bpm');ax.set_title(f'{start}-{start+10}s');ax.legend(fontsize=7)
    axes[0].set_ylabel('PSD after canonical detrend + bandpass');fig.suptitle(name+' / POS; PR timing unknown');fig.tight_layout();fig.savefig(HERE/(name+'_PSD.png'),dpi=150);plt.close(fig)
print('Rendered six montages and six PSD panels')
