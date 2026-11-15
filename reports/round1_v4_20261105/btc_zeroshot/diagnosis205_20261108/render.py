from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent
rows=list(csv.DictReader((O/'TOP20.csv').open()))
for r in rows:
    name=r['recording'];a=list(csv.DictReader((O/(name+'_PSD.csv')).open()))
    fig,ax=plt.subplots(figsize=(8,4),facecolor='white')
    ax.plot([60*float(x['hz']) for x in a],[float(x['power']) for x in a],color='#174a7e')
    ax.axvline(float(r['truth']),color='#ba2430',ls='--',label=f"PR scalar: {float(r['truth']):.0f} bpm")
    ax.axvline(float(r['prediction']),color='#26702d',ls=':',label=f"POS window: {float(r['prediction']):.2f}")
    ax.set_xlim(30,190);ax.set_xlabel('Heart rate (bpm)');ax.set_ylabel('POS PSD, canonical filtering')
    ax.set_title(f"{name} | {float(r['start']):g}–{float(r['start'])+10:g}s\nVideo POS − PR = {float(r['video_error']):+.2f} bpm; SNR median = {float(r['video_snr_median']):.2f}")
    ax.legend();fig.tight_layout();fig.savefig(O/(name+'_PSD.png'),dpi=130);plt.close(fig)
for f in O.glob('*_montage.npz'):
    with np.load(f) as a:
        fig,axes=plt.subplots(2,4,figsize=(9,6),facecolor='white')
        for ax,rgb,t,v in zip(axes.flat,a['rgb'],a['t'],a['valid']):
            ax.imshow(rgb,interpolation='nearest');ax.axis('off');ax.set_title(f'{t:.2f}s | valid={bool(v)}',fontsize=9)
        fig.suptitle(f.stem+' | actual cached 36×36');fig.tight_layout();fig.savefig(f.with_suffix('.png'),dpi=110);plt.close(fig)
# Contact sheets keep all special-case videos reviewable, native and C5 paired.
from PIL import Image,ImageDraw
files=sorted(O.glob('*_montage.png'))
for page in range((len(files)+7)//8):
    sheet=Image.new('RGB',(1200,1600),'white')
    for k,f in enumerate(files[page*8:page*8+8]):
        im=Image.open(f);im.thumbnail((600,400));sheet.paste(im,((k%2)*600,(k//2)*400))
    sheet.save(O/f'CONTACT_{page+1}.jpg')
print('Rendered',len(rows),'spectra and',len(files),'montages')
