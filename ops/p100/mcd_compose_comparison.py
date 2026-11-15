"""Local scientific figure: author Figure5 beside empirical dev60 fractions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--author-image',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--partial',action='store_true');a=p.parse_args()
rr=json.loads(a.results.read_text())
fig,ax=plt.subplots(1,2,figsize=(16,8),constrained_layout=True)
ax[0].imshow(plt.imread(a.author_image));ax[0].axis('off');ax[0].set_title('Published Figure 5: 600 subjects (counts)',fontsize=13)
for cam,label,color in [('IriunWebcam','cam1 / Iriun','#6666ff'),('FullHDwebcam','cam2 / FullHD','#ff7777'),('USBVideo','cam3 / USB','#bba188')]:
    r=[x for x in rr if x['camera']==cam];v=[x['fullrecord']['lag_frames'] for x in r if x.get('fullrecord') is not None]
    if v:ax[1].hist(v,bins=np.arange(-20.5,21.5),weights=np.ones(len(v))/len(v),histtype='stepfilled',alpha=.35,color=color,edgecolor=color,label=f'{label}: {len(v)}/{len(r)} evaluable')
ax[1].set(title='Dev60 subset: fractions within each camera'+(' (PARTIAL)' if a.partial else ''),xlabel='Lag (native-grid frames; POS delayed = positive)',ylabel='Fraction of evaluable recordings',xlim=(-21,21))
if any(x.get('fullrecord') is not None for x in rr):ax[1].legend(fontsize=10)
fig.suptitle('POS–PPG lag comparison | no offset correction | partial method reproduction',fontsize=16)
fig.supxlabel('Author: Egorov et al., arXiv:2508.17924v1, Fig. 5. ROI/filter details and cohort differ; histogram similarity does not prove synchronization.',fontsize=10)
a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=170);plt.close(fig)
