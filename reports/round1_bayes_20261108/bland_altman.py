"""Descriptive per-person Bland–Altman panels, with the exact plotted numerical table."""
from pathlib import Path
from collections import defaultdict
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
rows=list(csv.DictReader((P/'VIDEO_PREDICTIONS.csv').open()))
matched=list(csv.DictReader((P/'BTC_COVERAGE_BY_VIDEO.csv').open()))
plotted=[];stats=[]
fig,axes=plt.subplots(2,2,figsize=(11,8),facecolor='white')
for i,cond in enumerate(['native','c5']):
    keys={r['recording'] for r in matched if r['condition']==cond and r['included_all_methods']=='True'}
    for j,method in enumerate(['constant','Bayes_none']):
        people=defaultdict(list)
        for r in rows:
            if r['condition']==cond and r['method']==method and r['recording'] in keys:people[r['person']].append(r)
        a=[]
        for person,rr in sorted(people.items()):
            truth=np.mean([float(r['truth']) for r in rr]);pred=np.mean([float(r['prediction']) for r in rr])
            r=dict(condition=cond,method=method,person=person,videos=len(rr),reference_mean_bpm=truth,prediction_mean_bpm=pred,pair_mean_bpm=(truth+pred)/2,error_mean_bpm=pred-truth);plotted.append(r);a.append(r)
        x=np.array([r['pair_mean_bpm'] for r in a]);y=np.array([r['error_mean_bpm'] for r in a]);bias=y.mean();sd=y.std(ddof=1);lo,hi=bias+np.array([-1.96,1.96])*sd
        stats.append(dict(condition=cond,method=method,people=len(a),bias=float(bias),loa_low=float(lo),loa_high=float(hi)))
        ax=axes[i,j];ax.set_facecolor('white');ax.scatter(x,y,s=20,c='#176b91',alpha=.75,edgecolors='none')
        ax.axhline(bias,color='#111111',lw=1.3,label=f'Bias {bias:.2f} bpm');ax.axhline(lo,color='#b33b20',ls='--',lw=1.2,label=f'LoA [{lo:.2f}, {hi:.2f}]');ax.axhline(hi,color='#b33b20',ls='--',lw=1.2)
        ax.set_title(f"{cond.upper()} / {'Constant' if method=='constant' else 'Bayes'} / n={len(a)} people",fontsize=11)
        ax.set_xlabel('Mean of prediction and reference (bpm)');ax.set_ylabel('Prediction - reference (bpm)');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower left')
fig.suptitle('BTC development — person-level Bland–Altman\nSame videos across methods; descriptive, not clinical agreement limits',fontsize=13)
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(P/'BTC_BLAND_ALTMAN.png',dpi=180,facecolor='white');plt.close(fig)
with (P/'BTC_BLAND_ALTMAN_PEOPLE.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(plotted[0]));w.writeheader();w.writerows(plotted)
(P/'BTC_BLAND_ALTMAN_STATS.json').write_text(json.dumps(stats,indent=2))
print('Bland–Altman figure and numerical tables written')
