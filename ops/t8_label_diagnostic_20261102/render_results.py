import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]/'reports/t8_label_diagnostic_20261102'
p=root/'signals';cycles=np.load(p/'cycle_templates.npz')
records=list(csv.DictReader((p/'record_correlations.csv').open()))
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ds,color in [('UBFC2','tab:blue'),('MCD','tab:orange')]:
    arr=cycles[ds];xx=np.linspace(0,1,101)
    axs[0].plot(xx,arr.mean(axis=0),label=f'{ds} ({len(arr)} recordings)',color=color)
    axs[0].fill_between(xx,*np.quantile(arr,[.25,.75],axis=0),alpha=.15,color=color)
    rs=[float(r['r_best_abs']) for r in records if r['dataset']==ds]
    axs[1].hist(rs,bins=np.linspace(-1,1,21),alpha=.5,label=ds,color=color)
axs[0].set(xlabel='Trough-to-trough phase',ylabel='PPG z-score',title='Raw sign retained; mean and recording IQR')
axs[1].set(xlabel='Signed Pearson r at max |r| within +/-5 frames',ylabel='Recordings',title='Native oval POS versus contact PPG')
for ax in axs:ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(p/'polarity_cycles.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4))
lp=root/'loss_probe'
for arm in ['ubfc_only','joint']:
    h=json.loads((lp/(arm+'_aggregate_history.json')).read_text())
    axs[0].plot([x['epoch'] for x in h],[x['train_mse'] for x in h],label=arm)
summ=json.loads((lp/'summary.json').read_text())['summaries']
for i,arm in enumerate(['ubfc_only','joint']):
    axs[1].bar(np.arange(2)+(i-.5)*.36,[next(x['mean_eval_mse'] for x in summ if x['arm']==arm and x['dataset']==ds) for ds in ['UBFC2','MCD']],width=.36,label=arm)
axs[1].set_xticks([0,1],['UBFC2','MCD'])
axs[0].set(xlabel='Epoch',ylabel='Aggregate training MSE',title='Historical mixed-batch loss (not dataset-specific)')
axs[1].set(ylabel='Final checkpoint eval-mode MSE',title='8 fixed native train-subject probes per dataset')
for ax in axs:ax.legend();ax.grid(axis='y',alpha=.2)
fig.suptitle('Fold0 seed2027; CPU-only diagnostic, no model updates')
fig.tight_layout();fig.savefig(lp/'loss_diagnostic.png',dpi=170)
