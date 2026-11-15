"""Report frozen T7 outputs without changing their support or predictions."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['POS','CHROM','POS_ROI0','POS_ROI1','POS_ROI2']
def metrics(rows,name,reference='reference_hr'):
    good=[r for r in rows if r.get(name) is not None and r.get(reference) is not None]
    err=np.array([r[name]-r[reference] for r in good]);by=defaultdict(list)
    for r,e in zip(good,err):by[r['subject']].append(abs(e))
    if not len(err):return {'expected':len(rows),'valid':0}
    return dict(expected=len(rows),valid=len(good),subjects=len(by),window_MAE=float(abs(err).mean()),subject_macro_MAE=float(np.mean([np.mean(v) for v in by.values()])),RMSE=float(np.sqrt(np.mean(err**2))),bias=float(err.mean()),P95=float(np.quantile(abs(err),.95)),over20=int((abs(err)>20).sum()),over20_positive=int((err>20).sum()),over20_negative=int((err< -20).sum()),positive=int((err>0).sum()),negative=int((err<0).sum()),zero=int((err==0).sum()))

def main(root, raw_v2=False):
    data=json.loads((root/'results.json').read_text());rows=data['rows']
    suffix='_raw_v2' if raw_v2 else ''
    if raw_v2:
        corrected=json.loads((root/'raw_reference_v2.json').read_text())['rows']
        assert {(r['video'],r['index']) for r in rows}=={(r['video'],r['index']) for r in corrected}
        lookup={(r['video'],r['index']):r for r in corrected}
        for r in rows:
            cr=lookup[(r['video'],r['index'])]
            assert all(r[k]==cr[k] for k in METHODS+['reference_hr'])
            r['raw_reference_hr']=cr['raw_reference_hr']
    summary={'expected_windows':1860,'actual_windows':len(rows),'record_errors':data['failures'],'groups':{}}
    for group in ['all','before','after']:
        subset=[r for r in rows if group=='all' or r['step']==group]
        summary['groups'][group]={m:metrics(subset,m) for m in METHODS}
        summary['groups'][group]['raw_vs_sync_reference']=metrics(subset,'raw_reference_hr')
        summary['groups'][group]['POS_vs_raw']=metrics(subset,'POS','raw_reference_hr')
    subs=[]
    for sid in sorted({r['subject'] for r in rows}):
        ss=[r for r in rows if r['subject']==sid]
        subs.append({'subject':sid,**{m:metrics(ss,m) for m in METHODS}})
    summary['subjects']=subs
    (root/('summary'+suffix+'.json')).write_text(json.dumps(summary,indent=2))
    flat=[{k:v for k,v in r.items() if k!='failures'} for r in rows]
    with (root/('windows'+suffix+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for m,col in [('POS','tab:blue'),('CHROM','tab:orange')]:
        good=[r for r in rows if r.get(m) is not None]
        err=np.array([r[m]-r['reference_hr'] for r in good])
        ax[0,0].hist(err,bins=np.arange(-140,145,5),histtype='step',label=m,color=col)
        ax[0,1].scatter([r['reference_hr'] for r in good],[r[m] for r in good],s=5,alpha=.2,label=m,color=col)
    ax[0,0].set(xlabel='Prediction minus reference (bpm)',ylabel='10-second windows',title='Signed errors: same 1860 E0 windows');ax[0,0].legend()
    ax[0,1].plot([42,180],[42,180],'k--');ax[0,1].set(xlabel='Reference HR (bpm)',ylabel='Predicted HR (bpm)',title='Native/native FullHD');ax[0,1].legend()
    for j,m in enumerate(['POS','CHROM']):
        ax[1,0].bar(np.arange(2)+j*.35,[summary['groups'][g][m]['subject_macro_MAE'] for g in ['before','after']],width=.35,label=m)
    ax[1,0].set_xticks([.175,1.175],['Before','After']);ax[1,0].set(ylabel='Subject-macro MAE (bpm)',title='Rest versus post-exercise');ax[1,0].legend()
    raw=[r for r in rows if r['raw_reference_hr'] is not None]
    ax[1,1].hist([r['raw_reference_hr']-r['reference_hr'] for r in raw],bins=np.arange(-100,102,2))
    ax[1,1].set(xlabel='Raw PPG HR minus synchronized PPG HR (bpm)',ylabel='Windows',title=f'Reference diagnostic; {len(raw)}/1860 supported')
    fig.suptitle('T7: POS replay and CHROM comparison — development cohort, no offset correction')
    fig.savefig(root/('t7_diagnostics'+suffix+'.png'),dpi=150);plt.close(fig)
    print(json.dumps(summary['groups'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--raw-v2',action='store_true');a=p.parse_args();main(a.root,a.raw_v2)
