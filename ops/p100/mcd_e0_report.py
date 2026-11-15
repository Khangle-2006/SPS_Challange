"""Read-only scientific report from the fixed E0 outputs."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mcd_e0_metrics import ARMS,summarize

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    status=json.loads((a.output/'status.json').read_text())
    records=[json.loads(p.read_text()) for p in sorted(a.output.glob('*/result.json'))]
    summary=summarize(records)
    summary['complete']=status['state']=='complete' and len(records)==119 and len({r['video'] for r in records})==119
    summary['status']=status
    out=a.output/'report';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'records.csv').open('w',newline='') as f:
        fields=['video','subject','step','candidate_windows','analyzed_windows','error','container_average_bps']
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        for r in records:wr.writerow({**{k:r.get(k) for k in fields[:4]},'analyzed_windows':len(r.get('windows',[])),
                'error':r.get('error'),'container_average_bps':r.get('encode',{}).get('container_average_bps')})
    with (out/'windows.csv').open('w',newline='') as f:
        fields=['video','subject','step','index','start_s','reference_hr','reference_failure','matched']+ARMS+[a+'_failure' for a in ARMS]
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        for r in records:
            for w in r.get('windows',[]):
                wr.writerow({**{k:r[k] for k in ['video','subject','step']},
                    **{k:w[k] for k in ['index','start_s','reference_hr','reference_failure']},
                    'matched':w['reference_hr'] is not None and all(w['predictions'][a] is not None for a in ARMS),
                    **w['predictions'],**{a+'_failure':w['failures'][a] for a in ARMS}})
    labels=['Native pixels\nNative ROI','Native pixels\nH264 ROI','H264 pixels\nNative ROI','H264 pixels\nH264 ROI']
    fig,ax=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    val=[summary['matched'][a]['subject_macro_MAE'] for a in ARMS]
    ax[0].bar(labels,[v if v is not None else np.nan for v in val],color=['#287d8e','#779aae','#dca451','#bd7038'])
    ax[0].set(ylabel='Subject-macro MAE (bpm)',title=f"Matched windows: {summary['matched_windows']}")
    c=summary['coverage'][0]
    ax[1].bar(labels,[c['predictions_available'][a] for a in ARMS],color='#287d8e')
    ax[1].axhline(c['windows'],ls='--',color='gray',label='Analyzed candidate windows')
    ax[1].set(ylabel='Windows with a prediction',title=f"Reference-valid: {summary['reference_windows']}");ax[1].legend()
    fig.suptitle('MCD FullHD E0: native MPEG4 vs additional H264 500 kbps'+('' if summary['complete'] else ' — PARTIAL'))
    fig.savefig(out/'e0_cells.png',dpi=160);plt.close(fig)
    effects=summary['paired_effects']
    if effects:
        names=list(effects);means=[effects[k]['mean_bpm'] for k in names]
        lo=[effects[k]['ci95_bpm'][0] for k in names];hi=[effects[k]['ci95_bpm'][1] for k in names]
        fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
        y=np.arange(len(names));ax.hlines(y,lo,hi,color='#287d8e');ax.scatter(means,y,color='#287d8e')
        ax.axvline(0,color='gray',ls='--');ax.set(yticks=y,yticklabels=[k.replace('_',' ') for k in names],
            xlabel='Change in absolute HR error (bpm); positive = worse',title='Paired subject means; 95% subject-bootstrap intervals')
        fig.savefig(out/'e0_effects.png',dpi=160);plt.close(fig)
    print(json.dumps({'complete':summary['complete'],'processed':len(records),'report':str(out)}))

if __name__=='__main__':main()
