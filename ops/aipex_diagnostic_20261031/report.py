"""Report complete AiPEX cohort; preserves HR failures and unknown ITA."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BINS=['<=-30','(-30,10]','(10,28]','(28,41]','(41,55]','>55','unknown']
def metrics(rows):
    valid=[r for r in rows if r['prediction_bpm'] is not None]
    e=np.array([r['prediction_bpm']-r['reference_bpm'] for r in valid])
    m=dict(expected=len(rows),n=len(valid),unknown_ita=sum(r['ita'] is None for r in rows),sparse=len(valid)<5)
    if not len(e): return m
    m.update(MAE=float(np.abs(e).mean()),RMSE=float(np.sqrt((e**2).mean())),bias=float(e.mean()),
             P95=float(np.percentile(abs(e),95)),over20_fraction=float(np.mean(abs(e)>20)))
    if len(e)>1:
        rng=np.random.default_rng(20261031);boot=e[rng.integers(0,len(e),(10000,len(e)))]
        m['MAE_CI95']=np.percentile(abs(boot).mean(1),[2.5,97.5]).tolist()
        m['bias_CI95']=np.percentile(boot.mean(1),[2.5,97.5]).tolist()
    return m

def report(source,out):
    status=json.loads((source/'status.json').read_text())
    if status['phase']!='complete' or status['done']!=88: raise ValueError('Cohort incomplete')
    rows=json.loads((source/'records.json').read_text())
    if len(rows)!=88 or len({(r['country'],r['subject']) for r in rows})!=88: raise ValueError('Cohort mismatch')
    out.mkdir(parents=True,exist_ok=False)
    groups={'all':rows}
    for c in ('India','Sierra_Leone'): groups['country:'+c]=[r for r in rows if r['country']==c]
    for b in BINS:
        groups['ITA:'+b]=[r for r in rows if r['ita_bin']==b]
        for c in ('India','Sierra_Leone'):
            groups[c+' / '+b]=[r for r in rows if r['ita_bin']==b and r['country']==c]
    result={k:metrics(v) for k,v in groups.items()}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fields=['country','subject','reference_bpm','prediction_bpm','signed_error_bpm','ita','ita_bin','hr_error','ita_error']
    with (out/'clips.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            v={k:r.get(k) for k in fields}
            v['signed_error_bpm']=None if r['prediction_bpm'] is None else r['prediction_bpm']-r['reference_bpm']
            w.writerow(v)
    fields=['stratum','expected','n','unknown_ita','sparse','MAE','MAE_CI95','RMSE','bias','bias_CI95','P95','over20_fraction']
    with (out/'strata.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for k,v in result.items():w.writerow(dict(stratum=k,**v))
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for c,color in [('India','#2674b8'),('Sierra_Leone','#c67a23')]:
        rr=[r for r in rows if r['country']==c and r['ita'] is not None and r['prediction_bpm'] is not None]
        axes[0].scatter([r['ita'] for r in rr],[r['prediction_bpm']-r['reference_bpm'] for r in rr],label=c.replace('_',' '),s=24,alpha=.75,color=color)
    axes[0].axhline(0,color='gray',lw=1);axes[0].set(xlabel='Image-derived ITA (degrees)',ylabel='Predicted minus reference HR (bpm)');axes[0].legend()
    cats=['all','country:India','country:Sierra_Leone']
    for i,k in enumerate(cats):
        m=result[k]
        if m['n']:
            axes[1].bar(i,m['MAE'],color=['#607b8b','#2674b8','#c67a23'][i])
            if 'MAE_CI95' in m:
                lo,hi=m['MAE_CI95'];axes[1].errorbar(i,m['MAE'],yerr=[[m['MAE']-lo],[hi-m['MAE']]],fmt='none',color='black',capsize=4)
            axes[1].text(i,m['MAE']+2,f"n={m['n']}/{m['expected']}",ha='center')
    axes[1].set_xticks(range(3),['All','India','Sierra Leone']);axes[1].set_ylabel('Clip-mean HR MAE (bpm; 95% bootstrap CI)')
    fig.suptitle('AiPEX limited diagnostic | assumed 15 fps; acquisition timing unverified')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    lines=['AiPEX LIMITED DIAGNOSTIC — NOT T8 BENCHMARK','Assumed 15 fps; acquisition timing unverified. Fixed 24-sample POS / 150-sample HR windows.',
        'Three released ROI fragments; mean window HR per ROI, median three ROI means. Scalar clip-mean reference.',
        'ITA is camera/lighting-dependent image color, not calibrated skin tone; country denotes collection site.',
        'No FPS fitting, no model selection, no causal skin-effect claim. No waveform metrics. CI conditional on timing assumption.',
        'P95 and >20 use one error per person/clip. Sparse n<5 strata are unstable; singleton CI omitted.','']
    for k,m in result.items():
        lines.append(k+': '+json.dumps(m))
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.input,a.output)
