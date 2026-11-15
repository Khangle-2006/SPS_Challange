"""User-requested descriptive ratios from frozen predictions; no new inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BINS = ['<=-30','(-30,10]','(10,28]','(28,41]','(41,55]','>55','unknown']
COUNTRIES = ['India','Sierra_Leone']

def summarize(rows):
    valid = [r for r in rows if r['prediction_bpm'] is not None and r['reference_bpm'] > 0]
    out = dict(expected=len(rows), n=len(valid), sparse=len(valid)<5,
               unknown_ita=sum(r['ita'] is None for r in rows))
    if not valid:
        return out
    ref = np.array([r['reference_bpm'] for r in valid])
    pred = np.array([r['prediction_bpm'] for r in valid])
    ratio = pred/ref
    out.update(mean_ratio=float(ratio.mean()), median_ratio=float(np.median(ratio)),
               mean_relative_bias_percent=float(100*(ratio.mean()-1)),
               ratio_of_means_secondary=float(pred.mean()/ref.mean()),
               mean_reference_bpm=float(ref.mean()), mean_prediction_bpm=float(pred.mean()))
    if len(valid)>1:
        rng = np.random.default_rng(20261101)
        out['mean_ratio_CI95'] = np.percentile(ratio[rng.integers(len(valid), size=(10000,len(valid)))].mean(1), [2.5,97.5]).tolist()
    return out

def contrast(rows):
    groups = [[r for r in rows if r['country']==c and r['prediction_bpm'] is not None and r['reference_bpm']>0] for c in COUNTRIES]
    result = dict(direction='India minus Sierra Leone', n=[len(g) for g in groups], paired=False)
    if not all(groups):
        return result
    rng = np.random.default_rng(20261101)
    ix = [rng.integers(len(g), size=(10000,len(g))) for g in groups]
    for metric in ['ratio', 'signed_error_bpm']:
        vals = [np.array([r['prediction_bpm']/r['reference_bpm'] if metric=='ratio' else r['prediction_bpm']-r['reference_bpm'] for r in g]) for g in groups]
        delta = float(vals[0].mean()-vals[1].mean())
        boot = vals[0][ix[0]].mean(1)-vals[1][ix[1]].mean(1)
        result[metric] = dict(mean_difference=delta, CI95=np.percentile(boot,[2.5,97.5]).tolist())
    return result

def main(source, output):
    rows = json.loads(source.read_text(encoding='utf-8'))
    assert len(rows)==88 and len({(r['country'],r['subject']) for r in rows})==88
    assert all(sum(r['country']==c for r in rows)==44 for c in COUNTRIES)
    groups = {'all':rows}
    groups.update({'country:'+c:[r for r in rows if r['country']==c] for c in COUNTRIES})
    for b in BINS:
        groups['ITA:'+b] = [r for r in rows if r['ita_bin']==b]
        for c in COUNTRIES:
            groups[c+' / '+b] = [r for r in rows if r['ita_bin']==b and r['country']==c]
    summary = {k:summarize(g) for k,g in groups.items()}
    differences = {'all':contrast(rows)}
    differences.update({'ITA:'+b:contrast(groups['ITA:'+b]) for b in BINS})
    output.mkdir(parents=True, exist_ok=False)
    result = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  bootstrap_seed=20261101, bootstrap_draws=10000, groups=summary, contrasts=differences,
                  primary_estimand='Arithmetic mean of per-person predicted/reference HR ratios',
                  timing='Assumed 15 fps; acquisition timing unresolved; ratios are NOT timing invariant',
                  limitations=['Descriptive collection-site contrast, not a causal skin effect or equivalence test.',
                               'Independent country resampling; ITA missing India 1, Sierra Leone 10; sparse bins flagged.',
                               'Different reference-HR distributions affect ratio contrasts; no FPS fitting or model selection.',
                               'Post-hoc reporting requested by user; fixed predictions and all 88 people retained.'])
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (output/'per_person_ratios.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['country','subject','reference_bpm','prediction_bpm','ratio','relative_bias_percent','ita_bin'])
        for r in rows:
            q=None if r['prediction_bpm'] is None else r['prediction_bpm']/r['reference_bpm']
            w.writerow([r['country'],r['subject'],r['reference_bpm'],r['prediction_bpm'],q,None if q is None else (q-1)*100,r['ita_bin']])
    lines=['AiPEX ratio addendum: fixed predictions at assumed 15 fps; no re-inference or retuning.',
           result['primary_estimand'],result['timing'],*result['limitations'],'']
    lines += [k+': '+json.dumps(v) for k,v in summary.items()]
    lines += ['COUNTRY CONTRASTS: '+json.dumps(differences,indent=2)]
    (output/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig, axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    colors=['#2674b8','#c67a23']
    for i,c in enumerate(COUNTRIES):
        m=summary['country:'+c];lo,hi=m['mean_ratio_CI95']
        axes[0].errorbar(i,m['mean_ratio'],yerr=[[m['mean_ratio']-lo],[hi-m['mean_ratio']]],fmt='o',color=colors[i],capsize=5)
        rr=[r for r in rows if r['country']==c and r['ita'] is not None and r['prediction_bpm'] is not None]
        axes[1].scatter([r['ita'] for r in rr],[r['prediction_bpm']/r['reference_bpm'] for r in rr],s=22,alpha=.7,color=colors[i],label=c.replace('_',' '))
    for ax in axes:
        ax.axhline(1,color='gray',ls='--',lw=1);ax.set_ylabel('Estimated HR / reference HR')
    axes[0].set_xticks([0,1],['India (44/44)','Sierra Leone (44/44)'])
    axes[0].set_title('Mean person ratio; 95% bootstrap CI')
    axes[1].set_xlabel('Image-derived ITA (camera / lighting dependent)');axes[1].legend()
    axes[1].set_title('ITA available: 43 India / 34 Sierra Leone')
    fig.suptitle('AiPEX descriptive ratios | assumed 15 fps; acquisition timing unresolved')
    fig.savefig(output/'ratios.png',dpi=160);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['all','country:India','country:Sierra_Leone']}))
    print(json.dumps(differences['all']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.input,a.output)
