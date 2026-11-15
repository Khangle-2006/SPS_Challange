"""Replay saved matched validation waveforms, never train or retune."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from decode_hr_v2 import ablation_config,decode_sequence

STAGES=['legacy','a','b','c','d','e']
SEEDS=[2027,2028,2029]
CONDITIONS=['clean','h264_250k','h264_500k','h264_1000k']
SUBJECTS={'subject9','subject18','subject26','subject32','subject35','subject43','subject45'}

def metrics(rows,accepted=False):
    selected=[r for r in rows if np.isfinite(r['pred_hr']) and (not accepted or not r['abstain'])]
    err=np.array([abs(r['pred_hr']-r['gt_hr']) for r in selected])
    subs=sorted({r['subject'] for r in selected})
    return dict(n_total=len(rows),n_valid=len(selected),coverage=len(selected)/len(rows),
        subject_macro_MAE=float(np.mean([np.mean([abs(r['pred_hr']-r['gt_hr']) for r in selected if r['subject']==s]) for s in subs])) if subs else None,
        window_MAE=float(err.mean()) if len(err) else None,P95=float(np.percentile(err,95)) if len(err) else None,
        errors_over20=int((err>20).sum()),subjects_evaluable=len(subs))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def replay(waves,rows,fs=30.):
    answer={}
    for stage in STAGES:
        out=[]
        for sub in sorted({r['subject'] for r in rows}):
            indices=sorted([i for i,r in enumerate(rows) if r['subject']==sub],key=lambda i:rows[i]['start_s'])
            decoded=decode_sequence(waves[indices,None,:],fs,config=ablation_config(stage),starts=[rows[i]['start_s'] for i in indices])
            for i,r in zip(indices,decoded):
                out.append({**rows[i],'pred_hr':r['hr_bpm'],'abstain':r['abstain'],'confidence':r['confidence'],'reset':r.get('reset',False)})
        answer[stage]=dict(all=metrics(out),accepted=metrics(out,True),records=out)
    return answer

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    base=Path(a.base);out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    protocol=out/'protocol_frozen.json';assert protocol.exists()
    p=json.loads(protocol.read_text(encoding='utf-8-sig'));assert (p['beta'],p['kappa'],p['tau'])==(.5,10.,2.)
    saved=json.loads((base/'results.json').read_text());ref=saved['mse']['clean']['2027']['records']
    assert len(ref)==42 and {r['subject'] for r in ref}==SUBJECTS and all(r['split']=='val' for r in ref)
    results={};inputs={str(base/'results.json'):sha(base/'results.json')};flat=[]
    for family in ['mse','spectral']:
        for condition in CONDITIONS:
            file=base/f'evaluation/{family}_{condition}_waves.npz';inputs[str(file)]=sha(file)
            waves=np.load(file)['waves'];assert waves.shape==(3,42,300) and np.isfinite(waves).all()
            for si,seed in enumerate(SEEDS):
                rows=saved[family][condition][str(seed)]['records']
                assert all(r['split']=='val' and all(r[k]==ref[i][k] for k in ['subject','window','start_s','gt_hr']) for i,r in enumerate(rows))
                key=f'{family}/{seed}/{condition}';results[key]=replay(waves[si],rows)
                # Reproduce historical argmax before trusting new metrics.
                old={ (r['subject'],r['window']):r['pred_hr'] for r in rows }
                assert all(abs(r['pred_hr']-old[r['subject'],r['window']])<1e-10 for r in results[key]['legacy']['records'])
                for stage,v in results[key].items():
                    for scope in ['all','accepted']:
                        flat.append(dict(family=family,seed=seed,condition=condition,stage=stage,scope=scope,**v[scope]))
                print(key,'done',flush=True)
    acceptance={}
    for family in ['mse','spectral']:
        for seed in SEEDS:
            c=results[f'{family}/{seed}/clean'];h=results[f'{family}/{seed}/h264_500k']
            b=h['legacy']['all'];v=h['e']['accepted']
            checks=dict(p95_reduction_30=v['P95'] is not None and v['P95']<=.7*b['P95'],
                over20_reduction_30=(v['errors_over20']<=.7*b['errors_over20']) if b['errors_over20'] else None,
                coverage_90=v['coverage']>=.9,
                clean_within_5pct=c['e']['accepted']['window_MAE'] is not None and c['e']['accepted']['window_MAE']<=1.05*c['legacy']['all']['window_MAE'])
            acceptance[f'{family}/{seed}']=dict(checks=checks,passed=all(v is True for v in checks.values()))
    save(out/'matched_results.json',results);save(out/'acceptance.json',acceptance)
    with (out/'ablation.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    fig,axes=plt.subplots(1,2,figsize=(11,4));curves={}
    for ax,family in zip(axes,['mse','spectral']):
        for condition in CONDITIONS:
            rows=sum([results[f'{family}/{s}/{condition}']['e']['records'] for s in SEEDS],[])
            rows=sorted(rows,key=lambda r:r['confidence'],reverse=True)
            xs=[];ys=[]
            for n in range(1,len(rows)+1):
                xs.append(n/len(rows));ys.append(metrics(rows[:n])['window_MAE'])
            ax.plot(xs,ys,label=condition);curves[f'{family}/{condition}']={'coverage':xs,'MAE':ys}
        ax.set(title=family,xlabel='Coverage (ranked confidence; descriptive only)',ylabel='Window MAE (bpm)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'coverage_accuracy.png',dpi=160);plt.close(fig)
    save(out/'coverage_accuracy.json',curves)
    save(out/'provenance.json',dict(protocol_sha256=sha(protocol),inputs=inputs,python=sys.version,
        scripts={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('decode_hr_v2.py'),Path(__file__).with_name('test_decode_hr_v2.py')]}))
    save(out/'status.json',dict(state='matched_complete_pos_pending',matched_models=6,matched_model_conditions=24,
       matched_windows_per_condition=42,validation_subjects=sorted(SUBJECTS),overall_acceptance=False,
       remaining=['UBFC POS saved traces availability/replay','MCD POS native and completed H264 traces','final report'],no_training=True))

if __name__=='__main__':main()
