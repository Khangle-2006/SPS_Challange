"""Summarize completed or partial diagnostics without converting similarity to proof."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,csv
from pathlib import Path
from collections import Counter
from mcd_lag_reading import endpoint_drift, drift_counts, READING_CRITERIA_SHA256
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def quant(values):
    v=np.asarray([x for x in values if x is not None],float)
    return dict(zip(['min','p05','p25','median','p75','p95','max'],np.quantile(v,[0,.05,.25,.5,.75,.95,1]).tolist())) if len(v) else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);a=ap.parse_args()
    gd=a.audit/'dev60_gate_v3';ld=a.audit/'lag_pos_ppg_v3';out=ld/'report';out.mkdir(parents=True,exist_ok=True)
    gate=json.loads((gd/'status.json').read_text());lag=json.loads((ld/'status.json').read_text())
    results=[json.loads(p.read_text()) for p in sorted(ld.glob('*/result.json'))]
    windows=[{'video':r['video'],'subject':r['subject'],'camera':r['camera'],'step':r['step'],**w} for r in results for w in r['windows']]
    groups=[]
    for cam in ['FullHDwebcam','USBVideo','IriunWebcam']:
        for step in ['all','before','after']:
            rr=[r for r in results if r['camera']==cam and (step=='all' or r['step']==step)]
            ff=[r['fullrecord'] for r in rr if r.get('fullrecord') is not None]
            ww=[w for r in rr for w in r['windows']];vv=[w['result'] for w in ww if w['result'] is not None]
            groups.append({'camera':cam,'condition':step,'records_processed':len(rr),'processing_errors':sum('error'in r for r in rr),
                'fullrecord_evaluable':len(ff),'fullrecord_missing':len(rr)-len(ff),
                'full_lag_frames':quant([x['lag_frames'] for x in ff]),'full_lag_seconds':quant([x['lag_seconds'] for x in ff]),
                'full_peak_correlation':quant([x['pearson_max'] for x in ff]),
                'full_zero_correlation':quant([x['pearson_zero'] for x in ff]),
                'full_weak':sum(x['weak_peak'] for x in ff),'full_ambiguous':sum(x['ambiguous_peak'] for x in ff),
                'full_boundary':sum(x['search_boundary'] for x in ff),
                'windows_total':len(ww),'windows_evaluable':len(vv),'window_lag_s':quant([x['lag_seconds'] for x in vv]),
                'window_weak':sum(x['weak_peak'] for x in vv),'window_ambiguous':sum(x['ambiguous_peak'] for x in vv),
                'window_boundary':sum(x['search_boundary'] for x in vv),
                'drift_ms_per_min':quant([r.get('drift',{}).get('slope_ms_per_min') for r in rr]),
                'drift_range_ms':quant([r.get('drift',{}).get('lag_range_ms') for r in rr])})
    complete=(gate.get('state')=='complete' and lag.get('state')=='complete'
              and gate.get('processed')==360 and len(results)==357
              and lag.get('processed')==357 and lag.get('eligible_total')==357
              and len({r['video'] for r in results})==357)
    for group in groups:
        rr=[r for r in results if r['camera']==group['camera'] and
            (group['condition']=='all' or r['step']==group['condition'])]
        expected={'all':119,'before':60,'after':59}[group['condition']]
        group['endpoint_drift']=drift_counts(rr,expected)
        group['fullrecord_failure_reasons']=dict(Counter(
            r.get('error') or r.get('fullrecord_failure') or 'unspecified'
            for r in rr if r.get('fullrecord') is None))
        for unit in ['frames','seconds']:
            q=group['full_lag_'+unit]
            group['full_lag_'+unit+'_iqr']=q['p75']-q['p25'] if q else None
            group['full_lag_'+unit+'_p90_span']=q['p95']-q['p05'] if q else None
    widths={g['camera']:g['full_lag_frames_iqr'] for g in groups if g['condition']=='all'}
    width_agreement=(all(v is not None for v in widths.values()) and
                     all(widths['FullHDwebcam'] < widths[c] for c in ['USBVideo','IriunWebcam']))
    summary={'gate':gate,'lag':lag,'results_found':len(results),'groups':groups,'complete':complete,
       'known_reference_errata':gate.get('reference_errata_count'),
       'fullrecord_failure_reasons':dict(Counter(r.get('error') or r.get('fullrecord_failure') for r in results if r.get('fullrecord') is None)),
       'window_failure_reasons':dict(Counter(w['failure'] for w in windows if w['result'] is None)),
       'e0_gate_count_pass':bool(complete and gate.get('below_five_percent_gate',False)),
       'e0_readiness':'requires_final_scientific_review' if complete else 'pending_complete_results',
       'offset_corrected':False,'training_started':False,
       'reading_criteria_sha256':READING_CRITERIA_SHA256,
       'primary_endpoint_drift':drift_counts(results,357),
       'fullhd_strictly_smallest_frame_iqr':width_agreement,
       'figure5_review_required':True,
       'interpretation':'Paper-aligned partial reproduction. Histogram similarity is not a synchronization proof. No thresholds fitted after results.'}
    save(out/'summary.json',summary)
    endpoint_fields=list(endpoint_drift({}).keys())
    recfields=['video','subject','camera','step','official_fold','fullrecord_evaluable','full_lag_frames','full_lag_s','peak_corr','zero_corr','weak','ambiguous','boundary','windows_total','windows_valid','drift_ms_per_min','drift_range_ms','failure']+endpoint_fields
    with (out/'records.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=recfields);wr.writeheader()
        for r in results:
            m=r.get('fullrecord') or {};d=r.get('drift',{});ww=r['windows']
            wr.writerow({**{k:r[k] for k in ['video','subject','camera','step','official_fold']},
              'fullrecord_evaluable':bool(m),'full_lag_frames':m.get('lag_frames'),'full_lag_s':m.get('lag_seconds'),
              'peak_corr':m.get('pearson_max'),'zero_corr':m.get('pearson_zero'),'weak':m.get('weak_peak'),
              'ambiguous':m.get('ambiguous_peak'),'boundary':m.get('search_boundary'),'windows_total':len(ww),
              'windows_valid':sum(w['result'] is not None for w in ww),'drift_ms_per_min':d.get('slope_ms_per_min'),
              'drift_range_ms':d.get('lag_range_ms'),'failure':r.get('error') or r.get('fullrecord_failure'),
              **endpoint_drift(r)})
    with (out/'windows.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['video','subject','camera','step','index','midpoint_s','lag_frames','lag_seconds','pearson_max','pearson_zero','weak_peak','ambiguous_peak','search_boundary','failure']);wr.writeheader()
        for w in windows:
            wr.writerow({**{k:w[k] for k in ['video','subject','camera','step','index','midpoint_s','failure']},
                         **{k:(w['result'] or {}).get(k) for k in ['lag_frames','lag_seconds','pearson_max','pearson_zero','weak_peak','ambiguous_peak','search_boundary']}})
    fig,ax=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    for j,cam in enumerate(['FullHDwebcam','USBVideo','IriunWebcam']):
        ff=[r['fullrecord'] for r in results if r['camera']==cam and r.get('fullrecord') is not None]
        vals=[r['lag_frames'] for r in ff];secs=[r['lag_seconds'] for r in ff]
        ax[j,0].hist(vals,bins=np.arange(-20.5,21.5),color='#24718d');ax[j,0].set(title=f'{cam}, full-record n={len(vals)}',xlabel='Lag (native-grid frames)',ylabel='Records',xlim=(-21,21))
        if secs:ax[j,1].hist(secs,bins=np.linspace(-.85,.85,35),weights=np.ones(len(secs))/len(secs),color='#24718d')
        ax[j,1].set(xlabel='Lag (seconds)',ylabel='Fraction of evaluable records',xlim=(-.85,.85))
    fig.suptitle('MCD dev60: POS–PPG signed-Pearson lag; no offset correction'+('' if complete else ' — PARTIAL'))
    fig.savefig(out/'lag_histograms.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for j,cam in enumerate(['FullHDwebcam','USBVideo','IriunWebcam']):
        for r in results:
            if r['camera']!=cam:continue
            ww=[w for w in r['windows'] if w['result'] is not None]
            ax[j].plot([w['midpoint_s'] for w in ww],[1000*w['result']['lag_seconds'] for w in ww],alpha=.23,lw=.8)
        ax[j].set(title=cam,xlabel='Window midpoint (s)',ylabel='Lag (ms)',ylim=(-900,900))
    fig.suptitle('30 s window lag: periodic-peak jumps retained, no unwrapping')
    fig.savefig(out/'lag_drift.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for j,cam in enumerate(['FullHDwebcam','USBVideo','IriunWebcam']):
        rr=[r for r in results if r['camera']==cam]
        dd=[endpoint_drift(r) for r in rr]
        valid=[(i,d) for i,d in enumerate(dd) if d['endpoint_evaluable']]
        ax[j].scatter([i for i,d in valid],[d['endpoint_delta_frames'] for i,d in valid],
                      c=['#b3261e' if d['endpoint_drift_flag'] else '#24718d' for i,d in valid],s=16)
        ax[j].axhline(3,color='gray',ls='--');ax[j].axhline(-3,color='gray',ls='--')
        count=sum(d['endpoint_drift_flag'] is True for d in dd)
        ax[j].set(title=f'{cam}: {count} flagged / {len(valid)} evaluable',
                  xlabel='Record index (fixed file order)',ylabel='Last minus first window lag (frames)',ylim=(-42,42))
    fig.suptitle('Actual 30 s endpoints; missing endpoints are unknown'+('' if complete else ' — PARTIAL'))
    fig.savefig(out/'lag_endpoint_drift.png',dpi=160);plt.close(fig)
    print(json.dumps({'complete':complete,'results':len(results),'groups':groups},indent=2))

if __name__=='__main__':main()
