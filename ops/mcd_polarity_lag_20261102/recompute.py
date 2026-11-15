"""Exact sign-only lag recalculation from retained full signed Pearson curves."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'reports/mcd_live_run_20261030/final_results.json'
out=ROOT/'reports/mcd_polarity_lag_20261102';out.mkdir(exist_ok=True)
rows=json.loads(source.read_text());assert len(rows)==357

def peak(m,multiplier):
    s=np.asarray(m['shifts_frames']);c=np.asarray(m['pearson_curve'])*multiplier
    cand=np.flatnonzero(c>=c.max()-1e-12);j=min(cand,key=lambda j:(abs(int(s[j])),int(s[j])))
    peaks=find_peaks(c)[0].tolist()
    if c[0]>c[1]:peaks.append(0)
    if c[-1]>c[-2]:peaks.append(len(c)-1)
    secondary=[i for i in peaks if abs(int(s[i]-s[j]))>=int(np.ceil(.25*m['fs']))]
    second=max((float(c[i]) for i in secondary),default=None)
    return dict(lag_frames=int(s[j]),lag_seconds=float(s[j]/m['fs']),pearson_max=float(c[j]),pearson_zero=float(c[np.flatnonzero(s==0)[0]]),weak_peak=bool(c[j]<.3),search_boundary=j in (0,len(s)-1),ambiguous_peak=bool(second is not None and second>=.95*c[j]))

records=[];windows=[]
for r in rows:
    for unit,items in [('record',[(None,r.get('fullrecord'))]),('window',[(w['index'],w['result']) for w in r['windows']])]:
        for wi,m in items:
            if m is None:continue
            old=peak(m,1);assert old['lag_frames']==m['lag_frames'] and abs(old['pearson_max']-m['pearson_max'])<1e-12
            for mode,mult in [('original_POS_raw_PPG',1),('original_POS_corrected_PPG',-1),('canonical_negative_POS_corrected_PPG',1)]:
                row=dict(video=r['video'],subject=r['subject'],camera=r['camera'],step=r['step'],unit=unit,window_index=wi,mode=mode,**peak(m,mult))
                (records if unit=='record' else windows).append(row)

for name,rs in [('records',records),('windows',windows)]:
    with (out/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
groups=[]
for camera in ['FullHDwebcam','USBVideo','IriunWebcam']:
    for mode in ['original_POS_raw_PPG','original_POS_corrected_PPG','canonical_negative_POS_corrected_PPG']:
        for unit,rr in [('record',records),('window',windows)]:
            a=[r for r in rr if r['camera']==camera and r['mode']==mode]
            lag=np.asarray([r['lag_frames'] for r in a]);sec=np.asarray([r['lag_seconds'] for r in a])
            groups.append(dict(camera=camera,mode=mode,unit=unit,n=len(a),lag_frames_median=float(np.median(lag)),lag_frames_q25=float(np.quantile(lag,.25)),lag_frames_q75=float(np.quantile(lag,.75)),lag_seconds_median=float(np.median(sec)),abs_lag_lt3_fraction=float(np.mean(abs(lag)<3)),weak=sum(r['weak_peak'] for r in a),ambiguous=sum(r['ambiguous_peak'] for r in a),boundary=sum(r['search_boundary'] for r in a)))
summary=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records_in_source=357,subjects=len(set(r['subject'] for r in rows)),D3_opened=False,method='Pearson linearity: corr(POS,-PPG)=-corr(POS,PPG); corr(-POS,-PPG)=corr(POS,PPG). Recompute full original -20..20 search, original tie-breaking, weak/ambiguous/boundary flags. No new tracking, resampling, filters or offsets.',lag_convention='POS(t+lag) versus PPG(t), positive means POS delayed. Native diagnostic grids30Hz FullHD/USB,24Hz Iriun. Not sensorFPS assertion.',groups=groups,source_processing_errors=[{'video':r['video'],'error':r['error']} for r in rows if 'error' in r],limitations=['Both-flip result is algebraically unchanged; not new independent synchronization evidence.','Original3-region POS traces differ from T8 oval.','Original signed Pearson has periodic ambiguities; median is not proof each clip has small phase error.','Do not use original-POS/corrected-PPG half-cycle optimum as offset.'])
(out/'summary.json').write_text(json.dumps(summary,indent=2))
fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True)
for ax,cam in zip(axs,['FullHDwebcam','USBVideo','IriunWebcam']):
    for mode,label in [('original_POS_corrected_PPG','Flip PPG only'),('canonical_negative_POS_corrected_PPG','Both in canonical sign')]:
        a=[r['lag_frames'] for r in records if r['camera']==cam and r['mode']==mode]
        ax.hist(a,bins=np.arange(-20.5,21.5),alpha=.5,label=label)
    ax.set(title=cam,xlabel='Lag (diagnostic-grid frames)');ax.legend(fontsize=8)
axs[0].set_ylabel('Evaluable full recordings');fig.suptitle('Dev60 sign-only lag recalculation; timestamps unchanged');fig.tight_layout();fig.savefig(out/'lag_polarity.png',dpi=160)
print(json.dumps([g for g in groups if g['unit']=='record'],indent=2))
