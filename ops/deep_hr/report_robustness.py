"""Recompute saved metrics and render a reproducible development-results figure."""
import argparse,csv,json,math,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
stress=root/'results/stress_ubfc_v1'; dev=root/'results/development_comparison_v1'
audit=[]
for folder in (stress,dev):
    for path in sorted(folder.glob('*.json')):
        data=json.loads(path.read_text())
        if not isinstance(data,dict) or 'records' not in data:continue
        rows=data['records'];assert data['n']==len(rows)
        assert all(r['split'] in ('train','val') for r in rows)
        errors=[abs(r['pred_hr']-r['gt_hr']) if r['pred_hr'] is not None else 135. for r in rows]
        assert all(abs(e-r['abs_error'])<1e-8 for e,r in zip(errors,rows))
        subjects=sorted({r['subject'] for r in rows})
        macro=st.mean(st.mean(e for e,r in zip(errors,rows) if r['subject']==s) for s in subjects)
        computed={'window_MAE':st.mean(errors),'window_RMSE':math.sqrt(st.mean(e*e for e in errors)),'subject_macro_MAE':macro}
        for k,v in computed.items():assert abs(v-data[k])<1e-8
        assert sum(r.get('invalid',False) for r in rows)==data['invalid']
        audit.append(str(path.relative_to(root)))
summary=json.loads((stress/'summary.json').read_text())
assert len(summary)==72
development=json.loads((dev/'summary.json').read_text())
table=[]
for cond in json.loads((stress/'protocol.json').read_text())['conditions']:
    vals=[r for r in summary if r['split']=='val' and r['condition']==cond]
    train=next(r for r in summary if r['split']=='train' and r['condition']==cond)
    assert len(vals)==3
    values=[r['window_MAE'] for r in vals]
    table.append({'condition':cond,'val_mean_MAE':st.mean(values),'val_seed_SD':st.stdev(values),
                  'train_2027_MAE':train['window_MAE'],'invalid_val_total':sum(r['invalid'] for r in vals)})
h264=[r for r in development if r['model'].startswith('original_')]
assert len(h264)==3
table.append({'condition':'h264_500k','val_mean_MAE':st.mean(r['window_MAE'] for r in h264),
              'val_seed_SD':st.stdev(r['window_MAE'] for r in h264),'train_2027_MAE':'',
              'invalid_val_total':sum(r['invalid'] for r in h264)})
with (root/'robustness_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
display=[r for r in table if r['condition']!='frozen_frame']
fig,ax=plt.subplots(figsize=(11,8))
y=list(range(len(display)));means=[r['val_mean_MAE'] for r in display]
for yi,r in zip(y,display):
    color='#1d7668' if r['condition'] in ('clean','h264_500k') else '#507aa5'
    ax.errorbar(r['val_mean_MAE'],yi,xerr=r['val_seed_SD'],fmt='o',color=color,capsize=3,markersize=6)
ax.set_yticks(y,[r['condition'] for r in display]);ax.invert_yaxis()
ax.set_xscale('log');ax.set_xlim(.3,max(means)*2)
ax.set_xlabel('Validation HR MAE (bpm), logarithmic scale; mean +/- sample SD over 3 seeds')
ax.set_title('EfficientPhys: development robustness audit\n7 validation subjects, 42 windows per condition')
for yi,r in zip(y,display):
    ax.text((r['val_mean_MAE']+r['val_seed_SD'])*1.10,yi,f"{r['val_mean_MAE']:.2f}",va='center',fontsize=9)
ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
fig.text(.02,.02,'Crop-level synthetic tests except full-frame H.264 with fixed original ROIs.\nFrozen-frame negative control excluded: its HR is not a physiological measurement.',fontsize=9)
fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(root/'robustness_validation.png',dpi=180);plt.close(fig)
comparisons=[]
for condition in ('clean','jpeg90','jpeg50','h264_500k','noise5'):
    c=next(r for r in development if r['model']=='control_2027' and r['condition']==condition)
    b=next(r for r in development if r['model']=='augmentation_2027' and r['condition']==condition)
    comparisons.append({'condition':condition,'control_MAE':c['window_MAE'],'augmentation_MAE':b['window_MAE'],
                        'change_bpm':b['window_MAE']-c['window_MAE']})
aipex=json.loads((root/'results/aipex_audit_v1/records.json').read_text())
aipex_summary=json.loads((root/'results/aipex_audit_v1/summary.json').read_text())
assert len(aipex)==88 and len({(r['group'],r['subject']) for r in aipex})==88
for key,saved in aipex_summary['metrics'].items():
    errors=[abs(r[key]-r['gt']) if r[key] is not None else 135 for r in aipex]
    assert abs(st.mean(errors)-saved['MAE'])<1e-8
    assert abs(math.sqrt(st.mean(e*e for e in errors))-saved['RMSE'])<1e-8
    assert sum(r[key] is None for r in aipex)==saved['invalid']
output={'verified_metric_files':len(audit),'all_metrics_recomputed':True,'aipex_metrics_recomputed':True,
        'stress_conditions':table,'pilot_comparison':comparisons}
(root/'independent_audit.json').write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2))
