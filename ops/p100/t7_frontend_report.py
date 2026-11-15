"""Matched frontend diagnostic report; missing windows remain in denominator."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from t7_report import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(root):
    status=json.loads((root/'status.json').read_text())
    if status['completed']!=119 or status['state'] not in ['complete','complete_with_errors']:raise RuntimeError('Not complete; keep monitoring')
    data=json.loads((root/'results.json').read_text());rows=data['rows']
    names=['POS','control_small','large_oval'];paired=[r for r in rows if all(r.get(k) is not None for k in names)]
    groups={}
    for g in ['all','before','after']:
        rr=[r for r in paired if g=='all' or r['step']==g]
        groups[g]={k:metrics(rr,k) for k in names}
    effects={}
    for name,new,old in [('large_minus_small','large_oval','control_small'),('control_minus_original','control_small','POS'),('large_minus_original','large_oval','POS')]:
        by=defaultdict(list)
        for r in paired:by[r['subject']].append(abs(r[new]-r['reference_hr'])-abs(r[old]-r['reference_hr']))
        v=np.array([np.mean(by[s]) for s in sorted(by)])
        if len(v):
            boot=np.random.default_rng(20261031).choice(v,(1000,len(v)),replace=True).mean(1)
            effects[name]={'mean_bpm':float(v.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'subjects':len(v)}
    summary={'expected_windows':1860,'processed_windows':len(rows),'record_errors':data['errors'],'paired_windows':len(paired),'coverage':{k:sum(r.get(k) is not None for r in rows)/1860 for k in names},'matched_groups':groups,'effects':effects}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    if rows:
        with (root/'windows.csv').open('w',newline='') as f:
            names_csv=[k for k in rows[0] if k!='failures'];w=csv.DictWriter(f,fieldnames=names_csv,extrasaction='ignore');w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    ax.bar(['Original E0','Small-region control','Large face oval'],[groups['all'][k].get('subject_macro_MAE',np.nan) for k in names],color=['gray','tab:blue','tab:orange'])
    ax.set(ylabel='Subject-macro MAE (bpm)',title=f'T7b native POS: matched {len(paired)}/1860 windows, 10 seconds')
    fig.savefig(root/'frontend_comparison.png',dpi=150);plt.close(fig)
    (root/'REPORT.txt').write_text('T7b region diagnostic on development cohort, not untouched test.\nOriginal E0 versus two region definitions computed from the same T3 float16 landmarks.\nNot an exact reproduction of author face_alignment frontend. No offset or labels changed.\n'+json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
