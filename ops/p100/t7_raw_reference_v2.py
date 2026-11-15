"""Diagnostic-only repair: merge equal timestamps exactly as T1; no offset."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,argparse,hashlib
from pathlib import Path
import numpy as np
from t7_pipeline import save
from mcd_errata_audit import timed
from mcd_e0_metrics import sample_grid
from hr_baseline import estimate_hr
from t7_report import metrics

def main(a):
    original=json.loads((a.output/'results.json').read_text());rows=[];records=[]
    for c in json.loads((a.e0/'cohort.json').read_text()):
        pv,pt=timed(a.root/c['ppg']);_,mt=timed(a.root/c['meta'])
        desc={'video':c['video'],'negative_steps':int((np.diff(pt)<0).sum()),'duplicate_steps':int((np.diff(pt)==0).sum()),'raw_ppg_sha256':hashlib.sha256((a.root/c['ppg']).read_bytes()).hexdigest()}
        records.append(desc)
        tt,inverse,count=np.unique(pt,return_inverse=True,return_counts=True)
        yy=np.zeros(len(tt));np.add.at(yy,inverse,pv);yy/=count
        vv=np.ones(len(tt),bool);np.logical_and.at(vv,inverse,np.isfinite(pv))
        t=(tt-mt[0])/1e6
        for old in [r for r in original['rows'] if r['video']==c['video']]:
            r=dict(old);r['failures']=dict(old['failures']);r['raw_reference_hr']=None
            q=r['start_s']+np.arange(300)/30
            if desc['negative_steps']:r['failures']['raw_reference_hr']='Reversed timestamps'
            else:
                y,valid=sample_grid(t,yy,vv,q)
                if not valid.all():r['failures']['raw_reference_hr']='Raw support or gap>0.2s'
                else:
                    try:r['raw_reference_hr']=estimate_hr(y,30);r['failures'].pop('raw_reference_hr',None)
                    except ValueError as e:r['failures']['raw_reference_hr']=str(e)
            rows.append(r)
    summary={g:{'raw_vs_sync':metrics([r for r in rows if g=='all' or r['step']==g],'raw_reference_hr'), 'POS_vs_raw':metrics([r for r in rows if g=='all' or r['step']==g],'POS','raw_reference_hr')} for g in ['all','before','after']}
    save(a.output/'raw_reference_v2.json',dict(rows=rows,records=records,summary=summary,primary_predictions_unchanged=True,source_labels_unchanged=True,offset_corrected=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','e0','output']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
