"""Verify reported metadata errors; build immutable-source curated metadata."""
import csv,json,hashlib,argparse
from pathlib import Path
from datetime import datetime,timedelta
from collections import Counter
import numpy as np

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [x.split(maxsplit=1) for x in p.read_text().splitlines() if x.strip()]
def timed(p):
    rr=rows(p);epoch=datetime(1970,1,1)
    return np.array([float(x[0]) for x in rr]),np.array([(datetime.fromisoformat(x[1])-epoch)//timedelta(microseconds=1) for x in rr],dtype=np.int64)
def nearest(meta,pt,pv):
    # Source order establishes tie priority. Sorting doesn't alter source file.
    order=np.argsort(pt,kind='stable');t=pt[order];v=pv[order]
    t,first=np.unique(t,return_index=True);v=v[first]
    right=np.searchsorted(t,meta).clip(0,len(t)-1);left=(np.searchsorted(t,meta)-1).clip(0,len(t)-1)
    idx=np.where(abs(meta-t[left])<=abs(meta-t[right]),left,right)
    return v[idx],abs(meta-t[idx])/1e6

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--fold-csv',type=Path,required=True);a=ap.parse_args()
    out=a.audit/'curated_v3';out.mkdir(exist_ok=True)
    raw=list(csv.DictReader((a.root/'db.csv').open()));original_hash=digest(a.root/'db.csv')
    foldrows=list(csv.DictReader(a.fold_csv.open()));folds={}
    for r in foldrows:
        sid=str(int(float(r['patient_id'])));v=int(r['fold'])
        assert sid not in folds or folds[sid]==v,('subject crosses folds',sid)
        folds[sid]=v
    evidence=[];corrected=[];changes=[]
    for r0 in raw:
        r=dict(r0);sid=r['patient_id'];step=r['step']
        r['official_fold']=str(folds[sid]);r['hr_label_source']='contact_ppg';r['pulse_hr_label_allowed']='false'
        r['reference_eligible']='true';r['reference_exclusion_reason']=''
        if sid in ('4952','7412'):
            _,mt=timed(a.root/r['meta']);pv,pt=timed(a.root/r['ppg'])
            sy=np.loadtxt(a.root/r['ppg_sync'],ndmin=2)
            e={'video':r['video'],'step':step,'old_ppg':r['ppg'],'old_ecg':r['ecg'],
               'meta_start':datetime.fromtimestamp(int(mt[0])/1e6).isoformat(),
               'ppg_start':datetime.fromtimestamp(int(pt[0])/1e6).isoformat(),
               'start_difference_s':float((pt[0]-mt[0])/1e6),'supplied_sync_std':float(sy[:,0].std()),
               'source_meta_sha256':digest(a.root/r['meta']),'source_ppg_sha256':digest(a.root/r['ppg'])}
            if sid=='4952':
                newppg=f'ppg/4952_{step}.PW';newecg=f'ecg/4952_{step}.json'
                nv,nt=timed(a.root/newppg)
                assert abs(int(nt[0])-int(mt[0]))<1_000_000
                assert abs(int(pt[0])-int(mt[0]))>600_000_000
                assert (a.root/newecg).exists()
                e.update(corrected_ppg=newppg,corrected_ecg=newecg,corrected_start_difference_s=float((nt[0]-mt[0])/1e6))
                r['ppg']=newppg;r['ecg']=newecg
                newvalues,dist=nearest(mt,nt,nv)
                sync=out/'ppg_sync'/Path(r['ppg_sync']).name;sync.parent.mkdir(exist_ok=True)
                np.savetxt(sync,np.column_stack([newvalues,dist]),fmt=['%.10g','%.6f'])
                r['ppg_sync']=str(sync)
                e.update(corrected_sync_std=float(newvalues.std()),corrected_nearest_dt_max_s=float(dist.max()))
                changes.append({'video':r['video'],'old_ppg':r0['ppg'],'new_ppg':r['ppg'],'old_ecg':r0['ecg'],'new_ecg':r['ecg'],'new_sync':str(sync)})
            if sid=='7412' and step=='after':
                assert abs(int(pt[0])-int(mt[0]))>86400_000_000 and float(sy[:,0].std())==0
                r['reference_eligible']='false';r['reference_exclusion_reason']='7412_after_no_matching_ppg'
            evidence.append(e)
        corrected.append(r)
    fields=list(corrected[0])
    with (out/'db.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(corrected)
    (out/'errata_evidence.json').write_text(json.dumps(evidence,indent=2))
    byvideo={r['video']:r for r in corrected}
    plan=json.loads((a.audit/'development_plan.json').read_text())
    dev=[byvideo[r['video']] for r in plan['development_records']]
    assert len(dev)==360 and len({r['patient_id'] for r in dev})==60
    plan.update(development_records=dev,source_db_sha256=original_hash,curated_db_sha256=digest(out/'db.csv'),
        official_fold_source_sha256=digest(a.fold_csv),curated_metadata_path=str(out/'db.csv'),
        no_training=True,hr_label_source='contact_ppg_only')
    (a.audit/'development_plan_v3_curated.json').write_text(json.dumps(plan,indent=2))
    with (out/'dev60_official_folds.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['subject','official_fold','role_fold0_test_fold1_validation'])
        w.writeheader()
        for sid in plan['development_subjects']:
            fold=folds[sid];w.writerow({'subject':sid,'official_fold':fold,'role_fold0_test_fold1_validation':('test' if fold==0 else 'validation' if fold==1 else 'train')})
    counts=[]
    for d in json.loads((a.audit/'combined_diagnostics.json').read_text()):
        r=byvideo[d['video']];sy=np.loadtxt(a.root/r['ppg_sync'],ndmin=2);mr=rows(a.root/r['meta'])
        counts.append({'video':r['video'],'camera':r['camera'],'meta_rows':len(mr),'ppg_sync_rows':len(sy),
             'decoded_frames':d['frames_default'],'sync_equals_meta':len(sy)==len(mr),
             'sync_minus_decoded':len(sy)-d['frames_default'],'sync_std':float(sy[:,0].std()),
             'nearest_dt_max_s':float(sy[:,1].max()),'source_sync_sha256':digest(a.root/r['ppg_sync'])})
    with (out/'sync_lengths36.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(counts[0]));w.writeheader();w.writerows(counts)
    (out/'sync_lengths36.json').write_text(json.dumps(counts,indent=2))
    summary={'publisher_revision':'bf59b3cd1414c2a510d83c3125f16bb87d469ebb','source_db_sha256':original_hash,
      'source_unchanged':digest(a.root/'db.csv')==original_hash,'patched_4952_rows':len(changes),
      'reference_excluded_all':sum(r['reference_eligible']=='false' for r in corrected),
      'reference_excluded_dev60':sum(r['reference_eligible']=='false' for r in dev),
      '4952_in_dev60':'4952' in plan['development_subjects'],
      'dev60_subjects_by_official_fold':dict(Counter(folds[s] for s in plan['development_subjects'])),
      'dev60_videos_by_official_fold':dict(Counter(r['official_fold'] for r in dev)),
      'sync_equal_meta_36':sum(r['sync_equals_meta'] for r in counts),
      'sync_minus_decode_range':[min(r['sync_minus_decoded'] for r in counts),max(r['sync_minus_decoded'] for r in counts)],
      'pulse_hr_label_allowed':False,'train_started':False,'changes':changes}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
