"""Prepare, never train: label-blind dev60 split paired to immutable UBFC CV."""
import json,hashlib
from pathlib import Path
import numpy as np

def main():
    root=Path(__file__).resolve().parents[2]
    ub=root/'ops/deep_hr/reports/cv_5fold_20261020/folds_frozen.json'
    cohort=root/'reports/t7_pipeline_20261031/e0_cohort.json'
    u=json.loads(ub.read_text());c=json.loads(cohort.read_text())
    ids=sorted({r['patient_id'] for r in c},key=int)
    assert len(ids)==60 and '4952' not in ids
    assert not any(r['patient_id']=='7412' and r['step']=='after' for r in c)
    seed=20261031;perm=np.random.default_rng(seed).permutation(ids).tolist()
    folds=[]
    for i,f in enumerate(u['folds']):
        mv=perm[12*i:12*(i+1)];mt=[s for s in ids if s not in mv]
        ut=f['train_subjects'];uv=f['val_subjects']
        assert len(mt)==48 and len(mv)==12 and len(ut)==28 and len(uv)==7
        assert set(ut).isdisjoint(uv) and not set(ut+uv)&set(u['original_test_subjects'])
        folds.append(dict(fold=i,mcd_train_subjects=mt,mcd_val_subjects=mv,ubfc_train_subjects=ut,ubfc_val_subjects=uv))
    assert sorted(s for f in folds for s in f['mcd_val_subjects'])==sorted(ids)
    result=dict(status='prepared_only_T7_gate_pending_no_training',assignment_seed=seed,method='NumPy label-blind permutation of numeric-sorted dev60, groups of12; pair with existing UBFC fold index',ubfc_source_sha256=hashlib.sha256(ub.read_bytes()).hexdigest(),mcd_cohort_sha256=hashlib.sha256(cohort.read_bytes()).hexdigest(),original_ubfc_test_subjects=u['original_test_subjects'],folds=folds,training_seeds=[2027,2028,2029],D3_accessed=False)
    out=root/'reports/t8_preparation_20261031';out.mkdir(exist_ok=True)
    p=out/'folds_frozen.json'
    if p.exists():assert json.loads(p.read_text())==result
    else:p.write_text(json.dumps(result,indent=2))
    print('Prepared five folds: MCD48/12, UBFC28/7; seven original test subjects excluded. No training.')

if __name__=='__main__':main()
