"""Post-hoc negative control for multiple-peak matching, train data only."""
import json,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE/'reports/train_signal_diagnosis_20261019'
P=OUT/'results/train_compression_diagnosis_20261019/results.json'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    protocol={'seed':20271019,'replicates':1000,'halfwidth_bpm':6,'source_results_sha256':sha(P),
       'design':'For each train window independently draw an alternative HR from a different train subject, with replacement; use same draw for global and nine-patch peak matching. Preserve empirical HR distribution approximately. All windows retained.',
       'interpretation':'Descriptive background-match control, post-hoc after seeing multiple-peak results. Not a null experiment for absence of a physiological pulse, confidence interval, significance test or estimator; no tuning against results.'}
    (OUT/'matching_control_protocol.json').write_text(json.dumps(protocol,indent=2))
    data=json.loads(P.read_text());rows=data['records'];assert data['complete'] and len(rows)==161 and all(r['split']=='train' for r in rows)
    rng=np.random.default_rng(protocol['seed']);hr=np.array([r['gt_hr'] for r in rows]);subjects=np.array([r['subject'] for r in rows]);null=np.empty((1000,len(rows)))
    for i,r in enumerate(rows):null[:,i]=rng.choice(hr[subjects!=r['subject']],size=1000,replace=True)
    result={'protocol':protocol,'conditions':{}}
    for c in ['clean','h264_500k']:
        out={}
        for kind in ['global','nine_patches']:
            peakers=[r['input'][c]['pos']['top3'] if kind=='global' else [p for patch in r['input'][c]['patch_pos'] for p in patch['top3']] for r in rows]
            true=np.array([any(abs(p['bpm']-r['gt_hr'])<=6 for p in peaks) for r,peaks in zip(rows,peakers)])
            counts=np.zeros(1000)
            for i,peaks in enumerate(peakers):
                counts+=np.any(np.abs(null[:,i,None]-np.array([p['bpm'] for p in peaks])[None,:])<=6,axis=1)
            out[kind]={'actual_reference_hits':int(true.sum()),'actual_fraction':float(true.mean()),'reassigned_reference_mean_hits':float(counts.mean()),
                'reassigned_fraction_mean':float(counts.mean()/161),'reassigned_hits_p025_p975':np.percentile(counts,[2.5,97.5]).tolist()}
        result['conditions'][c]=out
    cases=json.loads((OUT/'severe_train_cases.json').read_text())
    result['severe_unique_windows']={family:sorted({(r['subject'],r['window']) for r in cases if r['family']==family}) for family in ['mse','spectral']}
    (OUT/'matching_control_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
