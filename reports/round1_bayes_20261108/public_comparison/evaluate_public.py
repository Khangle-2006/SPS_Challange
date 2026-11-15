"""Five-fold public comparison; run only after both feature sets are complete and verified."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,gzip,fcntl,csv,hashlib
from pathlib import Path
import numpy as np
from pipeline_frozen import fit,predict
from features import derive,read,sha,save,now
from metrics_frozen import summarize
P=Path(__file__).resolve().parent
ABLATIONS=['none','no_background','no_codec','no_viterbi','constant_epsilon','no_prior','K1']
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,list):return [serial(v) for v in x]
    return x
def main():
    import worker
    lock=(P/'evaluation_public.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (P/'PUBLIC_EVALUATION_COMPLETE.json').exists()
    for n,h in read(P/'EVALUATION_CODE_HASHES.json').items():assert sha(P/n)==h
    save(P/'EVALUATION_PREFLIGHT.json',worker.guard('P100',True))
    assert (P/'FEATURES_COMPLETE_P100.json').exists()
    # Relay is an exact checked copy of complete A100 features, not a partial count.
    relay=read(P/'A100_RELAY_RECEIPT.json');assert relay['verified_all_files'] and relay['feature_files']==70
    for name,h in relay['files'].items():assert sha(P/name)==h, ('relay hash mismatch',name)
    refs=read(P/'REFERENCE_COMPACT.json');frozen_path=P.parent.parent/'round1_final_20261105/revision_02/frozen_data/FROZEN_WINDOWS.json'
    assert sha(frozen_path)==refs['source_sha256'], 'Frozen reference source changed'
    records=[];baseline={};coverage=[]
    for item in refs['records']:
        p=P/'features'/item['dataset']/(item['recording_id']+'_'+item['variant']+'.json.gz')
        with gzip.open(p,'rt',encoding='utf8') as f:obj=json.load(f)
        assert obj['reference_sha256']==sha(P/'REFERENCE_COMPACT.json') and obj['protocol_sha256']==sha(P/'PROTOCOL.md')
        by={float(r['start']):r for r in obj['rows']};expected={float(r['window_start_s']) for r in item['windows'] if r['reference_eligible']}
        assert set(by)==expected, 'reference window coverage mismatch'
        for b in item['windows']:
            if not b['reference_eligible']:continue
            r=by[float(b['window_start_s'])]
            assert r['truth']==b['reference_hr_bpm'] and r['person']==b['subject_id'] and r['fold']==b['fold']
            baseline[r['condition'],r['recording']]=b['baseline'];records.append(r)
        coverage.append(dict(dataset=item['dataset'],recording=item['recording_id'],condition=item['variant'],all_windows=len(item['windows']),reference_eligible=len(expected),prior_only=sum(not r['candidates'] for r in by.values())))
    rows=[];fits=[]
    for cond in ['native','f10_b500']:
        original=[r for r in records if r['condition']==cond]
        for abl in ABLATIONS:
            transformed=[derive(r,abl) for r in original]
            for fold in range(5):
                train=[r for r in transformed if r['fold']!=fold];val=[r for r in transformed if r['fold']==fold]
                save(P/'EVALUATION_STATUS.json',dict(state='fitting',condition=cond,ablation=abl,fold=fold,completed_fits=len(fits),utc=now()))
                model=fit(train,abl);assert set(model['training_people']).isdisjoint(r['person'] for r in val)
                fits.append(dict(serial(model),condition=cond,fold=fold))
                for r in val:
                    ans=predict(r['candidates'],model)
                    rows.append({k:r[k] for k in ['recording','video','person','dataset','fold','condition','truth','start']}|dict(method='Bayes_'+abl,prediction=ans['hr'],pulse_observed=ans['pulse_observed'],lower90=ans['lower90'],upper90=ans['upper90']))
        for r in original:
            common={k:r[k] for k in ['recording','video','person','dataset','fold','condition','truth','start']}
            const=float(np.median([x['truth'] for x in original if x['fold']!=r['fold']]))
            rows.append(common|dict(method='constant',prediction=const,pulse_observed=None,lower90=None,upper90=None))
            for method,b in baseline[cond,r['recording']].items():
                if b['accepted']:rows.append(common|dict(method=method+'_seed2027' if method in ['v3_joint','ubfc_only'] else method,prediction=b['prediction_hr_bpm'],pulse_observed=None,lower90=None,upper90=None))
    stats=[]
    for ds in ['MCD','UBFC2']:
        for cond in ['native','f10_b500']:
            rr=[r for r in rows if r['dataset']==ds and r['condition']==cond];methods=sorted({r['method'] for r in rr})
            keys=set.intersection(*({r['recording'] for r in rr if r['method']==m} for m in methods))
            constant=[r for r in rr if r['method']=='constant' and r['recording'] in keys];pos=[r for r in rr if r['method']=='POS' and r['recording'] in keys]
            for method in methods:
                selected=[r for r in rr if r['method']==method and r['recording'] in keys];a=summarize(selected)
                a['windows']=a.pop('videos');a['recordings']=len({r['video'] for r in selected})
                dc=summarize(selected,constant);dp=summarize(selected,pos)
                for delta in [dc,dp]:delta['windows']=delta.pop('videos')
                stats.append(dict(dataset=ds,condition=cond,method=method,**a,delta_constant=dc,delta_POS=dp))
    save(P/'PUBLIC_FITS.json',fits);save(P/'PUBLIC_METRICS.json',stats);save(P/'PUBLIC_COVERAGE.json',coverage)
    with (P/'PUBLIC_PREDICTIONS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(P/'PUBLIC_EVALUATION_COMPLETE.json',dict(utc=now(),folds=5,fits=len(fits),automatic_promotion=False,holdout_evaluations=0,scope='10-second matched public windows; Viterbi inactive by design'))
if __name__=='__main__':main()
