"""Read-only source evaluation; matched classical/constant baselines on frozen oval caches."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse, csv, hashlib, importlib.util, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

p=argparse.ArgumentParser()
for n in ('code','controls','joint','index','output','classical_code'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.code))
from t8_plan import read,write,sha,SEEDS
from score_round1 import reference_hr,stats,key
from t8_t1_loader import supported
from spectral import snr
from polarity_guard import EXPECTED_INDEX
assert sha(a.index)==EXPECTED_INDEX
i=read(a.index);folds=i['folds']['folds']
split_checks=[]
for ds,prefix,n in [('UBFC2','ubfc',35),('MCD','mcd',60)]:
    seen=set()
    for f in folds:
        tr,ev=map(set,(f[prefix+'_train_subjects'],f[prefix+'_val_subjects']))
        assert tr.isdisjoint(ev),f'PERSON LEAKAGE {ds} fold{f["fold"]}'
        assert seen.isdisjoint(ev),'Repeated OOF person'
        seen|=ev;split_checks.append(dict(dataset=ds,fold=f['fold'],train=sorted(tr),eval=sorted(ev),disjoint=True))
    assert len(seen)==n
a.output.mkdir(parents=True,exist_ok=True)
assert not (a.output/'FROZEN_WINDOWS.json').exists(),'Preserve existing results'
write(a.output/'SPLIT_ASSERTIONS.json',split_checks)
# Load only function source from existing trusted local modules, without their CLI imports.
import ast
namespace=dict(np=np,json=json)
import math
from scipy import signal
namespace.update(math=math,signal=signal)
source_receipts={}
for name,file in [('pos','hr_baseline.py'),('chrom','t7_pipeline.py')]:
    src=(a.classical_code/file).read_text();tree=ast.parse(src)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(a.classical_code/file),'exec'),namespace)
    source_receipts[file]=sha(a.classical_code/file)

conditions={'UBFC2_f10_b500':'UBFC C5','MCD_native':'MCD native','UBFC2_native':'UBFC clean','MCD_f10_b500':'MCD-C5'}
windows=[];canonical={};artifacts=[]
for f in range(5):
    for seed in SEEDS:
        for arm,root,label in [('ubfc_only',a.controls,'ubfc_only'),('joint',a.joint,'v3_joint')]:
            run=root/f'fold{f}'/f'seed{seed}'/arm
            state=read(run/'status.json');prov=read(run/'provenance.json')
            assert state['state']=='complete' and state['completed_epochs']==30
            assert sha(run/'final.pt')==state['checkpoint_sha256']
            assert prov['index_sha256']==EXPECTED_INDEX
            assert [r['epoch'] for r in read(run/'history.json')]==list(range(1,31))
            artifacts.append(dict(fold=f,seed=seed,branch=label,checkpoint=str(run/'final.pt'),checkpoint_sha256=state['checkpoint_sha256'],provenance_sha256=sha(run/'provenance.json')))
            for c,labelc in conditions.items():
                d=run/'evaluation'/c;m=read(d/'metrics.json')
                assert m['reference_sha256']==sha(d/'reference_signal.csv')
                assert m['output_sha256']==sha(d/'processed_output.csv')
                assert m['scorer_sha256']==sha(a.code/'score_round1.py')
                if arm=='joint':
                    u=a.controls/f'fold{f}'/f'seed{seed}'/'ubfc_only'/'evaluation'/c
                    r=read(d/'REFERENCE_POLARITY.json')
                    assert sha(u/'reference_signal.csv')==sha(d/'raw_reference_signal.csv')==r['raw_reference_sha256']
                    assert m['reference_sha256']==r['canonical_reference_sha256'] and r['all_hr_and_failure_parity']
                for r in m['windows']:
                    w=dict(r,fold=f,seed=seed,branch=label,condition=labelc,dataset=c.split('_')[0])
                    windows.append(w)
                    ident=(labelc,r['subject_id'],r['recording_id'],r['window_start_s'])
                    ref={k:r[k] for k in ('reference_hr_bpm','reference_eligible','reference_reasons')}
                    if ident in canonical:assert canonical[ident]==ref,'Reference outcomes changed'
                    else:canonical[ident]=ref
write(a.output/'CHECKPOINT_MANIFEST.json',artifacts)

# Only native training people contribute to constants; all labels remain on their own clock.
native=defaultdict(list)
for (c,sid,rid,start),r in canonical.items():
    if c in ('MCD native','UBFC clean') and r['reference_eligible']:
        native[sid].append(r['reference_hr_bpm'])
constants={}
for f in folds:
    for branch,datasets in [('constant_ubfc',[('UBFC2','ubfc')]),('constant_joint',[('UBFC2','ubfc'),('MCD','mcd')])]:
        ids=[ds+':'+sid for ds,prefix in datasets for sid in f[prefix+'_train_subjects']]
        assert all(native[sid] for sid in ids)
        constants[f['fold'],branch]=float(np.mean([np.mean(native[sid]) for sid in ids]))
write(a.output/'CONSTANT_HR_FIT.json',[dict(fold=f,branch=b,train_person_mean_hr=v) for (f,b),v in constants.items()])

classical={};failures=[];commands=[];pos_snr=[]
baseline_lookup={(r['condition'],r['subject_id'],r['recording_id'],r['window_start_s']):r for r in windows if r['seed']==2027 and r['branch']=='ubfc_only'}
for record in i['records']:
    ds,rid=record['dataset'],record['recording_id'];sid=ds+':'+record['subject_id']
    for variant in ('native','f10_b500'):
        labelc=conditions[ds+'_'+variant];folder=Path(i['roots'][ds])/rid/variant/'crop'
        encoded=read(folder.parent/'complete.json')
        if variant=='f10_b500':
            e=encoded['encoding'];assert e['fps_requested']==10 and e['bitrate_pass'] and e['actual']['whole_file_kbps']<=500
            commands.append(dict(dataset=ds,recording=rid,receipt_sha256=sha(folder.parent/'complete.json'),command=e['command'],effective_sensor_fps=e['effective_sensor_fps'],mux_fps=e['mux_fps'],whole_file_kbps=e['actual']['whole_file_kbps']))
        arr=np.load(folder/'rgb.npy',mmap_mode='r')
        rgb=np.concatenate([np.asarray(arr[k:k+128],float).mean(axis=(1,2)) for k in range(0,len(arr),128)])
        with np.load(folder/'meta.npz') as meta:
            t,valid=meta['t'].copy(),meta['valid'].copy()
            sd=a.output/'spatial_rgb'/ds/rid
            sd.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(sd/(variant+'.npz'),rgb=rgb,t=t,valid=valid,ppg_t=meta['ppg_t'],ppg=meta['ppg'],ppg_valid=meta['ppg_valid'])
        for w in record['candidates']:
            start=float(w['start_s']);ident=(labelc,sid,rid,start)
            # Exactly the prediction support accepted by both completed model branches.
            reference=canonical[ident]
            baseline=baseline_lookup[ident]
            common=baseline['accepted']
            predictions={}
            if common:
                q=start+np.arange(300)/30
                assert supported(t,valid,start,start+10),'Accepted source has invalid image support'
                x=np.stack([np.interp(q,t,rgb[:,ch]) for ch in range(3)],axis=1)
                for method in ('POS','CHROM'):
                    try:
                        wave=namespace[method.lower()](x,30)
                        hr=reference_hr(dict(window_start_s=0.,ppg_timestamps_json=json.dumps((np.arange(301)/30).tolist()),ppg_values_json=json.dumps(np.r_[wave,wave[-1]].tolist())))
                        predictions[method]=hr
                        if method=='POS':pos_snr.append(dict(condition=labelc,person=sid,recording=rid,start=start,hr=hr,snr=snr(wave)))
                    except ValueError as exc:
                        failures.append(dict(condition=labelc,person=sid,recording=rid,start=start,method=method,error=str(exc)));predictions[method]=None
            classical[ident]=predictions
        print('Classical',ds,rid,flush=True)
base=[r for r in windows if r['branch']=='ubfc_only']
for r in base:
    ident=(r['condition'],r['subject_id'],r['recording_id'],r['window_start_s'])
    for branch in ('constant_ubfc','constant_joint','POS','CHROM'):
        prediction=constants[r['fold'],branch] if branch.startswith('constant') else classical[ident].get(branch)
        accepted=bool(r['accepted'] and prediction is not None)
        windows.append(dict(r,branch=branch,prediction_hr_bpm=prediction if accepted else None,accepted=accepted,prediction_reason=None if accepted else 'matched_support_or_classical_failure'))
write(a.output/'CLASSICAL_FAILURES.json',failures)
write(a.output/'CLASSICAL_SOURCE_HASHES.json',source_receipts)
write(a.output/'C5_ENCODING_RECEIPTS.json',commands)
write(a.output/'POS_SNR.json',pos_snr)
write(a.output/'FROZEN_WINDOWS.json',windows)
print('COMPLETE',len(windows),'rows; classical failures',len(failures),flush=True)
