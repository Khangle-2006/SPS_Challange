"""Freeze subject folds and primary cache views before training; no test data reads."""
import json
import numpy as np
from cv_common import ROOT,DATA,CLEAN,WORK,OUT,SEED,read,write,sha,CVClips
from train_efficientphys import Clips as OriginalClips
def main():
    WORK.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    if (OUT/'PREFLIGHT_COMPLETE.json').exists():raise RuntimeError('Preflight already frozen; refusing regeneration')
    split=read(CLEAN/'subject_split.json');pool=sorted(split['train']+split['val'],key=lambda s:int(s[7:]));test=split['test']
    assert len(set(pool))==35 and len(set(test))==7 and not set(pool)&set(test)
    rows=read(CLEAN/'clips.json');assert len(rows)==203 and {r['subject'] for r in rows}==set(pool)
    rows=sorted(rows,key=lambda r:(int(r['subject'][7:]),r['window']))
    perm=np.random.default_rng(SEED).permutation(pool).tolist();folds=[]
    for k in range(5):
        val=perm[k*7:(k+1)*7];train=[s for s in pool if s not in val]
        assert len(train)==28 and len(val)==7 and not set(train)&set(val)
        folds.append({'fold':k,'train_subjects':train,'val_subjects':val})
    assert sorted(s for f in folds for s in f['val_subjects'])==sorted(pool)
    write(WORK/'folds.json',{'assignment_seed':SEED,'method':'label-blind NumPy permutation of numerically sorted subjects, contiguous groups of7','original_test_subjects':test,'pool_subjects':pool,'folds':folds,'rows':rows})
    definitions={'clean':{s:CLEAN for s in pool},
        'native_500k':{s:DATA/('efficientphys36_train_h264_500k_v1' if s in split['train'] else 'efficientphys36_val_h264_500k_v1') for s in pool},
        'detected_500k':{s:DATA/('detected_on_h264_train500_20261019' if s in split['train'] else 'detected_on_h264_20261019/h264_500k') for s in pool}}
    manifests={};inventory_hashes={}
    for condition,mapping in definitions.items():
        view=WORK/'cache'/condition;view.mkdir(parents=True,exist_ok=True);manifests[condition]={}
        for s,source in mapping.items():
            inv=read(source/'inventory.json');inventory_hashes[str(source)]=sha(source/'inventory.json');manifests[condition][s]={}
            for suffix in ['_rgb.npy','_meta.npz']:
                name=s+suffix;p=source/name;expected=inv[name];assert p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256']
                dest=view/name
                if dest.exists():assert dest.resolve()==p.resolve()
                else:dest.symlink_to(p)
                manifests[condition][s][suffix]={'path':str(p),'bytes':expected['bytes'],'sha256':expected['sha256']}
            a=np.load(CLEAN/(s+'_meta.npz'));b=np.load(source/(s+'_meta.npz'))
            for key in ['t','ppg','original_t']:np.testing.assert_array_equal(a[key],b[key])
    write(WORK/'primary_manifest.json',{'conditions':manifests,'source_inventory_sha256':inventory_hashes,'test_opened':False})
    # Both the old train and old validation subjects can now train in some folds.
    # Check unchanged interpolation/normalization against the original loader for every clean sample.
    check=CVClips(WORK/'cache/clean',rows);assert not check.invalid
    legacy={tag:OriginalClips(CLEAN,tag) for tag in ['train','val']}
    indices={tag:{(r['subject'],r['window']):i for i,r in enumerate(ds.rows)} for tag,ds in legacy.items()}
    for i,r in enumerate(rows):
        x,y,_=check[i];a,b,_=legacy[r['split']][indices[r['split']][(r['subject'],r['window'])]]
        np.testing.assert_array_equal(x.numpy(),a.numpy());np.testing.assert_array_equal(y.numpy(),b.numpy())
    coverage={}
    for c in definitions:
        ds=CVClips(WORK/'cache'/c,rows);coverage[c]={'windows':len(ds),'unusable':len(ds.invalid)}
        if c in ['clean','native_500k']:assert not ds.invalid
    sources=['cv_common.py','prepare_cv_primary.py','train_cv_loss.py','spectral_objective.py','train_efficientphys.py','EfficientPhys_fixed.py','hr_baseline.py']
    protocol={'folds':5,'seed':SEED,'fresh_runs':10,'epochs':30,'checkpoint':'final epoch30 ONLY; no heldout evaluation during training; no early stopping',
        'losses':['MSE','MSE +0.1 KL(target PSD||predicted PSD)'],'training':'50% clean/50% native-derived H264500, batch2, AdamW wd0, OneCycleLR max0.009',
        'primary':'equal mean clean and actual-detector H264500 out-of-fold subject-macro MAE, pool35 subjects once each',
        'secondary':'actual-detector H264250/1000, no ensemble across folds (would include models trained on heldout subject)',
        'keep_gate':{'primary_relative_improvement_min':.10,'clean_mean_increase_max':.25,'improved_fold_count_min':3,'h264500_pooled_P95_increase_max':1.,'h264500_over20_increase_max':0},
        'invalid_policy':'all heldout windows retained, invalid135bpm; no condition-specific dropping',
        'folds_sha256':sha(WORK/'folds.json'),'primary_manifest_sha256':sha(WORK/'primary_manifest.json'),
        'code_sha256':{s:sha(ROOT/'code/deep_hr'/s) for s in sources},'test_opened':False,
        'limits':'Development CV of previously used35 subjects, not independent external validation. Single seed; fold scores correlated because train sets overlap. No bitwise CUDA determinism claim.'}
    write(OUT/'protocol.json',protocol);write(OUT/'PREFLIGHT_COMPLETE.json',{'coverage':coverage,'exact_clean_loader_equivalence_windows':203,'test_data_opened':False,'folds_sha256':protocol['folds_sha256']})
    print(json.dumps({'coverage':coverage,'folds':folds,'protocol_sha256':sha(OUT/'protocol.json')},indent=2),flush=True)
if __name__=='__main__':main()
