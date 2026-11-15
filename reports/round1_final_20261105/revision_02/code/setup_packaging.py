import json,hashlib,shutil,sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]);base=Path('/mnt/samsung_t5/khangle/facevitals');code=Path('/data_pool/khangle/facevitals/code')
vendor=r/'code/vendor';vendor.mkdir(exist_ok=False)
for source,name in [(code/'t8_polarity_v3_20261102','v3'),(code/'t8_oval_20261031','oval'),(base/'reports/t4_submission_20261030/mcd_e2e_v1/artifact/code/submission','submission')]:
    assert source.is_dir(),source
    dest=vendor/name;dest.mkdir()
    for p in source.iterdir():
        if p.suffix in ('.py','.json','.txt','.md'):shutil.copy2(p,dest/p.name)
shutil.copy2(code/'hr_baseline.py',vendor/'hr_baseline.py')
idxpath=base/'t8_oval_20261031/training_index.json';idx=json.loads(idxpath.read_text())
inp=r/'packaging_inputs';inp.mkdir()
shutil.copy2(idxpath,inp/'training_index.json')
windows=json.loads((r/'frozen_data/FROZEN_WINDOWS.json').read_text())
for ds,rid,pref in [('MCD','1115_FullHDwebcam_after','mcd'),('UBFC2','subject1','ubfc')]:
    rec=next(x for x in idx['records'] if x['dataset']==ds and x['recording_id']==rid)
    fold=next(f['fold'] for f in idx['folds']['folds'] if rec['subject_id'] in f[pref+'_val_subjects'])
    ck=base/f't8_polarity_v3_20261102/fold{fold}/seed2027/joint/final.pt'
    dst=inp/ds;dst.mkdir();shutil.copy2(ck,dst/'final.pt')
    (dst/'checkpoint_manifest.json').write_text(json.dumps(dict(branch='v3_joint',fold=fold,seed=2027,recording=rid,checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest()),indent=2))
    (dst/'oof_windows.json').write_text(json.dumps([x for x in windows if x['recording_id']==rid and x['dataset']==ds and x['branch']=='v3_joint' and x['seed']==2027]))
hashes={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in vendor.rglob('*') if p.is_file()}
(r/'PACKAGING_SOURCE_HASHES.json').write_text(json.dumps(hashes,indent=2))
with tarfile.open(r/'packaging_code.tar.gz','w:gz') as t:
    for p in [r/'code',inp,r/'PACKAGING_SOURCE_HASHES.json']:t.add(p,arcname=p.name,filter=lambda x:None if '__pycache__' in x.name else x)
print('PACKAGING_SETUP_COMPLETE')
