"""Apply frozen preprocessing to the seven held-out test subjects."""
import json, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import prepare_efficientphys as prep
if __name__=='__main__':
    prep.OUT=prep.R/'datasets/UBFC-rPPG/efficientphys36_test_v1'
    prep.OUT.mkdir(parents=True,exist_ok=True)
    (prep.OUT/'subject_split.json').write_text(json.dumps(prep.SPLIT,indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(prep.run,s,('test',)) for s in prep.SPLIT['test']]
        for f in as_completed(jobs):
            r=f.result();results.append(r)
            print(json.dumps({'subject':r['subject'],'frames':r['frames'],'clips':len(r['clips'])}),flush=True)
    assert len(results)==7
    (prep.OUT/'clips.json').write_text(json.dumps([c for r in results for c in r['clips']],indent=2))
    # Do not use or export test error diagnostics to guide any implementation change.
    inventory={}
    for f in sorted(prep.OUT.glob('*')):
        if f.name=='inventory.json':continue
        with f.open('rb') as stream:
            inventory[f.name]={'bytes':f.stat().st_size,'sha256':hashlib.file_digest(stream,'sha256').hexdigest()}
    (prep.OUT/'inventory.json').write_text(json.dumps(inventory,indent=2))
    print('Test input cache complete; no learned-model predictions made.')
