"""Verified small report inputs, excludes weights and full RGB caches."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path('/data_pool/khangle/facevitals');OUT=ROOT/'results/matched_spectral_20261019';CODE=ROOT/'code/deep_hr'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    done=json.loads((OUT/'COMPLETE.json').read_text());assert done['fresh_runs']==3 and done['paired_verified']
    assert (OUT/'subject32/diagnosis.json').exists()
    frozen=json.loads((OUT/'frozen_checkpoints.json').read_text());checks={}
    for name,item in frozen.items():
        p=Path(item['path']);actual=sha(p);assert actual==item['sha256']
        checks[name]={'sha256':actual,'bytes':p.stat().st_size,'verified':True}
    (OUT/'checkpoint_recheck.json').write_text(json.dumps(checks,indent=2))
    path=ROOT/'results/matched_spectral_analysis_inputs_20261019.tar.gz'
    sources=['train_matched_spectral.py','train_matched_detector_roi.py','spectral_objective.py','train_efficientphys.py','EfficientPhys_fixed.py','hr_baseline.py','evaluate_matched_spectral.py','diagnose_subject32_matched.py','MATCHED_SPECTRAL_PROTOCOL.json','run_matched_spectral.sh','package_matched_spectral.py']
    with tarfile.open(path,'w:gz') as tar:
        for p in OUT.rglob('*'):
            if p.is_file() and p.suffix not in ['.pt','.tmp']:tar.add(p,arcname=str(p.relative_to(ROOT)))
        for name in sources:tar.add(CODE/name,arcname='code/deep_hr/'+name)
    print(json.dumps({'bytes':path.stat().st_size,'sha256':sha(path),'checkpoint_count_verified':len(checks)}),flush=True)
if __name__=='__main__':main()
