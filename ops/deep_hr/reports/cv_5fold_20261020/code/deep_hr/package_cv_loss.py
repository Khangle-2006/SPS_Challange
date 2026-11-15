"""Small auditable CV report package; preserve weights and RGB on server."""
import json,tarfile
from pathlib import Path
from cv_common import ROOT,WORK,OUT,read,write,sha,roots
def main():
    done=read(OUT/'COMPLETE.json');assert done['runs']==10 and done['paired_verified'] and done['final_epoch_only']
    frozen=read(OUT/'frozen_checkpoints.json');recheck={}
    for key,item in frozen.items():
        p=Path(item['path']);actual=sha(p);assert actual==item['sha256'];recheck[key]={'sha256':actual,'bytes':p.stat().st_size,'verified':True}
    write(OUT/'checkpoint_recheck.json',recheck)
    encoded=ROOT/'datasets/UBFC-rPPG/cv_secondary_inputs_20261020'
    write(OUT/'secondary_encoding_reports.json',{f'h264_{rate}k':read(encoded/f'h264_{rate}k/encoding_report.json') for rate in [250,1000]})
    path=ROOT/'results/cv_5fold_analysis_inputs_20261020.tar.gz'
    sources=set(read(OUT/'protocol.json')['code_sha256'])|set(read(OUT/'evaluation_manifest.json')['code_sha256'])|{'run_cv_folds.sh','finish_cv.sh','package_cv_loss.py','encode_cv_secondary.py','encode_cv_secondary.sbatch'}
    with tarfile.open(path,'w:gz') as tar:
        for p in OUT.rglob('*'):
            if p.is_file() and p.suffix not in ['.pt','.tmp']:tar.add(p,arcname=str(p.relative_to(ROOT)))
        for name in ['folds.json','primary_manifest.json','secondary_manifest.json']:tar.add(WORK/name,arcname='inputs/'+name)
        for c,root in roots().items():
            for s in read(WORK/'folds.json')['pool_subjects']:
                p=root/(s+'_meta.npz');tar.add(p.resolve(),arcname=f'inputs/meta/{c}/{p.name}')
        for name in sorted(sources):tar.add(ROOT/'code/deep_hr'/name,arcname='code/deep_hr/'+name)
    write(OUT/'package_manifest.json',{'bytes':path.stat().st_size,'sha256':sha(path),'verified_checkpoints':len(recheck),'test_opened':False})
    print(json.dumps(read(OUT/'package_manifest.json')),flush=True)
if __name__=='__main__':main()
