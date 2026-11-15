from pathlib import Path
import json,shutil,hashlib
root=Path(__file__).resolve().parents[2];out=root/'ops/t8_20261031';out.mkdir(exist_ok=True)
for name in ['prepare_t8_cache.py','t8_t1_loader.py','data_adapters.py','window_support.py','prepare_recording.py','video_roi_frontend.py']:
    shutil.copy2(root/'ops/deep_hr'/name,out/name)
shutil.copy2(root/'ops/constraints/simulate.py',out/'simulate.py')
shutil.copy2(root/'reports/t8_preparation_20261031/PROTOCOL.txt',out/'PROTOCOL.txt')
shutil.copy2(root/'reports/t8_preparation_20261031/folds_frozen.json',out/'folds_frozen.json')
shutil.copy2(root/'reports/t7_pipeline_20261031/e0_cohort.json',out/'mcd_cohort.json')
p=json.loads((root/'ops/constraints/protocol.json').read_text())
p.update(version='T8-cache-v1-20261031',conditions=['native']+[f'f{f}_b{b}' for f in [10,15,30] for b in [300,500,800]],decoder={'band_hz':[.7,3.0],'implementation':'frozen Round1 argmax; old T5 scores not imported'})
p['windows']['reference_decoder']='Round1 fixed0.7-3Hz; independent T1 canonical clock'
p['storage']={'combined_budget_decimal_GB':60,'MCD_budget_decimal_GB':50,'UBFC2_budget_decimal_GB':10,'preserve_raw':True}
p['cohort']['UBFC2']=35;p['report']='T8 cache only; no inference scores; every canonical window retained with support status.'
(out/'protocol.json').write_text(json.dumps(p,indent=2))
hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file() and f.name!='SHA256.json'}
(out/'SHA256.json').write_text(json.dumps(hashes,indent=2));print(out)
