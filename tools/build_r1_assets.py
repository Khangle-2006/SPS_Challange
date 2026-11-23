"""Export R1 parameters with byte-identical, previously validated CNN checkpoints. No fit/data reads."""
import argparse,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(b): return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous-archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 old=json.loads((ROOT/'configs/inference-assets-20261123.json').read_text());assert sha(a.previous_archive.read_bytes())==old['sha256']
 a.output.mkdir(parents=True,exist_ok=False);name='facevitals-inference-assets-r1-20261124';data={};provenance=[]
 with zipfile.ZipFile(a.previous_archive) as z:
  prefix='facevitals-inference-assets-20261123/';m=json.loads(z.read(prefix+'MANIFEST.json'));by={x['path']:x for x in m['files']}
  names=[f'checkpoints/padding_fixed_fold{f}_seed2027.pt' for f in range(5)]+['code/EfficientPhys_fixed.py','code/replicate_tsm.py','code/public_fixed_math.py','TOOLBOX_LICENSE.txt']
  for n in names:
   b=z.read(prefix+n);assert sha(b)==by[n]['sha256'];data[n]=b
   provenance.append({'path':n,'previous_release_sha256':sha(b),'transformation':'None; identical bytes'})
 for n in ['btc_reader_v4_deploy.json','btc_reader_v4_folds.json','public_p4m_r1.json']:
  data['readers/'+n]=(ROOT/'models'/n).read_bytes()
 for n in ['reader_v4.py','reader_models_fixed.py','btc_spectral_math.py']:
  data['code/'+n]=(ROOT/'runtime'/n).read_bytes()
 cfg={'schema_version':1,'expert_mode':'selfwarm','cnn':'EfficientPhys frame_depth=10 img_size=36 replicate TSM; final epoch30 seed2027, matching person fold','expert_package':'open-rppg==0.1.1 (weights installed by bootstrap; no repackaging)','expert_state':'reset initial state per video, process same video twice, use second pass','btc_deploy_available':True,'fold_models_for_oof_only':True,'public_parameters':'public_p4m_r1.json; w_*/ro_* as defined in repository docs/PIPELINE.md','holdout_reads':0,'no_training_or_refit':True,'previous_archive_sha256':old['sha256']}
 data['CONFIG.json']=(json.dumps(cfg,indent=2)+'\n').encode()
 data['README.md']=b'R1 assets, 24 November 2026. Use with the matching SPS_Challange checkout. Five unchanged CNN checkpoints and current BTC deploy/fold/public parameters. See docs/INFERENCE_ASSETS.md. No participant data. Expert weights come from the pinned open-rppg package. OOF needs the correct fold; deployment is not OOF.\n'
 rows=[{'path':n,'bytes':len(b),'sha256':sha(b)} for n,b in sorted(data.items())]
 data['MANIFEST.json']=(json.dumps({'schema_version':1,'release':'inference-assets-r1-20261124','files':rows,'provenance':provenance},indent=2)+'\n').encode()
 archive=a.output/(name+'.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for n,b in sorted(data.items()): z.writestr(name+'/'+n,b)
 with zipfile.ZipFile(archive) as z: assert z.testzip() is None
 h=sha(archive.read_bytes());(a.output/'SHA256SUMS.txt').write_text(h+'  '+archive.name+'\n')
 release={'tag':'inference-assets-r1-20261124','asset':archive.name,'root':name,'url':'https://github.com/Khangle-2006/SPS_Challange/releases/download/inference-assets-r1-20261124/'+archive.name,'sha256':h,'bytes':archive.stat().st_size,'files':len(data),'expert_mode':'selfwarm','btc_all_dev_deploy_model_available':True,'previous_release_unchanged':True}
 (a.output/'RELEASE.json').write_text(json.dumps(release,indent=2)+'\n');print(json.dumps(release))
if __name__=='__main__': main()
