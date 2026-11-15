"""Encode only old training subjects for fixed CV secondary conditions; never test."""
import argparse,json,subprocess,hashlib,tarfile
from pathlib import Path
import imageio_ffmpeg
ROOT=Path('/data/khangle/facevitals');DATA=ROOT/'datasets/UBFC-rPPG'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rate',type=int,choices=[250,1000],required=True);a=ap.parse_args()
    split=json.loads((DATA/'efficientphys36_trainval_v1/subject_split.json').read_text());subjects=split['train'];assert len(subjects)==28 and not set(subjects)&set(split['test'])
    out=DATA/'cv_secondary_inputs_20261020'/f'h264_{a.rate}k';out.mkdir(parents=True,exist_ok=True)
    ff=imageio_ffmpeg.get_ffmpeg_exe();version=subprocess.check_output([ff,'-version'],text=True).splitlines()[0];reports=[]
    for s in subjects:
        dest=out/(s+'.mp4');source=DATA/'DATASET_2'/s/'vid.avi'
        cmd=[ff,'-nostdin','-y','-i',str(source),'-map','0:v:0','-an','-c:v','libx264','-preset','fast','-pix_fmt','yuv420p','-b:v',f'{a.rate}k','-maxrate',f'{a.rate}k','-bufsize',f'{2*a.rate}k','-threads','2','-fps_mode','passthrough',str(dest)]
        if not dest.exists():
            with (out/(s+'_encoder.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True)
        reports.append({'subject':s,'bytes':dest.stat().st_size,'sha256':sha(dest),'command':cmd});print('ENCODED',a.rate,s,flush=True)
    (out/'encoding_report.json').write_text(json.dumps({'rate':a.rate,'encoder_version':version,'videos':reports,'test_opened':False},indent=2))
    archive=DATA/f'cv_secondary_{a.rate}k_20261020.tar'
    with tarfile.open(archive,'w') as tar:tar.add(out,arcname=f'h264_{a.rate}k')
    (out/'COMPLETE.json').write_text(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'subjects':28,'test_opened':False},indent=2))
    print((out/'COMPLETE.json').read_text(),flush=True)
if __name__=='__main__':main()
