from pathlib import Path
import json, subprocess, hashlib

root=Path('/mnt/samsung_t5/khangle/facevitals/eda_examples_20261108')
root.mkdir(exist_ok=True)
specs=json.loads((root/'specs.json').read_text())
for row in specs:
    src=Path(row['source']); dst=root/row['preview']
    assert src.is_file(),src
    if not dst.exists():
        subprocess.run(['ffmpeg','-v','error','-nostdin','-threads','1','-i',str(src),'-t','10','-an','-vf',"scale=w='min(480,iw)':h=-2",'-c:v','libx264','-threads','1','-preset','fast','-crf','25','-pix_fmt','yuv420p','-movflags','+faststart',str(dst)],check=True)
    row['bytes']=dst.stat().st_size
    row['sha256']=hashlib.sha256(dst.read_bytes()).hexdigest()
    row['preview_note']='First 10 seconds, scaled to <=480 pixels wide, H264, no audio. Illustration only, not an analysis input.'
    row['probe']=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,avg_frame_rate:format=duration','-of','json',str(dst)]))
    print(row['preview'],row['bytes'],flush=True)
(root/'MANIFEST.json').write_text(json.dumps(specs,ensure_ascii=False,indent=2))
