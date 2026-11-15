import csv,json,hashlib
from pathlib import Path
import av,cv2
ROOT=Path('/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG')
out=Path('/mnt/samsung_t5/khangle/facevitals/storage_pilot_20261030')
result=[]
for r in list(csv.DictReader((ROOT/'db.csv').open()))[:6]:
    p=ROOT/r['video']
    if not p.exists():continue
    lines=(ROOT/r['meta']).read_text().splitlines()
    with av.open(str(p)) as c:
        s=c.streams.video[0];d={'video':r['video'],'meta':r['meta'],'metadata_rows':len(lines),'meta_first':lines[:2],'meta_last':lines[-2:],
            'declared_frames':s.frames,'fps':str(s.average_rate),'codec':s.codec_context.name,'time_base':str(s.time_base),'bytes':p.stat().st_size}
        n=0;first=None;last=None
        for f in c.decode(video=0):
            t=str(f.pts*f.time_base);first=t if n==0 else first;last=t;n+=1
        d.update(pyav_frames=n,pts_first=first,pts_last=last)
    cap=cv2.VideoCapture(str(p));n=0
    while cap.grab():n+=1
    cap.release();d['opencv_frames']=n
    result.append(d);print(json.dumps(d),flush=True)
(out/'clock_audit.json').write_text(json.dumps(result,indent=2))
