"""Header-based dev-only decode smoke; no PR or predictions used."""
import json,subprocess
from pathlib import Path
import numpy as np
from preprocess import Video,HERE,save,sha256
from oval_crop import OvalCrop
rows=json.loads((HERE/'dev_manifest.json').read_text())['videos']
selected={}
for label,predicate in [('HDR',lambda m:m['transfer'] in ['arib-std-b67','smpte2084']),('rotation',lambda m:float(m['rotation'])!=0),('240fps',lambda m:float(m['avg_fps'])>200),('19.99fps',lambda m:abs(float(m['avg_fps'])-19.99)<.15)]:
    candidates=[r for r in rows if predicate(r['media'])]
    if candidates:selected[label]=candidates[0]
results={}
for label,row in selected.items():
    source=Video(row['path'],HERE/(label+'_smoke_decode.log'))
    command=source.command[:-1]+['-frames:v','1',source.command[-1]]
    raw=subprocess.check_output(command)
    rgb=np.frombuffer(raw,np.uint8).reshape(source.height,source.width,3)
    det=OvalCrop()
    try:good,crop,_,_=det.step(rgb)
    finally:det.close()
    results[label]=dict(recording=row['recording'],person=row['person'],PTS_count=len(source.t),PTS_span=float(source.t[-1]),output_shape=list(rgb.shape),
        hdr=source.hdr,first_frame_detected=bool(good),finite_uint8=bool(np.isfinite(rgb).all()),source_sha256=row['sha256'],command=source.command)
save(HERE/'MEDIA_SMOKE.json',dict(pass_decode=True,dev_only=True,categories_without_dev_example=sorted(set(['HDR','rotation','240fps','19.99fps'])-selected.keys()),results=results))
print(json.dumps(results,indent=2))
