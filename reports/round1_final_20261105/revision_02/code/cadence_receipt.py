import json,sys
import numpy as np
from pathlib import Path
r=Path(sys.argv[1]);i=json.loads((r/'packaging_inputs/training_index.json').read_text());rows=[]
for rec in i['records']:
    with np.load(Path(i['roots'][rec['dataset']])/rec['recording_id']/'native/crop/meta.npz') as z:t=z['original_t'] if 'original_t' in z else z['t']
    rows.append(dict(dataset=rec['dataset'],recording=rec['recording_id'],native_frames=len(t),native_elapsed_s=float(t[-1]-t[0]),native_source_clock_effective_fps=float((len(t)-1)/(t[-1]-t[0]))))
summary={ds:dict(native_source_clock_fps_range=[min(x['native_source_clock_effective_fps'] for x in rows if x['dataset']==ds),max(x['native_source_clock_effective_fps'] for x in rows if x['dataset']==ds)],native_source_clock_median_fps=float(np.median([x['native_source_clock_effective_fps'] for x in rows if x['dataset']==ds]))) for ds in ('MCD','UBFC2')}
p=r/'SENSOR_CADENCE_CLARIFICATION.json';assert not p.exists();p.write_text(json.dumps(dict(summary=summary,rows=rows,note='Historical encoder effective_sensor_fps and COHORT_COUNTS C5 actual_sensor_fps_range describe selected C5 frames on the original clock, NOT original acquisition cadence. Native source-clock cadence is reported here separately. Sensor hardware clock accuracy is not independently established.'),indent=2));print(json.dumps(summary))
