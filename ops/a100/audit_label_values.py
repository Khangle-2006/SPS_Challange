import json
from pathlib import Path
import numpy as np
root = Path('/data/khangle/facevitals')
rows = []
for folder in sorted((root/'datasets/UBFC-rPPG/DATASET_2').glob('subject*')):
    gt = np.loadtxt(folder/'ground_truth.txt')
    sensor = gt[1]
    rows.append({'subject':folder.name, 'samples':len(sensor),
                 'sensor_min':float(sensor.min()), 'sensor_max':float(sensor.max()),
                 'sensor_zero_count':int((sensor==0).sum()),
                 'sensor_below_40_count':int((sensor<40).sum()),
                 'sensor_above_180_count':int((sensor>180).sum()),
                 'timestamp_gaps_over_0_2s':int((np.diff(gt[2])>.2).sum())})
out = root/'results/ubfc_pos_green_v1/label_value_audit.json'
out.write_text(json.dumps(rows,indent=2))
print(json.dumps([r for r in rows if r['sensor_below_40_count'] or r['sensor_above_180_count']],indent=2))
