"""Read-only compact status of the ten current CV runs."""
import json
from pathlib import Path
p=Path('/data_pool/khangle/facevitals/results/cv_5fold_20261020')
for k in range(5):
    parts=[]
    for loss in ['mse','spectral']:
        q=p/f'fold{k}'/loss/'status.json'
        if q.exists():
            try:
                r=json.loads(q.read_text());parts.append(f"{loss}:{r['state']}:{r['completed_epochs']}/30")
            except json.JSONDecodeError:parts.append(loss+':writing_status')
        else:parts.append(loss+':queued')
    print('fold'+str(k+1),' '.join(parts))
print('secondary_ready',(p/'SECONDARY_COMPLETE.json').exists(),'evaluation_complete',(p/'COMPLETE.json').exists())
