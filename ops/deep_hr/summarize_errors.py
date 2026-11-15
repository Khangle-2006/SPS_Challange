import json, statistics
from pathlib import Path
root=Path(__file__).parent/'reports'
subjects=json.loads((root/'error_audit_trainval.json').read_text())
windows=[w for r in subjects for w in r['windows']]
groups={}
for tag in ('train','val'):
    selected=[w for w in windows if w['split']==tag]
    groups[tag]={'n':len(selected),'MAE':statistics.mean(w['abs_pos_error_bpm'] for w in selected),
                 'errors_over5':sum(w['abs_pos_error_bpm']>5 for w in selected),
                 'errors_over20':sum(w['abs_pos_error_bpm']>20 for w in selected)}
for key,predicate in [('gap_over_0_2s',lambda w:w['max_gap_s']>.2),('gap_at_most_0_2s',lambda w:w['max_gap_s']<=.2),
                      ('roi_jump_over20px',lambda w:w['max_roi_center_step_px']>20),('roi_jump_at_most20px',lambda w:w['max_roi_center_step_px']<=20)]:
    selected=[w for w in windows if predicate(w)]
    groups[key]={'n':len(selected),'errors_over5':sum(w['abs_pos_error_bpm']>5 for w in selected),
                 'MAE':statistics.mean(w['abs_pos_error_bpm'] for w in selected) if selected else None}
out={'subjects':len(subjects),'windows':len(windows),'test_accessed':False,'groups':groups,
     'worst_windows':sorted(windows,key=lambda w:w['abs_pos_error_bpm'],reverse=True)[:10]}
(root/'error_summary_trainval.json').write_text(json.dumps(out,indent=2))
print(json.dumps(groups,indent=2))
