import json,collections,hashlib,sys
from pathlib import Path
r=Path(sys.argv[1]);i=json.loads((r/'packaging_inputs/training_index.json').read_text());ww=json.loads((r/'frozen_data/FROZEN_WINDOWS.json').read_text())
groups=collections.defaultdict(list)
for w in ww:groups[w['dataset'],w['recording_id'],w['condition'],w['window_start_s']].append(w)
for key,rows in groups.items():
    assert len({x['accepted'] for x in rows})==1,('common support mismatch',key)
    assert len({x['reference_eligible'] for x in rows})==1
    assert len({x['reference_hr_bpm'] for x in rows})==1
counts={}
for ds in ('MCD','UBFC2'):
    rr=[x for x in i['records'] if x['dataset']==ds]
    counts[ds]=dict(persons=len({x['subject_id'] for x in rr}),recordings=len(rr),recording_ids=[x['recording_id'] for x in rr],conditions={})
    if ds=='MCD':assert all('FullHDwebcam' in x['recording_id'] for x in rr);counts[ds]['camera']='FullHDwebcam'
    for c in sorted({x['condition'] for x in ww if x['dataset']==ds}):
        rows=[x for x in ww if x['dataset']==ds and x['condition']==c and x['branch']=='v3_joint' and x['seed']==2027]
        counts[ds]['conditions'][c]=dict(candidates=len(rows),reference_eligible=sum(x['reference_eligible'] for x in rows),accepted=sum(x['accepted'] for x in rows))
commands=json.loads((r/'frozen_data/C5_ENCODING_RECEIPTS.json').read_text())
counts['C5']={ds:dict(recordings=len(rr),mux_fps=sorted({x['mux_fps'] for x in rr}),measured_kbps_range=[min(x['whole_file_kbps'] for x in rr),max(x['whole_file_kbps'] for x in rr)],actual_sensor_fps_range=[min(x['effective_sensor_fps'] for x in rr),max(x['effective_sensor_fps'] for x in rr)]) for ds in ('MCD','UBFC2') if (rr:=[x for x in commands if x['dataset']==ds])}
counts['all_branches_all_seeds_common_support']=True
for f,v in [('COHORT_COUNTS.json',counts),('COMMON_SUPPORT_ASSERTIONS.json',dict(passed=True,unique_windows=len(groups),branches=sorted({x['branch'] for x in ww}),seeds=[2027,2028,2029]))]:
    p=r/f;assert not p.exists();p.write_text(json.dumps(v,indent=2))
print(json.dumps({ds:{k:v for k,v in d.items() if k!='recording_ids'} for ds,d in counts.items() if isinstance(d,dict)},indent=2))
