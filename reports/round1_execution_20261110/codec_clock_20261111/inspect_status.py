"""Light read-only snapshot; count encoded RESULT bf!=None, not before folders."""
import json,os,time
from pathlib import Path
P=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
workers=read(P/'WORKERS_V2.json')['workers'];out={'workers':[]}
if (P/'WORKER_V3.json').exists():workers=[dict(mode='real',**read(P/'WORKER_V3.json'))]+[w for w in workers if w['mode']=='synthetic']
if (P/'WORKER_V4.json').exists():workers=[dict(mode='real',**read(P/'WORKER_V4.json'))]+[w for w in workers if w['mode']=='synthetic']
for w in workers:
 proc=Path('/proc')/str(w['pid']);out['workers'].append(dict(w,alive=proc.exists(),cmdline=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None))
for name,pattern,total in [('real','*/*/f*/RESULT.json',72),('synthetic','*/*/RESULT.json',120)]:
 files=list((P/name).glob(pattern));rs=[read(p) for p in files];out[name]={'encoded_complete':sum(r['bf'] is not None for r in rs),'source_controls':sum(r['bf'] is None for r in rs),'planned_encoded':total,'complete_receipt':(P/(name.upper().replace('SYNTHETIC','SYNTH')+'_COMPLETE.json')).exists()}
out['historical_errors']=[p.name for p in P.glob('*_V2_ERROR.json')]+[p.name for p in P.glob('*_V3_ERROR.json')]
out['current_v4_errors']={p.name:read(p) for p in P.glob('*_V4_ERROR.json')}
out['excluded_sources']=[str(p.parent.relative_to(P/'real')) for p in (P/'real').glob('*/*/EXCLUDED.json')]
out['current']={k:v for k,v in read(P/'REAL_STATUS.json').items() if k not in ['completed_encodes','state']}
out['real']['expected_encoded_after_exclusions']=72-6*len(out['excluded_sources'])
out['bytes']=sum(p.stat().st_size for p in P.rglob('*') if p.is_file());print(json.dumps(out))
