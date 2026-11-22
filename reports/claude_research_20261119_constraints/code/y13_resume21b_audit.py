"""Read-only result audit and in-W delivery packaging, after queue terminal."""
import json,hashlib,zipfile,time
from pathlib import Path
import numpy as np
W=Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints');L=W/'logs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((L/'y13_resume21b_COMPLETE.json').read_text())
assert s['mode']=='complete'
for p,h in s['frozen'].items():assert sha(Path(p))==h,p
prior=json.loads((L/'y13_resume21b_EXISTING_OUTPUTS.json').read_text())
for p,a in prior.items():assert sha(Path(p))==a['sha'],p
models=['FacePhys.rlap','ME-chunk.pure','ME-flow.pure']
conds=['c5_split10','c4_qvga','c6_us1','aidtrack_c1_g25gv01','c2_vit','track_c1_g25p','track_c7_blue','c1_g50p','c1_g12p','c1_g25gv001','c7_white']
rows=[];npzerrors=[]
for c in conds:
 for m in models:
  files=sorted(p for p in (W/'out/y13'/c/m).glob('*.npz') if not p.name.endswith('.tmp.npz'))
  good=total=valid=0
  for p in files:
   try:
    with np.load(p,allow_pickle=False) as z:
     a=z['spec'];assert a.ndim==2 and a.shape[1]==314
     assert len(z['start'])==len(z['truth'])==len(a)
     v=np.isfinite(a).all(1);good+=int(v.any());total+=len(v);valid+=int(v.sum())
   except Exception as e:npzerrors.append(dict(file=str(p),error=str(e)))
  j=s['jobs'].get('y13_run_'+m+'_'+c,{})
  rows.append(dict(condition=c,model=m,expected=154 if c=='c5_split10' else 73,files=len(files),recordings_with_output=good,windows_total=total,windows_valid=valid,exit_code=j.get('exit_code'),peak_rss_bytes=j.get('peak_rss'),stop_reason=j.get('stop_reason')))
score=s['jobs'].get('y13_score',{})
scorepath=L/'y13_score.log'
r=dict(completed_utc=s['completed_utc'],rows=rows,errors=s['errors'],npz_errors=npzerrors,score_exit=score.get('exit_code'),score_log_exists=scorepath.exists(),source_hashes=s['frozen'],prior_outputs_preserved=True,accepted_track_mismatches=s.get('accepted_track_mismatches'),last_budget=s.get('last_budget'),peak_rss_bytes=max(j['peak_rss'] for j in s['jobs'].values()),holdout_evaluations=0)
(L/'y13_resume21b_FINAL_AUDIT.json').write_text(json.dumps(r,indent=2))
lines=['# Public conditions: authorized 33-run resume','',f"Completed UTC: {s['completed_utc']}",'','Supplied script unchanged during this execution. JPEG q60 excluded. Prior output hashes verified.','', '| Condition | Model | Files / expected | Recordings with finite output | Valid windows / total | Exit |','|---|---|---:|---:|---:|---:|']
for x in rows:lines.append(f"| {x['condition']} | {x['model']} | {x['files']}/{x['expected']} | {x['recordings_with_output']} | {x['windows_valid']}/{x['windows_total']} | {x['exit_code']} |")
lines+=['','## Errors',json.dumps(r['errors'],indent=2),json.dumps(npzerrors,indent=2),'','## Full y13_score.log','~~~text',scorepath.read_text(errors='replace') if scorepath.exists() else 'NOT RUN / log absent','~~~']
(L/'y13_resume21b_REPORT.md').write_text('\n'.join(lines))
paths=[Path(j['log']) for j in s['jobs'].values()]+[L/n for n in ['y13_resume21b_COMPLETE.json','y13_resume21b_FINAL_AUDIT.json','y13_resume21b_REPORT.md','y13_resume21b_EXISTING_OUTPUTS.json','x16_boxcrops.jsonl','y13_resume21b_resources.log','y13_resume21_HANDOFF.txt','y13_y14_PREFLIGHT.json','y13_y14_CAP_APPROVAL.json','y13_resume21_POWER_RECOVERY_AUDIT.json','y13_resume21_POWERLOSS_QUARANTINE.json','y13_resume21_POWER_INPUT_HEADERS.json']]
paths += [W/'code/y13_openrppg_public_conditions.py',W/'code/y13_resume21b_codex_runner.py',W/'code/y13_resume21b_audit.py']
if (W/'out/y13_public_conditions.csv').exists():paths.append(W/'out/y13_public_conditions.csv')
paths=sorted(set(p for p in paths if p.exists()))
manifest={str(p.relative_to(W)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths}
mp=L/'y13_resume21b_DELIVERY_MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2))
dest=L/'y13_resume21b_requested_logs_20261121.zip';assert not dest.exists()
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in paths+[mp]:z.write(p,str(p.relative_to(W)))
with zipfile.ZipFile(dest) as z:
 assert z.testzip() is None
 for name,a in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==a['sha256'],name
receipt=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest),members=len(paths)+1,all_member_hashes_verified=True)
(L/'y13_resume21b_DELIVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
print('ROWS',len(rows),'ERRORS',len(s['errors']),len(npzerrors),'SCORE_EXIT',score.get('exit_code'))
