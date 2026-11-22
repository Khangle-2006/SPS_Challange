"""Final evidence packaging only, never inference."""
import json,hashlib,zipfile
from pathlib import Path
import numpy as np
W=Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints');L=W/'logs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((L/'y05z09_COMPLETE_20261122.json').read_text());assert s['mode']=='complete'
for p,h in s['frozen'].items():assert sha(Path(p))==h
prior=json.loads((L/'y05z09_PRIOR_HASHES_20261122.json').read_text())
for p,h in prior.items():assert sha(Path(p))==h
rows=[]
for k,j in s['jobs'].items():
 rows.append(dict(job=k,exit_code=j['exit_code'],videos=j.get('videos_with_output'),rows=j.get('video_rows'),load_counts=j.get('weight_load_counts'),peak_rss_gb=j['peak_rss']/1e9,stop_reason=j.get('stop_reason'),output_sha=j.get('output_sha256')))
model_outputs={};masks=[]
for p in sorted((W/'out/y05').glob('*.npz')):
 with np.load(p,allow_pickle=False) as z:
  assert z['S'].shape==(218,277) and len(z['keys'])==218
  ok=np.isfinite(z['S']).all(1);masks.append(ok)
  model_outputs[p.stem]=dict(finite=int(ok.sum()),sha256=sha(p))
common=int(np.all(masks,axis=0).sum())
r=dict(completed_utc=s['completed_utc'],jobs=rows,errors=s['errors'],model_outputs=model_outputs,common_y05_videos=common,prior_hashes_preserved=True,source_hashes_preserved=True,last_budget=s.get('last_budget'),holdout_evaluations=0)
rp=L/'y05z09_FINAL_AUDIT_20261122.json';rp.write_text(json.dumps(r,indent=2))
lines=['# y05 six-model retry and FactorizePhys pilot','', 'CPU only; supplied scripts unchanged; holdout closed. Existing y05 outputs and original evaluation log hashes preserved.','',f"Finished UTC: {s['completed_utc']}",f"All-model shared BTC dev cohort: {common} / 218",'','| Job | Exit | Finite videos | Rows | missing/unexpected | Peak RSS GB |','|---|---:|---:|---:|---|---:|']
for x in rows:lines.append(f"| {x['job']} | {x['exit_code']} | {x['videos']} | {x['rows']} | {x['load_counts']} | {x['peak_rss_gb']:.3f} |")
lines+=['','## Errors',json.dumps(s['errors'],indent=2)]
for n in ['y05_eval_all.log','z09_score.log']:
 p=L/n;lines+=['','## Full '+n,'~~~text',p.read_text(errors='replace') if p.exists() else 'No log','~~~']
report=L/'y05z09_REPORT_20261122.md';report.write_text('\n'.join(lines))
paths=[Path(j['log']) for j in s['jobs'].values()]+[rp,report,L/'y05z09_COMPLETE_20261122.json',L/'y05z09_PREFLIGHT_20261122.json',L/'y05z09_PRIOR_HASHES_20261122.json',L/'y05z09_INPUT_STATS_20261122.json',L/'y05z09_RESOURCES_20261122.log',L/'y05z09_HANDOFF_20261122.txt',W/'code/y05_openrppg.py',W/'code/z09_factorizephys_pilot.py',W/'code/y05z09_codex_runner_20261122.py',W/'code/y05z09_audit_20261122.py']
paths=sorted(set(p for p in paths if p.exists()))
manifest={str(p.relative_to(W)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths}
mp=L/'y05z09_DELIVERY_MANIFEST_20261122.json';mp.write_text(json.dumps(manifest,indent=2))
dest=L/'y05z09_requested_logs_20261122.zip';assert not dest.exists()
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in paths+[mp]:z.write(p,str(p.relative_to(W)))
with zipfile.ZipFile(dest) as z:
 assert z.testzip() is None
 for p,a in manifest.items():assert hashlib.sha256(z.read(p)).hexdigest()==a['sha256']
receipt=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest),members=len(paths)+1,all_member_hashes_verified=True)
(L/'y05z09_DELIVERY_RECEIPT_20261122.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));print('JOBS',len(rows),'ERRORS',s['errors'],'COMMON',common)
