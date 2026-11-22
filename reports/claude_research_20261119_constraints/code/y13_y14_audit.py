"""Audit unchanged y13/y14 runs; writes reports only inside W."""
from pathlib import Path
import json,hashlib,re,zipfile,collections
import numpy as np
W=Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints');L=W/'logs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((L/'y13_y14_CODEX_COMPLETE.json').read_text())
assert s['mode']=='complete' and s['existing_outputs_preserved']
for p,h in s['frozen'].items():assert sha(Path(p))==h
models=['FacePhys.rlap','ME-chunk.pure','ME-flow.pure']
conds=['c5_split10','c5_split10_jpg60','c4_qvga','c6_us1','aidtrack_c1_g25gv01','c2_vit','track_c1_g25p','track_c7_blue','c1_g50p','c1_g12p','c1_g25gv001','c7_white']
rows=[];errors=list(s['errors']);warnings=collections.Counter()
for n,j in s['jobs'].items():
 for line in Path(j['log']).read_text(errors='replace').splitlines():
  if re.search(r'Traceback|\bERROR\b|Exception',line):errors.append(dict(job=n,log_error=line))
  if 'WARNING' in line:warnings[line]+=1
for m in models:
 n='y14_run_'+m+'_60';j=s['jobs'].get(n,{})
 p=W/'out/y11/c5_split10_jpg60'/(m+'.npz')
 r=dict(stage='BTC',model=m,condition='c5_split10_jpg60',exit=j.get('exit_code'),peak_GB=j.get('peak_rss',0)/1e9,expected=218)
 if p.exists():
  with np.load(p,allow_pickle=False) as z:
   r.update(attempted=int(z['done'].sum()),outputs=int(np.isfinite(z['S']).all(1).sum()),sha256=sha(p))
 else:r.update(outputs=0,missing=True)
 rows.append(r)
for c in conds:
 for m in models:
  n='y13_run_'+m+'_'+c;j=s['jobs'].get(n,{})
  ff=sorted(p for p in (W/'out/y13'/c/m).glob('*.npz') if not p.name.endswith('.tmp.npz'))
  nr=nv=nw=0
  for p in ff:
   with np.load(p,allow_pickle=False) as z:
    ok=np.isfinite(z['spec']).all(1);nr+=int(ok.any());nv+=int(ok.sum());nw+=len(ok)
  expected=154 if c.startswith('c5_split10') else 73
  r=dict(stage='public',model=m,condition=c,exit=j.get('exit_code'),expected=expected,outputs=len(ff),recordings_finite=nr,finite_windows=nv,eligible_windows=nw,peak_GB=j.get('peak_rss',0)/1e9,hashes={p.name:sha(p) for p in ff})
  marker=re.findall(r'done; recordings written (\d+)',Path(j['log']).read_text(errors='replace')) if j else []
  r['new_written']=int(marker[-1]) if marker else None
  if len(ff)!=expected:errors.append(dict(job=n,reason='Expected recording count not reached',expected=expected,actual=len(ff)))
  rows.append(r)
assert len(rows)==39
result=dict(rows=rows,errors=errors,warnings=dict(warnings),source_hashes=s['frozen'],existing_outputs_preserved=True,holdout_evaluations=0,crop_gate=s.get('crop_gate'),crop_gate_error=s.get('crop_gate_error'),max_rss_GB=max(j.get('peak_rss',0) for j in s['jobs'].values())/1e9)
(L/'y13_y14_FINAL_AUDIT.json').write_text(json.dumps(result,indent=2))
txt=['# y13/y14 execution report','','Supplied scripts and PREREG unchanged. CPU only; no holdout. Existing Claude results preserved.','','| Stage | Model | Condition | Files/videos | Finite recordings | Exit | Peak RSS GB |','|---|---|---|---:|---:|---:|---:|']
for r in rows:txt.append(f"| {r['stage']} | {r['model']} | {r['condition']} | {r['outputs']}/{r['expected']} | {r.get('recordings_finite','-')} | {r['exit']} | {r['peak_GB']:.3f} |")
txt+=['','## Errors','',json.dumps(errors,indent=2),'','## Warnings','',json.dumps(dict(warnings),indent=2)]
for name in ['y14_eval.log','y13_score.log']:
 p=L/name
 txt+=['','## Full '+name,'','~~~text',p.read_text(errors='replace') if p.exists() else 'NOT RUN: see errors and crop gate.','~~~']
(L/'y13_y14_EXECUTION_REPORT.md').write_text('\n'.join(txt))
files=[L/'y13_y14_EXECUTION_REPORT.md',L/'y13_y14_FINAL_AUDIT.json',L/'y13_y14_CODEX_COMPLETE.json',L/'y13_y14_EXISTING_OUTPUTS.json',L/'y13_y14_PREFLIGHT.json',L/'y13_y14_CAP_APPROVAL.json',L/'y13_y14_resources.log',W/'code/y13_openrppg_public_conditions.py',W/'code/y14_btc_split_jpeg.py']
files+=[p for p in [L/'y13_y14_X16_GATE_FAILURE.json',L/'y13_y14_X16_FAILURE_SNAPSHOT.jsonl',L/'y13_y14_RAM_INTERVENTION_NOT_NEEDED.json'] if p.exists()]
files+=[Path(j['log']) for j in s['jobs'].values() if Path(j['log']).exists()]
files=list(dict.fromkeys(files));manifest={str(p.relative_to(W)):sha(p) for p in files}
mp=L/'y13_y14_DELIVERY_MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2))
zp=L/'y13_y14_requested_logs_20261121.zip';assert not zp.exists()
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files+[mp]:z.write(p,str(p.relative_to(W)))
with zipfile.ZipFile(zp) as z:
 for p,h in manifest.items():assert hashlib.sha256(z.read(p)).hexdigest()==h
print(json.dumps(dict(zip=str(zp),bytes=zp.stat().st_size,sha256=sha(zp),rows=len(rows),errors=errors)))
