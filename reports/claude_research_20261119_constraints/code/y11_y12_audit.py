"""Read-only audit and report assembly after the supplied scripts terminate."""
import json,hashlib,re,zipfile
from pathlib import Path
import numpy as np
W=Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints')
L=W/'logs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((L/'y11_y12_CODEX_COMPLETE.json').read_text())
assert s['mode']=='complete' and s['existing_outputs_preserved']
for p,h in s['frozen'].items():assert sha(Path(p))==h
rows=[];issues=list(s['errors'])
for name,j in s['jobs'].items():
 if '_run_' not in name:continue
 stage,_,model,*rest=name.split('_',3);cond=rest[0] if rest else ''
 if stage=='y11':
  p=W/'out/y11'/cond/(model+'.npz')
  if p.exists():
   with np.load(p,allow_pickle=False) as z:
    with np.load(W/'out/y05'/(model+'.npz'),allow_pickle=False) as native:
     assert np.array_equal(z['keys'],native['keys'])
     union=int((np.isfinite(z['S']).all(1)|np.isfinite(native['S']).all(1)).sum()) if cond=='heavy' else None
    finite=int(np.isfinite(z['S']).all(1).sum());attempted=int(z['done'].sum())
   rows.append(dict(stage=stage,model=model,condition=cond,exit=j['exit_code'],attempted=attempted,outputs=finite,native_union=union,peak_GB=j['peak_rss']/1e9,sha256=sha(p)))
  else:rows.append(dict(stage=stage,model=model,condition=cond,exit=j['exit_code'],attempted=0,outputs=0,peak_GB=j['peak_rss']/1e9,missing=True))
 else:
  ff=[p for p in (W/'out/y12'/model).glob('*.npz') if not p.name.endswith('.tmp.npz')]
  have=0;windows=0;valid=0
  for p in ff:
   with np.load(p,allow_pickle=False) as z:
    ok=np.isfinite(z['spec']).all(1);have+=int(ok.any());windows+=len(ok);valid+=int(ok.sum())
  marker=re.findall(r'done; recordings written (\d+)',Path(j['log']).read_text(errors='replace'))
  rows.append(dict(stage=stage,model=model,condition='public',exit=j['exit_code'],outputs=len(ff),recordings_finite=have,eligible_windows=windows,finite_windows=valid,new_written=int(marker[-1]) if marker else None,peak_GB=j['peak_rss']/1e9,hashes={p.name:sha(p) for p in ff}))
  if len(ff)!=154:issues.append(dict(job=name,reason='public output count differs154',count=len(ff)))
assert len(rows)==30
result=dict(rows=rows,issues=issues,source_hashes=s['frozen'],existing_outputs_preserved=True,holdout_evaluations=0,all_processes_terminal=True)
(L/'y11_y12_FINAL_AUDIT.json').write_text(json.dumps(result,indent=2))
txt=['# y11/y12 — execution report','', 'Supplied scripts unchanged; CPU only. Holdout untouched. Counts are audited from actual output arrays/files, including earlier Claude partials.', '', '| Stage | Model | Condition | Outputs | Exit | Peak RSS GB |','|---|---|---|---:|---:|---:|']
for r in rows:txt.append(f"| {r['stage']} | {r['model']} | {r['condition']} | {r['outputs']} | {r['exit']} | {r['peak_GB']:.3f} |")
txt+=['','Heavy counts refer to additional4K/HDR videos, not the full native cohort. Public done-markers count newly written files; the table counts all files including preserved existing outputs. For finite-window/recording coverage see FINAL_AUDIT.json.','', '## Errors', '',json.dumps(issues,indent=2),'', '## Full y11_eval.log','', '~~~text',(L/'y11_eval.log').read_text(errors='replace'),'~~~','', '## Full y12_score.log','', '~~~text',(L/'y12_score.log').read_text(errors='replace'),'~~~','', 'Model set was selected after dev exploration. Paired CIs do not turn this into an independent holdout result. No promotion is implied.']
(L/'y11_y12_EXECUTION_REPORT.md').write_text('\n'.join(txt))
files=[L/'y11_y12_EXECUTION_REPORT.md',L/'y11_y12_FINAL_AUDIT.json',L/'y11_y12_CODEX_COMPLETE.json',L/'y11_y12_EXISTING_OUTPUTS.json',L/'y11_y12_PREFLIGHT.json',L/'y11_y12_resources.log',W/'code/y11_openrppg_conditions.py',W/'code/y12_openrppg_public.py',*list(L.glob('y11_run_*.log')),*list(L.glob('y12_run_*.log')),L/'y11_eval.log',L/'y12_score.log']
manifest={str(p.relative_to(W)):sha(p) for p in files};(L/'y11_y12_DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
zpath=L/'y11_y12_requested_logs_20261120.zip'
assert not zpath.exists()
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files+[L/'y11_y12_DELIVERY_MANIFEST.json']:z.write(p,str(p.relative_to(W)))
with zipfile.ZipFile(zpath) as z:
 for p,h in manifest.items():assert hashlib.sha256(z.read(p)).hexdigest()==h
print(json.dumps(dict(zip=str(zpath),bytes=zpath.stat().st_size,sha256=sha(zpath),rows=30,errors=issues)))
