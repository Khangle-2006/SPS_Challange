from pathlib import Path
import json
w=Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints')
s=json.loads((w/'logs/y11_y12_codex_status.json').read_text())
out=dict(mode=s['mode'],utc=s['updated_utc'],finished=sum('exit_code' in j for j in s['jobs'].values()),pending=len(s.get('pending',[])),errors=s['errors'],wait=s.get('wait_reason'),traceback=s.get('traceback'),active={})
for k in s.get('active',[]):
 j=s['jobs'][k];ll=Path(j['log']).read_text(errors='replace').splitlines()
 good=[l for l in ll if 'mean s/video' in l or 'videos to do' in l or 'recordings to do' in l or 'windows' in l or 'Traceback' in l or 'Error' in l]
 out['active'][k]=dict(peak_GB=round(j['peak_rss']/1e9,3),progress=good[-1:] if good else ll[-1:])
out['completed']={k:dict(exit=j['exit_code'],videos=j.get('videos_with_output'),recordings=j.get('recordings_total')) for k,j in s['jobs'].items() if 'exit_code' in j}
print(json.dumps(out))
