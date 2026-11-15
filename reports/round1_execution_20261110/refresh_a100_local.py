"""Local SSH inventory only; no job mutation or credentials on disk."""
import json,subprocess,shlex
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
code="""import pathlib,subprocess,json,datetime
p=pathlib.Path('/data/khangle/facevitals')
size=lambda p:sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.exists() else 0
print(json.dumps(dict(measured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),a100_bytes=int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]),a100_v4_bytes=size(p/'reports/round1_v4_20261105'),root=str(p),measurement_complete=True)))
"""
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','gpu-server','python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=45,check=True)
a=json.loads(r.stdout);assert a['measurement_complete'] and a['a100_bytes']>0
old=P/'A100_INVENTORY.json';history=P/'inventory_history';history.mkdir(exist_ok=True)
if old.exists():(history/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_A100_previous.json')).write_bytes(old.read_bytes())
old.write_text(json.dumps(a,indent=2),encoding='utf8');print(json.dumps(a))
