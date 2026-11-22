import sys,subprocess,json,datetime,os
from pathlib import Path
out=Path(sys.argv[1]); cmd=sys.argv[2:]
(out/'STARTED.json').write_text(json.dumps(dict(pid=os.getpid(),command=cmd,time=datetime.datetime.now(datetime.timezone.utc).isoformat())))
p=subprocess.run(cmd)
(out/'EXIT.json').write_text(json.dumps(dict(exit_code=p.returncode,time=datetime.datetime.now(datetime.timezone.utc).isoformat())))
if p.returncode:(out/'ERROR.json').write_text(json.dumps(dict(exit_code=p.returncode,inspect='process.log; do not blindly restart')))
sys.exit(p.returncode)
