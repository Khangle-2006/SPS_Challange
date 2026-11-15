"""Read-only A100 derived-video relay, exact frozen hashes, no source mutation."""
from pathlib import Path
import json,hashlib,subprocess,concurrent.futures,datetime
P=Path(__file__).resolve().parent
rows=json.loads((P/'LEGACY_UBFC_SOURCE_MANIFEST.json').read_text());assert len(rows)==35
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def one(r):
 target=P/'legacy_sources'/(r['recording']+'.mp4')
 if target.exists():assert digest(target)==r['sha256'];return dict(r,bytes=target.stat().st_size,verified=True)
 temp=target.with_suffix('.mp4.transfer_part');assert not temp.exists()
 subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15','gpu-server:'+r['source'],str(temp)],check=True,timeout=300)
 assert digest(temp)==r['sha256'];temp.rename(target);return dict(r,bytes=target.stat().st_size,verified=True)
with (P/'LEGACY_PULL_STARTED.json').open('x') as f:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),count=35),f)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:done=list(pool.map(one,rows))
(P/'LEGACY_PULL_COMPLETE.json').write_text(json.dumps(dict(rows=done,bytes=sum(r['bytes'] for r in done),all_sha256_verified=True),indent=2));print('VERIFIED',len(done),sum(r['bytes'] for r in done))
