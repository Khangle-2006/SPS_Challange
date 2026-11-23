"""Download the pinned R1 asset release and verify every file. No dataset access."""
import argparse,hashlib,json,re,urllib.request,zipfile
from pathlib import Path
from check_inference_assets import verify
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,default=ROOT/'checkpoints/r1');a=p.parse_args()
 c=json.loads((ROOT/'configs/inference-assets-r1-20261124.json').read_text());assert re.fullmatch('[0-9a-f]{64}',c['sha256']),'Malformed pinned SHA256';a.destination.mkdir(parents=True,exist_ok=True);archive=a.destination/c['asset'];bundle=a.destination/c['root']
 if not archive.exists():
  part=archive.with_suffix('.zip.part')
  if part.exists() and sha(part)!=c['sha256']:raise RuntimeError(f'Partial download preserved at {part}; choose a new destination to retry.')
  if not part.exists():
   with urllib.request.urlopen(c['url'],timeout=120) as response,part.open('xb') as f:
    for b in iter(lambda:response.read(1024*1024),b''):f.write(b)
  if sha(part)!=c['sha256']:raise RuntimeError('Archive hash mismatch; partial retained.')
  part.rename(archive)
 if sha(archive)!=c['sha256']:raise RuntimeError('Archive hash mismatch; no extraction.')
 if not bundle.exists():
  with zipfile.ZipFile(archive) as z:
   for n in z.namelist():
    q=Path(n)
    if q.is_absolute() or '..' in q.parts or '\\' in n or ':' in n or q.parts[0]!=c['root']:raise RuntimeError('Unsafe archive path')
   z.extractall(a.destination)
 count=verify(bundle)
 for n in ['btc_reader_v4_deploy.json','btc_reader_v4_folds.json','public_p4m_r1.json']:
  if (bundle/'readers'/n).read_bytes()!=(ROOT/'models'/n).read_bytes():raise RuntimeError('Release parameters differ from this checkout: '+n)
 print(f'PASS: pinned archive, {count} file hashes, reader parameters match checkout. Assets: {bundle}')
if __name__=='__main__': main()
