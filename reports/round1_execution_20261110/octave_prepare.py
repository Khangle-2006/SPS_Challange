"""Extract distribution packages into a private prefix; no sudo/system changes."""
import os,re,shlex,urllib.request,urllib.parse,concurrent.futures,traceback
from common import *
def main():
 lease=lock('octave_prepare.lock');guard(True)
 # 60,920,094 archive bytes + 215,228,416 installed bytes; 300MB reserve.
 a=read(P/'BUDGET_AIPEX_ADDENDUM.json');assert a['new_total_estimate_bytes']+300000000<=84000000000
 save(P/'OCTAVE_BUDGET_ADDENDUM.json',dict(previous_estimate=a['new_total_estimate_bytes'],reserve_bytes=300000000,total_estimate=a['new_total_estimate_bytes']+300000000,cap_bytes=84000000000,no_deletion=True))
 root=P/'octave_private';assert not (root/'COMPLETE.json').exists();root.mkdir(exist_ok=True);deb=root/'debs';deb.mkdir(exist_ok=True);prefix=root/'prefix';prefix.mkdir(exist_ok=True)
 lines=[shlex.split(l) for l in (P/'OCTAVE_APT_PLAN.txt').read_text().splitlines() if l.startswith("'")]
 jobs=[]
 for url,filename,n,checksum in lines:
  name,version,arch=filename.rsplit('_',2);version=urllib.parse.unquote(version)
  metadata=subprocess.check_output(['apt-cache','show',f'{name}={version}'],text=True)
  expected=re.search(r'^SHA256: (\w+)',metadata,re.M);assert expected,(name,version)
  jobs.append(dict(url=url,filename=filename,bytes=int(n),sha256=expected[1],package=name,version=version))
 assert sum(j['bytes'] for j in jobs)==60920094
 def download(j):
  path=deb/j['filename']
  if not path.exists():
   with urllib.request.urlopen(j['url'],timeout=45) as src,path.open('wb') as dst:
    while chunk:=src.read(1<<20):dst.write(chunk)
  assert path.stat().st_size==j['bytes'] and sha(path)==j['sha256'],j['filename']
  return j
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(download,jobs))
 for j in receipts:subprocess.run(['dpkg-deb','-x',str(deb/j['filename']),str(prefix)],check=True)
 save(root/'COMPLETE.json',dict(utc=now(),packages=receipts,install='private extraction; no maintainer scripts or system changes',root=str(prefix)))
 print('OCTAVE_EXTRACTED',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'OCTAVE_PREPARE_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
