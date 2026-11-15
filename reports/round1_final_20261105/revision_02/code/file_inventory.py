import csv,hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);parent=r.parent
local=[]
for p in sorted(parent.rglob('*')):
    if not p.is_file():continue
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    local.append(dict(host='Windows',path=p.resolve().as_posix(),bytes=p.stat().st_size,sha256=h,note=''))
remote=[]
for host in ('P100','A100'):
    remote.extend(csv.DictReader((r/f'REMOTE_FILES_{host}.csv').open()))
for host in ('P100','A100'):
    root='/mnt/samsung_t5/khangle/facevitals' if host=='P100' else '/data/khangle/facevitals'
    remote.append(dict(host=host,path=root+'/reports/round1_final_20261105/revision_02/REMOTE_FILES_'+host+'.csv',bytes='',sha256='',note='inventory itself; downloaded separately'))
for name in ('LOCAL_FILES.csv','REMOTE_FILES.csv','FILE_LIST.md'):
    local.append(dict(host='Windows',path=(r/name).resolve().as_posix(),bytes='',sha256='',note='inventory itself; excludes recursive hashing'))
for name,rows in [('LOCAL_FILES.csv',local),('REMOTE_FILES.csv',remote)]:
    with (r/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['host','path','bytes','sha256','note']);w.writeheader();w.writerows(rows)
text='# Every created file\n\nLocal files include the earlier superseded draft retained in the same output folder. Remote lists capture both hosts after their work completed, including large MAT artifacts. Venv and symlink entries have no binary hash; raw evidence and outputs are hashed. Inventory files exclude recursive self-hashing.\n\n'
for host,rows in [('Windows',local),('P100',[x for x in remote if x['host']=='P100']),('A100',[x for x in remote if x['host']=='A100'])]:
    text+='## '+host+'\n\n'+''.join('- `'+x['path']+'`\n' for x in rows)+'\n'
(r/'FILE_LIST.md').write_text(text,encoding='utf-8')
print(json.dumps(dict(local_files=len(local),remote_files=len(remote))))
