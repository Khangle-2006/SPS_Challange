"""Complete measured native GOP inputs; reuse verified205 and inspect missing16 only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import ast,shutil,traceback
import numpy as np
from common import *
def main():
 lease=lock('gop_complete.lock');guard(True);source=B/'codec_audit.py';tree=ast.parse(source.read_text())
 funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['stream_probe','analyze_frames']];assert len(funcs)==2
 ns=dict(np=np,json=json,subprocess=subprocess);exec(compile(ast.Module(body=funcs,type_ignores=[]),str(source),'exec'),ns)
 dev=read(V/'btc_zeroshot/dev_manifest.json')['videos'];split=read(V/'btc_zeroshot/btc_split.json');assert sha(V/'btc_zeroshot/btc_split.json')==SPLIT
 manifest=[]
 for v in dev:
  assert split['people'][v['person']]['partition']=='dev';old=B/'native_gop'/(v['recording']+'.json');new=P/'native_gop'/old.name
  if new.exists():a=read(new);assert a['source_sha256']==v['sha256']
  elif old.exists():
   a=read(old);assert a['source_sha256']==v['sha256'];new.parent.mkdir(exist_ok=True);shutil.copyfile(old,new)
  else:
   assert sha(v['path'])==v['sha256'];a=ns['analyze_frames'](ns['stream_probe'](v['path'],True)['frames']);a.update(source_sha256=v['sha256'],recording=v['recording'],source=v['path'],code_sha256=sha(source),holdout_evaluations=0);save(new,a)
  assert 'artifact_frequencies' in a and 'keyint_frames' in a and 'miniGOP_frames' in a
  manifest.append(dict(recording=v['recording'],source_sha256=v['sha256'],gop_sha256=sha(new),origin='verified_existing' if old.exists() else 'new_bitstream_measurement'))
  save(P/'GOP_STATUS.json',dict(completed=len(manifest),total=221,pid=os.getpid(),utc=now()))
 save(P/'GOP_COMPLETE.json',dict(utc=now(),rows=manifest,source_code_sha256=sha(source),holdout_evaluations=0))
 print('GOP_COMPLETE',len(manifest),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'GOP_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
