"""Explicitly inspected shard continuation; reject noncompliant codecs, never retune."""
import os,sys,traceback,hashlib
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common import *
from training_provider import DiskProvider

def main(shard):
 lease=lock(f'warm_constraint_{shard}.lock');guard(True)
 old=read(P/f'parallel/warm_{shard}_ERROR.json');assert old['error'].startswith('C5 proxy bitrate failed:') and 'no automatic rate retuning' in old['error']
 oldpid=next(r['pid'] for r in read(P/'parallel/STARTED.json')['children'] if r['kind']=='warm' and r['shard']==shard)
 assert not Path(f'/proc/{oldpid}').exists(),'Original worker still alive'
 dest=P/'constraint_preparation';dest.mkdir(exist_ok=True);assert not (dest/f'SHARD{shard}_STARTED.json').exists(),'Inspect any previous continuation; no restart'
 save(dest/f'SHARD{shard}_STARTED.json',dict(utc=now(),pid=os.getpid(),original_pid=oldpid,original_error_sha256=sha(P/f'parallel/warm_{shard}_ERROR.json'),code_sha256=sha(__file__)))
 ordinal=0;passed=0;failed=0
 for run in read(P/'PLAN.json')['runs']:
  provider=DiskProvider(read(R/'t8_oval_20261031/training_index.json'),run['btc_bpp_quartiles'],300000000)
  for ds,jobs in run['public_jobs'].items():
   for j in jobs:
    i=ordinal;ordinal+=1
    if i%8!=shard:continue
    guard();key=json.dumps([ds,j['recording'],float(j['start']),j['recipe'],bool(j['lowres']),run['fold']],separators=(',',':'));digest=hashlib.sha256(key.encode()).hexdigest();failure=dest/'failures'/f'{digest}.json'
    assert not failure.exists(),'Continuation should process each ordinal once'
    try:provider.prepare(ds,j['recording'],j['start'],j['recipe'],j['lowres'],run['fold']);passed+=1
    except RuntimeError as e:
     if not str(e).startswith('C5 proxy bitrate failed:') or 'no automatic rate retuning' not in str(e):raise
     tb=e.__traceback__;frame=None
     while tb:
      if tb.tb_frame.f_code.co_filename==str(P/'training_provider.py') and 'compressed' in tb.tb_frame.f_locals:frame=tb.tb_frame
      tb=tb.tb_next
     assert frame is not None;v=frame.f_locals;assert j['recipe']=='c5_proxy' and v['kbps']>500
     a=dict(utc=now(),global_index=i,key=key,fold=run['fold'],dataset=ds,job=j,eligible=False,reason='measured_whole_file_bitrate_above_500',whole_file_kbps=v['kbps'],limit_kbps=500,encoded_bytes=len(v['compressed']),encoded_sha256=hashlib.sha256(v['compressed']).hexdigest(),source_sha256=v['rec']['source_sha256'],source_decode_command=v['cmd'],encode_command=v['enc'],fs=v['fs'],frames=len(v['frames']),encoder_source_sha256=sha(P/'training_provider.py'),no_retuning=True,no_metric_selection=True)
     # One bitstream is retained as direct evidence; remaining failures have complete reproducibility receipts.
     sample=dest/'first_noncompliant.mp4'
     if not sample.exists():sample.write_bytes(v['compressed']);a['retained_bitstream']=str(sample)
     save(failure,a);failed+=1;frame=None;v=None
    save(dest/f'SHARD{shard}_STATUS.json',dict(utc=now(),pid=os.getpid(),last_global_index=i,passed=passed,rejected_constraint=failed))
  del provider
 save(dest/f'SHARD{shard}_COMPLETE.json',dict(utc=now(),passed=passed,rejected_constraint=failed))
if __name__=='__main__':
 shard=int(sys.argv[1])
 try:main(shard)
 except Exception as e:save(P/f'constraint_preparation/SHARD{shard}_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
