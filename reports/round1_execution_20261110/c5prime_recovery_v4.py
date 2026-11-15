"""Two frozen full-frame ffmpeg proxies; original source clock survives encoding."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LIBGL_ALWAYS_SOFTWARE='1',EGL_PLATFORM='surfaceless',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
import sys,json,math,subprocess,traceback,argparse
import numpy as np
from runtime_shared_v4 import install
install()
from common import *
sys.path.insert(0,str(V/'btc_zeroshot'))
import preprocess as pre
from codec_provider import c5_selection
pre.guard=guard
# Only cleanup changes: avoid an unbounded wait after a guard exception.
import inspect
_crop_source=inspect.getsource(pre.crop)
_old_cleanup="if p.poll() is None:p.terminate();p.wait()"
_new_cleanup="""if p.poll() is None:
                try:p.stdin.close()
                except (BrokenPipeError, OSError):pass
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()"""
assert _crop_source.count(_old_cleanup)==1
_namespace=dict(pre.__dict__)
exec(compile(_crop_source.replace(_old_cleanup,_new_cleanup),str(P/'c5prime_recovery_v4.py')+':crop_cleanup','exec'),_namespace)
pre.crop=_namespace['crop']
pre.crop.__globals__['guard']=guard
def jobs():
 index=read(R/'t8_oval_20261031/training_index.json');pub=read(B/'public_comparison/MANIFEST.json')['records'];out=[]
 for r in pub:
  ds,rid=r['dataset'],r['recording'];d=Path(index['roots'][ds])/rid/'native';rec=read(d/'complete.json')
  source=Path('/data_pool/khangle/facevitals/datasets/UBFC-rPPG/DATASET_2')/rid/'vid.avi' if ds=='UBFC2' else Path(r['source'])
  out.append(dict(dataset=ds,recording=rid,person=r['person'],fold=r['fold'],source=str(source),source_sha256=rec['source_sha256'],meta=str(d/'crop/meta.npz')))
 split=read(V/'btc_zeroshot/btc_split.json');assert sha(V/'btc_zeroshot/btc_split.json')==SPLIT
 for v in read(V/'btc_zeroshot/dev_manifest.json')['videos']:
  assert split['people'][v['person']]['partition']=='dev'
  out.append(dict(dataset='BTC',recording=v['recording'],person=v['person'],fold=v['fold'],source=v['path'],source_sha256=v['sha256'],drive_id=v['drive_id']))
 assert len(out)==375
 return out
def one(j,variant):
 d=P/'c5prime'/j['dataset']/j['recording']/variant;d.mkdir(parents=True,exist_ok=True)
 if (d/'COMPLETE.json').exists():
  a=read(d/'COMPLETE.json');assert a['source_sha256']==j['source_sha256'];return a
 reuse=(d/'video.mp4').exists()
 if reuse:
  receipt=read(P/'recovery_v4/ENCODED_PARTIAL_APPROVED.json')
  assert str(d)==receipt['directory'] and sha(d/'video.mp4')==receipt['encoded_sha256']
  assert sha(d/'clock.npz')==receipt['clock_sha256'] and j['source_sha256']==receipt['source_sha256']
  assert not (d/'crop').exists(),'Preserve inspected partial crop before retry'
 guard();assert sha(j['source'])==j['source_sha256']
 native=pre.Video(j['source'],d/'source_decode.log')
 if j['dataset']=='BTC':t=native.t
 else:
  with np.load(j['meta']) as m:t=m['original_t'].copy()
  cm=read(Path(j['meta']).with_name('manifest.json'));drop=int(cm['metadata'].get('dropped_final_decoded_frames',0))
  assert len(native.t)-len(t)==drop,'undeclared source clock/frame ordinal mismatch'
 if t[-1]-t[0]<10:
  a=dict(**j,state='unscorable',reason='source_under_10s_no_padding',variant=variant);save(d/'COMPLETE.json',a);return a
 ix=c5_selection(t);stride=int(ix[1]-ix[0]);times=t[ix];fs=float((len(times)-1)/(times[-1]-times[0]));assert fs<=10+1e-8
 vf=native.vf+f',select=lt(n\\,{len(t)})*not(mod(n\\,{stride})),setpts=N/({fs:.12f}*TB)'
 cmd=['ffmpeg','-v','error','-threads','1','-i',j['source'],'-map','0:v:0','-an','-sn','-dn','-vf',vf,'-filter_threads','1','-r',f'{fs:.12f}','-fps_mode','cfr','-c:v','libx264','-preset','medium','-crf','23','-maxrate','500k','-bufsize','1000k','-pix_fmt','yuv420p','-threads','1']
 if variant=='bf0':cmd+=['-bf','0']
 cmd+=['-movflags','+faststart',str(d/'video.mp4')]
 if not reuse:
  with (d/'encode.log').open('wb') as log:subprocess.run(cmd,stderr=log,check=True)
 encoded=pre.Video(d/'video.mp4',d/'encoded_decode.log');assert len(encoded.t)==len(times),'postencode frame count mismatch'
 duration=len(encoded.t)/fs;kbps=(d/'video.mp4').stat().st_size*8/duration/1000
 np.savez_compressed(d/'clock.npz',source_frame_indices=ix,source_t=times,encoded_t=encoded.t)
 # Same frozen landmark/oval implementation used by original BTC preprocessing.
 # It is rerun on the encoded frames, as in the historical C5 frontend.
 crop=pre.crop(encoded,times,d/'crop')
 g=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=pict_type,key_frame,best_effort_timestamp_time','-of','json',str(d/'video.mp4')]))
 save(d/'GOP.json',g)
 a=dict(**j,state='complete',variant=variant,command=cmd,command_sha256=hashlib.sha256(json.dumps(cmd).encode()).hexdigest(),encoded_sha256=sha(d/'video.mp4'),clock_sha256=sha(d/'clock.npz'),crop=crop,source_span_s=float(t[-1]-t[0]),analysis_clock='selected original source timestamps; encoded CFR clock recorded separately',fps=fs,requested_fps_ceiling=10,frame_skip=stride,whole_file_kbps=kbps,bitrate_pass=kbps<=500,official_MATLAB_parity=False,holdout_evaluations=0,code_sha256=sha(__file__))
 save(d/'COMPLETE.json',a);return a
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();lease=lock('c5prime.lock');guard(True)
 todo=jobs();save(P/'C5_JOBS.json',todo);save(P/'C5_CODE_HASHES.json',hashes());done=[]
 if args.smoke:todo=[next(x for x in todo if x['dataset']==ds) for ds in ['UBFC2','MCD','BTC']]
 for i,j in enumerate(todo):
  for variant in ['default','bf0']:
   save(P/'C5_STATUS.json',dict(state='running',pid=os.getpid(),index=i,total=len(todo),dataset=j['dataset'],recording=j['recording'],variant=variant,utc=now()))
   a=one(j,variant);done.append(dict(dataset=j['dataset'],recording=j['recording'],variant=variant,state=a['state'],bitrate_pass=a.get('bitrate_pass')))
   print(json.dumps(done[-1]),flush=True)
 save(P/('C5_SMOKE.json' if args.smoke else 'C5_COMPLETE.json'),dict(utc=now(),rows=done,holdout_evaluations=0))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'C5_ERROR_V2.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
