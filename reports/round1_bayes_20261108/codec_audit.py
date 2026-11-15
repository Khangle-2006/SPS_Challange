"""Native bitstream audit; codec/light hypothesis inspired by Tarassenko2014.
No claim that GOP coincidence proves codec causality. No reference labels used to alter GOP.
"""
from common import *
import numpy as np,fcntl,traceback
def stream_probe(p,frames=False):
 cmd=['ffprobe','-v','error','-threads','1','-select_streams','v:0','-show_streams','-show_format']
 if frames:cmd+=['-show_frames','-show_entries','frame=best_effort_timestamp_time,pict_type,pkt_size,key_frame']
 return json.loads(subprocess.check_output(cmd+['-of','json',str(p)]))
def analyze_frames(frames):
 t=np.array([float(x['best_effort_timestamp_time']) for x in frames]);assert np.all(np.diff(t)>0)
 fs=1/np.median(np.diff(t));ty=np.array([x.get('pict_type','?') for x in frames]);I=np.where(ty=='I')[0];nonB=np.where(ty!='B')[0]
 key=float(np.median(np.diff(I))) if len(I)>1 else None
 period=float(np.median(np.diff(nonB))) if (ty=='B').any() and len(nonB)>=4 else None
 freq=[]
 for name,p in [('I',key),('miniGOP',period)]:
  if p is None:continue
  for k in range(1,7):
   hz=abs((k*fs/p+fs/2)%fs-fs/2)
   if .7<=hz<=3:freq.append(dict(kind=name,harmonic=k,hz=float(hz)))
 return dict(frames=len(t),fps=float(fs),keyint_frames=key,I_period_s=float(np.median(np.diff(t[I]))) if len(I)>1 else None,miniGOP_frames=period,B_fraction=float(np.mean(ty=='B')),pattern_first120=''.join(ty[:120]),artifact_frequencies=freq)
def main():
 lock=(OUT/'codec_audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prereg()
 assert not (OUT/'codec_audit_COMPLETE.json').exists(),'Complete audit must not rerun'
 save(OUT/'codec_audit_PREFLIGHT.json',guard())
 cohort=read(OLD/'diagnosis205_20261108/COHORT.json');vs={v['recording']:v for v in dev_manifest()};rr=[]
 mcd=read('/data_pool/khangle/facevitals/code/t8_oval_20261031/mcd_cohort.json')
 assert len(mcd)==119
 index=read(ROOT/'t8_oval_20261031/training_index.json');assert {Path(x['video']).stem for x in mcd}=={x['recording_id'] for x in index['records'] if x['dataset']=='MCD'}
 existing=rows(OUT/'MCD_SOURCE_STREAMS.csv') if (OUT/'MCD_SOURCE_STREAMS.csv').exists() else []
 for x in ([] if existing else mcd):
  p=ROOT/'datasets/MCD-rPPG'/x['video'];z=stream_probe(p);s=z['streams'][0]
  rr.append(dict(recording=Path(x['video']).stem,path=str(p),width=s['width'],height=s['height'],codec=s['codec_name'],pixel=s['pix_fmt'],video_bitrate=s.get('bit_rate'),container_bitrate=z['format'].get('bit_rate'),size_bytes=p.stat().st_size,source_sha256=read(ROOT/'t8_oval_20261031/MCD'/Path(x['video']).stem/'native/complete.json')['source_sha256']))
 if existing:
  rr=existing;assert len(rr)==119 and {x['recording'] for x in rr}=={Path(x['video']).stem for x in mcd}
 else:csvout(OUT/'MCD_SOURCE_STREAMS.csv',rr)
 br=[float(x['video_bitrate']) for x in rr if x['video_bitrate']];cb=[float(x['container_bitrate']) for x in rr if x['container_bitrate']]
 save(OUT/'MCD_SOURCE_SUMMARY.json',dict(records=len(rr),formats=sorted({(int(x['width']),int(x['height']),x['codec'],x['pixel']) for x in rr}),median_video_bitrate=float(np.median(br)) if br else None,median_container_bitrate=float(np.median(cb)) if cb else None,video_bitrate_missing=len(rr)-len(br)))
 results=[];dst=OUT/'native_gop';dst.mkdir(exist_ok=True)
 for j,rec in enumerate(cohort['recordings']):
  v=vs[rec];p=Path(v['path']);fp=dst/(rec+'.json')
  if fp.exists():a=read(fp);assert a['source_sha256']==v['sha256']
  else:
   z=stream_probe(p,True);ff=z['frames'];a=dict(recording=rec,person=v['person'],source_sha256=v['sha256'],**analyze_frames(ff));save(fp,a)
   csvout(dst/(rec+'_frames.csv'),[dict(frame=i,pts=f.get('best_effort_timestamp_time'),type=f.get('pict_type'),pkt_size=f.get('pkt_size'),key=f.get('key_frame')) for i,f in enumerate(ff)])
  results.append(a);save(OUT/'codec_audit_STATUS.json',dict(state='native_GOP',completed=j+1,total=205,pid=os.getpid(),utc=now()))
  if j%25==0:guard(False)
 save(OUT/'BTC_NATIVE_GOP.json',results);save(OUT/'codec_audit_COMPLETE.json',dict(utc=now(),mcd=119,btc=205,holdout_evaluations=0,code_sha256=sha(__file__)))
if __name__=='__main__':
 try:main()
 except Exception as e:save(OUT/'codec_audit_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=now()));raise
