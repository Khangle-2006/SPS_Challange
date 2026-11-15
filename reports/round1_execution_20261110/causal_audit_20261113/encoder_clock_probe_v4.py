"""Inspect encoder timebase on the failed MCD source, no inference or disk video."""
from common_causal import *
import numpy as np,traceback
from ladder_physical_v3 import MemoryVideo,Video

def main():
 lease=lock('clock_probe_v4.lock');check_deadline();b=budget(True)
 assert not (C/'ENCODER_CLOCK_PROBE_V4.json').exists()
 case=next(c for c in read(C/'physical_v3/LADDER_SELECTION.json') if c['job']['dataset']=='MCD')
 source=Video(case['job']['source'],C/'clock_probe_source.log');results=[]
 for name,extra in [('default',[]),('demux',['-enc_time_base','demux'])]:
  with MemoryVideo('clock_probe_'+name) as m:
   cmd=['ffmpeg','-v','error','-y','-threads','1','-i',case['job']['source'],'-t','2.1','-an','-vf','format=yuv420p','-filter_threads','1','-fps_mode','passthrough','-c:v','libx265','-preset','fast','-b:v','8M','-maxrate','8M','-bufsize','16M','-x265-params','pools=none:frame-threads=1:log-level=error','-tag:v','hvc1']+extra+['-f','mp4',m.path]
   subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=open(C/('clock_probe_'+name+'.log'),'wb'))
   v=Video(m.path,C/('clock_probe_'+name+'_decode.log'));expected=source.t[:len(v.t)];delta=v.t-expected
   results.append(dict(name=name,command=cmd,frames=len(v.t),expected_prefix_frames=int(np.searchsorted(source.t,2.1,side='left')),source_time_base=source.stream['time_base'],encoded_time_base=v.stream['time_base'],max_abs_PTS_difference_s=float(abs(delta).max()),source_PTS=expected.tolist(),encoded_PTS=v.t.tolist()))
 assert results[1]['max_abs_PTS_difference_s']<2e-5 and results[1]['frames']==results[1]['expected_prefix_frames']
 save(C/'ENCODER_CLOCK_PROBE_V4.json',dict(utc=now(),source_sha256=case['job']['source_sha256'],budget=b,results=results,not_HR_evidence=True,no_training=True))
if __name__=='__main__':
 try:main()
 except Exception as e:save(C/'ENCODER_CLOCK_PROBE_V4_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
