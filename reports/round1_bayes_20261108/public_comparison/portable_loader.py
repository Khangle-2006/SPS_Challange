"""Exact BTC Video class; import-only portability extraction, no decode changes."""
import json,subprocess
from pathlib import Path
import numpy as np
class Video:
    def __init__(self,path,log):
        self.path=Path(path);self.log=Path(log)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)]))
        self.stream=probe['streams'][0]
        self.t=np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']]);self.t-=self.t[0]
        if len(self.t)<2 or not np.isfinite(self.t).all() or np.any(np.diff(self.t)<=0):raise ValueError('invalid_or_duplicate_display_PTS')
        rotation=next((float(s['rotation']) for s in self.stream.get('side_data_list',[]) if 'rotation' in s),0)
        self.width=int(self.stream['width']);self.height=int(self.stream['height'])
        if abs(rotation)%180==90:self.width,self.height=self.height,self.width
        elif rotation%180:raise ValueError('unsupported_non_right_angle_rotation')
        tr=self.stream.get('color_transfer','unknown');self.hdr=tr in ['arib-std-b67','smpte2084']
        if self.hdr:
            if self.stream.get('color_primaries')!='bt2020' or self.stream.get('color_space') not in ['bt2020nc','bt2020c']:raise ValueError('HDR_color_metadata_incomplete')
            self.vf='zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0:peak=10,zscale=t=bt709:m=bt709:r=full,format=rgb24'
        else:self.vf='format=rgb24'
        self.command=['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-i',str(path),'-map','0:v:0','-an','-sn','-dn','-filter_threads','1','-vf',self.vf,'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    def __iter__(self):
        with self.log.open('wb') as log:
            p=subprocess.Popen(self.command,stdout=subprocess.PIPE,stderr=log)
            n=0;framebytes=self.width*self.height*3
            try:
                while True:
                    raw=p.stdout.read(framebytes)
                    if not raw:break
                    if len(raw)!=framebytes:raise ValueError('partial_decode_frame')
                    yield np.frombuffer(raw,np.uint8).reshape(self.height,self.width,3);n+=1
                if p.wait()!=0:raise RuntimeError('decode_failed: '+str(self.log))
                if n!=len(self.t):raise ValueError(f'decode_PTS_mismatch:{n}/{len(self.t)}')
            finally:
                p.stdout.close()
                if p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
