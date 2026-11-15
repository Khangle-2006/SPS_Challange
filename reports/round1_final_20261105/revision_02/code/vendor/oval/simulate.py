"""Frozen, streaming constraint simulations. Never modify original recordings."""
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)


def select_frames(timestamps, fps):
    """Nearest actual source samples to a target grid; no blend/duplication."""
    t = np.asarray(timestamps, float)
    if len(t) < 2 or np.any(np.diff(t) < 0) or not np.isfinite(t).all():
        raise ValueError('Invalid source times')
    if fps <= 0: raise ValueError('Invalid FPS')
    q = t[0] + np.arange(int(np.floor((t[-1]-t[0])*fps+1e-9))+1)/fps
    right = np.searchsorted(t, q).clip(0, len(t)-1)
    left = np.maximum(0, right-1)
    indices = np.where(abs(t[left]-q) <= abs(t[right]-q), left, right)
    return np.unique(indices)


def noise_float(rgb, gain, rng, a=None, b=None):
    a = PROTOCOL['c1']['a'] if a is None else a
    b = PROTOCOL['c1']['b'] if b is None else b
    mu = np.asarray(rgb, dtype=np.float32) * (gain/255.)
    if not 0 < gain < 1 or a <= 0 or b < 0: raise ValueError('Invalid noise parameters')
    return rng.poisson(mu/a).astype(np.float32)*a + rng.normal(0, math.sqrt(b), mu.shape).astype(np.float32)


def c1_pixels(rgb, gain, rng):
    return np.rint(np.clip(noise_float(rgb, gain, rng), 0, 1)*255).astype(np.uint8)


def probe(path):
    cmd = ['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_format','-of','json',str(path)]
    raw = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    stream = raw['streams'][0]; duration = float(raw['format']['duration'])
    return {'width':stream['width'], 'height':stream['height'], 'codec':stream['codec_name'],
            'pix_fmt':stream['pix_fmt'], 'r_frame_rate':stream['r_frame_rate'],
            'avg_frame_rate':stream['avg_frame_rate'], 'duration_s':duration,
            'frames':int(stream['nb_frames']) if str(stream.get('nb_frames','')).isdigit() else None,
            'bytes':Path(path).stat().st_size, 'whole_file_kbps':Path(path).stat().st_size*8/duration/1000,
            'raw':raw}


def qp_summary(text):
    # Actual libx264 summary, weighted by frame counts. Never infer from CRF.
    vals = re.findall(r'frame\s+([IPB]):\s*(\d+)\s+Avg QP:\s*([\d.]+)', text)
    n = sum(int(v[1]) for v in vals)
    return {'mean_qp':sum(int(v[1])*float(v[2]) for v in vals)/n if n else None,
            'source':'libx264 encoder summary weighted by I/P/B counts' if n else 'NA: encoder did not expose I/P/B average QP',
            'frame_types':[{'type':t,'count':int(c),'mean_qp':float(q)} for t,c,q in vals]}


class FileVideo:
    def __init__(self, path):
        import cv2
        self.path = Path(path); p = probe(path)
        cap = cv2.VideoCapture(str(path)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        # API callers should pass authoritative times whenever provided by a dataset.
        cmd=['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)]
        f=json.loads(subprocess.run(cmd,check=True,capture_output=True,text=True).stdout)['frames']
        self.timestamps=np.asarray([float(x['best_effort_timestamp_time']) for x in f])
        self.timestamps-=self.timestamps[0]
        if len(self.timestamps)!=n: raise ValueError('Frame/PTS count mismatch')
    def __iter__(self):
        import cv2
        cap=cv2.VideoCapture(str(self.path)); n=0
        try:
            while True:
                ok,bgr=cap.read()
                if not ok:break
                n+=1; yield cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        finally:cap.release()
        if n!=len(self.timestamps):raise ValueError('Incomplete source decode')


def simulate(frames, timestamps, output_dir, kind, fps=None, bitrate_kbps=500,
             size=None, gain=None, seed=20261030, budget_root=None, budget_bytes=None):
    import cv2
    cv2.setNumThreads(1)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): raise FileExistsError('Refuse overwrite completed simulation')
    if kind not in ['c1','c4','c5','c4_c5']:raise ValueError(kind)
    t=np.asarray(timestamps,float)
    indices=select_frames(t,fps) if kind in ['c5','c4_c5'] else np.arange(len(t))
    muxfps=float(fps) if kind in ['c5','c4_c5'] else (len(t)-1)/(t[-1]-t[0])
    if kind in ['c5','c4_c5']:
        if fps not in PROTOCOL['c5']['allowed_fps'] or bitrate_kbps not in PROTOCOL['c5']['allowed_bitrate_kbps']:raise ValueError('Unsupported frozen C5 setting')
    if kind in ['c4','c4_c5'] and list(size or []) not in PROTOCOL['c4']['sizes']:raise ValueError('QVGA or VGA only')
    if kind=='c1' and gain not in PROTOCOL['c1']['gain_levels']:raise ValueError('Unsupported frozen gain')
    if budget_root is not None:
        used=sum(p.stat().st_size for p in Path(budget_root).rglob('*') if p.is_file())
        if used>=budget_bytes:raise RuntimeError('Derivative byte budget exhausted')
    else:used=0
    dest=out/('video.mkv' if kind=='c4' else 'video.mp4')
    log=out/'encoder.log'; rng=np.random.default_rng(seed); selected=set(indices.tolist()); count=0; proc=None
    cmd=[]; width=height=None
    with log.open('wb') as err:
        try:
            for i, rgb in enumerate(frames):
                if i>=len(t):raise ValueError('Extra source frame')
                count=i+1
                if i not in selected:continue
                if size:rgb=cv2.resize(rgb,tuple(size),interpolation=cv2.INTER_CUBIC)
                if kind=='c1':rgb=c1_pixels(rgb,gain,rng)
                if proc is None:
                    height,width=rgb.shape[:2]
                    cmd=['ffmpeg','-hide_banner','-loglevel','info','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
                         '-r',f'{muxfps:.12g}','-i','pipe:0','-an','-threads','1','-filter_threads','1']
                    if kind=='c4':cmd+=['-c:v','ffv1','-level','3','-pix_fmt','bgr0','-slicecrc','1']
                    else:
                        cmd+=['-c:v','libx264','-preset','veryfast','-pix_fmt','yuv420p','-g',str(round(2*muxfps)),'-keyint_min',str(round(2*muxfps)),'-sc_threshold','0']
                        if kind=='c1':cmd+=['-crf','18','-maxrate','2000k','-bufsize','4000k']
                        else:
                            rate=round(bitrate_kbps*PROTOCOL['c5']['headroom_fraction'])
                            cmd+=['-b:v',f'{rate}k','-maxrate',f'{rate}k','-bufsize',f'{2*rate}k']
                        cmd+=['-movflags','+faststart']
                    cmd+=[str(dest)]
                    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=err)
                proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
                if budget_root is not None and i%100==0 and dest.exists() and used+dest.stat().st_size>budget_bytes:
                    raise RuntimeError('Byte budget crossed during encode; partial file preserved')
            if count!=len(t):raise ValueError(f'Source decode {count}/{len(t)}')
            if proc is None:raise ValueError('No frames')
            proc.stdin.close(); rc=proc.wait()
            if rc:raise RuntimeError(f'ffmpeg exit {rc}; see encoder.log')
        except Exception:
            if proc is not None:
                try:proc.stdin.close()
                except Exception:pass
                proc.terminate();proc.wait()
            raise
    actual=probe(dest)
    if actual['frames'] is not None and actual['frames']!=len(indices):raise ValueError('Encoded frame count mismatch')
    np.savez_compressed(out/'frame_mapping.npz',source_frame_indices=indices,timestamps_s=t[indices])
    manifest={'protocol_version':PROTOCOL['version'],'protocol_sha256':sha256(HERE/'protocol.json'),'kind':kind,'fps_requested':fps,
      'mux_fps':muxfps,'effective_sensor_fps':(len(indices)-1)/(t[indices[-1]]-t[indices[0]]),
      'bitrate_limit_kbps':bitrate_kbps if kind in ['c5','c4_c5'] else None,
      'bitrate_pass':actual['whole_file_kbps']<=bitrate_kbps if kind in ['c5','c4_c5'] else None,
      'size_requested':size,'gain':gain,'seed':seed,'source_frames':len(t),'selected_frames':len(indices),
      'timestamp_authority':'frame_mapping.npz source clock, not CFR container time','no_blending':True,'no_duplicates':True,
      'actual':actual,'command':cmd,'qp':qp_summary(log.read_text(errors='replace')),
      'ffmpeg_version':subprocess.run(['ffmpeg','-version'],capture_output=True,text=True).stdout.splitlines()[0],
      'video':str(dest),'script_sha256':sha256(__file__)}
    write_json(out/'manifest.json',manifest)
    return dest,manifest


def _api(video,output_dir,kind,timestamps=None,**kwargs):
    source=FileVideo(video)
    return simulate(source,source.timestamps if timestamps is None else timestamps,output_dir,kind,**kwargs)

def c5(video,output_dir,fps=10,bitrate_kbps=500,timestamps=None,**kwargs):
    return _api(video,output_dir,'c5',timestamps,fps=fps,bitrate_kbps=bitrate_kbps,**kwargs)

def c4(video,output_dir,size=(320,240),timestamps=None,**kwargs):
    return _api(video,output_dir,'c4',timestamps,size=size,**kwargs)

def c1(video,output_dir,lux_scale=0.5,timestamps=None,**kwargs):
    """lux_scale is relative synthetic gain, not calibrated illuminance."""
    return _api(video,output_dir,'c1',timestamps,gain=lux_scale,**kwargs)

def c4_c5(video,output_dir,size=(320,240),fps=10,bitrate_kbps=500,timestamps=None,**kwargs):
    return _api(video,output_dir,'c4_c5',timestamps,size=size,fps=fps,bitrate_kbps=bitrate_kbps,**kwargs)
