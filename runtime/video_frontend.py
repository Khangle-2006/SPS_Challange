"""Decode display PTS and run the frozen frontend; write only to the requested run."""
import collections
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
from common import save, sha
from primitives import BTCVideo, Mesh, fast_crop
from frontend_guard_v2 import Frontend


def main(out):
    req = json.loads((out/'request.json').read_text()); path = Path(req['video'])
    stream = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_streams','-of','json',str(path)]))['streams'][0]
    bitrate = req['source_bitrate'] or float(stream.get('bit_rate', 0))
    if not np.isfinite(bitrate) or bitrate <= 0:
        raise ValueError('Source stream bitrate unavailable. Supply --source-bitrate in bit/s; no guessed class.')
    video = BTCVideo(path)
    t = video.t
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Non-monotonic/missing display timestamps; cannot safely infer PR.')
    if t[-1] < 10: raise ValueError('Need at least 10 seconds of video; no padding of short clips.')
    # Bound on crop storage before allocation (240-fps clips included).
    if len(t)*72*72*3 > 2*1024**3: raise ValueError('Video too long for this single-clip CPU runner (crop limit 2 GiB).')
    n = len(t)
    crops = np.lib.format.open_memmap(out/'crop72.npy', mode='w+', dtype=np.uint8, shape=(n,72,72,3))
    valid = np.zeros(n, bool); rgb = np.zeros((n,3)); boxes = np.zeros((n,4), np.int32)
    guard = Frontend(seed_box=req['seed_box']) if req['frontend']=='guarded' else None
    mesh = Mesh() if guard is None else None
    branches = collections.Counter(); repeated = True; previous = None; count=0
    # Capture ffmpeg errors and exit status: a truncated decode is never accepted.
    with (out/'decode.log').open('w+') as err, (out/'trace.jsonl').open('w') as trace:
        process = subprocess.Popen(video.cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            for i in range(n):
                raw = process.stdout.read(video.w*video.h*3)
                if len(raw) != video.w*video.h*3: raise ValueError(f'Decoder ended at {i}/{n} frames.')
                frame = np.frombuffer(raw,np.uint8).reshape(video.h,video.w,3)
                if previous is not None and not np.array_equal(frame, previous): repeated=False
                previous=frame.copy()
                if guard:
                    r=guard.step(frame,float(t[i])); valid[i]=r['valid']; crops[i]=r['crop72']
                    # Float32 block means, then float64 mean: same frozen x01 evidence path.
                    rgb[i]=r['blk'].astype(float).mean((0,1)); boxes[i]=r['box']; tr=r['trace']
                else:
                    good, oval, box=fast_crop(frame,mesh.step(frame)); boxes[i]=box; valid[i]=good
                    crops[i]=0
                    x0,y0,x1,y1=map(int,box)
                    if good and x1-x0>=8 and y1-y0>=8:
                        crops[i]=cv2.resize(frame[y0:y1,x0:x1],(72,72),interpolation=cv2.INTER_AREA)
                    rgb[i]=oval.reshape(6,6,6,6,3).mean((1,3),dtype=np.float32).astype(float).mean((0,1))
                    tr={'branch':'native' if good else 'invalid_lost'}
                branches[tr['branch']]+=1
                trace.write(json.dumps(dict(frame=i,t=float(t[i]),**tr),allow_nan=False)+'\n')
                count+=1
                if count%300==0: print(f'frames {count}/{n}',flush=True)
            if process.stdout.read(1): raise ValueError('More decoded frames than timestamp entries.')
            if process.wait()!=0: raise ValueError('ffmpeg failed; see decode.log.')
        finally:
            process.stdout.close()
            if process.poll() is None: process.terminate(); process.wait()
            if guard: guard.close()
            if mesh: mesh.close()
    crops.flush()
    np.savez_compressed(out/'trace.npz',t=t,valid=valid,rgb_mean=rgb,boxes=boxes)
    save(out/'frontend.json',dict(frames=n,valid_frames=int(valid.sum()),duration_seconds=float(t[-1]),
        source_width=int(stream["width"]), source_height=int(stream["height"]),
        source_codec=stream["codec_name"], source_avg_frame_rate=stream.get("avg_frame_rate"),
        source_bitrate=bitrate,capture_class_low=bool(bitrate<4e6),hdr=video.hdr,
        display_size=[video.w,video.h],branches=dict(branches),static_input=bool(repeated),
        video_sha256=sha(path),frontend=req['frontend']))


if __name__=='__main__': main(Path(sys.argv[1]))
