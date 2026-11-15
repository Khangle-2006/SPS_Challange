"""Team's original C5 encoder, varying only declared FPS and max B frames."""
import os,sys,json,traceback,fcntl
from pathlib import Path
P=Path(__file__).resolve().parent;E=P.parent;OLD=E/'codec_clock_20261111'
sys.path.insert(0,str(OLD))
import exact_real_v2 as a
import padding_queue as q
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
OUT=P/'codec_exact';OUT.mkdir(exist_ok=True)
def encode(frames,fs,bf,d):
    assert not (d/'video.mp4').exists(),'Preserve partial; no automatic restart'
    h,w=frames[0].shape[:2]
    protocol=a.read('/data_pool/khangle/facevitals/code/t8_oval_20261031/protocol.json')
    rate=round(500*protocol['c5']['headroom_fraction'])
    cmd=['ffmpeg','-hide_banner','-loglevel','info','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',f'{fs:.12g}','-i','pipe:0','-an','-threads','1','-filter_threads','1','-c:v','libx264','-preset','veryfast','-pix_fmt','yuv420p','-g',str(round(2*fs)),'-keyint_min',str(round(2*fs)),'-sc_threshold','0','-b:v',f'{rate}k','-maxrate',f'{rate}k','-bufsize',f'{2*rate}k','-bf',str(bf),'-movflags','+faststart',str(d/'video.mp4')]
    a.save(d/'COMMAND.json',dict(command=cmd,source_recipe='/data_pool/khangle/facevitals/code/t8_oval_20261031/simulate.py',recipe_sha=a.sha('/data_pool/khangle/facevitals/code/t8_oval_20261031/simulate.py'),only_encoder_intervention='explicit -bf 0/3/4; target8/10fps',B_adapt_default=True,official_MATLAB_parity=False,prefix_seconds=30))
    hashed=a.hashlib.sha256()
    with (d/'encode.log').open('wb') as log:
        proc=a.subprocess.Popen(cmd,stdin=a.subprocess.PIPE,stderr=log)
        try:
            for x in frames:b=x.tobytes();hashed.update(b);proc.stdin.write(b)
            proc.stdin.close();assert proc.wait(timeout=600)==0
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
    g=a.framemeta(d/'video.mp4');a.save(d/'GOP.json',g);sch=a.schedule(g,fs)
    assert len(g['frames'])==len(frames) and np.allclose(sch['PTS'],np.arange(len(frames))/fs,atol=2e-5)
    return dict(encoded_sha=a.sha(d/'video.mp4'),input_rgb_sha=hashed.hexdigest(),wholefile_kbps=(d/'video.mp4').stat().st_size*8/(len(frames)/fs)/1000,**sch)
def main():
    lease=q.lock(OUT/'worker.lock');assert not (OUT/'LAUNCH_V2.json').exists();assert not Path('/proc/2609418').exists();assert not list(OUT.glob('real/*/*/f*_b*/RESULT.json'))
    plan=a.read(OLD/'PLAN.json')
    for j in plan['videos']:
        excluded=OLD/'real'/j['dataset']/j['recording']/'EXCLUDED.json'
        if excluded.exists():
            target=OUT/'real'/j['dataset']/j['recording'];target.mkdir(parents=True,exist_ok=True);(target/'EXCLUDED.json').write_bytes(excluded.read_bytes())
    a.save(OUT/'PLAN_V2.json',dict(**plan,new_recipe='original T8 C5 rate/GOP/default B-adapt, explicit B0/3/4',source_plan_sha256=a.sha(OLD/'PLAN.json'),mapping_sha256=a.sha(P/'exact_real_v2.py'),wrapper_sha256=a.sha(__file__),operational_source_sha256=a.sha(OLD/'run_v4.py')))
    a.save(OUT/'LAUNCH_V2.json',dict(pid=os.getpid(),utc=a.now(),budget=q.budget(fresh=True)))
    a.P=OUT;a.encode=encode;a.guard=lambda:q.budget();a.real(plan)
if __name__=='__main__':
    try:main()
    except Exception as e:a.save(OUT/'ERROR_V2.json',dict(error=str(e),traceback=traceback.format_exc()));raise
