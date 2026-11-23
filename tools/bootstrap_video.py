"""One-time CPU setup for predict.py; isolated locked environments and weight verification."""
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    if sys.version_info[:2]!=(3,12) or sys.platform!='linux' or platform.machine()!='x86_64':
        raise SystemExit('Use Linux x86_64 with Python 3.12. This is the tested platform.')
    if not all(shutil.which(x) for x in ['ffmpeg','ffprobe']):
        raise SystemExit('Install ffmpeg first: sudo apt-get install ffmpeg python3.12-venv')
    for profile in ['core','frontend','experts']:
        target=ROOT/'.venv'/profile; py=target/'bin/python'
        if not target.exists():
            subprocess.run([sys.executable,str(ROOT/'tools/setup_env.py'),profile],check=True)
        elif not py.is_file():
            raise SystemExit(f'Incomplete environment {target}. Use a fresh clone; no environment overwritten.')
        # Check every pinned package; do not silently accept a stale environment.
        pins=[]
        for line in (ROOT/'environments/locks'/f'{profile}-linux-py312.txt').read_text().splitlines():
            if '==' in line and not line.lstrip().startswith('#'): pins.append(line.strip().split('==',1))
        code='import importlib.metadata as m,json,sys; p=json.loads(sys.argv[1]); bad=[(n,v,m.version(n)) for n,v in p if m.version(n)!=v]; assert not bad,bad'
        subprocess.run([str(py),'-c',code,json.dumps(pins)],check=True)
        subprocess.run([str(py),'-m','pip','check'],check=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',KERAS_BACKEND='jax',PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    subprocess.run([str(ROOT/'.venv/experts/bin/python'),str(ROOT/'runtime/check_install.py')],env=env,check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/fetch_inference_assets.py')],check=True)
    print('\nReady. Run: python3.12 predict.py --video /path/video.mp4 --activity Resting --output output/my_video')


if __name__=='__main__': main()
