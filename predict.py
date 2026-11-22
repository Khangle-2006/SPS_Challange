"""Run the fixed BTC PR estimator on one user-supplied video, CPU only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True, type=Path)
    p.add_argument('--activity', required=True, choices=['Resting', 'Task', 'Walking'],
                   help='Known recording activity; never inferred from the filename.')
    p.add_argument('--output', required=True, type=Path,
                   help='A new directory for result.json, result.csv and processing evidence.')
    p.add_argument('--frontend', choices=['guarded', 'reference'], default='guarded',
                   help='guarded keeps target identity; reference is the historical single-face reproduction route.')
    p.add_argument('--fold', type=int, choices=range(5),
                   help='Research only: use the matching held-out dev fold instead of the deployment fit.')
    p.add_argument('--source-bitrate', type=float,
                   help='Original video stream bit/s, required if absent or input is a transformed derivative.')
    p.add_argument('--seed-box', type=float, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'),
                   help='Optional target face box in display-oriented source pixels (guarded only).')
    a = p.parse_args()
    if not a.video.is_file(): p.error('Video does not exist.')
    if a.output.exists(): p.error('Output already exists. Choose a new directory; existing results are never overwritten.')
    if a.seed_box and a.frontend != 'guarded': p.error('--seed-box requires guarded frontend.')
    if a.source_bitrate is not None and a.source_bitrate <= 0: p.error('Bitrate must be positive bit/s.')
    for exe in ['ffmpeg', 'ffprobe']:
        if not shutil.which(exe): p.error(f'Missing {exe}; install ffmpeg first.')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', KERAS_BACKEND='jax',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
    for k in ['PYTHONPATH', 'PYTHONHOME']: env.pop(k, None)
    interpreters = {s: ROOT/'.venv'/s/'bin/python' for s in ['frontend', 'experts', 'core']}
    for exe in interpreters.values():
        if not exe.is_file(): p.error('Run python3.12 tools/bootstrap_video.py first.')
    if not (ROOT/'models/MANIFEST.json').is_file(): p.error('Missing shipped reader models; obtain the complete repository.')
    # Resource check is repeated before every child; never kill unrelated jobs.
    def check_ram():
        m = Path('/proc/meminfo')
        if m.exists():
            d = dict(x.split(':', 1) for x in m.read_text().splitlines())
            if int(d['MemAvailable'].split()[0]) < 15*1024**2:
                raise RuntimeError('Available RAM <15 GiB. Retry later with a new output directory.')
    check_ram()
    a.output.mkdir(parents=True)
    out = a.output.resolve()
    (out/'.gitignore').write_text('*\n', encoding='utf-8')
    request = dict(video=str(a.video.resolve()), activity=a.activity, frontend=a.frontend,
                   fold=a.fold, source_bitrate=a.source_bitrate, seed_box=a.seed_box)
    (out/'request.json').write_text(json.dumps(request, indent=2)+'\n')
    started = time.monotonic()
    try:
        for stage in ['frontend', 'experts', 'core']:
            check_ram()
            print(f'[{stage}] processing; log: {out / (stage+".log")}', flush=True)
            cmd = [str(interpreters[stage]), str(ROOT/'runtime'/('video_'+stage+'.py')), str(out)]
            if shutil.which('nice'): cmd = ['nice', '-n', '10'] + cmd
            with (out/(stage+'.log')).open('w') as log:
                child=subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                peak=0
                try:
                    while child.poll() is None:
                        status=Path(f'/proc/{child.pid}/status')
                        if status.exists():
                            lines=status.read_text().splitlines()
                            rss=next((int(x.split()[1])*1024 for x in lines if x.startswith('VmRSS:')),0)
                            peak=max(peak,rss)
                            if rss>8*1024**3: raise RuntimeError(f'{stage} exceeds 8 GiB RSS; stopped this run only.')
                        time.sleep(.5)
                    if child.returncode: raise subprocess.CalledProcessError(child.returncode,cmd)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    (out/(stage+'_resources.json')).write_text(json.dumps({'peak_rss_bytes':peak,'returncode':child.returncode})+'\n')
        result = json.loads((out/'result.json').read_text())
        result['elapsed_seconds'] = round(time.monotonic()-started, 3)
        (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(result, indent=2))
    except Exception as e:
        (out/'ERROR.json').write_text(json.dumps({'error': str(e), 'seconds': time.monotonic()-started}, indent=2)+'\n')
        print(f'FAILED: {e}. See logs in {out}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
