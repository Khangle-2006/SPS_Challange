"""CPU P4m raw-video runner. S1 validation status is recorded in docs/SUBMISSION_ROUND1.md."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--dataset', choices=['UBFC2', 'MCD'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fold', type=int, choices=range(5))
    p.add_argument('--clock', type=Path, help='Optional label-free acquisition-clock NPZ, bound to the source SHA256. See docs/SUBMISSION_ROUND1.md.')
    p.add_argument('--torch-python', type=Path, default=ROOT/'.venv/torch-cpu/bin/python')
    a = p.parse_args()
    if not a.video.is_file() or (a.clock and not a.clock.is_file()): p.error('Missing input video/clock.')
    if a.output.exists(): p.error('Output must be a new directory.')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', KERAS_BACKEND='jax',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
    for k in ['PYTHONPATH', 'PYTHONHOME']: env.pop(k, None)
    exes = {'frontend': ROOT/'.venv/frontend/bin/python', 'experts': ROOT/'.venv/experts/bin/python', 'core': a.torch_python}
    for x in exes.values():
        if not x.is_file(): p.error('Missing environment: '+str(x))
    a.output.mkdir(parents=True); out = a.output.resolve()
    req = dict(video=str(a.video.resolve()), dataset=a.dataset, fold=a.fold,
               clock=str(a.clock.resolve()) if a.clock else None, stride_s=10,
               cnn_fold=0 if a.fold is None else a.fold)
    (out/'request.json').write_text(json.dumps(req, indent=2)+'\n')
    try:
        for stage, exe in exes.items():
            mem = dict(x.split(':', 1) for x in Path('/proc/meminfo').read_text().splitlines())
            if int(mem['MemAvailable'].split()[0]) < 15*1024**2: raise RuntimeError('Available RAM <15 GiB')
            with (out/(stage+'.log')).open('w') as log:
                child = subprocess.Popen(['nice', '-n', '19', str(exe), str(ROOT/'runtime'/('public_'+stage+'.py')), str(out)], env=env, stdout=log, stderr=subprocess.STDOUT)
                peak = 0
                try:
                    while child.poll() is None:
                        stat = Path(f'/proc/{child.pid}/status')
                        try:
                            rss = next((int(x.split()[1])*1024 for x in stat.read_text().splitlines() if x.startswith('VmRSS:')), 0)
                        except FileNotFoundError: rss = 0
                        peak = max(peak, rss)
                        if rss > 8*1024**3: raise RuntimeError(stage+' RSS exceeds 8 GiB')
                        time.sleep(.5)
                    if child.returncode: raise RuntimeError(stage+' failed; see '+str(out/(stage+'.log')))
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try: child.wait(timeout=10)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
                    (out/(stage+'_resources.json')).write_text(json.dumps(dict(peak_rss_bytes=peak, returncode=child.returncode))+'\n')
            print(stage, 'complete', flush=True)
    except Exception as e:
        (out/'ERROR.json').write_text(json.dumps({'error':str(e)})+'\n')
        raise
    print((out/'result.json').read_text())


if __name__ == '__main__': main()
