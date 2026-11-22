"""Create an isolated Python 3.12 environment; never alter an existing environment."""
import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('core', 'frontend', 'torch-cpu', 'experts')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('profile', choices=PROFILES)
    p.add_argument('--destination', type=Path)
    a = p.parse_args()
    if sys.version_info[:2] != (3, 12):
        p.error('Use Python 3.12 (Linux x86_64 for the complete research environments).')
    target = (a.destination or ROOT / '.venv' / a.profile).resolve()
    if target.exists():
        p.error(f'Refusing to modify existing environment: {target}')
    if sys.platform == 'linux':
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(mem['MemAvailable'].split()[0]) < 15 * 1024**2:
            p.error('Need at least 15 GiB available RAM; retry later.')
    target.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(target)
    py = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    env = dict(os.environ, PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='',
               JAX_PLATFORMS='cpu', KERAS_BACKEND='jax', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    requirements = ROOT / 'environments' / (a.profile + '.txt')
    lock = ROOT / 'environments' / 'locks' / (a.profile + '-linux-py312.txt')
    if sys.platform == 'linux' and platform.machine() == 'x86_64' and lock.is_file():
        requirements = lock
    subprocess.run([str(py), '-m', 'pip', 'install', '--no-cache-dir', '-r',
                    str(requirements)], env=env, check=True)
    subprocess.run([str(py), '-m', 'pip', 'check'], env=env, check=True)
    # Keep installation evidence outside tracked source; never export credentials/environment variables.
    out = ROOT / 'output' / 'setup' / a.profile
    out.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.check_output([str(py), '-m', 'pip', 'freeze'], env=env, text=True)
    (out / 'resolved.txt').write_text(freeze, encoding='utf-8')
    (out / 'environment.json').write_text(json.dumps({'profile': a.profile,
        'python': str(py), 'python_version': sys.version.split()[0],
        'requirements': str(requirements), 'system_site_packages': False}, indent=2) + '\n', encoding='utf-8')
    print(f'Installed {a.profile}; interpreter: {py}')

if __name__ == '__main__':
    main()
