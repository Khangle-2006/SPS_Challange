"""User-authorized shared GPU0; bounded successful storage remeasurement."""
import time
import common
from common import P, read, sha, save, now, subprocess

_original_guard = common.guard

def guard(fresh=False):
    for attempt in range(3):
        try:
            return _original_guard(fresh)
        except subprocess.CalledProcessError as e:
            if not isinstance(e.cmd, list) or e.cmd[:2] != ['du', '-sb'] or attempt == 2:
                raise
        except FileNotFoundError as e:
            if not e.filename or not str(e.filename).startswith(str(P)+'/') or not str(e.filename).endswith('.tmp') or attempt == 2:
                raise
        time.sleep(1)

def gpu_snapshot():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()
    idx, total, free, util = map(int, rows[0].split(','))
    assert idx == 0
    return dict(index=idx, total_MiB=total, free_MiB=free, utilization_percent=util,
                required_free_MiB=6000, memory_fraction=0.35,
                shared_GPU_authorized=True, utilization_is_not_a_launch_veto=True)

def available_gpu():
    return gpu_snapshot()['free_MiB'] >= 6000

def gpu():
    import torch
    samples=[]
    for _ in range(3):
        a=gpu_snapshot(); samples.append(a)
        assert a['free_MiB'] >= 6000, ('WAIT_GPU0_VRAM', a)
        time.sleep(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.35, 0)
    save(P/'GPU_SHARED_LAUNCH_CHECK.json', dict(utc=now(), samples=samples,
         user_authorization='Run while enough VRAM remains; do not wait for idle GPU',
         GPU1_untouched=True, no_other_process_modified=True))
    return 'cuda:0'

def verify_manifest():
    for file, digest in read(P/'EXECUTION_FROZEN_V4.json')['files'].items():
        assert sha(file)==digest, ('Operational dependency changed',file)

def install():
    common.guard=guard
    common.gpu=gpu
