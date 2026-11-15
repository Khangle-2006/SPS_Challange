"""Approved 65GB operational guard around the unchanged frozen T8 runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

CODE = Path('/data_pool/khangle/facevitals/code/t8_oval_training_20261031')
CAP = 65_000_000_000
HEADROOM = 100_000_000

def check_total(p100_bytes, a100_origin_bytes):
    if p100_bytes + a100_origin_bytes > CAP-HEADROOM:
        raise RuntimeError('T8 approved65GB actual-storage budget including old/new/origin/replica/checkpoints')

def storage_guard(index, output):
    roots = [Path(p) for p in index['roots'].values()]
    size = sum(p.stat().st_size for root in roots+[Path(output)] for p in root.rglob('*') if p.is_file())
    origin_status = json.loads((Path(index['roots']['UBFC2'])/'status.json').read_text())
    origin_size = origin_status['bytes']+origin_status['legacy_bytes']
    legacy = Path(json.loads((Path(index['roots']['MCD'])/'status.json').read_text())['legacy_root'])
    legacy_size = sum(p.stat().st_size for p in legacy.rglob('*') if p.is_file())
    size += legacy_size
    check_total(size, origin_size)
    return dict(P100_bytes=size, A100_UBFC_origin_bytes_including_legacy=origin_size,
                P100_legacy_included=legacy_size, approved_T8_cap_bytes=CAP,
                guard_headroom_bytes=HEADROOM,
                operational_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def main(a):
    sys.path.insert(0, str(CODE))
    import t8_run
    from t8_plan import code_fingerprint, read, sha, write
    assert code_fingerprint(CODE) == read(CODE/'FROZEN_SHA256.json')
    binding = dict(approved_by='Explicit user55GBcache/65GBwhole approval',
                   approved_T8_cap_bytes=CAP, headroom_bytes=HEADROOM,
                   operational_wrapper_sha256=sha(Path(__file__)),
                   frozen_runner_sha256=sha(CODE/'t8_run.py'),
                   method_changes=False, only_override='storage_guard',
                   index_sha256=sha(a.index))
    a.output.mkdir(parents=True,exist_ok=True)
    sidecar = a.output/'OPERATIONAL_BUDGET.json'
    if sidecar.exists():
        assert read(sidecar)==binding
    else:
        write(sidecar,binding)
    t8_run.storage_guard = storage_guard
    try:
        t8_run.main(a)
    except Exception as exc:
        write(a.output/'queue_error.json', dict(state='waiting_resource' if 'RESOURCE_WAIT:' in str(exc) else 'blocked_error',
              error=str(exc),traceback=traceback.format_exc(),pid=os.getpid()))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke-only',action='store_true')
    main(p.parse_args())
