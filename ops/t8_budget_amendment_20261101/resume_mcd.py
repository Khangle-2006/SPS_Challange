"""Operational quota amendment only; execute the unchanged frozen cache worker."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

BASE = Path('/mnt/samsung_t5/khangle/facevitals')
RUNTIME = Path('/data_pool/khangle/facevitals')
CODE = RUNTIME/'code/t8_oval_20261031'
OUT = BASE/'t8_oval_20261031/MCD'
OLD = BASE/'t8_20261031/MCD'
CAP = 55_000_000_000
WHOLE_CAP = 65_000_000_000
# Verified UBFC origins, prepared replica, and checkpoint/training reservation.
UBFC_ORIGINS = 5_742_780_105
REPLICA = 3_576_196_685
TRAINING = 1_000_000_000
# Additional allowance for metadata/log growth on A100 and relay.
OVERHEAD = 10_000_000
EFFECTIVE_CAP = min(CAP, WHOLE_CAP-UBFC_ORIGINS-REPLICA-TRAINING-OVERHEAD)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    launch_lock = (OUT/'budget_resume.lock').open('a')
    fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = json.loads((OUT/'status.json').read_text())
    for pid in (1459060, 1699902, status['pid']):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f'Existing worker PID {pid}; no duplicate launch')
    assert status['state'] == 'blocked_error' and status['completed'] == 1150
    assert len(status['errors']) == 1
    assert 'Cache/disk budget guard' in status['errors'][0]['traceback']
    worker_lock = (OUT/'worker.lock').open('a')
    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hashes = json.loads((CODE/'SHA256.json').read_text())
    assert hashes == json.loads((OUT/'fingerprint.json').read_text())
    assert all(digest(CODE/name) == expected for name, expected in hashes.items())
    cohort = json.loads((OUT/'cohort.json').read_text())
    missing = []
    for rid in cohort['recordings']:
        for variant in cohort['variants']:
            dest = OUT/rid/variant
            receipt = dest/'complete.json'
            if receipt.exists():
                assert json.loads(receipt.read_text())['fingerprint'] == hashes
            else:
                assert variant == 'f30_b800'
                assert not (dest/'crop').exists() and not (dest/'encoded').exists()
                missing.append(str(dest.relative_to(OUT)))
    assert len(missing) == 40
    actual = sum(p.stat().st_size for root in (OUT, OLD) for p in root.rglob('*') if p.is_file())
    assert actual < EFFECTIVE_CAP-150_000_000
    assert shutil.disk_usage(OUT).free > 5_000_000_000
    project = sum(int(subprocess.check_output(['du', '-sb', str(p)]).split()[0]) for p in (BASE, RUNTIME))
    assert project + 59_164_557_140 + (EFFECTIVE_CAP-actual) + REPLICA + 50_000_000_000 < 500_000_000_000
    archive = OUT/'operational_budget_amendment_20261101'
    archive.mkdir(exist_ok=False)
    shutil.copy2(OUT/'status.json', archive/'blocked_status.json')
    for relative in missing:
        error = OUT/relative/'error.json'
        if error.exists():
            shutil.copy2(error, archive/(relative.replace('/', '__')+'__error.json'))
    amendment = dict(user_approved=True, approved_P100_cache_bytes=CAP,
                     approved_T8_whole_bytes=WHOLE_CAP, project_cap_bytes=500_000_000_000,
                     effective_worker_budget_bytes=EFFECTIVE_CAP, overhead_reserve_bytes=OVERHEAD,
                     UBFC_origins_bytes=UBFC_ORIGINS, UBFC_replica_bytes=REPLICA,
                     training_reserve_bytes=TRAINING, actual_MCD_including_legacy_bytes=actual,
                     missing=missing, preserved_receipts=1150, frontend_hashes=hashes,
                     utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (archive/'amendment.json').write_text(json.dumps(amendment, indent=2))
    cmd = ['nice','-n','10',str(RUNTIME/'envs/storage_pilot/bin/python'),'-u',
           'prepare_t8_cache.py','--dataset','MCD','--root',str(BASE/'datasets/MCD-rPPG'),
           '--output',str(OUT),'--reuse-root',str(OLD),'--audit',str(BASE/'mapping_audit_20261030'),
           '--budget-bytes',str(EFFECTIVE_CAP)]
    fcntl.flock(worker_lock, fcntl.LOCK_UN)
    with (OUT/'worker.log').open('ab') as log:
        proc = subprocess.Popen(cmd, cwd=CODE, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    amendment.update(pid=proc.pid, command=cmd)
    (archive/'launch.json').write_text(json.dumps(amendment, indent=2))
    print(json.dumps({'pid':proc.pid, 'missing':len(missing), 'effective_cap':EFFECTIVE_CAP}), flush=True)

if __name__ == '__main__':
    main()
