"""Apply the user's two-primary-dataset choice; restrict deletion to reviewed downloads."""
import json, os, signal, shutil, subprocess, time
from pathlib import Path

BASE = Path('/mnt/samsung_t5/khangle/facevitals/datasets')
PROJECT = Path('/data_pool/khangle/facevitals')
REPORT = PROJECT / 'reports/storage_consolidation_20261030'
REPORT.mkdir(parents=True, exist_ok=True)
assert BASE.resolve() == BASE
targets = [BASE / 'SCAMPS', BASE / 'MPU-rPPG_sample']
inventory = []
for target in targets:
    assert target.is_dir() and not target.is_symlink()
    assert target.resolve().parent == BASE and target.stat().st_uid == os.getuid()
    for p in target.rglob('*'):
        assert not p.is_symlink(), f'Unexpected symlink: {p}'
        assert p.resolve().is_relative_to(target)
        if p.is_file():
            inventory.append({'path': str(p), 'bytes': p.stat().st_size})
    link = PROJECT / 'datasets' / target.name
    assert link.is_symlink() and link.resolve() == target

jobs = {2659537: 'download_direct_datasets.py', 2567696: 'download_mcd.py'}
for pid, expected in jobs.items():
    proc = Path('/proc') / str(pid)
    cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert proc.stat().st_uid == os.getuid() and expected in cmd, (pid, cmd)
for pid in jobs:
    os.kill(pid, signal.SIGKILL)  # Stopped processes cannot execute graceful handlers.
time.sleep(1)
for pid in jobs:
    stat = Path('/proc') / str(pid) / 'stat'
    assert not stat.exists() or stat.read_text().split(') ', 1)[1].startswith('Z ')

(REPORT / 'removed_file_inventory.json').write_text(json.dumps(inventory, indent=2))
before = subprocess.check_output(['du', '-s', '-B1', *map(str, targets)], text=True)
(REPORT / 'removed_disk_bytes.txt').write_text(before)
for target in targets:
    archive = REPORT / target.name
    archive.mkdir(exist_ok=True)
    for p in target.rglob('*.json'):
        dst = archive / p.relative_to(target)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    # Exact, resolved, allowlisted trees only. No old experiment outputs are touched.
    shutil.rmtree(target)
    (PROJECT / 'datasets' / target.name).unlink()
    (archive / 'REMOVED.txt').write_text('Removed 2026-10-30 per user request to retain only core datasets. Historical status JSONs are not live.\n')

policy = {
    'updated': '2026-10-30', 'units': 'decimal bytes, including raw/cache/results/environments',
    'combined_server_ceiling': 500_000_000_000,
    'p100_operating_budget': 300_000_000_000,
    'a100_operating_budget': 150_000_000_000,
    'unallocated_margin': 50_000_000_000,
    'primary_now': ['MCD-rPPG', 'UBFC-rPPG'],
    'small_official_auxiliary_retained': ['AiPEX'],
    'removed': ['SCAMPS', 'MPU-rPPG_sample'],
    'future_priority': 'Organizer dataset, after access, size and label audit; reassess second primary',
    'no_bulk_acquisition': ['SCAMPS', 'MPU-rPPG full', 'MMPD', 'UCLA', 'CHILL'],
    'new_cache_target_bytes': 60_000_000_000,
    'new_results_target_bytes': 10_000_000_000,
    'note': 'Application-level MCD guard, not an OS quota; measure all project roots before future jobs.'
}
(PROJECT / 'STORAGE_POLICY.json').write_text(json.dumps(policy, indent=2))
(REPORT / 'storage_policy.json').write_text(json.dumps(policy, indent=2))
note = BASE / 'DOWNLOADS_PAUSED_SHARED_STORAGE_20261030.txt'
if note.exists():
    note.write_text('Superseded by /data_pool/khangle/facevitals/STORAGE_POLICY.json. SCAMPS/MPU removed; only MCD resumed with budget guard.\n')
log = PROJECT / 'logs/mcd_download_budgeted_20261030.log'
with log.open('ab', buffering=0) as out:
    child = subprocess.Popen([str(PROJECT / 'envs/hr/bin/python'), '-u',
        str(PROJECT / 'code/download_mcd.py'), str(BASE / 'MCD-rPPG')],
        stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
        start_new_session=True, cwd=str(PROJECT))
result = {'removed_allocated_bytes': sum(int(x.split()[0]) for x in before.splitlines()),
          'mcd_pid': child.pid, 'mcd_log': str(log), 'removed': [str(x) for x in targets]}
(REPORT / 'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
print(subprocess.check_output(['du', '-s', '-B1', str(BASE.parent), str(PROJECT)], text=True))
