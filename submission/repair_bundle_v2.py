"""Packaging-only repair; refuses incomplete jobs and preserves original source/provenance."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys


def repair(artifact,source):
    artifact=Path(artifact);source=Path(source)
    if json.loads((artifact/'status.json').read_text())['state']!='complete':
        raise RuntimeError('Original E2E must finish before packaging repair')
    if (artifact/'packaging_v2_receipt.json').exists():raise FileExistsError('Repair already applied')
    dst=artifact/'code'/'submission';backup=artifact/'original_packaging_v1';backup.mkdir()
    shutil.copyfile(dst/'process_round1.py',backup/'process_round1.py')
    shutil.copyfile(artifact/'script_sha256.json',backup/'script_sha256.json')
    for name in ['process_round1.py','requirements.txt','SCHEMA.md']:
        shutil.copyfile(source/name,dst/name)
    # The delivered artifact must work without the original workspace on sys.path.
    p=subprocess.run([sys.executable,str(artifact/'process_round1.py'),'--help'],cwd=artifact,
                     check=True,capture_output=True,text=True)
    (artifact/'packaging_v2_help.log').write_text(p.stdout)
    for name in ['mat_writer.py','metrics.py','requirements.txt','SCHEMA.md','checkpoints/baseline_seed2027.pt']:
        if not (dst/name).is_file():raise AssertionError('Missing portable dependency '+name)
    hashes={str(p.relative_to(artifact)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    receipt={'state':'complete','change':'non-Python packaging dependencies and propagation fix only',
             'original_e2e_status_preserved':True,'inference_or_metrics_rerun':False,
             'mat_not_modified':True,'standalone_cli_help_passed':True,'updated_files_sha256':hashes}
    (artifact/'packaging_v2_receipt.json').write_text(json.dumps(receipt,indent=2))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('artifact');p.add_argument('--source',default=str(Path(__file__).parent))
    a=p.parse_args();print(json.dumps(repair(a.artifact,a.source),indent=2))
