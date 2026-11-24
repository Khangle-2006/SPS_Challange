"""One preselected record per machine; never restart existing outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops'/'deep_hr'),str(ROOT)]
from data_adapters import UBFC2,MCD


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['ubfc2','mcd'],required=True)
    p.add_argument('--root',required=True);p.add_argument('--audit-dir');p.add_argument('--recording',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--roi-cache',required=True)
    a=p.parse_args();backend=UBFC2(a.root) if a.dataset=='ubfc2' else MCD(a.root,a.audit_dir)
    rec=backend.load(a.recording);ppg=rec.require_synchronized_ppg();a.out.mkdir(parents=True,exist_ok=True)
    reference=a.out/'reference_input.csv'
    if reference.exists():raise FileExistsError('Preserve existing E2E run; create a versioned output to retry')
    np.savetxt(reference,np.column_stack([ppg.timestamps,ppg.values]),delimiter=',',header='timestamps_s,ppg',comments='',fmt='%.17g')
    command=[sys.executable,str(ROOT/'submission'/'process_round1.py'),'--video',str(rec.frames.path),
             '--reference_csv',str(reference),'--out_dir',str(a.out/'artifact'),'--dataset',a.dataset,
             '--root',a.root,'--recording',a.recording,'--roi-cache',a.roi_cache]
    if a.audit_dir:command+=['--audit-dir',a.audit_dir]
    (a.out/'command.json').write_text(json.dumps(command,indent=2))
    subprocess.run(command,check=True)


if __name__=='__main__':main()
