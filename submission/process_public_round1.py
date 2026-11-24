"""Round1_Baseline wrapper: invoke the unchanged public P4m CLI with relative paths."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--repo',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args();b=a.bundle.resolve();repo=(a.repo or b.parent/'repo').resolve()
    if not (repo/'predict_public.py').is_file():p.error('Clone SPS_Challange beside Round1_Baseline as repo, run bootstrap_video.py and setup_env.py torch-cpu, or pass --repo.')
    cfg=json.loads((b/'baseline_config.json').read_text());out=(a.output or b/'rerun').resolve()
    cmd=[sys.executable,str(repo/'predict_public.py'),'--video',str(b/'target_video.mp4'),'--dataset','UBFC2','--fold',str(cfg['fold']),'--clock',str(b/'acquisition_clock.npz'),'--output',str(out)]
    subprocess.run(cmd,check=True)
    print('Window HR output:',out/'processed_output.csv')


if __name__=='__main__':main()
