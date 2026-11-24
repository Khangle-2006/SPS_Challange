"""Predict only, one fresh BTC deployment process/video. Default permits DEV DRY-RUN only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cli_activity(activity):
    """Normalize locked protocol values, never infer activity from the video ID."""
    if activity in {'Task1','Task2','Task3','Task'}: return 'Task'
    if activity in {'Resting','Walking'}: return activity
    raise ValueError(f'Unknown locked activity: {activity!r}')


def fingerprint(repo):
    files=[repo/'predict.py']+sorted((repo/'runtime').glob('*.py'))+sorted((repo/'models').glob('*.json'))
    return {str(p.relative_to(repo)):sha(p) for p in files}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--authorization-file',type=Path)
    p.add_argument('--activity-lock',type=Path)
    a=p.parse_args();repo=a.repo.resolve()
    # No scan of dataset roots. Manifest contains no labels; a holdout manifest
    # is not even opened until explicit authorization is supplied.
    allowed=a.manifest.name.startswith('dev_')
    if not allowed:
        if not a.authorization_file or a.authorization_file.read_text(encoding='utf-8').strip()!='M\u1ede HOLDOUT S1':
            p.error('Holdout CLOSED. Requires the explicit user message M\u1ede HOLDOUT S1 and its saved authorization receipt.')
        if not a.activity_lock or sha(a.activity_lock)!='a0f58cc5b87888aa591ec07de6957d197a954909f85a7db6dfd95d78a664cd76':p.error('Frozen activity lock missing/changed')
    doc=json.loads(a.manifest.read_text());rows=doc['videos'];scope=doc['scope']
    if allowed and (scope!='dev_dryrun' or len(rows)!=3):p.error('Dry-run requires exactly 3 explicitly declared DEV videos')
    if not allowed and (scope!='holdout' or len(rows)!=72):p.error('Holdout manifest must contain exactly 72 videos')
    fields={'id','video','activity','source_bitrate','file_type'}
    if any(set(r)-fields or not {'id','video','activity','file_type'}<=set(r) for r in rows):p.error('Prediction manifest must contain only input metadata, never PR/person labels')
    if len({r['id'] for r in rows})!=len(rows):p.error('Duplicate video IDs')
    if not allowed:
        import csv
        with a.activity_lock.open(newline='') as f:locked=list(csv.DictReader(f))
        if len(locked)!=72:p.error('Activity lock row count changed')
        # Explicit two-column schema; do not infer/fall back to PR or filename.
        headers=set(locked[0]);idcol=next((k for k in ('recording','id','video_id') if k in headers),None);actcol=next((k for k in ('activity','PROTOCOL') if k in headers),None)
        if not idcol or not actcol or headers!={idcol,actcol}:p.error('Unexpected activity lock schema')
        if {(r['id'],r['activity']) for r in rows}!={(r[idcol],r[actcol]) for r in locked}:p.error('Holdout IDs/activity differ from lock')
    activities=[cli_activity(r['activity']) for r in rows]
    a.output.mkdir(parents=True,exist_ok=False)
    before=fingerprint(repo)
    pre=dict(scope=scope,manifest_sha256=sha(a.manifest),code_model_sha256=before,fold=None,expected_videos=len(rows),labels_read=False)
    (a.output/'PRE_RUN.json').write_text(json.dumps(pre,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    results=[]
    for i,row in enumerate(rows):
        dest=a.output/('video_%03d'%i)
        cmd=['nice','-n','19',sys.executable,str(repo/'predict.py'),'--video',row['video'],'--activity',activities[i],'--output',str(dest)]
        if row.get('source_bitrate') is not None:cmd+=['--source-bitrate',str(row['source_bitrate'])]
        with (a.output/('video_%03d.log'%i)).open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        r=json.loads((dest/'result.json').read_text())
        results.append(dict(id=row['id'],file_type=row['file_type'],pr_bpm=r['pr_bpm'],abstain=r['abstain'],result=str(dest.relative_to(a.output)/'result.json'),sha256=sha(dest/'result.json')))
        if fingerprint(repo)!=before:raise RuntimeError('Code/model changed during prediction; no seal issued')
    pred=a.output/'PREDICTIONS.json';pred.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    # Seal ONLY after every fresh-process prediction succeeds, before score script reads labels.
    seal=dict(predictions_sha256=sha(pred),pre_run_sha256=sha(a.output/'PRE_RUN.json'),scope=scope,videos=len(results),labels_read=False)
    with (a.output/'SEALED.json').open('x') as f:json.dump(seal,f,indent=2)
    print(json.dumps(seal))


if __name__=='__main__':main()
