"""Summarize three decoded grayscale thumbnails in memory, without saving images."""
from pathlib import Path
import csv
import statistics
import subprocess
import math

base=Path(__file__).resolve().parent
public=Path('/data_pool/khangle/facevitals/datasets')
files={r['drive_id']:Path(r['path']) for r in csv.DictReader((base.parent/'organiser_download/P100_verified_files.csv').open())}
records=[]
for row in csv.DictReader((base/'media.csv').open()):
    if row['dataset']=='BTC':
        path=files[row['id']]
    elif row['dataset']=='UBFC':
        path=public/'UBFC-rPPG'/('DATASET_'+row['group'][-1])/row['id']/row['file']
    else:
        path=public/'AiPEX/Data'/row['group']/row['file']
    samples=[]
    for frac in [.1,.5,.9]:
        cmd=['ffmpeg','-nostdin','-v','error','-threads','1','-ss',str(frac*float(row['seconds'])),'-i',str(path),'-map','0:v:0','-an','-sn','-filter_threads','1','-vf','scale=64:64:flags=area,format=gray','-frames:v','1','-f','rawvideo','-threads','1','pipe:1']
        pix=subprocess.check_output(cmd)
        if len(pix)==4096:
            avg=statistics.mean(pix)
            samples.append((avg,statistics.pstdev(pix),sum(p<=10 for p in pix)/len(pix),sum(p>=245 for p in pix)/len(pix)))
    vals=[statistics.mean(x) for x in zip(*samples)] if samples else [None]*4
    records.append(dict(group=row['group'],id=row['id'],file=row['file'],n=len(samples),luma=vals[0],contrast=vals[1],dark_fraction=vals[2],bright_fraction=vals[3]))
with (base/'sample_luma.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)

