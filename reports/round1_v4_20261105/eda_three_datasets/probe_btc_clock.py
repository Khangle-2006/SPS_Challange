"""Inspect display timestamps of compressed packets, without decoding frames."""
import csv
import json
import statistics
import subprocess
from pathlib import Path

base=Path(__file__).resolve().parent
rows=[]
for row in csv.DictReader((base.parent/'organiser_download/P100_verified_files.csv').open()):
    payload=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_packets','-show_entries','packet=pts_time','-of','json',row['path']])
    packets=json.loads(payload)['packets']
    times=sorted(float(p['pts_time']) for p in packets if 'pts_time' in p)
    dt=[round(1000*(b-a),3) for a,b in zip(times,times[1:])]
    med=statistics.median(dt)
    rows.append(dict(id=row['drive_id'],packets=len(packets),pts=len(times),duplicates=dt.count(0),median_dt_ms=med,max_dt_ms=max(dt),gt_1_5median=sum(x>1.5*med for x in dt),gt_2median=sum(x>2*med for x in dt),span=times[-1]-times[0]))
with (base/'btc_clock.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)

