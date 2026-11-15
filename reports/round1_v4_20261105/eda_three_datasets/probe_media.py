"""Read-only media and reference inventory. Run on P100 with one CPU thread."""
import csv
import json
import statistics as st
import subprocess
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent
PUBLIC = Path('/data_pool/khangle/facevitals/datasets')
def write(name, rows):
    with (OUT/name).open('w', newline='') as file:
        w = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)

def probe(path, dataset, group, identifier):
    opts = 'format=duration:stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,duration,nb_frames,bit_rate,color_space,color_transfer,color_primaries:stream_side_data=rotation'
    meta = json.loads(subprocess.check_output(['ffprobe','-v','error','-threads','1','-show_entries',opts,'-of','json',str(path)]))
    stream = next(s for s in meta['streams'] if s['codec_type']=='video')
    size = path.stat().st_size
    duration = float(stream.get('duration',meta['format']['duration']))
    return dict(dataset=dataset,group=group,id=identifier,file=path.name,bytes=size,seconds=duration,container_seconds=float(meta['format']['duration']),codec=stream['codec_name'],width=stream['width'],height=stream['height'],rotation=next((s['rotation'] for s in stream.get('side_data_list',[]) if 'rotation' in s),0),avg_fps=float(Fraction(stream['avg_frame_rate'])),nominal_rate=stream['r_frame_rate'],frames=stream.get('nb_frames'),pixel=stream['pix_fmt'],colour_space=stream.get('color_space',''),transfer=stream.get('color_transfer',''),audio=int(any(s['codec_type']=='audio' for s in meta['streams'])),video_kbps=float(stream['bit_rate'])/1000 if stream.get('bit_rate') else None,mux_kbps=size*8/float(meta['format']['duration'])/1000)

def gt(path, group):
    if group=='UBFC1':
        rows = list(csv.reader((path/'gtdump.xmp').read_text().splitlines()))
        t,h,oxy,ppg=zip(*[tuple(map(float,row[:4])) for row in rows if row])
        t=[x/1000 for x in t]
    else:
        ppg,h,t=[list(map(float,r.split())) for r in (path/'ground_truth.txt').read_text().splitlines() if r.strip()]
        oxy=[]
    dt=[b-a for a,b in zip(t,t[1:])]
    gap_limit=2*st.median(dt)
    return dict(group=group,id=path.name,ppg_n=len(ppg),hr_n=len(h),t_n=len(t),seconds=t[-1]-t[0],median_dt_ms=1000*st.median(dt),nonincreasing=sum(x<=0 for x in dt),gaps_gt_2median=sum(x>gap_limit for x in dt),hr_min=min(h),hr_median=st.median(h),hr_max=max(h),hr_zero=sum(x==0 for x in h),ppg_min=min(ppg),ppg_max=max(ppg),spo2_min=min(oxy) if oxy else None,spo2_max=max(oxy) if oxy else None)

def main():
    media=[]; traces=[]
    for r in csv.DictReader((OUT.parent/'organiser_download/P100_verified_files.csv').open()):
        media.append(probe(Path(r['path']),'BTC',r['group'],r['drive_id']))
    for i in [1,2]:
        base=PUBLIC/'UBFC-rPPG'/('DATASET_'+str(i))
        dirs=sorted(d for d in base.iterdir() if d.is_dir()) if i==1 else [base/k for k in json.loads((OUT/'cohort.json').read_text())['recordings']]
        for d in dirs:
            g='UBFC'+str(i)
            media.append(probe(d/'vid.avi','UBFC',g,d.name))
            traces.append(gt(d,g))
    for p in sorted((PUBLIC/'AiPEX/Data').glob('*/*.avi')):
        media.append(probe(p,'AiPEX',p.parent.name,p.stem.split('_')[0]))
    refs=[]
    for f in sorted((PUBLIC/'AiPEX/Ground_Truth').glob('*.csv')):
        for row in list(csv.reader(f.read_text(encoding='utf-8-sig').splitlines()))[1:]:
            if row and row[0].strip():
                refs.append(dict(group=f.stem.removesuffix('_GT'),id=row[0].strip(),hr=float(row[1])))
    assert len(media)==603 and len(refs)==88 and len(traces)==43
    write('media.csv',media);write('ubfc_reference.csv',traces);write('aipex_reference.csv',refs)
if __name__=='__main__':main()

