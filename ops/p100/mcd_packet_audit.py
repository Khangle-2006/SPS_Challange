"""Read-only packet/frame diagnostics. No automatic capture-clock repair."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse, csv, json, hashlib, struct, subprocess, time
from pathlib import Path
from datetime import datetime
from collections import Counter
import av, numpy as np
ROOT=Path('/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG')
OUT=Path('/mnt/samsung_t5/khangle/facevitals/mapping_audit_20261030')
OUT.mkdir(exist_ok=True)

def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def probe(path,folder,label,showall=False):
    cmd=['ffprobe','-v','warning','-threads','1']
    if showall:cmd+=['-flags2','+showall']
    cmd+=['-select_streams','v:0','-count_packets','-count_frames','-show_frames',
        '-show_entries','stream=codec_name,codec_tag_string,has_b_frames,avg_frame_rate,r_frame_rate,time_base,start_time,duration,nb_frames,nb_read_frames,nb_read_packets:frame=key_frame,pict_type,pkt_pos,pkt_size,pts,pts_time,best_effort_timestamp,best_effort_timestamp_time,pkt_dts,pkt_duration,decode_error_flags',
        '-of','json',str(path)]
    r=subprocess.run(cmd,capture_output=True,text=True)
    (folder/(label+'.stderr.txt')).write_text(r.stderr)
    (folder/(label+'.json')).write_text(r.stdout)
    assert r.returncode==0,(label,r.returncode,r.stderr[-1000:])
    return json.loads(r.stdout),r.stderr

def audit(row,manifest):
    path=ROOT/row['video'];folder=OUT/path.stem;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text())
    h=hashlib.sha256()
    with path.open('rb') as f:
        head=f.read(12);h.update(head)
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    source=manifest[row['video']]
    assert path.stat().st_size==source['size'] and h.hexdigest()==source['lfs_sha256']
    rawlines=[line for line in (ROOT/row['meta']).read_text().splitlines() if line.strip()]
    ids=[];clock=[]
    for line in rawlines:
        i,t=line.split(maxsplit=1);ids.append(int(i));clock.append(datetime.fromisoformat(t))
    rel=np.asarray([(t-clock[0]).total_seconds() for t in clock]);delta=np.diff(rel)
    med=float(np.median(delta));gaps=np.where(delta>1.5*med)[0]
    packets=[]
    with av.open(str(path)) as c:
        for p in c.demux(video=0):
            if p.size==0:continue # exclude decoder flush sentinel
            data=bytes(p);vops=[];at=0
            while True:
                at=data.find(b'\x00\x00\x01\xb6',at)
                if at<0:break
                vops.append({'offset':at,'coding_type':int(data[at+4]>>6) if at+4<len(data) else None})
                at+=4
            packets.append({'index':len(packets),'pts':p.pts,'dts':p.dts,'time_base':str(p.time_base),'pos':p.pos,
                'size':p.size,'key':p.is_keyframe,'is_corrupt':p.is_corrupt,'vops':vops,
                'hex_if_small':data.hex() if len(data)<=32 else None})
    save(folder/'packets.json',packets)
    default,err=probe(path,folder,'default')
    shown,serr=probe(path,folder,'showall',True)
    frames=default.get('frames',[]);sframes=shown.get('frames',[])
    default_times=[f.get('pts',f.get('best_effort_timestamp')) for f in frames]
    show_times=[f.get('pts',f.get('best_effort_timestamp')) for f in sframes]
    pos=set(int(f['pkt_pos']) for f in frames if 'pkt_pos' in f)
    unused=[p['index'] for p in packets if p['pos'] not in pos]
    riff_size=struct.unpack('<I',head[4:8])[0] if head[:4]==b'RIFF' else None
    riff_bytes=riff_size+8 if riff_size is not None and riff_size!=0xffffffff else None
    pts=np.asarray([float(f.get('pts_time',f.get('best_effort_timestamp_time','nan'))) for f in frames])
    dif=np.diff(pts);pmed=float(np.nanmedian(dif)) if len(dif) else None
    summary={'video':row['video'],'subject':row['patient_id'],'camera':row['camera'],'step':row['step'],
        'sha256':h.hexdigest(),'publisher_hash_match':True,'file_bytes':path.stat().st_size,
        'riff_length_is_sentinel':riff_size==0xffffffff,
        'riff_declared_bytes':riff_bytes,'riff_bytes_minus_file':riff_bytes-path.stat().st_size if riff_bytes else None,
        'metadata_rows':len(ids),'metadata_first_id':ids[0],'metadata_last_id':ids[-1],
        'metadata_first_time':clock[0].isoformat(),'metadata_last_time':clock[-1].isoformat(),
        'metadata_index_contiguous':bool(np.all(np.diff(ids)==1)),
        'metadata_delta_median_s':med,'metadata_gap_rows_1_5x':gaps.tolist(),
        'metadata_gap_seconds':delta[gaps].tolist(),'metadata_delta_max_s':float(delta.max()),
        'packets':len(packets),'packet_sizes_min':min(p['size'] for p in packets),
        'small_packets_le32':sum(p['size']<=32 for p in packets),
        'multi_vop_packets':sum(len(p['vops'])>1 for p in packets),
        'vop_coding_type_counts':dict(Counter(v['coding_type'] for p in packets for v in p['vops'])),
        'packet_positions_without_output_frame':unused,
        'frames_default':len(frames),'frames_showall':len(sframes),
        'showall_additional_frames':len(sframes)-len(frames),'same_pts_default_showall':default_times==show_times,
        'first_frame':frames[:1],'last_frame':frames[-1:],
        'first_keyframe_index':next((i for i,f in enumerate(frames) if f.get('key_frame')==1),None),
        'frame_types':dict(Counter(f.get('pict_type','?') for f in frames)),
        'frame_pts_gap_indices':np.where(dif>1.5*pmed)[0].tolist() if pmed else [],
        'stream':default.get('streams',[]),
        'decoder_warning_lines':err.splitlines(),'showall_warning_lines':serr.splitlines(),
        'acceptance':'reject_unresolved','reason':'No verified bridge from decoded presentation picture to capture metadata row; diagnostic only.'}
    save(folder/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['video','metadata_rows','packets','frames_default','frames_showall','riff_bytes_minus_file','small_packets_le32','frame_types']}),flush=True)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan');ap.add_argument('--subject',default='1020');a=ap.parse_args()
    rows=list(csv.DictReader((ROOT/'db.csv').open()))
    manifest={x['path']:x for x in json.loads((ROOT/'download_manifest.json').read_text())['files']}
    selected=json.loads(Path(a.plan).read_text())['audit_records'] if a.plan else [r for r in rows if r['patient_id']==a.subject]
    result=[]
    for row in selected:
        result.append(audit(row,manifest))
        save(OUT/('expanded_results.json' if a.plan else 'subject1020_results.json'),result)
    print('COMPLETE',len(result),flush=True)
