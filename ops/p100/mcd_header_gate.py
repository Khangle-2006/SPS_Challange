"""Read AVI headers and audit the user-specified ordinal-prefix rule (no training)."""
import argparse, csv, hashlib, json, statistics, struct
from datetime import datetime, timezone
from pathlib import Path

def avi_header(path):
    result = {'file_bytes': path.stat().st_size, 'avih': [], 'strh': [], 'dmlh': []}
    with path.open('rb') as f:
        first = f.read(12)
        if first[:4] != b'RIFF' or first[8:] != b'AVI ':
            raise ValueError('Not RIFF AVI; requires a different parser')
        declared = struct.unpack_from('<I', first, 4)[0]
        result.update(riff_size_field=declared, riff_size_hex=hex(declared),
                      riff_is_placeholder=declared == 0xffffffff,
                      riff_declared_total_bytes=None if declared == 0xffffffff else declared + 8)
        def walk(start, end, trail):
            pos = start
            while pos + 8 <= end:
                f.seek(pos); tag, n = struct.unpack('<4sI', f.read(8))
                payload, stop = pos + 8, pos + 8 + n
                if tag == b'LIST':
                    kind = f.read(4)
                    if kind != b'movi':
                        walk(payload + 4, min(stop, end), trail + [kind.decode('latin1')])
                elif tag in (b'avih', b'strh', b'dmlh'):
                    f.seek(payload); data = f.read(min(n, 64, end-payload))
                    base = {'chunk_offset': pos, 'payload_size': n, 'list_path': trail}
                    if tag == b'avih' and len(data) >= 40:
                        vals = struct.unpack_from('<10I', data)
                        base.update(dwMicroSecPerFrame=vals[0], dwFlags=vals[3],
                                    dwTotalFrames=vals[4], dwStreams=vals[6],
                                    dwWidth=vals[8], dwHeight=vals[9])
                    elif tag == b'strh' and len(data) >= 36:
                        base.update(fccType=data[:4].decode('latin1'), fccHandler=data[4:8].decode('latin1'),
                                    dwScale=struct.unpack_from('<I', data, 20)[0],
                                    dwRate=struct.unpack_from('<I', data, 24)[0],
                                    dwStart=struct.unpack_from('<I', data, 28)[0],
                                    dwLength=struct.unpack_from('<I', data, 32)[0])
                    elif tag == b'dmlh' and len(data) >= 4:
                        base['dwTotalFrames'] = struct.unpack_from('<I', data)[0]
                    result[tag.decode()].append(base)
                if stop > end: break
                pos = stop + (n & 1)
        walk(12, result['file_bytes'], [])
    return result

def metadata_gate(path, frames):
    lines = [x for x in path.read_text().splitlines() if x.strip()]
    pairs = [x.split(maxsplit=1) for x in lines]
    ids = [int(x[0]) for x in pairs]
    times = [datetime.fromisoformat(x[1]) for x in pairs]
    # Integer microseconds prevent floating-point threshold artifacts at exactly 1.5x.
    dt = [int((b-a).total_seconds()*1_000_000 + .5) if b>=a else -int((a-b).total_seconds()*1_000_000+.5)
          for a,b in zip(times, times[1:])]
    med = statistics.median(dt) if dt else 0
    gaps = [i+1 for i,x in enumerate(dt) if 2*x > 3*med]
    excess = len(lines) - frames  # count before dropping final decoded frame
    reasons = []
    if excess > 90: reasons.append('excess_metadata_rows_gt90')
    if gaps: reasons.append('metadata_gap_gt1.5_median')
    if excess < 0: reasons.append('metadata_shorter_than_decoded_frames')
    if any(x<=0 for x in dt): reasons.append('nonmonotonic_or_duplicate_timestamp')
    if any(b != a+1 for a,b in zip(ids,ids[1:])): reasons.append('nonconsecutive_log_index')
    if frames < 2: reasons.append('too_few_decoded_frames')
    return {'metadata_rows':len(lines), 'decoded_frames':frames, 'excess_rows_before_final_drop':excess,
            'unused_rows_after_final_drop':len(lines)-max(frames-1,0),
            'median_interval_us':med, 'gap_threshold_us':1.5*med,
            'max_interval_us':max(dt, default=0), 'gap_count':len(gaps),
            'gap_following_row_ordinals_0based':gaps, 'rejection_reasons':reasons,
            'accepted_operational_prefix':not reasons,
            'candidate_prefix_frames':max(min(frames-1,len(lines)),0),
            'independently_verified_capture_mapping':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True);args=p.parse_args()
    rows=json.loads((args.audit/'combined_diagnostics.json').read_text())
    assert len(rows)==36 and len({r['video'] for r in rows})==36
    out=[]
    for row in rows:
        h=avi_header(args.root/row['video'])
        g=metadata_gate(args.root/'meta'/f"{Path(row['video']).stem}.txt",row['frames_default'])
        out.append({'video':row['video'],'camera':row['camera'],'sample_set':row['sample_set'],
                    'publisher_hash_match_previous_audit':row['publisher_hash_match'],
                    'header':h,'gate':g})
    doc={'created_utc':datetime.now(timezone.utc).isoformat(),'records':out,
         'protocol_sha256':hashlib.sha256((args.audit/'ACCEPTANCE_V2.txt').read_bytes()).hexdigest(),
         'interpretation':'Zero counts plus sentinel RIFF are unfinalized/placeholder metadata; they do not uniquely identify a process kill, prove absence of interior loss, or establish capture/PPG synchronization.'}
    (args.audit/'headers36.json').write_text(json.dumps(doc,indent=2))
    table=[]
    for r in out:
        h,g=r['header'],r['gate']
        table.append({'video':r['video'],'camera':r['camera'],
                      'avih_dwTotalFrames':';'.join(str(x.get('dwTotalFrames')) for x in h['avih']),
                      'strh_video_dwLength':';'.join(str(x.get('dwLength')) for x in h['strh'] if x.get('fccType')=='vids'),
                      'riff_size':h['riff_size_hex'],'file_bytes':h['file_bytes'],
                      **{k:g[k] for k in ['decoded_frames','metadata_rows','excess_rows_before_final_drop','median_interval_us','gap_count','accepted_operational_prefix']}})
    with (args.audit/'headers36.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    for cam in sorted({r['camera'] for r in out}):
        rr=[r for r in out if r['camera']==cam]
        print(json.dumps({'camera':cam,'n':len(rr),'accepted':sum(r['gate']['accepted_operational_prefix'] for r in rr),
             'avih_counts':sorted({x['dwTotalFrames'] for r in rr for x in r['header']['avih']}),
             'video_stream_lengths':sorted({x['dwLength'] for r in rr for x in r['header']['strh'] if x.get('fccType')=='vids'}),
             'riff_fields':sorted({r['header']['riff_size_hex'] for r in rr}),
             'gap_count_range':[min(r['gate']['gap_count'] for r in rr),max(r['gate']['gap_count'] for r in rr)]}),flush=True)

if __name__=='__main__':main()
