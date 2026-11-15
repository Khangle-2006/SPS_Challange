"""Audit the frozen 360-record cohort. No training, label repair, or raw deletion."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse, csv, hashlib, json, time
from pathlib import Path
from collections import Counter
import av
from mcd_header_gate import avi_header
from mcd_v3_rules import metadata_gate
from mcd_riff_tail import inspect as riff_inspect

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path, obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def inspect(row, root, manifest, protocol_hash):
    source=root/row['video'];entry=manifest[row['video']]
    result={'video':row['video'],'subject':row['patient_id'],'camera':row['camera'],
            'step':row['step'],'protocol_sha256':protocol_hash,'av_version':av.__version__}
    result['source_sha256']=sha(source)
    if source.stat().st_size!=entry['size'] or result['source_sha256']!=entry['lfs_sha256']:
        raise ValueError('Publisher size/hash mismatch')
    result['metadata_sha256']=sha(root/row['meta'])
    result['header']=avi_header(source)
    n=0;bad=[];pts=[];packet_errors=[];corrupt_packets=[]
    tail=riff_inspect(source)
    last_media_pos=tail['last_media_chunks'][-1]['pos'] if tail['last_media_chunks'] else None
    with av.open(str(source)) as c:
        c.streams.video[0].codec_context.thread_count=1
        for packet in c.demux(video=0):
            pos=packet.pos
            if packet.is_corrupt:corrupt_packets.append(pos)
            try:
                frames=packet.decode()
            except av.error.FFmpegError as e:
                packet_errors.append({'pos':pos,'decoded_frames_before':n,'error':str(e)})
                continue
            for frame in frames:
                if frame.is_corrupt:bad.append(n)
                pts.append(str(frame.pts*frame.time_base) if frame.pts is not None else None)
                n+=1
    gate=metadata_gate(root/row['meta'],n)
    interior=[i for i in bad if i<n-1]
    if interior:
        gate['rejection_reasons'].append('corrupt_interior_frame')
        gate['accepted_operational_prefix']=False
    error_positions=corrupt_packets+[x['pos'] for x in packet_errors]
    if any(p is None or p<0 or last_media_pos is None for p in error_positions):
        raise ValueError('Unlocalized decoder error; not assessable')
    if any(p<last_media_pos for p in error_positions):
        gate['rejection_reasons'].append('interior_packet_decode_error')
        gate['accepted_operational_prefix']=False
    result.update(gate=gate,decoded_corrupt_indices=bad,container_pts_rational=pts,
        packet_decode_errors=packet_errors,corrupt_packet_positions=corrupt_packets,
        last_media_chunk_pos=last_media_pos,
        accepted_operational_prefix=gate['accepted_operational_prefix'],
        training_authorized=False,ppg_alignment_verified=False)
    result['accepted_frame_mapping']=gate['accepted_operational_prefix']
    result['official_fold']=int(row['official_fold'])
    result['reference_eligible']=row['reference_eligible']=='true'
    result['reference_exclusion_reason']=row['reference_exclusion_reason']
    result['hr_label_source']='contact_ppg_only'
    result['accepted_operational_prefix']=result['accepted_frame_mapping'] and result['reference_eligible']
    if not result['reference_eligible']:
        gate['rejection_reasons'].append('reference_errata:'+row['reference_exclusion_reason'])
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=360);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(a.output/'worker.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan_path=a.audit/'development_plan_v3_curated.json'
    plan=json.loads(plan_path.read_text())
    rows=plan['development_records'];assert len(rows)==360
    assert len({r['patient_id'] for r in rows})==60
    assert len({r['video'] for r in rows})==360
    # Bundle hash prevents resuming results under stale metadata/errata policy.
    ph=hashlib.sha256((a.audit/'ACCEPTANCE_V3.txt').read_bytes()+
        (a.audit/'ERRATA_ADDENDUM.txt').read_bytes()+plan_path.read_bytes()).hexdigest()
    manifest={r['path']:r for r in json.loads((a.root/'download_manifest.json').read_text())['files']}
    histograms=[]
    for row in rows[:a.limit]:
        try:
            m=sum(bool(x.strip()) for x in (a.root/row['meta']).read_text().splitlines())
            h=metadata_gate(a.root/row['meta'],m)
            histograms.append({'video':row['video'],**{k:v for k,v in h.items() if k.startswith(('metadata_fps','interval_','lattice_','median_interval','max_interval','gap_count'))}})
        except Exception as e:histograms.append({'video':row['video'],'error':str(e)})
    save(a.output/'metadata_histograms_before_decode.json',histograms)
    results=[]
    for row in rows[:a.limit]:
        path=a.output/(Path(row['video']).stem+'.json')
        if path.exists():
            result=json.loads(path.read_text())
            assert result['protocol_sha256']==ph,'Changed gate; refuse mixed results'
            assert result['video']==row['video']
        else:
            try:result=inspect(row,a.root,manifest,ph)
            except Exception as e:
                result={'video':row['video'],'subject':row['patient_id'],'camera':row['camera'],
                        'step':row['step'],'protocol_sha256':ph,'accepted_operational_prefix':False,
                        'official_fold':int(row['official_fold']),'reference_eligible':row['reference_eligible']=='true',
                        'error':str(e),'gate':{'rejection_reasons':['processing_error']}}
            save(path,result)
        results.append(result)
        accepted=sum(r['accepted_operational_prefix'] for r in results)
        state={'processed':len(results),'expected':360,'accepted':accepted,
               'rejected':len(results)-accepted,'protocol_sha256':ph,
               'state':'running','updated_unix':time.time(),'training_authorized':False}
        save(a.output/'status.json',state)
        print(json.dumps({**state,'video':row['video']}),flush=True)
    failed=sum('error' in r for r in results)
    rejected=sum(not r['accepted_operational_prefix'] and 'error' not in r for r in results)
    state.update(state='complete' if len(results)==360 else 'partial_smoke',
                 reference_errata_count=sum(not r['reference_eligible'] for r in results),
                 frame_mapping_accepted=sum(r.get('accepted_frame_mapping',False) for r in results),
                 rejected=rejected,processing_failures=failed,total_nonaccepted=rejected+failed,
                 exclusion_fraction=rejected/len(results),processing_failure_fraction=failed/len(results),
                 total_nonaccepted_fraction=(rejected+failed)/len(results),
                 below_five_percent_gate=len(results)==360 and rejected+failed<18,
                 rejection_reasons=dict(Counter(x for r in results for x in r['gate']['rejection_reasons'])),
                 camera_summary=[{'camera':cam,'n':sum(r['camera']==cam for r in results),
                   'accepted':sum(r['camera']==cam and r['accepted_operational_prefix'] for r in results)}
                   for cam in sorted({r['camera'] for r in results})],
                 condition_summary=[{'condition':step,'n':sum(r['step']==step for r in results),
                   'accepted':sum(r['step']==step and r['accepted_operational_prefix'] for r in results)}
                   for step in ['before','after']])
    save(a.output/'status.json',state)
    save(a.output/'summary.json',state)
    with (a.output/'records.csv').open('w',newline='') as f:
        extra=['decoded_frames','metadata_rows','excess_rows_before_final_drop','decoded_deficit_fraction',
               'candidate_prefix_frames','metadata_fps_elapsed','metadata_fps_from_median',
               'median_interval_us','max_interval_us','gap_count','nonpositive_intervals','log_index_discontinuities']
        writer=csv.DictWriter(f,fieldnames=['video','subject','camera','step','official_fold','reference_eligible','accepted','reasons']+extra)
        writer.writeheader()
        for r in results:
            g=r['gate'];writer.writerow({**{k:r[k] for k in ['video','subject','camera','step','official_fold','reference_eligible']},
                'accepted':r['accepted_operational_prefix'],'reasons':';'.join(g['rejection_reasons']),
                **{k:g.get(k) for k in extra}})
    print(json.dumps(state),flush=True)

if __name__=='__main__':main()
