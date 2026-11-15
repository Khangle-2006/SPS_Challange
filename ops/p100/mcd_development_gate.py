"""Audit the frozen 360-record cohort. No training, label repair, or raw deletion."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse, csv, hashlib, json, time
from pathlib import Path
from collections import Counter
import av
from mcd_header_gate import avi_header, metadata_gate

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
    n=0;bad=[];pts=[]
    with av.open(str(source)) as c:
        c.streams.video[0].codec_context.thread_count=1
        for frame in c.decode(video=0):
            if frame.is_corrupt:bad.append(n)
            pts.append(str(frame.pts*frame.time_base) if frame.pts is not None else None)
            n+=1
    gate=metadata_gate(root/row['meta'],n)
    interior=[i for i in bad if i<n-1]
    if interior:
        gate['rejection_reasons'].append('corrupt_interior_frame')
        gate['accepted_operational_prefix']=False
    result.update(gate=gate,decoded_corrupt_indices=bad,container_pts_rational=pts,
        accepted_operational_prefix=gate['accepted_operational_prefix'],
        training_authorized=False,ppg_alignment_verified=False)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=360);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(a.output/'worker.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((a.audit/'development_plan.json').read_text())
    rows=plan['development_records'];assert len(rows)==360
    assert len({r['patient_id'] for r in rows})==60
    assert len({r['video'] for r in rows})==360
    ph=sha(a.audit/'ACCEPTANCE_V2.txt')
    manifest={r['path']:r for r in json.loads((a.root/'download_manifest.json').read_text())['files']}
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
                        'error':str(e),'gate':{'rejection_reasons':['processing_error']}}
            save(path,result)
        results.append(result)
        accepted=sum(r['accepted_operational_prefix'] for r in results)
        state={'processed':len(results),'expected':360,'accepted':accepted,
               'rejected':len(results)-accepted,'protocol_sha256':ph,
               'state':'running','updated_unix':time.time(),'training_authorized':False}
        save(a.output/'status.json',state)
        print(json.dumps({**state,'video':row['video']}),flush=True)
    rejected=len(results)-accepted
    state.update(state='complete' if len(results)==360 else 'partial_smoke',
                 exclusion_fraction=rejected/len(results),
                 below_five_percent_gate=len(results)==360 and rejected<18,
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
        writer=csv.DictWriter(f,fieldnames=['video','subject','camera','step','accepted','reasons','decoded_frames','metadata_rows','gap_count'])
        writer.writeheader()
        for r in results:
            g=r['gate'];writer.writerow({**{k:r[k] for k in ['video','subject','camera','step']},
                'accepted':r['accepted_operational_prefix'],'reasons':';'.join(g['rejection_reasons']),
                **{k:g.get(k) for k in ['decoded_frames','metadata_rows','gap_count']}})
    print(json.dumps(state),flush=True)

if __name__=='__main__':main()
