"""Count T1 loader windows from existing timestamps/receipts; no new video audit."""
import argparse,json,hashlib,time,traceback
from pathlib import Path
from collections import Counter
from data_adapters import UBFC2,MCD
from window_support import recording_windows

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['ubfc2','mcd'],required=True)
    p.add_argument('--root',required=True);p.add_argument('--audit-dir');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();protocol_path=a.out/'protocol_frozen.json'
    if not protocol_path.exists():raise RuntimeError('Freeze protocol before real-data execution')
    protocol=json.loads(protocol_path.read_text());start=time.time()
    files=['data_adapters.py','window_support.py','prepare_recording.py','test_window_support.py','count_loader_windows.py']
    hashes={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in files}
    (a.out/'script_sha256.json').write_text(json.dumps(hashes,indent=2))
    backend=UBFC2(a.root) if a.dataset=='ubfc2' else MCD(a.root,a.audit_dir)
    ids=backend.recordings()
    if a.dataset=='mcd':ids=[i for i in ids if backend.rows[i]['camera']=='FullHDwebcam']
    result=[]
    for name in ids:
        try:
            rec=backend.load(name)
            w=recording_windows(rec,duration_s=protocol['window_seconds'],stride_s=protocol['stride_seconds'],max_gap_s=protocol['gap_threshold_seconds'])
            result.append(dict(recording=name,subject=rec.subject_id,windows=w,
                fps_effective=rec.metadata.get('fps_effective',(len(rec.timestamps)-1)/(rec.timestamps[-1]-rec.timestamps[0])),
                duplicate_video_timestamps=int(sum(rec.timestamps[1:]==rec.timestamps[:-1])),
                labels={k:{'scope':v.scope,'unit':v.unit,'valid_samples':int(v.valid_mask.sum())} for k,v in rec.labels.items()}))
        except Exception as e:result.append(dict(recording=name,error=str(e),traceback=traceback.format_exc()))
    ok=[r for r in result if 'windows' in r]
    summary=dict(dataset=a.dataset,records=len(ids),processed=len(result),errors=[r for r in result if 'error' in r],
        legacy_video_only_windows=sum(r['windows']['legacy_video_only_windows'] for r in ok),
        intersection_candidates=sum(r['windows']['candidate_windows'] for r in ok),
        valid_windows=sum(len(r['windows']['valid_windows']) for r in ok),
        rejected_interior_windows=sum(len(r['windows']['rejected_windows']) for r in ok),
        reasons=dict(Counter(reason for r in ok for w in r['windows']['rejected_windows'] for reason in w['reasons'])),
        seconds=time.time()-start,source_pixels_decoded=False,
        scope='Timestamp/source-validity window inventory, not frontend-dependent ROI coverage',
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),script_sha256=hashes)
    (a.out/'results.json').write_text(json.dumps(result,indent=2));(a.out/'summary.json').write_text(json.dumps(summary,indent=2))
    (a.out/'status.json').write_text(json.dumps({'state':'complete' if not summary['errors'] else 'complete_with_errors','records':len(result)},indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
