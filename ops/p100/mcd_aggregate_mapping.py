import json,csv,hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
from mcd_riff_tail import inspect
ROOT=Path('/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG')
OUT=ROOT.parent.parent/'mapping_audit_20261030'
plan=json.loads((OUT/'development_plan.json').read_text())
summary=[]
for setname,rows in [('diagnostic1020',json.loads((OUT/'subject1020_results.json').read_text())),('random30',json.loads((OUT/'expanded_results.json').read_text()))]:
    for row in rows:
        tail=inspect(ROOT/row['video'])
        sentinel=any(c['tag']=='RIFF' and c['pos']==0 for c in tail['sentinel_container_lengths'])
        row['riff_length_is_sentinel']=sentinel
        if sentinel:
            row['riff_declared_bytes']=None;row['riff_bytes_minus_file']=None
        lines=[s for s in (ROOT/('meta/'+Path(row['video']).stem+'.txt')).read_text().splitlines() if s.strip()]
        times=[datetime.fromisoformat(s.split(maxsplit=1)[1]) for s in lines]
        delta=np.array([(b-a).total_seconds() for a,b in zip(times,times[1:])]);med=np.median(delta)
        n=row['frames_default'];m=row['metadata_rows']
        final_bad=[p for p in tail['truncated_payloads'] if p['tag'].endswith(('dc','db'))]
        row.update(sample_set=setname,meta_minus_frames=m-n,
                   tail_media_payload_truncated=bool(final_bad),riff_details=tail,
                   metadata_gap_count_ge_2median=int(np.sum(delta>=2*med)),
                   metadata_interval_quantiles=np.quantile(delta,[0,.01,.5,.99,1]).tolist(),
                   head_loss_decoder_evidence=bool(row['showall_additional_frames']>0 or row['first_keyframe_index']!=0),
                   packed_b_evidence=bool(row['multi_vop_packets']>0 or row['frame_types'].get('B',0)>0))
        # Container damage is diagnosed; capture-log mapping remains a separate gate.
        row.update(acceptance='reject_unresolved',reason='No independent capture-row anchor. Tail truncation, where present, does not prove no earlier capture loss.')
        summary.append(row)
(OUT/'combined_diagnostics.json').write_text(json.dumps(summary,indent=2))
counts=[]
for setname in ['diagnostic1020','random30']:
    for cam in ['FullHDwebcam','USBVideo','IriunWebcam']:
        rs=[r for r in summary if r['sample_set']==setname and r['camera']==cam]
        if not rs:continue
        counts.append({'set':setname,'camera':cam,'videos':len(rs),'people':len(set(r['subject'] for r in rs)),
            'hash_match':sum(r['publisher_hash_match'] for r in rs),
            'first_I_keyframe_at_pts0':sum(r['first_keyframe_index']==0 and r['first_frame'][0].get('pts')==0 for r in rs),
            'showall_recovers_frames':sum(r['showall_additional_frames']>0 for r in rs),
            'B_frame_or_packed_candidates':sum(r['packed_b_evidence'] for r in rs),
            'small_packet_candidates':sum(r['small_packets_le32']>0 for r in rs),
            'tail_truncated':sum(r['tail_media_payload_truncated'] for r in rs),
            'packet_eq_frame':sum(r['packets']==r['frames_default'] for r in rs),
            'meta_minus_frames_min':min(r['meta_minus_frames'] for r in rs),
            'meta_minus_frames_max':max(r['meta_minus_frames'] for r in rs),
            'accepted_mapping':sum(r['acceptance']=='accepted_mapping' for r in rs)})
out={'random_records_complete':sum(r['sample_set']=='random30' for r in summary),
     'random_records_expected':30,'all_records':len(summary),'counts':counts,
     'sample_size_extrapolation_allowed':False,'training_cache_pruning_allowed':False}
(OUT/'aggregate.json').write_text(json.dumps(out,indent=2))
with (OUT/'audit_table.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(counts[0]));w.writeheader();w.writerows(counts)
print(json.dumps(out),flush=True)
