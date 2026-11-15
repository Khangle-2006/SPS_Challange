"""Summarize downloaded worker receipts; never invent completion or fill samples."""
import argparse
import json
from pathlib import Path


def summarize(results):
    n=sum(r['n_frames'] for r in results);det=sum(r['detected_frames'] for r in results)
    if not n:return {'completed':0,'frames':0}
    finite=sum(r['trace_finite_fraction_all_slots']*r['n_frames'] for r in results)/n
    return {'completed':len(results),'frames':n,'detected_frames':det,'detected_fraction':det/n,
       'trace_finite_fraction_all_slots':finite,
       'region_valid_counts':[sum(r['region_valid_counts'][i] for r in results) for i in range(4)],
       'crop_valid_frames':sum(r['crop_valid_frames'] for r in results),
       'small_face_frames':sum(r['small_face_frames'] for r in results),
       'output_bytes_completed':sum(r['output_bytes'] for r in results),
       'timing_ms_frame_weighted_mean':{k:sum(r['timing_ms'][k]['mean']*r['n_frames'] for r in results)/n for k in results[0]['timing_ms']}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);a=p.parse_args();root=Path(a.report)
    summary={};lines=['# T3 landmark frontend — 30/10/2026','',
      'Implementation: `video_landmark_frontend.py`; frozen T3-v1 before processing data. No HR labels used.',
      'Old 36×36 cache and raw files remain unchanged. All frames use actual timestamps provided by T1.',
      '', '| Dataset | Completed / requested | Frames | Detected | Finite RGB (all slots) |',
      '|---|---:|---:|---:|---:|']
    for name,total in [('MCD',119),('UBFC2',42)]:
        rfile=root/(name+'_results.json');sfile=root/(name+'_status.json')
        results=json.loads(rfile.read_text()) if rfile.exists() else []
        state=json.loads(sfile.read_text()) if sfile.exists() else {'state':'unknown'}
        s=summarize(results);summary[name]={'status':state,'summary':s}
        lines.append('| %s | %s / %s | %s | %s | %s |'%(name,s['completed'],total,s['frames'],
          f"{s.get('detected_fraction',0):.4%}",f"{s.get('trace_finite_fraction_all_slots',0):.4%}"))
    lines+=['','## CPU timings (frame-weighted mean, ms/frame)',
      'P100 node: Intel Xeon E5-2686 v4 @2.30GHz, affinity CPU2. A100 node: Xeon Silver 4316 @2.30GHz, affinity CPU2.',
      'These are host CPU measurements, not P100/A100 GPU inference benchmarks. Auxiliary detector is timed separately; FaceMesh graph timing includes its own internal detection/tracking.',
      '', '| Dataset | Aux detector | FaceMesh graph | Rasterize | Affine | Encode | Total including decode |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name,item in summary.items():
        m=item['summary'].get('timing_ms_frame_weighted_mean',{})
        lines.append('| '+name+' | '+' | '.join(f'{m[k]:.3f}' if k in m else 'pending' for k in
          ['detector_aux_ms','landmark_graph_ms','rasterize_ms','affine_ms','encode_ms','total_with_decode_ms'])+' |')
    lines+=['','## Contract and limitations',
      '- `trace.npz`: landmarks `[T,468,2]` float16, polygons `[T,4,36,2]` padded with NaN plus polygon_lengths, RGB `[T,4,3]` float32 and region_valid; timestamps_s, affine, crop_valid, detected, face_small.',
      '- ROI order: forehead, cheek_left, cheek_right, medial periocular/nasal bridge. Twelve-percent centroid shrink. Small faces use slot0 whole-face approximation with fixed eye/mouth holes; other slots are intentionally invalid.',
      '- Affine: ten fixed landmarks fitted by least squares to the official canonical model projected into a centered 60px extent inside 72×72. FFV1 bgr0, lossless after affine resampling.',
      '- Seven unit tests passed on both nodes, including exact FFV1 RGB roundtrip, known affine recovery, full-mask versus local raster agreement, nonempty canonical polygons, and invalid-value preservation.',
      '- FaceMesh 0.10.21 exports no per-landmark confidence. landmark_confidence is NaN, availability=false, lower_face_occluded=-1 unknown. C7 confidence-based occlusion detection is **not implemented**; no fake visibility/confidence score is used.',
      '- Detector confidence belongs to a separate FaceDetection graph; it is not per-landmark or lower-face occlusion confidence.',
      '- Missing detections/polygons remain NaN with validity=false. Black missing crops are storage frames only. The literal 100% finite-trace acceptance is not claimed if any measurement is missing.',
      '- Periocular ROI covers medial skin/bridge, not eyeballs. It cannot guarantee avoidance of all hair, facial hair, glasses, masks or background for every frame; visual QA is a fixed sample, not population proof.',
      '- MCD is native already-compressed FullHD dev60; no untouched-test generalization claim. Excludes4952 and7412_after; keeps9401. T1 downstream support masks remain mandatory.',
      '- Cache cap: 10GB/node, 20GB combined. Stops before cap with 300MB reserve. Source raw files and old cache are not deleted.',
      '- Figures/manual visual checks: see QA_REVIEW.txt; exactly first five sorted IDs per dataset, frame100.',
      '- All 10 production QA images were opened and inspected. All show nonempty polygons and upright crops, but strict no-hair acceptance FAILED: definite forehead bangs in MCD1115_after and UBFCsubject11, conservative hairline flag in subject10. Subject13 has glasses contamination. No ROI was tuned or sample removed to hide these cases.',
      '- region_valid is a raster/measurement-support flag, not a visible-skin certificate. This V1 cache must not be presented as safe unconditional four-region physiological fusion.',
      '', '## Status',
      'Results are partial while either status is running. No completion, >=98% full-UBFC detection, or full-data finite acceptance is asserted early.',
      '', '## Sources/license',
      '- [Official FaceMesh v0.10.21 API](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/python/solutions/face_mesh.py): output is multi_face_landmarks; configured thresholds are not exported scores.',
      '- [Official Apache-2.0 license](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/LICENSE); retained locally in assets/MEDIAPIPE_LICENSE.',
      '- [Official canonical face model](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/modules/face_geometry/data/canonical_face_model.obj).',
      '', 'No training, new datasets, raw deletion, offset adjustment, or E0 modification.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'status.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
