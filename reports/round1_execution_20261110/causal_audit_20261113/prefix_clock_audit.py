"""Prefix support on acquisition clocks. No inference, no outcome selection."""
from common_causal import *
import numpy as np
from fractions import Fraction

def main():
 assert not (C/'PREFIX_CLOCK_AUDIT.json').exists()
 cases=read(C/'LADDER_SELECTION.json')
 # These five actual ffprobe observations were printed successfully by attempt1.
 # That attempt stopped at MCD's known retained-prefix versus full-file length
 # difference before saving the combined table. Do not repeat the five decodes.
 observed={
 'UBFC2_subject1':(9.961,10.65901),
 'UBFC2_subject10':(10.182,10.509874),
 'UBFC2_subject11':(10.321,10.335446),
 'UBFC2_subject12':(10.414,10.262012),
 'UBFC2_subject14':(20.003,20.663804000000003)}
 rows=[]
 for case in cases:
  j=case['job'];w=case['window'];end=w['start']+10.25;target=w['start']+10
  if j['dataset']=='UBFC2':
   last,newend=observed[j['key']];provenance=dict(source='actual ffprobe output from prefix probe attempt1, source frame count exact; retained printed observations',source_sha256=j['source_sha256'])
  else:
   path=R/'mapping_audit_20261030/dev60_gate_v3'/(j['recording']+'.json');g=read(path);assert g['source_sha256']==j['source_sha256'];m=np.load(Path(j['crop'])/'meta.npz');physical=m['original_t'];n=g['gate']['candidate_prefix_frames'];assert len(physical)==n and g['accepted_operational_prefix'];pts=np.array([float(Fraction(x)) for x in g['container_pts_rational'][:n]]);pts-=pts[0]
   k=int(np.searchsorted(pts,end,side='right'));assert 0<k<=n;last=float(physical[k-1]);i=min(n-1,int(np.searchsorted(physical,target+.25,side='right'))+2);newend=max(end,float(pts[i]+.1))
   provenance=dict(source='previously verified V3 retained-frame rational PTS with exact unchanged video SHA; final decoded frame excluded by frozen public contract',source_sha256=j['source_sha256'],gate_sha256=sha(path),retained_frames=n,physical_sync_verified=False)
  rows.append(dict(key=j['key'],old_container_end=end,last_available_reference_time=last,target_reference_end=target,required_reference_buffer_s=.25,window_itself_truncated=bool(last<target),insufficient=bool(last<target+.25),new_container_end=newend,provenance=provenance))
 save(C/'PREFIX_CLOCK_AUDIT.json',rows)
 save(C/'PREFIX_CLOCK_AUDIT_RECEIPT.json',dict(utc=now(),code_sha256=sha(__file__),selection_sha256=sha(C/'LADDER_SELECTION.json'),cases=10,repair_cases=[r['key'] for r in rows if r['insufficient']],selection_uses_timestamps_only=True,reason='Original intended0.25s trailing buffer must be on the public acquisition clock, not assumed container clock. Supports10fps endpoint and denoising context.',failed_attempt='Full decode count was compared to MCD retained prefix count; preserved known final-frame exclusion, not a relaxed clock tolerance.'))
 print(json.dumps(rows))

if __name__=='__main__':main()
