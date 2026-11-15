"""User's V3: count deficit + interior decoder errors; gaps are descriptive."""
from datetime import datetime,timedelta
from collections import Counter
import statistics
import numpy as np

def metadata_gate(path,frames):
    rows=[x.split(maxsplit=1) for x in path.read_text().splitlines() if x.strip()]
    ids=[int(x[0]) for x in rows];times=[datetime.fromisoformat(x[1]) for x in rows]
    if not times or frames<2:raise ValueError('No usable common prefix')
    dt=[(b-a)//timedelta(microseconds=1) for a,b in zip(times,times[1:])]
    med=statistics.median(dt) if dt else 0
    excess=len(rows)-frames; reasons=[]
    if excess>90:reasons.append('excess_metadata_rows_gt90')
    if 20*excess>len(rows):reasons.append('decoded_deficit_gt5percent_of_log')
    duration=(times[-1]-times[0]).total_seconds()
    return {'metadata_rows':len(rows),'decoded_frames':frames,
      'excess_rows_before_final_drop':excess,'decoded_deficit_fraction':excess/len(rows),
      'unused_rows_after_final_drop':max(len(rows)-max(frames-1,0),0),
      'surplus_decoded_frames':max(frames-len(rows),0),
      'candidate_prefix_frames':min(len(rows),max(frames-1,0)),
      'metadata_fps_elapsed':(len(rows)-1)/duration if duration>0 else None,
      'metadata_fps_from_median':1e6/med if med>0 else None,
      'median_interval_us':med,'max_interval_us':max(dt,default=0),
      'interval_us_quantiles':np.quantile(dt,[0,.01,.5,.95,.99,1]).tolist() if dt else [],
      'gap_count':sum(2*x>3*med for x in dt),
      'nonpositive_intervals':sum(x<=0 for x in dt),
      'log_index_discontinuities':sum(b!=a+1 for a,b in zip(ids,ids[1:])),
      'top_interval_250us_bins':Counter(int(x//250)*250 for x in dt).most_common(8),
      'interval_histogram_250us_counts':sorted(Counter(int(x//250)*250 for x in dt).items()),
      'lattice_15p625ms_within_0p5ms_fraction':float(np.mean(abs(np.asarray(dt)-15625*np.rint(np.asarray(dt)/15625))<=500)) if dt else None,
      'lattice_15p625ms_within_1ms_fraction':float(np.mean(abs(np.asarray(dt)-15625*np.rint(np.asarray(dt)/15625))<=1000)) if dt else None,
      'rejection_reasons':reasons,'accepted_operational_prefix':not reasons,
      'independently_verified_capture_mapping':False}
