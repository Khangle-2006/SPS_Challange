from mcd_lag_reading import endpoint_drift, drift_counts

def record(lags):
    return {'windows':[{'index':i,'result':None if v is None else
             {'lag_frames':v,'weak_peak':False,'ambiguous_peak':False,'search_boundary':False}}
             for i,v in enumerate(lags)]}

assert endpoint_drift(record([0,3]))['endpoint_drift_flag'] is False
assert endpoint_drift(record([0,-4]))['endpoint_drift_flag'] is True
assert not endpoint_drift(record([None,0,1]))['endpoint_evaluable']
assert not endpoint_drift(record([0,1,None]))['endpoint_evaluable']
assert not endpoint_drift(record([0]))['endpoint_evaluable']
assert endpoint_drift(record([0,None,4]))['endpoint_drift_flag'] is True
assert drift_counts([record([0,0])]*19+[record([0,4])],20)['decision']=='at_or_above_5_percent'
assert drift_counts([record([0,0])]*19,20)['decision']=='unresolved_missing_endpoints'
assert drift_counts([record([0,0])]*20,20)['decision']=='below_5_percent'
assert drift_counts([record([0,0])]*19+[record([None,0])],20)['unknown_total']==1
print('PASS endpoint drift: strict >3, real endpoints, missing outcomes, 5% boundary')
