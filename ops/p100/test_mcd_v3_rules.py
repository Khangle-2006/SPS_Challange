import tempfile
from pathlib import Path
from datetime import datetime,timedelta
from mcd_v3_rules import metadata_gate
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'meta.txt'
    def write(n,largegap=False):
        start=datetime(2023,1,1)
        path.write_text('\n'.join(f'{i+1} '+(start+timedelta(microseconds=i*32000+(1000000 if largegap and i>30 else 0))).isoformat() for i in range(n)))
    write(100)
    assert metadata_gate(path,95)['accepted_operational_prefix'] # exactly5%
    assert metadata_gate(path,94)['rejection_reasons']==['decoded_deficit_gt5percent_of_log']
    write(2000)
    assert metadata_gate(path,1910)['accepted_operational_prefix'] # exactly90
    assert metadata_gate(path,1909)['rejection_reasons']==['excess_metadata_rows_gt90']
    write(100,True)
    g=metadata_gate(path,100)
    assert g['gap_count']==1 and g['accepted_operational_prefix']
    assert g['candidate_prefix_frames']==99 and g['unused_rows_after_final_drop']==1
    assert metadata_gate(path,103)['candidate_prefix_frames']==100
print('PASS V3: 5% and90-row boundaries; gap diagnostic only; final drop/common prefix')
