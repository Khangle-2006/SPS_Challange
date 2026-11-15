"""Replay existing whole-recording POS waves, preserving historical validation ROI."""
import argparse,json
from pathlib import Path
import numpy as np
from evaluate_t2_saved import replay,save,sha,SUBJECTS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();src=Path(a.source);out=Path(a.out)
    assert (out/'protocol_frozen.json').exists()
    old=json.loads((src/'pos_h264_results.json').read_text());rows=old['records']
    assert len(rows)==42 and {r['subject'] for r in rows}==SUBJECTS and all(r['split']=='val' for r in rows)
    result={};inputs={}
    for condition,key in [('clean','clean_pos'),('h264_500k','h264_pos')]:
        waves=[]
        for r in rows:
            path=src/(r['subject']+'_traces.npz');inputs[str(path)]=sha(path);z=np.load(path)
            q=r['start_s']+np.arange(300)/30
            assert q[0]>=z['t'][0] and q[-1]<=z['t'][-1]
            waves.append(np.interp(q,z['t'],z[key]))
        result[condition]=replay(np.array(waves),rows)
        by={(r['subject'],r['window']):r for r in rows}
        predkey='clean_pos_hr' if condition=='clean' else 'h264_pos_hr'
        assert all(abs(r['pred_hr']-by[r['subject'],r['window']][predkey])<1e-8 for r in result[condition]['legacy']['records'])
    save(out/'ubfc_historical_pos_results.json',result);save(out/'ubfc_historical_pos_provenance.json',dict(inputs=inputs,script_sha256=sha(__file__),legacy_exactly_reproduced=True,missing_conditions=['h264_250k','h264_1000k']))

if __name__=='__main__':main()
