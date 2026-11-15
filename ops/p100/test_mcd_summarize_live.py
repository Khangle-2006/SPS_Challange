"""Check report denominators using explicitly synthetic data in a temporary directory."""
import tempfile,json,subprocess,sys
from pathlib import Path
with tempfile.TemporaryDirectory() as td:
    root=Path(td);gd=root/'dev60_gate_v3';ld=root/'lag_pos_ppg_v3';gd.mkdir();ld.mkdir()
    (gd/'status.json').write_text(json.dumps({'state':'complete','processed':360,'below_five_percent_gate':True,'reference_errata_count':3}))
    (ld/'status.json').write_text(json.dumps({'state':'running','eligible_total':357,'processed':2}))
    metric={'lag_frames':3,'lag_seconds':.1,'pearson_max':.8,'pearson_zero':.4,'weak_peak':False,'ambiguous_peak':True,'search_boundary':False}
    base={'subject':'synthetic','camera':'FullHDwebcam','step':'after','official_fold':0,'drift':{'slope_ms_per_min':None}}
    r1={**base,'video':'synthetic1','fullrecord':metric,'windows':[{'index':0,'midpoint_s':15,'result':metric,'failure':None}]}
    r2={**base,'video':'synthetic2','fullrecord':None,'error':'synthetic processing failure','windows':[]}
    for name,r in [('a',r1),('b',r2)]:
        (ld/name).mkdir();(ld/name/'result.json').write_text(json.dumps(r))
    subprocess.run([sys.executable,str(Path(__file__).with_name('mcd_summarize_live.py')),'--audit',str(root)],check=True,stdout=subprocess.DEVNULL)
    s=json.loads((ld/'report/summary.json').read_text());g=s['groups'][0]
    assert not s['complete'] and not s['e0_gate_count_pass']
    assert g['records_processed']==2 and g['fullrecord_missing']==1 and g['processing_errors']==1
    assert g['full_ambiguous']==1 and g['windows_evaluable']==1
    assert (ld/'report/lag_histograms.png').stat().st_size>1000
print('PASS summary: partial not final; errors remain denominator; ambiguity and window counts')
