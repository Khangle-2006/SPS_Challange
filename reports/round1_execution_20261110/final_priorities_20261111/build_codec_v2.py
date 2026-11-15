from pathlib import Path
P=Path(__file__).resolve().parent
s=(P.parent/'codec_clock_20261111/run_v4.py').read_text()
s=s[:s.index('\ndef check():')]
s=s.replace(" assert len(raw['oval_points'])==n and len(raw['t'])==n\n assert np.array_equal(raw['t'],raw['original_t']) and np.all(np.diff(raw['original_t'])>0)"," assert len(raw['oval_points'])==n\n assert np.all(np.diff(raw['original_t'])>=0)\n if j['dataset']=='MCD':assert np.array_equal(raw['t'],raw['original_t'])\n else:assert np.array_equal(raw['t'],np.unique(raw['original_t']))")
s=s.replace("c=int(np.argmin(abs(raw['t']-(t[k]+origin))));assert abs(raw['t'][c]-(t[k]+origin))<1e-5;points[k]=raw['oval_points'][c].copy()","c=k;assert abs(raw['original_t'][c]-(t[k]+origin))<1e-5;points[k]=raw['oval_points'][c].copy()")
s=s.replace("meta_sha=sha(j['meta'])","landmark_mapping='original decoded frame ordinal, before timestamp deduplication',meta_sha=sha(j['meta'])")
(P/'exact_real_v2.py').write_text(s)
t=(P/'codec_exact.py').read_text().replace('import run_v4 as a','import exact_real_v2 as a')
t=t.replace("assert not (OUT/'LAUNCH.json').exists()","assert not (OUT/'LAUNCH_V2.json').exists();assert not Path('/proc/2609418').exists();assert not list(OUT.glob('real/*/*/f*_b*/RESULT.json'))")
t=t.replace("OUT/'LAUNCH.json'","OUT/'LAUNCH_V2.json'").replace("OUT/'ERROR.json'","OUT/'ERROR_V2.json'")
t=t.replace("source_plan_sha256=a.sha(OLD/'PLAN.json')","source_plan_sha256=a.sha(OLD/'PLAN.json'),mapping_sha256=a.sha(P/'exact_real_v2.py')")
t=t.replace("OUT/'PLAN.json'","OUT/'PLAN_V2.json'")
(P/'codec_exact_v2.py').write_text(t)
