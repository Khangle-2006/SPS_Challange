from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=read(P/'SELECTION.json');out=[]
for g in sorted({r['group'] for r in rows}):
    rr=[r for r in rows if r['group']==g];gm=[];pm=[];prs=[];corr=[];windows=0;hrs=[]
    for r in rr:
        d=P/'data'/r['key'];m=P/'model'/r['key'];s=P/'signal'/r['key']
        ready=read(d/'READY.json');mc=read(m/'COMPLETE.json');sc=read(s/'COMPLETE.json');met=read(s/'METRICS.json')
        assert sha(m/'outputs.npz')==mc['outputs_sha256']==sc['input_hash']
        assert sha(s/'METRICS.json')==sc['metrics_hash']
        assert sha(s/'signal.npz')==met['hash']
        assert sha(d/'geometry.npz')==ready['geometry_sha256']
        assert mc['static_hr'] is None
        a=np.load(s/'signal.npz')['pulse_maps'];gm.append(float(np.nanmean(a[:,0])));pm.append(float(np.nanmean(a[:,1])));windows+=len(a)
        prs+=met['pr'];corr += [c for c in met['correlations'] if c is not None]
        hrs+=mc['predicted_hr']
    out.append(dict(group=g,n_video=len(rr),n_windows=windows,label_windows=sum(p is not None for p in prs),in_band_label_windows=sum(p is not None and 42<=p<=180 for p in prs),G_ratio_n=int(np.isfinite(gm).sum()),POS_ratio_n=int(np.isfinite(pm).sum()),G_ratio_mean=float(np.nanmean(gm)),POS_ratio_mean=float(np.nanmean(pm)),model_hr_mean=float(np.nanmean(hrs)),model_hr_sd=float(np.nanstd(hrs)),model_ge175=sum(h is not None and h>=175 for h in hrs),correlation_n=len(corr),corr_zero_mean=float(np.mean([c['zero'] for c in corr])) if corr else None,corr_max_mean=float(np.mean([c['max_signed'] for c in corr])) if corr else None,corr_boundary=sum(c['boundary'] for c in corr)))
(P/'READING_TABLES.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
