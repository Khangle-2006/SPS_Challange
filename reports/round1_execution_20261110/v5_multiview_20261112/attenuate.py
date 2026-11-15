"""Fixed 5% nuisance attenuation; true150 probe gates diagnostic evaluation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,traceback
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;E=P.parent;N=E/'final_priorities_20261111'
sys.path.insert(0,str(N));import classical_screen_v2 as c
os.environ['CUDA_VISIBLE_DEVICES']=''
from nuisance import regress
from probe_regression import amplitude
ALPHA=.05
def attenuate(rgb,types,boxes=None,valid=None):
    full,fit=regress(rgb,types,boxes,valid);return np.asarray(rgb,float)+ALPHA*(full-rgb),dict(**fit,alpha=ALPHA)
def probe():
    import probe_regression as old
    out=[]
    for root in sorted((E/'codec_clock_20261111/synthetic').glob('scene*')):
        parts=root.name.split('_');pulse=int(parts[2][5:]);fs=int(parts[3][1:])
        if pulse!=150:continue
        for bf in [0,3,4]:
            d=root/f'b{bf}';types=[x['pict_type'] for x in c.read(d/'GOP.json')['frames']]
            with np.load(d/'traces.npz') as z:rgb=z['skin'];t=z['t']
            boxes=np.stack([.8*np.sin(2*np.pi*.73*t),.8*np.cos(2*np.pi*.41*t),np.full(len(t),110),np.full(len(t),90)],1)
            clean,_=attenuate(rgb,types,boxes);a=amplitude(rgb[:,1],t,2.5);b=amplitude(clean[:,1],t,2.5)
            out.append(dict(case=root.name,bf=bf,retained_fraction=b/a if a>1e-10 else None,source_sha256=c.sha(d/'traces.npz')))
    fs=10;t=np.arange(301)/fs;types=['P' if i%4==0 else 'B' for i in range(len(t))];rgb=np.array([180,130,110])[None,:]+np.sin(2*np.pi*2.5*t)[:,None]*np.array([.2,1,.1]);clean,_=attenuate(rgb,types)
    ideal=amplitude(clean[:,1],t,2.5)/amplitude(rgb[:,1],t,2.5)
    good=len(out)==24 and all(r['retained_fraction'] is not None and r['retained_fraction']>=.9 for r in out) and ideal>=.9
    return dict(alpha=ALPHA,cases=out,ideal150_retention=ideal,passed=good,selection='fixed before results; no alpha tuning',code_sha256=c.sha(__file__))
def main():
    out=P/'attenuation';out.mkdir(exist_ok=True);lease=c.guard.lock(out/'worker.lock');assert not (out/'START.json').exists()
    c.save(out/'START.json',dict(pid=os.getpid(),budget=c.guard.budget(fresh=True),code_sha256=c.sha(__file__)))
    paired=c.read(P/'CODEC_COMMON_V2.json')['table']
    for m in ['POS','CHROM']:
        a={r['condition']:r['fraction'] for r in paired if r['dataset']=='BTC' and r['method']==m and r['cohort']=='common_window'};assert a['default']>a['bf0']
    result=probe();c.save(out/'PULSE_PROBE.json',result);assert result['passed'],'True150 pulse preservation failed; no evaluation'
    allrows=[];files=sorted((N/'classical_screen/predictions').glob('*/*/c5prime_*.json'))
    for i,file in enumerate(files):
        prior=c.read(file);sample=prior['rows'][0];variant=sample['condition'].removeprefix('c5prime_');d=E/'c5prime'/sample['dataset']/sample['recording']/variant/'crop'
        assert c.sha(d/'meta.npz')==prior['meta_sha256'];g=Path(prior['evidence']['path']);assert c.sha(g)==prior['evidence']['sha256']
        types=[v['pict_type'] for v in c.read(g)['frames']]
        with np.load(d/'meta.npz') as z:t=z['t'];valid=z['valid'];rgb=z['rgb_mean'];boxes=z['boxes']
        assert len(types)==len(t);clean,fit=attenuate(rgb,types,boxes,valid);rows=[]
        for old in prior['rows']:
            if old['method'] not in ['POS','CHROM']:continue
            rows.append(old);pred=None;reason=old['reason']
            if old['prediction'] is not None:
                try:
                    start=old['start'];assert c.supported(t,valid,start,start+10);grid=start+np.arange(300)/30;x=np.stack([np.interp(grid,t,clean[:,k]) for k in range(3)],1);pred=c.decode_wave(c.NS[old['method'].lower()](x,30));reason=None
                except ValueError as e:reason=str(e)
            rows.append(dict(old,method=old['method']+'_attenuation05',prediction=pred,reason=reason))
        c.save(out/'predictions'/sample['dataset']/sample['recording']/(sample['condition']+'.json'),dict(rows=rows,fit=fit,input_sha256=c.sha(file),code_sha256=c.sha(__file__)))
        allrows.extend(rows);c.save(out/'STATUS.json',dict(processed=i+1,total=len(files),pid=os.getpid()))
    baseline=[r for r in c.read(E/'C5_BTC_VIDEO.json') if r['method'] in ['constant','calibrated']];video=[]
    for rid,cond,m in sorted({(r['recording'],r['condition'],r['method']) for r in allrows if r['dataset']=='BTC'}):
        rr=[r for r in allrows if (r['recording'],r['condition'],r['method'])==(rid,cond,m) and r['prediction'] is not None]
        if rr:video.append(dict(rr[0],prediction=float(np.median([r['prediction'] for r in rr]))))
    tables=[];methods=['constant','calibrated','POS','CHROM','POS_attenuation05','CHROM_attenuation05']
    for cond in ['c5prime_default','c5prime_bf0']:
        for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
            rr=[r for r in baseline+video if r['condition']==cond and c.belongs(r,group)]
            tables.extend(dict(condition=cond,group=group,**r) for r in c.metric(rr,methods))
    from report_c5 import public_score
    public=[]
    for ds in ['UBFC2','MCD']:
        for cond in ['c5prime_default','c5prime_bf0']:
            look={m:{(r['recording'],r['start']):r for r in allrows if r['dataset']==ds and r['condition']==cond and r['method']==m and r['prediction'] is not None and r['truth'] is not None} for m in methods[2:]};common=set.intersection(*(set(x) for x in look.values()))
            for m in look:public.append(dict(dataset=ds,condition=cond,method=m,**public_score([look[m][k] for k in sorted(common)])))
    c.save(out/'BTC_METRICS.json',tables);c.save(out/'PUBLIC_METRICS.json',public);c.save(out/'VIDEO.json',video);c.save(out/'COMPLETE.json',dict(files=len(files),alpha=ALPHA,probe_passed=True,holdout_evaluations=0,diagnostic_only=True))
if __name__=='__main__':
    try:main()
    except Exception as e:c.save(P/'attenuation/ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
