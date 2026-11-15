import numpy as np
from scipy.signal import find_peaks
GRID=np.arange(42.,180.01,.5)

def top3(score):
    score=np.asarray(score,float)
    if not np.isfinite(score).all() or np.max(score)<=0 or np.ptp(score)<1e-12:return []
    ids=find_peaks(score)[0].tolist()
    if score[0]>score[1]:ids.append(0)
    if score[-1]>score[-2]:ids.append(len(score)-1)
    selected=[]
    for j in sorted(set(ids),key=lambda j:(-score[j],GRID[j])):
        if all(abs(GRID[j]-GRID[k])>=6-1e-10 for k in selected):selected.append(j)
        if len(selected)==3:break
    return [float(GRID[j]) for j in selected]

def extract(t,rgb,pipeline):
    """Only arrays and fixed signal operators; no labels/metadata inputs."""
    t=np.asarray(t,float);rgb=np.asarray(rgb,float)
    assert t.ndim==1 and rgb.shape==(len(t),4,3)
    assert np.isfinite(t).all() and np.all(np.diff(t)>0)
    dt=float(np.median(np.diff(t)));fs=1/dt
    assert fs>6
    windows=[];fail=[];spectra=[];valid_masks=[]
    end=float(t[-1]+dt)
    for st in np.arange(0,np.floor((end+1e-6)/10)*10,10):
        mask=(t>=st)&(t<st+10);tt=t[mask];rr=rgb[mask,:3]
        channels=np.zeros((9,len(GRID)));available=np.zeros(9,dtype=bool)
        for ri in range(3):
            good=np.isfinite(rr[:,ri]).all(1);tg=tt[good];rg=rr[good,ri]
            if len(tg)<2 or tg[0]>st+1.5*dt or tg[-1]<st+10-1.5*dt or np.max(np.diff(tg))>.2:
                fail.append(dict(start=float(st),roi=ri,reason='incomplete_edge_support_or_gap'));continue
            q=tg[0]+np.arange(int(np.floor((tg[-1]-tg[0])*fs))+1)/fs
            assert q[0]>=tg[0] and q[-1]<=tg[-1]+1e-9
            x=np.stack([np.interp(q,tg,rg[:,c]) for c in range(3)],1)
            for mi,method in enumerate(['POS','CHROM','G']):
                wave=pipeline.pulse(x,fs,method);f,p=pipeline.psd(wave,fs)
                pp=np.interp(GRID/60,f,p)
                if not np.isfinite(pp).all() or pp.max()<1e-20:
                    fail.append(dict(start=float(st),roi=ri,method=method,reason='static_or_zero_spectrum'));continue
                channels[ri*3+mi]=np.log1p(pp/max(float(np.median(pp)),1e-30));available[ri*3+mi]=True
        if available.any():
            windows.append(dict(start=float(st),available_channels=int(available.sum())))
            spectra.append(channels);valid_masks.append(available)
    if not windows:return dict(candidates=[],windows=[],failures=fail,fs=fs),np.empty((0,9,len(GRID))),np.empty((0,9),bool)
    video_score=np.mean([a[m].mean(0) for a,m in zip(spectra,valid_masks)],0)
    result=dict(candidates=top3(video_score),windows=windows,failures=fail,fs=fs,score=video_score.tolist())
    return result,np.asarray(spectra),np.asarray(valid_masks)

def permute_blocks(rows,y,rng):
    people=sorted({r['person'] for r in rows});idx={p:[i for i,r in enumerate(rows) if r['person']==p] for p in people}
    out=np.asarray(y).copy();mapping={};strata={}
    for p in people:strata.setdefault(len(idx[p]),[]).append(p)
    for count,ps in sorted(strata.items()):
        donors=rng.permutation(ps).tolist()
        for p,d in zip(ps,donors):out[idx[p]]=y[idx[d]];mapping[p]=d
    assert np.array_equal(np.sort(out),np.sort(y))
    return out,mapping,{str(k):len(v) for k,v in strata.items()}

def stats(rows,candidates,y):
    valid=np.array([bool(c) for c in candidates]);errors=np.array([min(abs(np.asarray(c)-v)) if c else np.inf for c,v in zip(candidates,y)])
    people=sorted({r['person'] for r,ok in zip(rows,valid) if ok})
    person_errors=np.array([np.mean([e for r,e,ok in zip(rows,errors,valid) if ok and r['person']==p]) for p in people])
    return dict(videos=len(rows),people=len({r['person'] for r in rows}),evaluable_videos=int(valid.sum()),evaluable_people=len(people),
        oracle_MAE=float(person_errors.mean()) if len(people) else None,
        within5=float(np.mean(errors<=5)) if len(rows) else None,within10=float(np.mean(errors<=10)) if len(rows) else None),person_errors
