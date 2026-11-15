"""POS replay of existing RGB caches/traces only; no video re-encoding/training."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
import numpy as np
from decode_hr_v2 import decode,ablation_config
from hr_decoder import estimate_hr
from evaluate_t2_saved import STAGES,SUBJECTS,metrics,save,sha,replay

def pos(rgb,fs):
    rgb=np.asarray(rgb,float);length=int(np.ceil(1.6*fs));out=np.zeros(len(rgb))
    if rgb.shape!=(len(rgb),3) or not np.isfinite(rgb).all():raise ValueError('invalid RGB')
    for end in range(length,len(rgb)+1):
        x=rgb[end-length:end];x=x/np.maximum(x.mean(axis=0),1e-8)
        s=np.array([[0,1,-1],[-2,1,1]])@x.T
        h=s[0]+np.std(s[0])/max(np.std(s[1]),1e-8)*s[1]
        out[end-length:end]+=h-h.mean()
    return out

def ubfc(root,out,reference):
    ref=json.loads(reference.read_text())['mse']['clean']['2027']['records']
    assert len(ref)==42 and {r['subject'] for r in ref}==SUBJECTS and all(r['split']=='val' for r in ref)
    roots={'clean':root/'efficientphys36_trainval_v1',**{f'h264_{r}k':root/f'detected_on_h264_20261019/h264_{r}k' for r in [250,500,1000]}}
    summary={};inputs={}
    for condition,folder in roots.items():
        waves=[]
        for row in ref:
            meta=folder/(row['subject']+'_meta.npz');rgbpath=folder/(row['subject']+'_rgb.npy')
            m=np.load(meta);raw=np.load(rgbpath,mmap_mode='r');t=m['t']
            q=row['start_s']+np.arange(301)/30
            if q[0]<t[0] or q[-1]>t[-1]:raise ValueError('Saved clip outside support')
            j=np.clip(np.searchsorted(t,q,side='right'),1,len(t)-1)
            weight=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)[:,None,None,None]
            # Match pre-existing subject32 diagnostic: interpolate image then mean.
            clip=raw[j-1].astype(np.float32)*(1-weight)+raw[j].astype(np.float32)*weight
            waves.append(pos(clip.mean(axis=(1,2)),30.)[:300])
            inputs[str(meta)]=sha(meta)
            inputs[str(rgbpath)]={'size':rgbpath.stat().st_size,'inventory':str(folder/'inventory.json')}
        waves=np.array(waves);np.savez_compressed(out/f'ubfc_{condition}_pos_waves.npz',waves=waves)
        summary[condition]=replay(waves,ref)
        save(out/'ubfc_pos_results.json',summary)
        print('UBFC',condition,'42 windows',flush=True)
    save(out/'ubfc_pos_inputs.json',inputs)

def mcd(root,audit,e0,out,condition):
    import data_adapters,window_support
    from data_adapters import MCD
    from window_support import recording_windows
    adapter=MCD(root,audit)
    stems=[s for s in adapter.recordings() if adapter.rows[s]['camera']=='FullHDwebcam']
    assert len(stems)==119 and len({adapter.rows[s]['patient_id'] for s in stems})==60
    if condition=='h264_500k':
        state=json.loads((e0/'status.json').read_text())
        if state.get('processed',0)!=119 or state.get('state') not in ['complete','completed']:
            save(out/'mcd_h264_pending.json',{'reason':'E0 not complete; no partial cohort treated as full','e0_status':state})
            return
    dest=out/f'mcd_{condition}';dest.mkdir(exist_ok=True)
    manifest={'scripts':{n:sha(Path(__file__).with_name(n)) for n in ['evaluate_t2_pos.py','decode_hr_v2.py']},'condition':condition}
    manifest['scripts']['data_adapters.py']=sha(data_adapters.__file__)
    manifest['scripts']['window_support.py']=sha(window_support.__file__)
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    statesum=[]
    for index,stem in enumerate(stems):
        target=dest/(stem+'.json')
        if target.exists():
            item=json.loads(target.read_text());assert item['fingerprint']==fingerprint;statesum.append(item);continue
        rec=adapter.load(stem);t=rec.timestamps;n=len(t)
        trace=(audit/'lag_pos_ppg_v3'/stem/'trace.npz') if condition=='native' else (e0/stem/'trace.npz')
        z=np.load(trace);rgb=z['rgb'][:n]
        if condition!='native':rgb=rgb[:,3] # P1R1; actual compressed-video ROI.
        valid=np.isfinite(rgb).all(axis=(1,2))
        if 'valid' in z and z['valid'].ndim==1:valid &=z['valid'][:n]
        support=recording_windows(rec,video_valid=valid)
        reference=rec.require_synchronized_ppg()
        stage_rows={s:[] for s in STAGES};states={s:None for s in STAGES};last=None
        for w in support['valid_windows']:
            q=w['start_s']+np.arange(300)/30
            sampled=np.stack([np.stack([np.interp(q,t,rgb[:,r,c]) for c in range(3)],axis=1) for r in range(3)])
            ppg=np.interp(q,reference.timestamps,reference.values)
            row=dict(subject=rec.subject_id,recording=stem,window=w['index'],start_s=w['start_s'],split='development',gt_hr=None)
            try:row['gt_hr']=estimate_hr(ppg,30)
            except ValueError:pass
            waves=np.array([pos(r,30) for r in sampled])
            dt=10 if last is None else w['start_s']-last
            # Missing support creates a discontinuity: do not smooth across gap.
            if last is not None and dt>10+1e-8:states={s:None for s in STAGES}
            for stage in STAGES:
                result=decode(waves,30,state=states[stage],config=ablation_config(stage),step_seconds=dt)
                states[stage]=result['state'] if not result['invalid'] else None
                stage_rows[stage].append(dict(row,pred_hr=result['hr_bpm'],abstain=result['abstain'],confidence=result['confidence'],reset=result.get('reset',False)))
            last=w['start_s']
        item=dict(recording=stem,subject=rec.subject_id,fingerprint=fingerprint,support=support,stages=stage_rows,trace_sha256=sha(trace))
        save(target,item);statesum.append(item)
        save(out/'pos_status.json',dict(state='running',condition=condition,processed=index+1,total=119,pid=os.getpid(),updated=time.time()))
        print('MCD',condition,index+1,stem,len(support['valid_windows']),flush=True)
    result={}
    for stage in STAGES:
        rows=sum([r['stages'][stage] for r in statesum],[])
        evaluable=[r for r in rows if r['gt_hr'] is not None]
        result[stage]=dict(all=metrics(evaluable),accepted=metrics(evaluable,True),reference_failures=len(rows)-len(evaluable),records=rows)
    save(out/f'mcd_{condition}_results.json',dict(records=len(stems),subjects=60,stages=result,
        candidate_windows=sum(r['support']['candidate_windows'] for r in statesum),
        support_rejected=sum(len(r['support']['rejected_windows']) for r in statesum),
        manifest=manifest,fingerprint=fingerprint))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--ubfc-root');ap.add_argument('--reference');ap.add_argument('--mcd-root');ap.add_argument('--audit');ap.add_argument('--e0');ap.add_argument('--condition',default='native');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert (out/'protocol_frozen.json').exists() and (out/'pos_replay_addendum.json').exists()
    if a.ubfc_root:ubfc(Path(a.ubfc_root),out,Path(a.reference))
    if a.mcd_root:mcd(Path(a.mcd_root),Path(a.audit),Path(a.e0),out,a.condition)
    save(out/'pos_status.json',dict(state='complete_available_inputs',pid=os.getpid(),updated=time.time(),condition=a.condition))

if __name__=='__main__':main()
