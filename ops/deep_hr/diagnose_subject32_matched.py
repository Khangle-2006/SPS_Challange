"""Posthoc window0/window2 reference, compression and model diagnostics; no tuning."""
import argparse,json
import numpy as np
from scipy import signal
from train_matched_spectral import CLEAN,DATA,OUT,Clips,SEEDS,sha
from spectral_ensemble import normalized_psd
from hr_baseline import detrend,pos

D=OUT/'subject32'
def read(p):return json.loads(p.read_text())
def stats(w,gt):
    w=np.asarray(w,dtype=float);f,p=normalized_psd(w)
    ix=signal.find_peaks(p)[0];ix=sorted(ix,key=lambda i:p[i],reverse=True)[:5]
    filt=signal.sosfiltfilt(signal.butter(1,[.75,3.],fs=30,btype='bandpass',output='sos'),detrend(w))
    return {'hr':float(f[np.argmax(p)]),'power_within_reference_pm6':float(p[np.abs(f-gt)<=6].sum()),
        'peaks':[{'bpm':float(f[i]),'power_relative_to_max':float(p[i]/p.max())} for i in ix],
        'first_half_hr':float(normalized_psd(w[:150])[0][np.argmax(normalized_psd(w[:150])[1])]),
        'second_half_hr':float(normalized_psd(w[150:300])[0][np.argmax(normalized_psd(w[150:300])[1])])},f,p,filt
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs-only',action='store_true');a=ap.parse_args();D.mkdir(exist_ok=True)
    clean=Clips(CLEAN,'val');meta=np.load(CLEAN/'subject32_meta.npz');gt=np.loadtxt(D/'ground_truth.txt');times=gt[2]-gt[2,0]
    legacy=np.load(D/'original_signals.npz')
    np.testing.assert_array_equal(times,meta['original_t'])
    u,inv,count=np.unique(times,return_inverse=True,return_counts=True);ppg=np.zeros(len(u));np.add.at(ppg,inv,gt[0]);ppg/=count
    np.testing.assert_array_equal(u,meta['t']);np.testing.assert_array_equal(ppg,meta['ppg'])
    roots={'clean':CLEAN,**{f'h264_{r}k':DATA/f'detected_on_h264_20261019/h264_{r}k' for r in [500,250,1000]}}
    rows=[r for r in clean.rows if r['subject']=='subject32' and r['window'] in [0,2]];report=[];traces={}
    for row in rows:
        window=row['window'];prefix=f'w{window}_';q=row['start_s']+np.arange(301)/30
        ref=np.interp(q,u,ppg);idx=next(i for i,r in enumerate(clean.rows) if r==row);_,dy,_=clean[idx]
        reconstructed=np.r_[0.,np.cumsum(dy.numpy()[:-1])]
        sensor=gt[1,(times>=q[0])&(times<q[-1])];dt=np.diff(times);mask=(times[:-1]<q[-1])&(times[1:]>q[0])
        rr={**row,'reference':{},'inputs':{},'models':{},'sensor_HR':{'mean':float(sensor.mean()),'median':float(np.median(sensor)),'min':float(sensor.min()),'max':float(sensor.max()),'below40_count':int((sensor<40).sum())},
            'timing':{'max_gap_s':float(dt[mask].max()),'duplicate_intervals':int((dt[mask]==0).sum())}}
        legacy_ref=np.interp(q[:-1],legacy['t'],legacy['ppg'])
        for name,w in [('ppg',ref[:300]),('legacy_reference',legacy_ref),('training_target_reconstructed',reconstructed)]:
            st,f,p,filtered=stats(w,row['gt_hr']);rr['reference'][name]=st
            traces[prefix+name]=w;traces[prefix+name+'_psd']=p;traces[prefix+name+'_filtered']=filtered;traces['bpm']=f
        assert rr['reference']['legacy_reference']['hr']==row['gt_hr']
        for c,root in roots.items():
            m=np.load(root/'subject32_meta.npz');raw=np.load(root/'subject32_rgb.npy',mmap_mode='r');t=m['t']
            j=np.clip(np.searchsorted(t,q,side='right'),1,len(t)-1);w=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)[:,None,None,None]
            clip=raw[j-1].astype(np.float32)*(1-w)+raw[j].astype(np.float32)*w;rgb=clip.mean(axis=(1,2))
            roi_mask=(m['original_t']>=q[0])&(m['original_t']<=q[-1]);boxes=m['boxes'][roi_mask];rr['inputs'][c]={'roi_max_coordinate_step_px':int(np.abs(np.diff(boxes,axis=0)).max())}
            for name,wave in [('pos',pos(rgb,30.)[:300]),('green',rgb[:300,1])]:
                st,f,p,filtered=stats(wave,row['gt_hr']);rr['inputs'][c][name]=st
                traces[prefix+c+'_'+name]=wave;traces[prefix+c+'_'+name+'_psd']=p
            traces[prefix+c+'_rgb']=rgb;traces[prefix+c+'_boxes']=boxes
        if not a.inputs_only:
            results=read(OUT/'results.json')
            for family in ['mse','spectral']:
                rr['models'][family]={}
                for c in roots:
                    z=np.load(OUT/f'evaluation/{family}_{c}_waves.npz')['waves'];records=results[family][c]['2027']['records']
                    i=next(i for i,r in enumerate(records) if r['subject']=='subject32' and r['window']==window)
                    rr['models'][family][c]={}
                    for k,seed in enumerate(SEEDS):
                        st,f,p,filtered=stats(z[k,i],row['gt_hr']);stored=results[family][c][str(seed)]['records'][i];assert st['hr']==stored['pred_hr']
                        rr['models'][family][c][str(seed)]={**st,'abs_error':stored['abs_error']}
                        traces[prefix+family+'_'+c+'_'+str(seed)]=z[k,i];traces[prefix+family+'_'+c+'_'+str(seed)+'_psd']=p
        report.append(rr)
    name='input_diagnosis' if a.inputs_only else 'diagnosis'
    np.savez(D/(name+'_traces.npz'),**traces)
    (D/(name+'.json')).write_text(json.dumps({'records':report,'source_ground_truth_sha256':sha(D/'ground_truth.txt'),'raw_ppg_timestamps_match_training_cache':True,'test_opened':False,
        'limits':'Posthoc reference-conditioned peak power is oracle diagnosis, never inference. Five-second-half peak estimates have poor frequency resolution and cannot alone establish physiological HR variation. Sensor HR has undocumented smoothing/alignment; disagreement does not prove a wrong label. No target or decoder changes.'},indent=2))
    print('SUBJECT32',name,'COMPLETE',flush=True)
if __name__=='__main__':main()
