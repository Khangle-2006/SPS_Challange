"""Read-only audit of three predeclared DEV caches, original PTS and encoded PTS."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,subprocess,hashlib
from pathlib import Path
import numpy as np
from scipy import signal,sparse
from scipy.sparse.linalg import spsolve
OUT=Path(__file__).resolve().parent;BASE=OUT.parent
sys.path.insert(0,str(BASE))
from infer_dev import crop_rgb,classical_functions,decode_wave,supported
from distribution import read,save,sha,csvout
SELECTED=['S_N1_Resting','S_N2_Resting','S_N3_Resting']
def probe(p):
    d=json.loads(subprocess.check_output(['ffprobe','-v','error','-threads','1','-select_streams','v:0','-show_streams','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(p)]))
    t=np.array([float(f['best_effort_timestamp_time']) for f in d['frames']]);t-=t[0]
    assert np.isfinite(t).all() and np.all(np.diff(t)>0)
    return d['streams'][0],t
def filtered_psd(w):
    n=len(w);D=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')
    y=w-spsolve(sparse.eye(n,format='csc')+100**2*(D.T@D),w)
    y=signal.sosfiltfilt(signal.butter(1,[.7,3.],btype='bandpass',fs=30,output='sos'),y)
    return signal.periodogram(y,fs=30,window='hann',nfft=4096,detrend=False)
def main():
    dest=OUT/'loader';dest.mkdir(exist_ok=True)
    man=read(BASE/'dev_manifest.json');split=read(BASE/'btc_split.json');ns,_=classical_functions();qc=[];spectra=[]
    for rec in SELECTED:
        v=next(v for v in man['videos'] if v['recording']==rec);assert split['people'][v['person']]['partition']=='dev'
        d=BASE/'dev'/v['drive_id'];src=read(d/'SOURCE.json');stream,source_t=probe(v['path'])
        assert len(source_t)==src['display_frames'];assert sha(v['path'])==v['sha256']
        rot=next((float(s['rotation']) for s in stream.get('side_data_list',[]) if 'rotation' in s),0)
        encstream,mux_t=probe(d/'c5_encoded/video.mp4')
        with np.load(d/'c5_encoded/frame_mapping.npz') as m:mapping={k:m[k].copy() for k in m.files}
        selected=mapping['timestamps_s'];indices=np.searchsorted(source_t,selected)
        assert np.all(indices<len(source_t)) and np.allclose(source_t[indices],selected,atol=1e-6,rtol=0)
        assert len(selected)==len(mux_t) and len(set(indices))==len(indices)
        assert np.max(np.abs(mux_t-np.arange(len(mux_t))/10))<1e-5
        for cond in ['native','c5']:
            dd=d/cond;receipt=read(dd/'COMPLETE.json');assert sha(dd/'crop.mkv')==receipt['crop_sha256'] and sha(dd/'meta.npz')==receipt['meta_sha256']
            with np.load(dd/'meta.npz') as m:t=m['t'].copy();valid=m['valid'].copy();mean=m['rgb_mean'].copy()
            assert np.array_equal(t,source_t if cond=='native' else selected)
            rgb=crop_rgb(dd,len(t));assert hashlib.sha256(rgb.tobytes()).hexdigest()==receipt['decoded_rgb_sha256']
            ids=np.linspace(0,len(t)-1,8).astype(int)
            np.savez_compressed(dest/(rec+'_'+cond+'_frames.npz'),rgb=rgb[ids],indices=ids,t=t[ids],valid=valid[ids])
            qc.append(dict(recording=rec,condition=cond,frames=len(t),valid_frames=int(valid.sum()),face_fraction=float(valid.mean()),rotation_degrees=rot,
                source_width=stream['width'],source_height=stream['height'],source_transfer=stream.get('color_transfer'),source_pixel=stream.get('pix_fmt'),
                pts_span_s=float(t[-1]),max_pts_gap_s=float(np.diff(t).max()),count_and_source_PTS_match=True,ffv1_rgb_hash_verified=True,
                C5_mux_count=len(mux_t) if cond=='c5' else None,C5_mux_CFR10_verified=True if cond=='c5' else None,
                C5_mux_vs_source_max_difference_s=float(np.max(np.abs(mux_t-t))) if cond=='c5' else None,
                face_definition=receipt['face_success_definition']))
            for start in [0,10,20]:
                ok=supported(t,valid,start,start+10)
                if not ok:continue
                q=start+np.arange(300)/30;x=np.stack([np.interp(q,t,mean[:,c]) for c in range(3)],axis=1)
                w=ns['pos'](x,30);f,p=filtered_psd(w);sel=(f>=.7)&(f<=3);hr=decode_wave(w)
                ix=int(np.argmin(np.abs(f*60-v['pr_bpm'])));peak=float(p[sel].max())
                spectra.append(dict(recording=rec,condition=cond,start_s=start,PR=v['pr_bpm'],POS_HR=hr,power_at_PR_nearest_bin=float(p[ix]),max_inband_power=peak,PR_power_fraction_of_peak=float(p[ix]/peak) if peak else None))
                csvout(dest/f'{rec}_{cond}_{start}_PSD.csv',[dict(bpm=float(a*60),power=float(b)) for a,b in zip(f,p)])
        save(dest/(rec+'_SOURCE_AUDIT.json'),dict(source=src,source_probe=stream,c5_probe=encstream,mapping_keys=list(mapping),selected_indices_verified=indices.tolist(),split_sha256=man['split_sha256']))
    csvout(dest/'QC.csv',qc);csvout(dest/'PSD_SUMMARY.csv',spectra);save(dest/'SUMMARY.json',dict(qc=qc,spectra=spectra,code_sha256=sha(__file__),note='3 predeclared dev videos only; no inference of HDR/rotation correctness outside this sample.'))
    print('LOADER_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':main()
