"""Additive CPU audit of a frozen dev-only cohort. Never modifies inference caches."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv,hashlib,fcntl,traceback,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy import signal,sparse
from scipy.sparse.linalg import spsolve
OUT=Path(__file__).resolve().parent;BASE=OUT.parent
sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'diagnostics_20261108'))
from infer_dev import read,sha,save,classical_functions,decode_wave,crop_rgb,ROOT
from loader_audit import filtered_psd
from polarity_guard import storage_guard

def csvout(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def status(s,**kw):save(OUT/'STATUS.json',dict(state=s,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw))
def guard():
    r=read(OUT/'INVENTORY.json');age=(datetime.now(timezone.utc)-datetime.fromisoformat(r['measured_utc'])).total_seconds()
    assert -60<=age<=3600,'Fresh cross-server inventory required'
    b=storage_guard(read(ROOT/'t8_oval_20261031/training_index.json'),ROOT/'t8_oval_20261031',ROOT/'t8_polarity_v3_20261102')
    size=lambda p:sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file()) if Path(p).exists() else 0
    whole=b['total_bytes']+size(BASE.parent)+size('/data_pool/khangle/facevitals/reports/round1_v4_20261105')+r['a100_v4_bytes']
    assert whole+250_000_000<64_900_000_000,'T8 budget'
    project=sum(int(subprocess.check_output(['du','-sb',p],text=True).split()[0]) for p in [str(ROOT),'/data_pool/khangle/facevitals'])+r['a100_bytes']
    assert project+250_000_000<499_900_000_000,'project budget'
    assert size(BASE)+30_000_000<1_200_000_000,'BTC derivatives reserve'
    return dict(T8_bytes=whole,project_bytes=project,btc_bytes=size(BASE),reserve_bytes=30_000_000)

def main():
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'COMPLETE.json').exists(),'Completed: do not rerun'
    budget=guard();save(OUT/'PREFLIGHT.json',budget)
    man=read(BASE/'dev_manifest.json');split=read(BASE/'btc_split.json');cohort=read(OUT/'COHORT.json')
    assert sha(BASE/'btc_split.json')==man['split_sha256']==cohort['split_sha256']
    videos={v['recording']:v for v in man['videos']};selected=cohort['recordings'];assert len(selected)==205
    assert all(split['people'][videos[r]['person']]['partition']=='dev' for r in selected)
    ns,_=classical_functions();wins=[];metas=[];c5=[];spectra={};hashes={}
    for j,rec in enumerate(selected):
        v=videos[rec];d=BASE/'dev'/v['drive_id'];src=read(d/'SOURCE.json');s=src['stream']
        rot=next((float(x['rotation']) for x in s.get('side_data_list',[]) if 'rotation' in x),0.)
        metas.append(dict(recording=rec,person=v['person'],fold=v['fold'],truth=v['pr_bpm'],group=v['group'],task=v['task'],skin=v['skin'],lighting=v['lighting'],resolution=f"{s['width']}x{s['height']}",codec=s['codec_name'],hdr=src['hdr'],rotation=rot,fps=src['actual_frame_rate'],pixel=s['pix_fmt'],transfer=s.get('color_transfer','unknown')))
        dd=d/'native';receipt=read(dd/'COMPLETE.json');assert sha(dd/'meta.npz')==receipt['meta_sha256'];hashes[rec]=dict(meta=receipt['meta_sha256'],source=sha(d/'SOURCE.json'))
        with np.load(dd/'meta.npz') as m:t=m['t'];rgb=m['rgb_mean']
        saved=read(BASE/'predictions'/v['drive_id']/'native/classical.json')
        hashes[rec]['classical']=sha(BASE/'predictions'/v['drive_id']/'native/classical.json')
        for w in saved['rows']:
            if w['scheme']!='10s' or w['method']!='POS' or w['prediction'] is None:continue
            q=w['start']+np.arange(300)/30;x=np.stack([np.interp(q,t,rgb[:,c]) for c in range(3)],axis=1)
            wave=ns['pos'](x,30);hr=decode_wave(wave);assert abs(hr-w['prediction'])<1e-8,'POS frozen prediction parity'
            f,p=filtered_psd(wave);band=(f>=.7)&(f<=3);peak=f[band][np.argmax(p[band])];near=band&(abs(f-peak)<=.1);a=float(p[near].sum());b=float(p[band&~near].sum());ratio=a/b if b>0 else None
            assert ratio is not None and np.isfinite(ratio)
            wins.append(dict(recording=rec,person=v['person'],start=w['start'],truth=v['pr_bpm'],prediction=hr,snr_ratio=ratio,snr_db=float(10*np.log10(ratio)),peak_hz=float(peak),peak_power=a,remaining_power=b))
            spectra[rec,w['start']]=(f,p)
        for cond in ['native','c5']:
            for w in read(BASE/'predictions'/v['drive_id']/cond/'classical.json')['rows']:
                if w['scheme']=='10s':c5.append(dict(recording=rec,person=v['person'],condition=cond,method=w['method'],start=w['start'],prediction=w['prediction'],reason=w['reason']))
        status('spectral',completed=j+1,total=205)
    csvout('SNR_WINDOWS.csv',wins);csvout('VIDEO_METADATA.csv',metas);csvout('CLASSICAL_WINDOWS.csv',c5);save(OUT/'SOURCES.json',hashes)
    byrec={r:[w for w in wins if w['recording']==r] for r in selected}
    top=sorted(selected,key=lambda r:(-np.median([w['snr_ratio'] for w in byrec[r]]),r))[:20]
    top_rows=[]
    for rec in top:
        ww=sorted(byrec[rec],key=lambda w:(w['snr_ratio'],w['start']));rep=ww[len(ww)//2]
        f,p=spectra[rec,rep['start']]
        csvout(rec+'_PSD.csv',[dict(hz=float(x),power=float(y)) for x,y in zip(f,p)])
        top_rows.append(dict(rep,video_snr_median=float(np.median([w['snr_ratio'] for w in ww])),video_POS=float(np.median([w['prediction'] for w in ww])),video_error=float(np.median([w['prediction'] for w in ww])-rep['truth'])))
    csvout('TOP20.csv',top_rows);status('spectral_complete_qc_starting',windows=len(wins),parity=True)
    # Decode existing tiny lossless crops only; never raw or holdout, no tracking rerun.
    qc=[];special=[]
    for j,meta in enumerate(metas):
        rec=meta['recording'];v=videos[rec];d=BASE/'dev'/v['drive_id']
        if j%25==0:guard()
        for cond in ['native','c5']:
            dd=d/cond;receipt=read(dd/'COMPLETE.json');assert sha(dd/'crop.mkv')==receipt['crop_sha256'] and sha(dd/'meta.npz')==receipt['meta_sha256']
            with np.load(dd/'meta.npz') as a:t=a['t'];valid=a['valid'].astype(bool);box=a['boxes']
            rgb=crop_rgb(dd,len(t));assert hashlib.sha256(rgb.tobytes()).hexdigest()==receipt['decoded_rgb_sha256']
            pix=rgb[valid];mask=np.any(pix>0,axis=-1);p=pix[mask].astype(float)
            luma=p@np.array([.299,.587,.114]);bb=box[valid];wh=bb[:,2:]-bb[:,:2]
            row=dict(recording=rec,person=v['person'],condition=cond,frames=len(t),valid_frames=int(valid.sum()),face_fraction=float(valid.mean()),box_width_mean=float(wh[:,0].mean()),box_height_mean=float(wh[:,1].mean()),box_area_mean=float(np.prod(wh,axis=1).mean()),nonzero_pixel_count=len(p),luma_mean=float(luma.mean()),saturated_fraction=float(np.mean(np.any(p>=254,axis=1))),dark_fraction=float(np.mean(np.all(p<=1,axis=1))))
            qc.append(row)
            if meta['rotation']%360!=0 or meta['hdr']:
                ids=np.linspace(0,len(t)-1,8).astype(int)
                np.savez_compressed(OUT/(rec+'_'+cond+'_montage.npz'),rgb=rgb[ids],t=t[ids],valid=valid[ids],indices=ids)
                special.append(dict(recording=rec,condition=cond,rotation=meta['rotation'],hdr=meta['hdr']))
        csvout('QC_VIDEO.csv',qc);status('qc',completed=j+1,total=205)
    csvout('MONTAGE_INDEX.csv',special)
    save(OUT/'COMPLETE.json',dict(videos=205,windows=len(wins),source_hashes=hashes,code_sha256=sha(__file__),holdout_evaluations=0,training=False))
    status('complete',videos=205,windows=len(wins))
if __name__=='__main__':
    try:main()
    except Exception as e:status('error',error=repr(e),traceback=traceback.format_exc());raise
