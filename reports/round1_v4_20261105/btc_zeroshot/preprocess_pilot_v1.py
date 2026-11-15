"""CPU-only dev decoding, exact T8 C5, oval36 FFV1 and timestamp receipts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,fcntl,hashlib,json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/samsung_t5/khangle/facevitals')
CODE=Path('/data_pool/khangle/facevitals/code/t8_oval_20261031')
V3=Path('/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102')
sys.path.insert(0,str(CODE));sys.path.insert(0,str(V3))
from simulate import simulate,sha256
from oval_crop import OvalCrop
from polarity_guard import storage_guard
import cv2
cv2.setNumThreads(1)

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def size(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file()) if Path(p).exists() else 0
def guard():
    r=read(HERE/'INVENTORY.json')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(r['measured_utc'])).total_seconds()
    if not -60<=age<=3600:raise RuntimeError('WAIT_FRESH_INVENTORY')
    index=read(ROOT/'t8_oval_20261031/training_index.json')
    b=storage_guard(index,ROOT/'t8_oval_20261031',ROOT/'t8_polarity_v3_20261102')
    whole=b['total_bytes']+size(HERE.parent)+size('/data_pool/khangle/facevitals/reports/round1_v4_20261105')+r['a100_v4_bytes']
    if whole+250_000_000>64_900_000_000:raise RuntimeError('WAIT_T8_STORAGE')
    project=sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in [ROOT,Path('/data_pool/khangle/facevitals')])+r['a100_bytes']
    if project+250_000_000>499_900_000_000:raise RuntimeError('WAIT_PROJECT_STORAGE')
    if size(HERE)>1_200_000_000:raise RuntimeError('WAIT_BTC_1_2GB_BUDGET')
    return dict(T8_bytes=whole,project_bytes=project,inventory_sha256=sha256(HERE/'INVENTORY.json'))

class Video:
    def __init__(self,path,log):
        self.path=Path(path);self.log=Path(log)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)]))
        self.stream=probe['streams'][0]
        self.t=np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']]);self.t-=self.t[0]
        if len(self.t)<2 or not np.isfinite(self.t).all() or np.any(np.diff(self.t)<=0):raise ValueError('invalid_or_duplicate_display_PTS')
        rotation=next((float(s['rotation']) for s in self.stream.get('side_data_list',[]) if 'rotation' in s),0)
        self.width=int(self.stream['width']);self.height=int(self.stream['height'])
        if abs(rotation)%180==90:self.width,self.height=self.height,self.width
        elif rotation%180:raise ValueError('unsupported_non_right_angle_rotation')
        tr=self.stream.get('color_transfer','unknown');self.hdr=tr in ['arib-std-b67','smpte2084']
        if self.hdr:
            if self.stream.get('color_primaries')!='bt2020' or self.stream.get('color_space') not in ['bt2020nc','bt2020c']:raise ValueError('HDR_color_metadata_incomplete')
            self.vf='zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0:peak=10,zscale=t=bt709:m=bt709:r=full,format=rgb24'
        else:self.vf='format=rgb24'
        self.command=['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-i',str(path),'-map','0:v:0','-an','-sn','-dn','-filter_threads','1','-vf',self.vf,'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    def __iter__(self):
        with self.log.open('wb') as log:
            p=subprocess.Popen(self.command,stdout=subprocess.PIPE,stderr=log)
            n=0;framebytes=self.width*self.height*3
            try:
                while True:
                    raw=p.stdout.read(framebytes)
                    if not raw:break
                    if len(raw)!=framebytes:raise ValueError('partial_decode_frame')
                    yield np.frombuffer(raw,np.uint8).reshape(self.height,self.width,3);n+=1
                if p.wait()!=0:raise RuntimeError('decode_failed: '+str(self.log))
                if n!=len(self.t):raise ValueError(f'decode_PTS_mismatch:{n}/{len(self.t)}')
            finally:
                if p.poll() is None:p.terminate();p.wait()

def crop(source,t,dest):
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        receipt=read(dest/'COMPLETE.json');assert sha256(dest/'crop.mkv')==receipt['crop_sha256'];return receipt
    if (dest/'crop.mkv').exists():raise RuntimeError('partial_crop_preserved_requires_inspection')
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','36x36','-r','30','-i','pipe:0','-an','-threads','1','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-slicecrc','1',str(dest/'crop.mkv')]
    valid=[];means=[];boxes=[];h=hashlib.sha256();det=OvalCrop()
    with (dest/'ffv1.log').open('wb') as log:
        p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        try:
            for n,rgb in enumerate(source):
                good,small,box,_=det.step(rgb)
                raw=small.tobytes();p.stdin.write(raw);h.update(raw)
                valid.append(bool(good));means.append(small.astype(float).mean((0,1)));boxes.append(box)
                if n%100==0:guard()
            p.stdin.close();assert p.wait()==0
        finally:
            det.close()
            if p.poll() is None:p.terminate();p.wait()
    assert len(valid)==len(t),'crop_clock_count_mismatch'
    # Verify every cached RGB byte after lossless decode, before a completion receipt.
    verify=['ffmpeg','-v','error','-threads','1','-i',str(dest/'crop.mkv'),'-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    q=subprocess.Popen(verify,stdout=subprocess.PIPE);hh=hashlib.sha256();count=0
    while True:
        b=q.stdout.read(1024*1024)
        if not b:break
        count+=len(b);hh.update(b)
    assert q.wait()==0 and count==len(t)*36*36*3 and hh.hexdigest()==h.hexdigest(),'FFV1_roundtrip_not_bit_exact'
    np.savez_compressed(dest/'meta.npz',t=t,valid=valid,rgb_mean=means,boxes=boxes)
    receipt=dict(frames=len(t),valid_frames=sum(valid),face_success_fraction=float(np.mean(valid)),valid_frame_fraction=float(np.mean(valid)),
        face_success_definition='nonempty detected 36-point oval; same condition as usable crop, not landmark accuracy',
        timestamp_authority='meta.npz source timestamps, FFV1 30fps is a storage clock only',crop_sha256=sha256(dest/'crop.mkv'),meta_sha256=sha256(dest/'meta.npz'),
        decoded_rgb_sha256=h.hexdigest(),ffv1_roundtrip_bit_exact=True,command=cmd)
    save(dest/'COMPLETE.json',receipt);return receipt

def process(row):
    dest=HERE/'dev'/row['drive_id'];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    assert sha256(row['path'])==row['sha256'],'raw_source_hash_mismatch'
    native=Video(row['path'],dest/'native_decode.log')
    info=dict(recording=row['recording'],person=row['person'],source_sha256=row['sha256'],display_frames=len(native.t),
        source_span_s=float(native.t[-1]),actual_frame_rate=float((len(native.t)-1)/native.t[-1]),stream=native.stream,
        source_decode_command=native.command,rotation_policy='FFmpeg default display-matrix autorotate exactly once',hdr=native.hdr,
        hdr_policy='HLG/PQ to linear100nit reference; BT709 primaries; Hable peak10 desat0; BT709 transfer full-rangeRGB8; fixed before outcomes',
        SDR_policy='FFmpeg tagged matrix/range conversion to RGB24; no HDR tone mapping for SDR; unknown tags retain decoder defaults and are disclosed',
        split_sha256=sha256(HERE/'btc_split.json'))
    save(dest/'SOURCE.json',info)
    if native.t[-1]<10:
        save(dest/'COMPLETE.json',dict(info,state='unscorable',reason='shorter_than_10s_source_PTS_span_no_padding'));return
    receipt=crop(native,native.t,dest/'native')
    enc=dest/'c5_encoded'
    if (enc/'manifest.json').exists():
        encoding=read(enc/'manifest.json');encoded=enc/'video.mp4'
    else:
        if enc.exists() and any(enc.iterdir()):raise RuntimeError('partial_C5_preserved_requires_inspection')
        encoded,encoding=simulate(native,native.t,enc,'c5',fps=10,bitrate_kbps=500,budget_root=HERE,budget_bytes=1_200_000_000)
    assert encoding['bitrate_pass'] and encoding['actual']['whole_file_kbps']<=500,'C5_actual_bitrate_exceeded'
    with np.load(enc/'frame_mapping.npz') as f:t=f['timestamps_s'].copy()
    c5=Video(encoded,dest/'c5_decode.log');assert len(c5.t)==len(t)
    c5receipt=crop(c5,t,dest/'c5')
    save(dest/'COMPLETE.json',dict(info,state='complete',native=receipt,c5=c5receipt,encoding_sha256=sha256(enc/'manifest.json'),
        code_sha256=sha256(__file__),completed_utc=now()))

def main(limit=None):
    lock=(HERE/'preprocess.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    man=read(HERE/'dev_manifest.json');split=read(HERE/'btc_split.json')
    assert man['split_sha256']==sha256(HERE/'btc_split.json')
    for r in man['videos']:assert split['people'][r['person']]['partition']=='dev'
    # Freeze the actual imported frontend and codec hashes before any predictions.
    src={str(p):sha256(p) for p in [Path(__file__),CODE/'simulate.py',CODE/'protocol.json',CODE/'oval_crop.py',CODE/'oval_frontend.py']}
    receipt=HERE/'PREPROCESS_SOURCES.json'
    if receipt.exists():assert read(receipt)==src,'Changed preprocessing code'
    else:save(receipt,src)
    for i,r in enumerate(man['videos'][:limit]):
        while True:
            try:budget=guard();break
            except RuntimeError as e:
                save(HERE/'PREPROCESS_STATUS.json',dict(utc=now(),state='waiting_guard',reason=str(e),pid=os.getpid()));time.sleep(30)
        save(HERE/'PREPROCESS_STATUS.json',dict(utc=now(),state='processing',video=r['recording'],position=i+1,total=len(man['videos']),pid=os.getpid(),budget=budget))
        try:process(r)
        except Exception as e:
            save(HERE/'PREPROCESS_ERROR.json',dict(utc=now(),video=r['recording'],error=repr(e),traceback=traceback.format_exc()));raise
        print('COMPLETE',r['recording'],flush=True)
    save(HERE/'PREPROCESS_STATUS.json',dict(utc=now(),state='complete' if limit is None else 'pilot_complete',pid=os.getpid(),videos=min(limit or len(man['videos']),len(man['videos']))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args();main(a.limit)
