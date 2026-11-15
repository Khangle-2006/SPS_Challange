"""Audit all AiPEX files and fixed temporal aggregations; never tune FPS to labels."""
import csv, hashlib, json, math, concurrent.futures
from pathlib import Path
import cv2
import numpy as np
from hr_baseline import pos, estimate_hr
ROOT=Path('/data_pool/khangle/facevitals')
DATA=ROOT/'datasets/AiPEX'
OLD=ROOT/'results/aipex_pos_green_v1'
OUT=ROOT/'results/aipex_audit_v1'

def process(task):
    group,subject,gt=task; rois=[]
    for region in ('forehead','leftcheek','rightcheek'):
        path=DATA/'Data'/group/f'{subject}_{region}.avi'
        marker=json.loads(path.with_name(path.name+'.complete.json').read_text())
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==marker['sha256']
        cap=cv2.VideoCapture(str(path));assert cap.isOpened();cv2.setNumThreads(1)
        fps=cap.get(cv2.CAP_PROP_FPS);declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc=int(cap.get(cv2.CAP_PROP_FOURCC));codec=''.join(chr((fourcc>>8*i)&255) for i in range(4))
        values=[];pts=[];duplicates=0;prev=None;clipped=0;pixel_count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            values.append(cv2.mean(frame)[:3][::-1]);pts.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
            if prev is not None and np.array_equal(prev,frame):duplicates+=1
            prev=frame;clipped+=int(((frame==0)|(frame==255)).sum());pixel_count+=frame.size
        cap.release();rgb=np.asarray(values);assert len(rgb)==declared and 1<fps<300
        cache=np.load(OLD/f'{group}_{subject}_{region}_rgb.npz',allow_pickle=False)
        assert float(cache['fs'])==fps and str(cache['sha256'])==marker['sha256']
        np.testing.assert_allclose(rgb,cache['rgb'],rtol=0,atol=1e-10)
        bvp=pos(rgb,fps);trim=math.ceil(1.6*fps)
        def hr(x):
            try:return estimate_hr(x,fps)
            except ValueError:return None
        result={'group':group,'subject':subject,'roi':region,'gt':gt,'fps':fps,'frames':len(rgb),
                'width':w,'height':h,'codec':codec,'container_duration_s':len(rgb)/fps,
                'pts_first_s':pts[0],'pts_last_s':pts[-1],
                'pts_nonpositive_steps':int(np.sum(np.diff(pts)<=0)),
                'identical_adjacent_frames':duplicates,'clipped_channel_fraction':clipped/pixel_count,
                'cache_reextraction_max_difference':float(np.max(np.abs(rgb-cache['rgb']))),
                'sha256':marker['sha256'],'channel_temporal_std':rgb.std(axis=0).tolist(),
                'pos_whole':hr(bvp[trim:-trim]),'green_whole':hr(rgb[trim:-trim,1])}
        for seconds in (5,10):
            n=round(seconds*fps);windows=[]
            for start in range(trim,len(rgb)-trim-n+1,n):
                windows.append({'start_s':start/fps,'pos':hr(bvp[start:start+n]),'green':hr(rgb[start:start+n,1])})
            assert windows
            result[f'windows{seconds}']=windows
            for method in ('pos','green'):
                vals=[r[method] for r in windows]
                result[f'{method}_{seconds}s']=float(np.mean(vals)) if all(v is not None for v in vals) else None
        rois.append(result)
    record={'group':group,'subject':subject,'gt':gt}
    for method in ('pos','green'):
        for aggregation in ('whole','5s','10s'):
            key=method+'_'+aggregation;vals=[r[key] for r in rois]
            record[key]=float(np.median(vals)) if all(v is not None for v in vals) else None
    return record,rois

def metrics(rows,key):
    errors=[abs(r[key]-r['gt']) if r[key] is not None else 135 for r in rows]
    return {'n':len(rows),'MAE':float(np.mean(errors)),'RMSE':float(np.sqrt(np.mean(np.square(errors)))),
            'invalid':sum(r[key] is None for r in rows),
            'bias':float(np.mean([r[key]-r['gt'] for r in rows])) if all(r[key] is not None for r in rows) else None}

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for group in ('India','Sierra_Leone'):
        path=DATA/'Ground_Truth'/f'{group}_GT.csv'
        marker=json.loads(path.with_name(path.name+'.complete.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==marker['sha256']
        with path.open(encoding='utf-8-sig') as f:
            rows=list(csv.reader(f))[1:]
        tasks += [(group,r[0].strip(),float(r[1])) for r in rows if r and r[0].strip()]
    assert len(tasks)==88 and len({(g,s) for g,s,_ in tasks})==88
    records=[];all_rois=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for rec,rois in pool.map(process,tasks):
            records.append(rec);all_rois.extend(rois)
            print(json.dumps({'done':len(records),'group':rec['group'],'subject':rec['subject']}),flush=True)
    old=list(csv.DictReader((OLD/'recordings.csv').open()))
    for r in records:
        prev=next(x for x in old if x['group']==r['group'] and x['subject']==r['subject'])
        assert abs(float(prev['pos_bpm'])-r['pos_10s'])<1e-8
        assert abs(float(prev['green_bpm'])-r['green_10s'])<1e-8
        assert float(prev['sensor_hr_bpm'])==r['gt']
    summary={'records':88,'videos':len(all_rois),'all_file_hashes_and_cached_RGB_verified':True,
             'old_protocol_reproduced':True,'fps_values':sorted({r['fps'] for r in all_rois}),
             'dimensions':sorted({f"{r['width']}x{r['height']}" for r in all_rois}),
             'codecs':sorted({r['codec'] for r in all_rois}),
             'duration_s_range':[min(r['container_duration_s'] for r in all_rois),max(r['container_duration_s'] for r in all_rois)],
             'decoded_frames':sum(r['frames'] for r in all_rois),
             'identical_adjacent_frames':sum(r['identical_adjacent_frames'] for r in all_rois),
             'pts_nonpositive_steps':sum(r['pts_nonpositive_steps'] for r in all_rois),
             'metrics':{k:metrics(records,k) for k in ('pos_10s','pos_5s','pos_whole','green_10s','green_5s','green_whole')},
             'by_group':{g:metrics([r for r in records if r['group']==g],'pos_10s') for g in ('India','Sierra_Leone')},
             'paper_protocol':{'fps_average':25,'recording_duration_s':120,'window_s':5,'reference':'mean sensor HR over recording'},
             'limits':'Container timestamps verify encoding time only, not original acquisition timing. No FPS inferred from labels. Released ROI videos may differ from original full recordings. Five-second and whole-record variants are descriptive, not tuned models or exact paper reproduction.'}
    (OUT/'records.json').write_text(json.dumps(records,indent=2))
    (OUT/'rois.json').write_text(json.dumps(all_rois,indent=2))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
