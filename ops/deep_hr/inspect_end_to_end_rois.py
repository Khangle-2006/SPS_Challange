"""Post-hoc paired ROI illustrations; never changes inference or evaluation."""
import json
from pathlib import Path
import cv2
import numpy as np
R=Path('/data_pool/khangle/facevitals');OUT=R/'results/end_to_end_detection_20261019';D=R/'datasets/UBFC-rPPG'
def main():
    data=json.loads((OUT/'results.json').read_text());rows=[]
    for family,conds in data['groups'].items():
        for c,d in conds.items():
            a=d['native_ROI']['psd_mean']['records'];b=d['detected_ROI']['psd_mean']['records']
            for old,new in zip(a,b):rows.append({'family':family,'condition':c,'subject':old['subject'],'window':old['window'],
                'start_s':old['start_s'],'error_change':new['abs_error']-old['abs_error'],'native_hr':old['pred_hr'],'detected_hr':new['pred_hr'],'gt_hr':old['gt_hr']})
    chosen=[];seen=set()
    for r in sorted(rows,key=lambda r:r['error_change'],reverse=True):
        key=(r['condition'],r['subject'],r['window'])
        if key not in seen:chosen.append(r);seen.add(key)
        if len(chosen)==4:break
    canvas=np.full((4*340,1000,3),245,np.uint8)
    for n,r in enumerate(chosen):
        s=r['subject'];c=r['condition'];a=np.load(OUT/'native_reference'/f'{s}_meta.npz');b=np.load(D/f'detected_on_h264_20261019/{c}/{s}_meta.npz')
        mask=(a['original_t']>=r['start_s'])&(a['original_t']<r['start_s']+10)&b['original_valid']
        idx=np.where(mask)[0]
        if len(idx):
            aa=a['boxes'][idx];bb=b['boxes'][idx];inter=np.prod(np.maximum(0,np.minimum(aa[:,2:],bb[:,2:])-np.maximum(aa[:,:2],bb[:,:2])),axis=1)
            iou=inter/(np.prod(aa[:,2:]-aa[:,:2],axis=1)+np.prod(bb[:,2:]-bb[:,:2],axis=1)-inter)
            index=int(idx[np.argmin(iou)]);r['shown_iou']=float(iou.min())
        else:index=int(np.argmin(np.abs(a['original_t']-r['start_s']-5)));r['shown_iou']=None
        cap=cv2.VideoCapture(str(D/f'end_to_end_detection_inputs_20261019/{c}'/(s+'.mp4')));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,f=cap.read()
        actual=round(cap.get(cv2.CAP_PROP_POS_FRAMES))-1;cap.release();assert ok and actual==index
        oldbox=a['boxes'][index];newbox=b['boxes'][index];base=f.copy()
        for box,color in [(oldbox,(0,170,0)),(newbox,(0,0,220))]:
            x1,y1,x2,y2=map(int,box)
            if x2>x1 and y2>y1:cv2.rectangle(base,(x1,y1),(x2,y2),color,2)
        y=n*340;canvas[y+40:y+340,:400]=cv2.resize(base,(400,300))
        for j,(label,box) in enumerate([('native crop',oldbox),('detected crop',newbox)]):
            x1,y1,x2,y2=map(int,box)
            if x2>x1 and y2>y1:canvas[y+80:y+320,430+j*280:670+j*280]=cv2.resize(f[y1:y2,x1:x2],(240,240))
            cv2.putText(canvas,label,(430+j*280,y+65),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1)
        caption=f"{r['family']} {c} {s} w{r['window']} dError={r['error_change']:.2f}; green=native/red=detected"
        cv2.putText(canvas,caption,(8,y+25),cv2.FONT_HERSHEY_SIMPLEX,.52,(20,20,20),1)
        r.update(frame_index=index,recorded_time=float(a['original_t'][index]),native_box=oldbox.tolist(),detected_box=newbox.tolist())
    cv2.imwrite(str(OUT/'roi_examples.png'),canvas)
    (OUT/'roi_examples.json').write_text(json.dumps({'selection':'Post-hoc largest PSD-ensemble error regressions, distinct condition/subject/window; within each window show smallest native/detected IoU among usable frames. Illustration, not causal frame localization or anatomical ground truth.','records':chosen},indent=2))
    print('ROI ILLUSTRATIONS SAVED',flush=True)
if __name__=='__main__':main()
