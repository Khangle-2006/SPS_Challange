"""Mid-window visual checks for three prior failure cases, no test subjects."""
from pathlib import Path
import cv2,numpy as np,json
r=Path('/data/khangle/facevitals');out=r/'results/spectral_roi_jump';out.mkdir(parents=True,exist_ok=True)
chosen=[('subject11',58.03),('subject11',58.10),('subject11',59.08),('subject11',59.115)]
canvas=np.full((len(chosen)*270,960,3),245,np.uint8);records=[]
for row,(s,time) in enumerate(chosen):
    m=np.load(r/'datasets/UBFC-rPPG/efficientphys36_trainval_v1'/(s+'_meta.npz'))
    index=int(np.argmin(np.abs(m['original_t']-time)));box=m['boxes'][index];x1,y1,x2,y2=map(int,box)
    for col,(label,path) in enumerate([('clean',r/'datasets/UBFC-rPPG/DATASET_2'/s/'vid.avi'),('H264500k',r/'datasets/UBFC-rPPG/training_h264_500k_v1'/(s+'.mp4') if s!='subject32' else r/'datasets/UBFC-rPPG/validation_h264_500k_v1'/(s+'.mp4'))]):
        cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,f=cap.read();cap.release();assert ok
        crop=f[y1:y2,x1:x2].copy();cv2.rectangle(f,(x1,y1),(x2,y2),(0,220,0),2)
        thumb=cv2.resize(f,(320,240));zoom=cv2.resize(crop,(160,240))
        panel=np.concatenate([thumb,zoom],axis=1);canvas[row*270+30:(row+1)*270,col*480:(col+1)*480]=panel
        cv2.putText(canvas,f'{s} {time:.2f}s {label}',(col*480+8,row*270+21),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
    records.append({'subject':s,'requested_time':time,'frame_index':index,'recorded_time':float(m['original_t'][index]),'box':box.tolist()})
cv2.imwrite(str(out/'roi_contact_sheet.png'),canvas)
(out/'frames.json').write_text(json.dumps(records,indent=2));print('ROI SNAPSHOTS COMPLETE')
