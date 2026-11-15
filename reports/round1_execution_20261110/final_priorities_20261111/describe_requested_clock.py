"""Descriptive requested-frequency test, separate from measured-cycle eligibility."""
import os,sys,json,hashlib,csv
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'codec_clock_20261111'))
import report_clock as r
import padding_queue as q
r.P=P/'codec_exact'
def main():
 q.budget(fresh=True)
 target=r.P/'REQUESTED_CLOCK_DESCRIPTION.json'
 assert not target.exists(),'Preserve prior descriptive output'
 rows,enc=r.loadrows('real');assert len(enc)==66
 selected=[x for x in rows if x.get('bf') in (3,4) and x['roi']=='oval36' and x['clock']=='source_clock' and x.get('peak_bpm') is not None]
 summaries=[]
 import numpy as np
 for ds,fs,bf,m in sorted({(x['dataset'],x['fs'],x['bf'],x['method']) for x in selected}):
  rr=[x for x in selected if (x['dataset'],x['fs'],x['bf'],x['method'])==(ds,fs,bf,m)];nominal=60*fs/(bf+1)
  summaries.append(dict(dataset=ds,fps=fs,bf=bf,method=m,windows=len(rr),requested_frequency_bpm=nominal,median_peak=float(np.median([x['peak_bpm'] for x in rr])),near_requested=sum(abs(x['peak_bpm']-nominal)<=2 for x in rr),measured_cycle_eligible=sum(x.get('cycle_bpm') is not None for x in rr)))
 look={(x['dataset'],x['recording'],x['fs'],x['method'],x['window'],x['bf']):x for x in selected};pairs=[]
 for k,x in look.items():
  if k[-1]!=3 or k[:-1]+(4,) not in look:continue
  y=look[k[:-1]+(4,)];f=x['fs'];pairs.append(dict(dataset=x['dataset'],recording=x['recording'],fps=f,method=x['method'],window=x['window'],peak_b3=x['peak_bpm'],peak_b4=y['peak_bpm'],requested_b3=60*f/4,requested_b4=60*f/5,follows_both_requested=abs(x['peak_bpm']-60*f/4)<=2 and abs(y['peak_bpm']-60*f/5)<=2,measured_cycle_pair_eligible=bool(x.get('cycle_bpm') and y.get('cycle_bpm'))))
 r.writecsv(r.P/'REQUESTED_CLOCK_PAIRS.csv',pairs)
 r.save(target,dict(utc=r.now(),summary=summaries,pairs=len(pairs),follows_both_requested=sum(x['follows_both_requested'] for x in pairs),measured_cycle_pairs=sum(x['measured_cycle_pair_eligible'] for x in pairs),unchanged_dominance_threshold=.8,code_sha256=r.sha(__file__),limitation='Requested bf is a maximum. Descriptive closeness to fs/(bf+1) is not observed periodicity or proof of physiological absence. No gate/encoding/prediction changed.'))
 lines=['# Dịch đỉnh: tần số đặt trước và lịch frame thực','', 'Cột đích yêu cầu dùng 60×fps/(bf+1), chỉ là giả thuyết cần kiểm. Nó KHÔNG được thay cho chu kỳ đo từ bitstream. Với GOP ngắn của recipe thật, khoảng cách frame tham chiếu bị ngắt ở biên GOP; không bản bf3/4 nào đạt ngưỡng chu kỳ nổi trội 80% đã đặt. Vì vậy bảng cũ có mẫu số 0/0 và không tạo CSV cặp chu kỳ đủ điều kiện. Đây không phải chứng cứ không có artifact. Không thay ngưỡng 80%.','', '| Dataset | FPS | bf | Phép chiếu | Đích yêu cầu bpm | Đỉnh trung vị bpm | Gần đích/cửa sổ |','|---|---:|---:|---|---:|---:|---:|']
 for x in summaries:lines.append(f"|{x['dataset']}|{x['fps']}|{x['bf']}|{x['method']}|{x['requested_frequency_bpm']:.2f}|{x['median_peak']:.2f}|{x['near_requested']}/{x['windows']}|")
 lines+=['','Không kết luận mọi đỉnh đi theo fs/(bf+1). Bảng cặp REQUESTED_CLOCK_PAIRS.csv giữ từng cửa sổ, kể cả không đi theo giả thuyết; đây là mô tả, không chọn mô hình hay tinh chỉnh.']
 (r.P/'REQUESTED_CLOCK_DESCRIPTION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('Requested-frequency description saved',len(pairs))
if __name__=='__main__':main()
