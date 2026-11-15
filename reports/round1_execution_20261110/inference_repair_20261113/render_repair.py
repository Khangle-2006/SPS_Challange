"""Render saved evidence only. Does not choose or promote an estimator."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common_repair import *
from readout import spectra
import numpy as np,csv,time,traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict,Counter
plt.rcParams.update({'font.size':10,'figure.facecolor':'white','axes.facecolor':'white','text.color':'#18252d','axes.labelcolor':'#18252d','xtick.color':'#18252d','ytick.color':'#18252d','savefig.facecolor':'white'})

def table(lines,headers,rows):
 lines.extend(['','|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|'])
 lines.extend('|'+'|'.join(str(x) for x in row)+'|' for row in rows);lines.append('')

def dumpcsv(path,rows):
 if not rows:return
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
 assert (P/'COMPUTE_COMPLETE.json').exists() and (P/'CODEC_BAND_V2_COMPLETE.json').exists() and (P/'BASELINE_AUDIT_COMPLETE.json').exists()
 dest=P/'rendered';assert not dest.exists();dest.mkdir();figdir=dest/'figures';figdir.mkdir();lines=['# Sửa suy luận FaceVitals — bằng chứng và cách đọc','','Bản này tổng hợp đầu ra đã tính, chưa tự động thay FREEZE. Không huấn luyện; holdout chưa mở. Những thay đổi được thử đồng thời là phân tích khám phá trên dev, không phải đánh giá độc lập trên test.','','## Đọc số trước khi đọc kết luận','','MAE là sai số tuyệt đối trung bình, đơn vị bpm (nhịp/phút), càng thấp càng tốt. Bias là dự đoán trừ nhãn: âm nghĩa là dự đoán thấp. Mỗi người được tính trước rồi mới lấy trung bình giữa người; người có nhiều video không được tăng trọng số. CI95% là khoảng bootstrap 10.000 lần theo người, seed 0. Delta âm nghĩa là ứng viên tốt hơn mốc. Khoảng delta chứa 0 thì chưa chứng minh được cải thiện.','','Các phép thử ảnh tĩnh kiểm tra đầu ra giả khi không có mạch; chúng không chứng minh thuật toán đo đúng trên video thật. Mặt nạ màu và vùng giải phẫu trong báo cáo là ước lượng, không phải nhãn da/ống thở đã được chuyên gia xác nhận.']
 # Orientation and full-frame masking, no PR in decision.
 orientations=[]
 for j in manifest():
  a=read(P/'orientation'/j['key']/'COMPLETE.json');prep=P/'spatial'/j['key']/'native/prepare/PREPARED.json';full=read(prep) if prep.exists() else {}
  orientations.append(dict(dataset=j['dataset'],recording=j['recording'],group=j.get('metadata',{}).get('group','public'),resolved=a['resolved'],degrees_ccw=a['degrees_ccw'],agreement=a['agreement'],metadata_rotation=a['metadata_rotation'],sample_mask_fraction=a['non_skin_fraction_mean'],full_frame_mask_fraction=full.get('mask_fraction_mean')))
 dumpcsv(dest/'ORIENTATION_MASK.csv',orientations);lines+=['','## 1. Hướng mặt và vật che','','Góc ở đây là góc giải phẫu bổ sung sau khi đã áp metadata xoay một lần. 0° nghĩa là giữ nguyên. Các ca không đủ đồng thuận được giữ nguyên và ghi chưa xác định. Mặt nạ thay pixel bị đánh dấu bằng trung bình phần da giữ lại trong cùng khung, không thay bằng 0. Nó có thể xóa nhầm môi/mắt và vẫn giữ một phần ống; không gọi đây là phân đoạn cannula.']
 rows=[]
 for g in ['Normal','Abnormal']:
  rr=[r for r in orientations if r['dataset']=='BTC' and r['group']==g];fr=[r['full_frame_mask_fraction'] for r in rr if r['full_frame_mask_fraction'] is not None]
  rows.append([g,len(rr),sum(r['resolved'] and r['degrees_ccw']!=0 for r in rr),sum(not r['resolved'] for r in rr),*[sum(r['resolved'] and r['degrees_ccw']==d for r in rr) for d in [0,90,180,270]],f'{100*np.median(fr):.2f}%' if fr else 'thiếu',sum(x>.1 for x in fr)])
 table(lines,['Nhóm','Video','Cần xoay','Chưa xác định','0°','90°','180°','270°','Mask trung vị','Mask >10%'],rows)
 for rid in ['BTC_S_AB51_Resting','BTC_S_AB59_Resting']:
  path=P/'orientation'/rid/'before_rotation_mask.png'
  if path.exists():
   import shutil
   shutil.copy2(path,figdir/(rid+'.png'));lines+=['',f'![{rid}](figures/{rid}.png)','Trái→phải: khung ban đầu; sau xoay; pixel bị mặt nạ đánh dấu màu tím; xoay + thay pixel. Đây là ảnh minh họa, không tự chứng minh MAE giảm.']
 # Full-cohort measured native factorial, fixed saved baseline rows included.
 metrics=read(P/'report_spatial/BTC_METRICS.json');lines+=['','## 2. Kết quả BTC trên cùng người/video','','Các mốc hằng số, hiệu chỉnh và padding đã lưu được ghép lại trên đúng tập video của từng ứng viên trong `report_spatial/BTC_MATCHED_TABLES.csv`. Bảng ngắn dưới đây chỉ hiển thị native và nhóm toàn bộ; các nhóm bệnh nhân, hoạt động, xoay và mask nằm trong file CSV đó. Không so hai hàng khác số video như một thử nghiệm ghép cặp.']
 rows=[]
 for a in metrics:
  if a['condition']=='native' and a['group']=='all':
   d=a['delta_saved_calibrated'];rows.append([a['input'],a['method'],a['readout'],f"{a['videos']}/{a['people']}",f"{a['MAE']:.3f}",f"{a['bias']:+.3f}",f"{d['MAE']:+.3f} [{d['CI95_MAE'][0]:+.3f}; {d['CI95_MAE'][1]:+.3f}]"])
 table(lines,['Đầu vào','Phương pháp','Đọc phổ','Video/người','MAE','Bias','Delta so hiệu chỉnh [CI95%]'],rows)
 lines+=['Mốc toàn bộ 218 video/158 người: hằng số 12,07068; hiệu chỉnh đã lưu 11,37385; padding-fixed đã lưu 17,29208 bpm. Các mốc không được dùng thay cho mốc ghép cặp khi coverage khác.','Cổng thành công không đổi: native ≤10 bpm, hơn hiệu chỉnh cùng cohort ≥1 bpm và CI delta loại 0; thêm kiểm soát âm. Không tự chọn hàng thấp nhất trong nhiều thử nghiệm để tuyên bố xác nhận.']
 # Show spectra for fixed representative windows, actual units retained.
 lines+=['','## 3. Đầu ra sai phân, tích phân và đọc phổ','','Ba đường là phổ sai phân d̂, phổ cumsum(d̂), và phổ sau detrend + bandpass. Đồ thị log giúp nhìn nền phổ; các đường không được chuẩn hóa riêng về cùng chiều cao. Độ nổi lấy công suất chia nền làm trơn; bù nghiêng nhân 4sin²(πf/fs). Cả hai vẫn đọc trên đại lượng đã tích phân.']
 jobs=manifest();chosen=[]
 for ds in ['UBFC2','MCD']:
  chosen.append(next(j for j in jobs if j['dataset']==ds))
 chosen += [j for j in jobs if j['recording'] in ['S_AB51_Resting','S_AB59_Resting']]
 for j in chosen:
  root=P/'original'/j['key']/'native';a=np.load(root/'waveforms.npz');ix=next(i for i,k in enumerate(a['keys']) if k[0].startswith('padding'));d=a['difference'][ix];f,pd,pc,pf=spectra(d,.6);fig,ax=plt.subplots(figsize=(8,4))
  for values,label in [(pd,'d_hat'),(pc,'cumsum'),(pf,'detrend + bandpass')]:ax.semilogy(f,np.maximum(values,1e-30),label=label)
  start=float(a['keys'][ix][1]);truth=next(w['truth'] for w in j['windows'] if w['start']==start)
  if truth is not None:ax.axvline(truth/60,color='black',ls='--',label='reference PR/HR')
  ax.set(xlim=(.3,5),xlabel='Frequency (Hz); bpm = Hz x 60',ylabel='PSD (model output units)',title=j['recording']);ax.legend();fig.tight_layout();name=j['key']+'_spectra.png';fig.savefig(figdir/name,dpi=150);plt.close(fig);lines.append(f'![Ba phổ {j["recording"]}](figures/{name})')
 probe=read(P/'READOUT_PROBE.json')['rows'];noise=[r for r in probe if r['kind']=='white_noise_difference'];table(lines,['Decoder','Trung vị dự đoán trên 100 nhiễu không mạch'],[[k,f"{np.median([r[k] for r in noise]):.3f} bpm"] for k in ['power_0.7','power_0.6','prominence_0.6','tilt_0.6']]);lines+=['Nhiễu không có nhịp thật: mọi giá trị bpm trên bảng này là kết quả bị ép chọn đỉnh. Dịch phân bố nhiễu lên cao không đồng nghĩa MAE thật tốt hơn. Hạ cận xuống 0,6 có thể tăng xu hướng chọn nhịp thấp.','Reference công khai 0,7 Hz được giữ làm cột chính trước/sau. Cột reference 0,6 Hz được báo riêng vì 71 cửa sổ đổi giá trị tham chiếu. Không tính thay đổi nhãn như cải thiện mô hình. CHROM 0,6 đã sửa cả bộ lọc trong thuật toán, không chỉ decoder.']
 # Raw gradients, signed occlusion, randomized layers and area normalization.
 lines+=['','## 4. Mô hình nhạy ở đâu — và bản đồ có phụ thuộc trọng số không?','','BTC được chọn lại theo lỗi padding-fixed, không theo hiệu chỉnh. Gradient là đạo hàm công suất tại tần số cố định của mô hình ban đầu, trên cumsum + linear detrend + Hann; đây là surrogate độ nhạy, không phải đạo hàm của argmax production. Giữ cùng functional giữa mô hình học và các lần ngẫu nhiên hóa. Occlusion dương nghĩa là che làm giảm công suất mục tiêu; âm nghĩa là che làm tăng công suất đó.','Các panel gradient dùng cùng thang trong một video, không chia từng ảnh cho max riêng. Mass/area >1 nghĩa là vùng nhận nhiều độ nhạy hơn tỉ lệ diện tích; không chứng minh vùng đó chứa mạch thật.']
 visualrows=[];regionalrows=[];selection=read(P/'VISUAL_SELECTION.json')
 for r in selection:
  root=P/'visual'/r['visual_key'];receipt=read(root/'COMPLETE.json');a=np.load(root/'MAPS.npz');maps=a['gradient_raw'];occ=a['occlusion_signed'];fig,axes=plt.subplots(1,4,figsize=(13,3));axes[0].imshow(np.clip(a['mean_rgb'],0,255).astype(np.uint8));axes[0].set_title(r['recording']);vmax=max(float(maps[0].max()),float(maps[-1].max()),1e-30)
  axes[1].imshow(maps[0],vmin=0,vmax=vmax,cmap='magma');axes[1].set_title('Raw |gradient|');limit=max(float(abs(occ).max()),1e-30);im=axes[2].imshow(occ,cmap='coolwarm',vmin=-limit,vmax=limit);axes[2].set_title('Signed occlusion');fig.colorbar(im,ax=axes[2],shrink=.7);axes[3].imshow(maps[-1],vmin=0,vmax=vmax,cmap='magma');axes[3].set_title('Randomized all stages')
  for ax in axes:ax.axis('off')
  fig.suptitle(f"Fixed frequency {receipt['fixed_target_hz']*60:.2f} bpm; gradient common max {vmax:.3g}");fig.tight_layout();name=r['visual_key']+'_sensitivity.png';fig.savefig(figdir/name,dpi=130);plt.close(fig)
  for c in receipt['randomization']:visualrows.append(dict(group=r['group'],recording=r['recording'],**c))
  for method,v in receipt['regional'].items():
   for i,name in enumerate(['estimated_skin','oval_edge','eyes_mouth','outside_oval']):regionalrows.append(dict(group=r['group'],recording=r['recording'],method=method,region=name,area=v['area'][i],mass=v['mass'][i],mass_per_area=v['mass_per_area'][i]))
 dumpcsv(dest/'REGION_MASS_AREA.csv',regionalrows)
 dumpcsv(dest/'RANDOMIZATION.csv',visualrows);rows=[]
 for group in sorted({r['group'] for r in visualrows}):
  for stage in [1,6]:
   rr=[r for r in visualrows if r['group']==group and r['stage']==stage];rho=[r['spearman'] for r in rr if r['spearman'] is not None];rows.append([group,stage,len(rr),f'{np.median(rho):.3f}' if rho else 'không xác định',f"{np.median([r['mass_ratio'] for r in rr]):.3g}"])
 table(lines,['Nhóm','Số bước randomize tích lũy','Video','Tương quan bản đồ trung vị','Tỉ lệ tổng gradient trung vị'],rows);lines+=['Tương quan còn cao sau khi phá trọng số là dấu hiệu phải dè dặt khi diễn giải bản đồ. Tương quan giảm cho thấy bản đồ phụ thuộc trọng số, nhưng không chứng minh mô hình dùng sinh lý đúng. Tất cả 100 panel nằm trong `figures/`. POS trong pulse-map nhận RGB trước z-score; audit không tìm thấy lỗi ở bước này.','Mass, diện tích và mass/area của từng vùng, từng video nằm trong REGION_MASS_AREA.csv. Vùng da ước lượng bằng landmark không phân biệt được ống thở và da; không dùng số này để bác bỏ một vật che nhìn thấy trên ảnh.']
 for group in sorted({r['group'] for r in selection}):
  r=next(r for r in selection if r['group']==group);lines.append(f"![{group}](figures/{r['visual_key']}_sensitivity.png)")
 # Four-condition codec evidence, consistent .6 band, explicit QP limitations.
 lines+=['','## 5. Codec bốn điều kiện','','Cùng source-frame index và PTS; FFV1 phải khôi phục RGB bit-exact. Native giữ cadence nguồn; ba nhánh còn lại giữ cùng các khung đã chọn về khoảng 10 fps. Đây là ROI hình chữ nhật cố định theo bbox native để cô lập encode, không phải trace oval dùng cho bảng MAE chính. QP là tham số lượng tử của encoder, không có cùng ý nghĩa ở lossless FFV1. HEVC/MPEG4 không được bịa QP khi parser H.264 không hỗ trợ.']
 allwindows=[];qprows=[]
 for j in read(P/'CODEC_SELECTION.json'):
  root=P/'codec_four'/j['key'];cp=read(root/'COMPLETE.json');assert cp['FFV1_RGB_bit_exact'];allwindows+=read(root/'WINDOWS_BAND_V2.json')
  for q in cp['QP']:qprows.append(dict(recording=j['recording'],dataset=j['dataset'],**q))
 dumpcsv(dest/'CODEC_WINDOWS.csv',allwindows);dumpcsv(dest/'CODEC_QP.csv',qprows);rows=[]
 for condition in ['native','lossless10','default','bf0']:
  for method in ['POS','CHROM','G']:
   rr=[r for r in allwindows if r['condition']==condition and r['method']==method and r['prediction'] is not None];rows.append([condition,method,len(rr),sum(r['near150'] for r in rr),f"{100*np.mean([r['near150'] for r in rr]):.2f}%" if rr else 'thiếu'])
 table(lines,['Điều kiện','Phương pháp','Cửa sổ hợp lệ','Trong 148–152 bpm','Tỉ lệ'],rows);lines+=['Thay cấu hình encoder còn thay nhiều yếu tố. Dù cụm 150 bpm thay đổi, không gọi đó là chứng minh riêng B-frame gây lỗi. C5′ vẫn là mô phỏng ffmpeg, chưa xác minh tương đương MATLAB VideoWriter.']
 # Integrity and disposition, do not silently certify a failed negative control.
 audit=read(P/'BASELINE_AUDIT.json');lines+=['','## 6. Kiểm tra tính so sánh và phần còn cần quyết định',f"Đối chiếu trước/sau: {audit['videos']} video/{audit['people']} người; mọi baseline sai khác ≤1e-8: {audit['all_within_1e8']}."]
 table(lines,['Baseline','Video đối chiếu','Sai khác lớn nhất','Video lệch >1e-8'],[[a['method'],a['compared'],f"{a['max_absolute_difference']:.9g}",a['changed_over_1e8']] for a in audit['comparisons']])
 static=Counter()
 for tree in ['original','spatial']:
  for path in (P/tree).glob('**/STATIC.json'):
   for r in read(path):static[('padding' if r['model'].startswith('padding') else 'V3',r['passed'])]+=1
 table(lines,['Loại checkpoint','Số phép thử PASS','Số phép thử FAIL'],[[m,static[m,True],static[m,False]] for m in ['padding','V3']]);lines+=['Các con số trên là phép thử checkpoint × video × điều kiện × đầu vào thực sự chạy, không phải số checkpoint độc lập; các nhánh alias không được đếm lại. V3 còn lỗi ảnh tĩnh cũ, không âm thầm sửa TSM. Kiểm soát âm toàn bộ bộ ước lượng hiệu chỉnh sau fit phải được xem riêng, chưa được thay bằng PASS của padding.','FREEZE cũ và checkpoint giữ nguyên cho đến khi người thực hiện đọc bảng đầy đủ, kiểm hình và xử lý mọi sai khác baseline. File này là bản dựng từ bằng chứng, không phải biên bản tự động thông qua thay ứng viên.','Tái lập: code và hash trong IMPLEMENTATION_BINDING*.json; cohort PROTOCOL.md/MANIFEST.json/FROZEN.json; CLASSICAL_BAND_V2 audit; report_spatial gồm dự đoán từng cửa sổ, từng video, hệ số calibration theo fold và CI. Không mở holdout.']
 (dest/'ANALYSIS_DRAFT.md').write_text('\n'.join(lines),encoding='utf8');save(P/'RENDER_COMPLETE.json',dict(utc=now(),images=len(list(figdir.glob('*.png'))),report_sha256=sha(dest/'ANALYSIS_DRAFT.md'),human_visual_review_pending=True,FREEZE_unchanged=True))

if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'RENDER_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
