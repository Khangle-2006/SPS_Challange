"""Read-only report from frozen codec pilot outputs. No model fitting."""
from audit_clock import *
from collections import defaultdict,Counter
from report_metrics import metric

def writecsv(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def nfmt(x):return '—' if x is None else f'{x:.3f}'
def pct(n,d):return f'{n}/{d} ({100*n/d:.1f}%)' if d else '0/0'
def loadrows(kind):
 out=[];enc=[]
 paths=(P/kind).glob('*/*/f*/RESULT.json') if kind=='real' else (P/kind).glob('*/*/RESULT.json')
 for p in paths:
  a=read(p);common={k:a[k] for k in ['dataset','recording','person','fs','bf','scene','seed','pulse_bpm','wholefile_kbps','bitrate_pass','cycle_bpm','dominance','face_valid_fraction','max_timing_error_s'] if k in a}
  if a['bf'] is not None:enc.append(common|dict(path=str(p),schedule=json.dumps(a['reference_spacing_counts'])))
  with np.load(p.parent/'spectra.npz') as z:
   for r in a['rows']:
    rr=common|r;rr['path']=str(p);rr['cycle_fraction']=None
    if a.get('cycle_bpm') and r.get('band_power',0)>0:
     key=f"{r['roi']}_{r['window']}_{r['clock']}_{r['method']}";f=z[key+'_f'];power=z[key+'_p'];rr['cycle_fraction']=float(power[abs(f-a['cycle_bpm']/60)<=.1].sum()/power[(f>=.7)&(f<=3)].sum())
    out.append(rr)
 return out,enc
def table_movement(rows,groupkeys):
 grouped=defaultdict(list)
 for r in rows:
  if r.get('bf') is not None and r['roi'] in ['oval36','skin'] and r.get('peak_bpm') is not None:grouped[tuple(r[k] for k in groupkeys)].append(r)
 lines=[]
 for key,rr in sorted(grouped.items(),key=lambda z:str(z[0])):
  at150=sum(abs(r['peak_bpm']-150)<=2 for r in rr);eligible=[r for r in rr if r.get('cycle_bpm') and .7<=r['cycle_bpm']/60<=3];near=sum(abs(r['peak_bpm']-r['cycle_bpm'])<=2 for r in eligible)
  lines.append(dict(zip(groupkeys,key))|dict(windows=len(rr),median_peak=float(np.median([r['peak_bpm'] for r in rr])),near150=at150,eligible_cycle=len(eligible),near_cycle=near,median_cycle_fraction=float(np.median([r['cycle_fraction'] for r in eligible])) if eligible else None))
 return lines
def paired_moves(rows):
 lookup={}
 for r in rows:
  if r['bf'] in [3,4] and r['roi']=='oval36' and r['clock']=='source_clock':lookup[(r['dataset'],r['recording'],r['fs'],r['method'],r['window'],r['bf'])]=r
 out=[]
 for key,a in lookup.items():
  if key[-1]!=3:continue
  b=lookup.get(key[:-1]+(4,))
  if not b or not a.get('cycle_bpm') or not b.get('cycle_bpm') or a.get('peak_bpm') is None or b.get('peak_bpm') is None:continue
  out.append(dict(dataset=key[0],recording=key[1],fs=key[2],method=key[3],window=key[4],peak_b3=a['peak_bpm'],peak_b4=b['peak_bpm'],cycle_b3=a['cycle_bpm'],cycle_b4=b['cycle_bpm'],follows_both=abs(a['peak_bpm']-a['cycle_bpm'])<=2 and abs(b['peak_bpm']-b['cycle_bpm'])<=2,delta_peak=a['peak_bpm']-b['peak_bpm'],delta_cycle=a['cycle_bpm']-b['cycle_bpm']))
 return out
def accuracy(rows,plan):
 dev=read(V/'btc_zeroshot/dev_manifest.json')['videos'];by={v['recording']:v for v in dev};pubref={};selected={j['recording']:j for j in plan['videos']}
 cal={r['recording']:float(r['prediction']) for r in csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open()) if r['condition']=='native' and r['method']=='calibrated_median_POS_CHROM_V3'}
 for j in plan['videos']:
  if j['dataset']=='BTC':continue
  with np.load(j['meta']) as z:meta=adapt_cached_meta(z,j['dataset'],source_convention='raw_device')[0];origin=float(z['original_t'][0])
  for wi in range(3):
   start=origin+wi*10;t=meta['ppg_t'];a=max(0,np.searchsorted(t,start,side='right')-1);b=min(len(t),np.searchsorted(t,start+10,side='left')+1);vals=meta['ppg'][a:b].astype(float).copy();vals[~meta['ppg_valid'][a:b].astype(bool)]=np.nan
   try:truth=reference_hr(dict(window_start_s=start,ppg_timestamps_json=json.dumps(t[a:b].tolist()),ppg_values_json=json.dumps(vals.tolist())));reason=None
   except ValueError as e:truth=None;reason=str(e)
   pubref[j['recording'],wi]=(truth,reason)
 detailed=[];grouped=defaultdict(list)
 for r in rows:
  if r['roi']!='oval36' or r['clock']!='source_clock':continue
  j=selected[r['recording']];truth=by[r['recording']]['pr_bpm'] if r['dataset']=='BTC' else pubref[r['recording'],r['window']][0]
  rr=r|dict(truth=truth,error=r['prediction']-truth if r['prediction'] is not None and truth is not None else None,person=j['person']);detailed.append(rr)
  if rr['error'] is not None:grouped[(r['dataset'],r['recording'],r['fs'],r['bf'],r['method'])].append(rr)
 video=[]
 for key,ww in grouped.items():
  ds,rid,fs,bf,method=key;j=selected[rid];prediction=float(np.median([r['prediction'] for r in ww]));truth=float(np.median([r['truth'] for r in ww]))
  video.append(dict(dataset=ds,recording=rid,person=j['person'],fs=fs,bf=bf,method=method,prediction=prediction,truth=truth,windows=len(ww),group=j.get('group','public'),task=by[rid]['task'] if ds=='BTC' else 'public',abs_error=abs(prediction-truth)))
 btc=[];summary=[]
 for fs in [8,10]:
  for bf in [None,0,3,4]:
   rr=[x for x in video if x['dataset']=='BTC' and x['fs']==fs and x['bf']==bf and x['recording'] in cal]
   for rid in sorted({x['recording'] for x in rr}):
    v=by[rid];r=next(x for x in rr if x['recording']==rid);c=float(np.median([x['pr_bpm'] for x in dev if x['fold']!=v['fold']]))
    rr+=[dict(r,method='constant',prediction=c,abs_error=abs(c-r['truth'])),dict(r,method='calibrated',prediction=cal[rid],abs_error=abs(cal[rid]-r['truth']))]
   btc.extend(rr)
   for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
    keep=[r for r in rr if group=='all' or selected[r['recording']]['group']==group]
    for m in metric(keep,['constant','calibrated','G','POS','CHROM']):summary.append(dict(fs=fs,bf=bf,group=group,**m))
 writecsv(P/'WINDOW_PREDICTIONS.csv',detailed);writecsv(P/'VIDEO_PREDICTIONS.csv',video);writecsv(P/'BTC_BASELINES_MATCHED.csv',btc);save(P/'BTC_METRICS.json',summary)
 return summary,video
def main():
 plan=read(P/'PLAN.json');real,enc=loadrows('real');synthetic,senc=loadrows('synthetic');audit=read(P/'READOUT_AUDIT.json')
 complete=(P/'REAL_COMPLETE.json').exists() and (P/'SYNTH_COMPLETE.json').exists();state='COMPLETE_WITH_EXCLUSION' if complete else 'PARTIAL'
 writecsv(P/'ALL_REAL_SPECTRA.csv',real);writecsv(P/'ALL_SYNTHETIC_SPECTRA.csv',synthetic);writecsv(P/'ENCODER_SCHEDULES.csv',enc+senc)
 moves=paired_moves(real);writecsv(P/'PAIRED_CLOCK_MOVEMENT.csv',moves)
 movement=table_movement(real,['dataset','fs','bf','roi','clock','method']);save(P/'MOVEMENT_SUMMARY.json',movement)
 btc,video=accuracy(real,plan)
 lines=[f'# Kiểm chứng đồng hồ codec — {state}', '', f'Cập nhật {now()}. {len(enc)}/72 encode video thật; {len(senc)}/120 encode tổng hợp. Không train, không mở holdout. Không thay ứng viên hay cổng.', '',
 'MCD 1474_FullHDwebcam_after không đủ điều kiện: khoảng mất timestamp0,337102s làm lấy mẫu8/10fps lặp một frame. Giữ nguyên mẫu khóa, không thay video; 6 encode dự kiến của nguồn này không chạy. Hoàn tất có nghĩa đã xử lý mọi nguồn, không phải72/72 encode thành công.', '', '## 1. Kiểm tra đường tính HR', '', 'Các đường đánh giá được kiểm tra đã tích phân sai phân bằng cumsum trước detrend/lọc/argmax, dải 0,7–3Hz. Không tìm thấy lỗi thiếu tích phân trong các đường này. Không suy rộng thành mọi lịch sử runtime đều đã được kiểm.', '',
 '| Đầu vào decoder | SD trước chuẩn hóa | HR trả về |','|---|---:|---:|']
 for name in ['zero','constant','difference','tiny_sine','near_threshold_sine']:
  a=audit['synthetic_controls'][name];lines.append(f"| {name} | {a['std']:.3g} | {nfmt(a['hr'])} |")
 lines+=['', 'Ảnh tĩnh SiNC epoch8 có SD 2,425e-9, lớn hơn ngưỡng 1e-10 nên decoder vẫn trả 61,52 bpm. Đây là bằng chứng kiểm tra ảnh tĩnh không đạt theo luật cũ, nhưng riêng con số này chưa chứng minh dao động giả biên độ lớn. Không đổi ngưỡng hậu nghiệm.', '', '## 2. Danh sách khóa trước và điều kiện', '', '| Dataset | Video | Nhóm |','|---|---|---|']
 for j in plan['videos']:lines.append(f"| {j['dataset']} | {j['recording']} | {j.get('group','public dev')} |")
 lines+=['', 'Mỗi video lấy 30 giây đầu + frame cuối hỗ trợ. Chọn frame gần nhất theo thời gian thật vào lưới 8/10fps, không đổi tốc độ phát. ROI lấy từ nguồn rồi giữ cố định giữa B0/B3/B4. Đây là thí nghiệm can thiệp có kiểm soát, không phải bản chạy lại chính xác recipe C5 trước đó. H264 ffmpeg chưa xác minh tương đương MATLAB.', '',
 f"Bitrate thực (bảng cơ chế giữ cả bản vượt500, không gọi bản đó là C5 hợp lệ): {min([x['wholefile_kbps'] for x in enc],default=0):.2f}–{max([x['wholefile_kbps'] for x in enc],default=0):.2f} kbps; {sum(not x['bitrate_pass'] for x in enc)}/{len(enc)} bản thật vượt500kbps được giữ và đánh dấu, không retune. Không coi cùng CRF là cùng bitrate.", '',
 '## 3. Đỉnh phổ và chu kỳ frame — video thật', '', 'Bảng chính dùng oval36, clock nguồn và phép chiếu trên lưới30Hz như evaluator cũ. Phổ trước nội suy G/POS nằm trong CSV; CHROM nguyên bản không hỗ trợ trực tiếp8/10Hz (padlen21 > cửa sổ1,6s), được ghi lỗi thay vì sửa công thức. Cycle chỉ được gán khi >=80% khoảng cách frame không-B cùng độ dài và chu kỳ >1 frame. Near-cycle: đỉnh cách chu kỳ quan sát <=2bpm. Đây là thống kê cơ chế, không phải MAE hay tỉ lệ đo mạch đúng.', '',
 '| Dataset | fps | B | Phép chiếu | n cửa sổ | Đỉnh trung vị bpm | Gần chu kỳ thực | Gần150 |','|---|---:|---:|---|---:|---:|---|---|']
 for a in movement:
  if a['roi']=='oval36' and a['clock']=='source_clock':lines.append(f"| {a['dataset']} | {a['fs']} | {a['bf']} | {a['method']} | {a['windows']} | {a['median_peak']:.2f} | {pct(a['near_cycle'],a['eligible_cycle'])} | {pct(a['near150'],a['windows'])} |")
 lines+=['', '### Ghép cặp B3 → B4, cùng video/fps/cửa sổ', '', '| Dataset | Phép chiếu | Đỉnh bám cả hai chu kỳ |','|---|---|---|']
 for ds in ['UBFC2','MCD','BTC']:
  for method in ['G','POS','CHROM']:
   rr=[r for r in moves if r['dataset']==ds and r['method']==method];lines.append(f"| {ds} | {method} | {pct(sum(r['follows_both'] for r in rr),len(rr))} |")
 lines+=['', 'Bảng đầy đủ ALL_REAL_SPECTRA.csv có cả mean trên pixel da thật, oval36, clock nguồn, clock mẫu và phổ native-rate. Các cửa sổ/ROI lỗi giữ nguyên; không thay video vì kết quả xấu.', '', '## 4. Đối chứng tổng hợp', '', 'Hai cảnh × hai seed, nhiễu/chuyển động/ánh sáng giữ nguyên giữa các pulse. Không có pulse không có nghĩa mọi argmax phải rỗng: thuật toán cổ điển luôn có thể chọn một đỉnh nhiễu. Vì vậy báo riêng tần số/biên độ và dương tính giả theo chu kỳ, không tự gọi nó là phát hiện sinh lý.', '', '| Pulse thật | fps | B | Phương pháp | n | Sai số đỉnh trước encode | Sai số sau encode | Gần chu kỳ codec sau encode |','|---:|---:|---:|---|---:|---:|---:|---|']
 before={(r['scene'],r['seed'],r['pulse_bpm'],r['fs'],r['window'],r['method']):r for r in synthetic if r['bf'] is None and r['clock']=='source_clock'}
 for bpm in [0,72,120,150,180]:
  for fs in [8,10]:
   for bf in [0,3,4]:
    for method in ['G','POS','CHROM']:
     rr=[r for r in synthetic if r['pulse_bpm']==bpm and r['fs']==fs and r['bf']==bf and r['method']==method and r['clock']=='source_clock' and r.get('peak_bpm') is not None]
     if not rr:continue
     ee=[abs(r['peak_bpm']-bpm) for r in rr];bb=[abs(before[r['scene'],r['seed'],bpm,fs,r['window'],method]['peak_bpm']-bpm) for r in rr];eligible=[r for r in rr if r.get('cycle_bpm')];near=sum(abs(r['peak_bpm']-r['cycle_bpm'])<=2 for r in eligible)
     lines.append(f"| {bpm or 'Không mạch'} | {fs} | {bf} | {method} | {len(rr)} | {np.mean(bb):.2f}"+f" | {np.mean(ee):.2f}"+f" | {pct(near,len(eligible))} |" if bpm else f"| Không mạch | {fs} | {bf} | {method} | {len(rr)} | — | — | {pct(near,len(eligible))} |")
 lines+=['', 'Nếu pulse đã mất trước encode thì không quy lỗi đó cho encoder. Nếu pulse trùng codec thì MAE thấp không chứng minh thuật toán theo đúng nguồn sinh lý. Không thử notch hay huấn luyện bộ khử nhiễu trong đợt này.', '', '## 5. BTC — bảng MAE thăm dò, cùng video và hai baseline', '', 'Chỉ4 người khóa trước, mỗi nhóm1 người; không đại diện218video/158người. Các cột là MAE theo người, bootstrap ghép cặp10000 seed0. Bản hiệu chỉnh là dự đoán OOF native đã lưu của chính video đó, không phải hiệu chỉnh được fit mới cho codec này. Bảng dùng giao video G/POS/CHROM/constant/calibrated trong từng điều kiện; không so cohort khác như nhân quả. Chi tiết nhóm và coverage trong BTC_METRICS.json và CSV.', '', '| fps | B | Nhóm | Phương pháp | Người/video | MAE | CI95 | Δ hiệu chỉnh [CI95] |','|---:|---|---|---|---:|---:|---|---|']
 for x in btc:
  lines.append(f"| {x['fs']} | {'Trước encode' if x['bf'] is None else x['bf']} | {x['group']} | {x['method']} | {x['people']}/{x['videos']} | {x['MAE']:.2f} | {str([round(v,2) for v in x['CI95']])} | {x['delta_calibrated']:+.2f} {str([round(v,2) for v in x['delta_calibrated_CI']])} |")
 lines+=['', '## 6. Giới hạn và quyết định', '', 'Đây là pilot chẩn đoán. Giữ ứng viên hiệu chỉnh11,37 và holdout đóng. Chưa có bộ tách nhiễu hay bằng chứng cải thiện BTC native. Kết quả nguồn tổng hợp cần lặp trên video thật; không chọn kết luận từ một cảnh/seed. Cấu trúc B cố định có thể khác lịch encoder thích ứng trong dữ liệu cũ. Source đã có nén; đối chứng mới chỉ thao tác lần encode bổ sung.', '', '## Tái lập', '', '`audit_clock.py prepare` khóa mẫu và hash; `audit_clock.py audit` kiểm readout; `launch.py` là lần đầu đã dừng; `run_v2.py` ghi nhận CHROM native-rate không hỗ trợ; `run_v3.py` khôi phục mapping MCD cache đã có; `run_v4.py` tiếp tục sau khi ghi loại nguồn có khoảng mất timestamp. Các AMENDMENT giữ hash và lý do, không chạy lại nguồn đã hoàn tất; `report_clock.py` chỉ đọc kết quả. Không chạy lại prepare/launch trên thư mục đã có bằng chứng. Lệnh từng encode, bitstream SHA, lịch I/P/B, PSD và trace nằm cạnh RESULT.json.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 save(P/'REPORT_STATUS.json',dict(utc=now(),state=state,real_encodes=len(enc),planned_real_encodes=72,expected_real_after_exclusion=66,excluded_source_count=1,synthetic_encodes=len(senc),real_windows=len(real),synthetic_windows=len(synthetic),report_sha=sha(P/'REPORT.md'),holdout_evaluations=0))
 print(json.dumps(read(P/'REPORT_STATUS.json')))
if __name__=='__main__':main()
