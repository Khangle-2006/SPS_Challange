"""Read-only final E0 branch decision under the previously frozen50% rule."""
import argparse,collections,csv,hashlib,json
from pathlib import Path
import numpy as np


def main(root):
    r=Path(root);s=json.loads((r/'final_summary.json').read_text());records=json.loads((r/'final_results.json').read_text())
    if not s['complete'] or s['processed']!=119:raise ValueError('Full cohort required')
    arms=['P0R0','P0R1','P1R0','P1R1'];by=collections.defaultdict(list);groups=collections.defaultdict(list)
    for rec in records:
        for w in rec.get('windows',[]):
            if w['reference_hr'] is not None and all(w['predictions'][a] is not None for a in arms):
                e=np.array([abs(w['predictions'][a]-w['reference_hr']) for a in arms]);by[rec['subject']].append(e)
                groups[(rec['subject'],rec['step'])].append(e)
    x=np.array([np.mean(by[k],axis=0) for k in sorted(by)])
    rng=np.random.default_rng(20261030);boots=x[rng.integers(0,len(x),size=(1000,len(x)))].mean(axis=1)
    gaps=boots[:,3]-boots[:,0];ratios=(boots[:,3]-boots[:,2])[gaps>0]/gaps[gaps>0]
    a,b,d=[s['matched'][k]['subject_macro_MAE'] for k in ['P0R0','P1R0','P1R1']]
    recovery=(d-b)/(d-a) if d>a else None;decision='ROI/E3' if recovery is not None and recovery>=.5 else 'QA-lite/T6'
    result={'complete':True,'records':119,'paired_windows':s['matched_windows'],'subjects':len(by),
            'recovered_fraction':recovery,'bootstrap_fraction_ci95_positive_denominator':np.quantile(ratios,[.025,.975]).tolist() if len(ratios) else None,
            'bootstrap_positive_denominators':int(sum(gaps>0)),'bootstrap_replicates':1000,'decision':decision,
            'training_started':False,'rule_sha256':hashlib.sha256((r/'BRANCH_RULE_20261030.json').read_bytes()).hexdigest(),
            'scope':'Original frozen E0 diagnostics, not new Round1 self-score or proof of codec mechanism'}
    (r/'FINAL_DECISION.json').write_text(json.dumps(result,indent=2))
    with open(r/'final_subject_condition.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['subject','condition','paired_windows']+[a+'_window_mae' for a in arms]+[a+'_gt20' for a in arms])
        for (sid,step),values in sorted(groups.items()):
            v=np.asarray(values);w.writerow([sid,step,len(v),*v.mean(axis=0),*(v>20).sum(axis=0)])
    lines=['# E0 FullHD — kết quả cuối, tổng hợp31/10/2026','',
      '**119/119 video,60người, không có lỗi xử lý. Nhánh ưu tiên: '+decision+'. Chưa chạy training.**','',
      'Đây là E0 theo protocol gốc, không phải bảng tự chấm Round1 mới. Band E0 đã là0,7–3Hz; khác T1 ở mốc cửa sổ và support tham chiếu. Không trộn số cửa sổ hoặc MAE của hai protocol.',
      '', '| Pixel | ROI | Subject-macro MAE | Window MAE | >20bpm /1860 |', '|---|---|---:|---:|---:|']
    for arm,pix,roi in [('P0R0','native','native'),('P0R1','native','H264'),('P1R0','H264','native'),('P1R1','H264','H264')]:
        m=s['matched'][arm];lines.append(f"| {pix} | {roi} | {m['subject_macro_MAE']:.3f} | {m['window_MAE']:.3f} | {m['errors_over20']} |")
    lines+=['',f"Gap diagonal={d-a:.3f}bpm. Đổi ROI nén về native thu hồi{d-b:.3f}bpm, tương đương{100*recovery:.2f}%. Bootstrap theo60người,1000lần seed20261030: CI95% tỷ lệ thu hồi [{100*np.quantile(ratios,.025):.2f}%,{100*np.quantile(ratios,.975):.2f}%]; mẫu bootstrap có denominator dương1000/1000. Không đạt ngưỡng thu hồi50% đã chốt.",
      '', 'Pixel effect tại nativeROI: +3,102bpm, CI95%[2,102;4,147]. ROI effect trên pixel nén: −0,137bpm, CI95%[−0,518;0,232]. Tác động pixel rõ trong thí nghiệm này; ROI effect nhỏ và khoảng bất định qua0. Không suy ra riêng cơ chế sample-and-hold/IIR hoặc bảo đảm QA-lite hiệu quả.',
      '', 'Coverage:2023ứng viên,1865tham chiếu hợp lệ,1860ghép đủ4ô. Mỗi ô có1961prediction hữu hạn trước giao với reference. Bỏ158cửa sổ thiếu support reference và5reference-valid thiếu prediction. Paired coverage=91,94% trên toànứng viên,99,73% trênreference-valid. Before:940/1020; after:920/1003 ghép đủ4ô. Không bỏ bản ghi khỏi inventory.',
      '', 'Toàn bộ video là FullHDwebcam thuộcdev60 đã dùng phát triển; không phải test chưa chạm. Native vốn MPEG4, H264500k là nén thêm ở FPS nguồn, không phảiC5@10fps. Đổi ROI là oracle chẩn đoán. Giữ nhãn và offset. Không mở audit dataset mới.',
      '', 'Artifacts: final_summary.json,final_windows.csv,final_records.csv,final_results.json,final_subject_condition.csv,FINAL_DECISION.json. Đã xem hai biểu đồ bằng mắt; trục/chú thích và khoảng bootstrap đọc được.',
      '', '![Four cells](final_e0_cells.png)', '', '![Paired effects](final_e0_effects.png)']
    (r/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (r/'FINAL_REPORT_SCRIPT_SHA256.txt').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+'  '+str(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);main(p.parse_args().report)
