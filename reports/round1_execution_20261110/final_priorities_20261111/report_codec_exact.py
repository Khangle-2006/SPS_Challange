"""Report exact-team-recipe controls; no new encodes or fitting."""
import os,sys
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'codec_clock_20261111'))
import report_clock as r
r.P=P/'codec_exact'
def main():
 assert (r.P/'REAL_COMPLETE.json').exists()
 plan=r.read(r.P/'PLAN_V2.json');rows,enc=r.loadrows('real');assert len(enc)==66
 verified=[]
 for a in enc:
  file=Path(a['path']);result=r.read(file);assert r.sha(file.parent/'video.mp4')==result['encoded_sha']
  verified.append(dict(path=str(file),sha256=result['encoded_sha']))
 r.writecsv(r.P/'ALL_REAL_SPECTRA.csv',rows);r.writecsv(r.P/'ENCODER_SCHEDULES.csv',enc)
 movement=r.table_movement(rows,['dataset','fs','bf','roi','clock','method']);r.save(r.P/'MOVEMENT_SUMMARY.json',movement)
 moves=r.paired_moves(rows);r.writecsv(r.P/'PAIRED_CLOCK_MOVEMENT.csv',moves)
 btc,_=r.accuracy(rows,plan)
 lines=['# Codec: đúng recipe C5 gốc của đội','', 'Hoàn tất 66 encode trên 11 nguồn / 12 ID khóa trước. Nguồn MCD 1474 có gap timestamp đã bị loại từ trước, không thay nguồn. Đây là đối chứng encoder trên đoạn 30 giây, không phải kết quả toàn bộ BTC.',
 '', 'Rate control/GOP/preset giữ đúng simulate.py của T8, thay FPS 8/10 và bf 0/3/4. B-adapt mặc định vẫn hoạt động: bf là số B tối đa, không bảo đảm chu kỳ bằng bf+1. Do đó bảng dùng chu kỳ đo từ bitstream; chỉ gọi một chu kỳ nổi trội khi ≥80% khoảng cách frame tham chiếu cùng độ dài.',
 '', 'C5 ffmpeg chưa được xác minh tương đương MATLAB VideoWriter. Không notch hoặc sửa nhãn. Roi lấy từ native và giữ cố định giữa các can thiệp encoder; UBFC dùng landmark theo ordinal decoded frame gốc, trước bước bỏ timestamp trùng.',
 '', f"Bitrate đo toàn file: {min(a['wholefile_kbps'] for a in enc):.2f}–{max(a['wholefile_kbps'] for a in enc):.2f} kbps; {sum(not a['bitrate_pass'] for a in enc)}/66 vượt 500 được giữ và đánh dấu, không retune. Bảng cơ chế có thể bao gồm các bản không đạt bitrate; không gọi chúng là C5 hợp lệ.",
 '', '| Dataset | FPS | bf tối đa | Phép chiếu | Cửa sổ | Đỉnh trung vị bpm | Gần chu kỳ thực / đủ điều kiện | Gần150 |','|---|---:|---:|---|---:|---:|---:|---:|']
 for a in movement:
  if a['roi']=='oval36' and a['clock']=='source_clock':lines.append(f"| {a['dataset']} | {a['fs']} | {a['bf']} | {a['method']} | {a['windows']} | {a['median_peak']:.2f} | {a['near_cycle']}/{a['eligible_cycle']} | {a['near150']}/{a['windows']} |")
 lines += ['', 'Gần chu kỳ và gần150 đều là trong ±2 bpm, không phải số cửa sổ đo mạch đúng. Bảng đầy đủ có skin/oval36, clock nguồn/clock mux và phổ trước nội suy. Native-rate CHROM không đủ hỗ trợ padlen được giữ lỗi, không đổi thuật toán.', '', '## BTC: so cùng video với hai baseline', '', 'Chỉ bốn người, mỗi nhóm một người. Mốc hiệu chỉnh lấy từ dự đoán native đã lưu của cùng người; không phải mô hình hiệu chỉnh fit mới cho từng encode. CI ghép cặp theo người 10.000 lần seed0. Không chọn ứng viên từ pilot này.', '', '| FPS | bf | Nhóm | Phương pháp | Video/người | MAE | CI95 | Δ hiệu chỉnh [CI95] |','|---:|---|---|---|---:|---:|---|---|']
 for a in btc:lines.append(f"| {a['fs']} | {a['bf'] if a['bf'] is not None else 'trước encode'} | {a['group']} | {a['method']} | {a['videos']}/{a['people']} | {a['MAE']:.3f} | {a['CI95']} | {a['delta_calibrated']:+.3f} {a['delta_calibrated_CI']} |")
 lines += ['', '## Giới hạn', '', 'Hồi quy theo lịch codec có thể xóa mạch thật trùng chu kỳ; REGRESSION_PULSE_PROBE.json đã ghi thất bại bảo toàn pulse 150 bpm. Vì thế đỉnh bám encoder không cho phép tự động xóa thành phần đó hoặc kết luận mọi đỉnh cùng tần số đều là artifact. Re-encode không loại artifact có từ video nguồn.', '', 'Tái lập báo cáo: python report_codec_exact.py. Lệnh encode/bitstream/GOP/PSD/hash nằm cạnh từng RESULT.json. Không chạy lại launcher trong thư mục có kết quả.']
 (r.P/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 r.save(r.P/'FINAL_VALIDATION.json',dict(utc=r.now(),encodes=66,verified_sha256=verified,holdout_evaluations=0,report_sha256=r.sha(r.P/'REPORT.md'),source_plan_sha256=r.sha(r.P/'PLAN_V2.json')))
 print('EXACT_CODEC_REPORT_COMPLETE',66)
if __name__=='__main__':main()
