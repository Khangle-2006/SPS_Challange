import json
from pathlib import Path
b=Path('reports/mcd_mapping_20261030')
a=json.loads((b/'aggregate.json').read_text())
diag=json.loads((b/'combined_diagnostics.json').read_text())
c=json.loads((b/'crop_storage/results.json').read_text())
cs=json.loads((b/'crop_storage/summary.json').read_text())
dev=json.loads((b/'development_plan.json').read_text())
dl=json.loads((b/'development_download_snapshot.json').read_text())
m=json.loads((b/'mmpd_60_files_plan.json').read_text())
u=json.loads(Path('reports/storage_pilot_20261030/ubfc2_ffv1_complete.json').read_text())
lines=['KẾT QUẢ CHẨN ĐOÁN FRAME/TIMESTAMP VÀ LƯU TRỮ — 30/10/2026','',
'1. PHẠM VI VÀ KẾT QUẢ MCD',
'Đã kiểm tra6 video của1020 và30 video từ30 người khác, seed20261030;',
'10 người/camera,5 before+5 after mỗi camera; không chọn theo lỗi mô hình.',
'Mọi file khớp size/SHA256 của nguồn pinned. Không sửa hoặc xóa raw.',
'Không thấy B-frame/multi-VOP/tiny-packet; showall không thêm frame.',
'Frame đầu là I/keyframe tại PTS0 trong cả36 video.',
'Cả36 có payload00dc cuối khai báo dài hơn phần byte còn lại.',
'RIFF length0xffffffff là sentinel, KHÔNG dùng nó để suy ra file phải dài4GB.',
'Các file đều dừng ở biên256KiB; có bằng chứng tail bị cắt/chưa flush đầy đủ,',
'nhưng chưa phân biệt được lỗi xảy ra lúc ghi, sao chép hay phát hành.',
'Không thể từ đây khẳng định mọi frame thiếu đều ở cuối log.',
'', 'Bảng30 người ngẫu nhiên (meta rows - decoded frames):']
for r in a['counts']:
    if r['set']=='random30':
        lines.append(f"{r['camera']}: n={r['videos']}; tail truncated={r['tail_truncated']}; thiếu={r['meta_minus_frames_min']}–{r['meta_minus_frames_max']}; packets=frames {r['packet_eq_frame']}/{r['videos']}; accepted mapping={r['accepted_mapping']}")
exceptions=[r for r in diag if r['packets']!=r['frames_default']]
for r in exceptions:lines.append(f"Ngoại lệ packet/frame: {r['video']}: packets={r['packets']},frames={r['frames_default']}; xem packet table/log, không ghép một packet thành một frame mặc định.")
equal=[r['video'] for r in diag if r['meta_minus_frames']==0]
lines += ['',f"Ngay cả số timestamp bằng số frame vẫn có tail hỏng: {equal}.",
'Chẩn đoán: không có bằng chứng decoder bỏ đầu trước I-frame hoặc packed B',
'trong mẫu này. Tail payload thiếu là quan sát trực tiếp; clock mapping chưa',
'được chứng minh. 36/36 tạm quarantine cho cache training theo gate đã chốt,',
'KHÔNG có nghĩa toàn bộ36 video vô dụng hoặc phải loại vĩnh viễn.',
'Không ngoại suy số video lỗi trong3600 từ mẫu này. Ngưỡng tỉ lệ chưa đủ để',
'quyết định bỏ dataset; cần writer/index convention hoặc mốc độc lập.',
'Metadata gap có thể là jitter; riêng chỗ gap không chứng minh row nào mất.',
'Xem PREREGISTRATION.txt và combined_diagnostics.json. PPG sync chưa xử lý.',
'', '2. TẬP PHÁT TRIỂN VÀ DOWNLOAD',
f"Chốt60 người,360 video đủ3camera x2state, {dev['development_raw_bytes']/1e9:.3f} GB; bảo vệ thêm1020 làm ca chẩn đoán.",
f"Snapshot download: {dl['completed']}/{dl['files_total']} video của tập60; state={dl['state']}; errors={len(dl['errors'])}.",
'Bộ đếm gồm cả video đã có từ trước; không phải toàn bộ đều vừa tải.',
'Chỉ nối lại tập60, audit30 được tải trước. Phần ngoài tập60 chưa tải thêm.',
'Toàn bộ raw đã có vẫn được giữ đến khi E3/frontend chốt và mapping/cache',
'được kiểm chứng. Không bật bulk-prune. Tập60 là development, không được',
'dùng để tuyên bố untouched external test sau khi đã chọn/sửa phương pháp.',
'', '3. ĐO CROP FFV1 (STORAGE-ONLY, KHÔNG PHẢI CACHE TRAINING)']
for r in c:
    lines.append(f"{r['camera']}: {r['frames']} frame; uint8={r['crop_raw_bytes']/1e6:.3f} MB; FFV1={r['ffv1_bytes']/1e6:.3f} MB; NPZ={r['npz_bytes']/1e6:.3f} MB; exact RGB/PTS={r['exact_rgb_and_pts']}; corrupt_source_frames={r['corrupt_frames']}")
lines += [f"Giảm gộp {cs['bytes_ratio_raw_over_ffv1']:.3f} lần. Ngoại suy theo mẫu: crop FFV1~{cs['projected_crop_ffv1_bytes']/1e9:.2f} GB.",
f"Trace RGB3vùng x3kênh float32~{cs['projected_trace_rgb_float32_only_bytes']/1e9:.3f} GB trước nén.",
'Ngoại suy chỉ từ1người/3camera, chưa phải dung lượng toàn bộ đo được.',
'Trace chưa gồm polygon, masks, label/PPG, clocks, provenance và metadata.',
'Giữ crop hash/decoded index/container PTS; acquisition log giữ riêng chưa',
'ánh xạ. FFV1 khớp pixel không biến frame bị decoder che lỗi thành frame sạch.',
'Các frame bị đánh dấu corrupt được lưu để tái lập nhưng validity=false.',
'', '4. UBFC2',
f"FFV1 hoàn tất{u['completed']}/{u['total']}; giải phóng{u['saved_bytes']/1e9:.3f} GB.",
'Đã kiểm tra RGB từng frame, PTS và decoder OpenCV trước mỗi lần thay file.',
'', '5. MMPD',
f"Giữ10 người2/2/3/3,60file, {m['total_bytes']/1e9:.3f} GB theo manifest.",
'LED-low / LED-high / Nature x Stationary / Rotation, không exercise.',
'Giữ lô2người; size lần lượt: '+', '.join(f"{x['bytes']/1e9:.2f}GB" for x in m['two_person_batches'])+'.',
'Chưa có quyền truy cập, chưa tải dữ liệu. Email TXT/EML đã cập nhật, chưa gửi.',
'Không gọi mini frontend thất bại khi chưa thử; ở đây chọn320x240 để giảm rủi ro.',
'MMPD subset chỉ là kế hoạch đánh giá nhóm3–6, không phải đạt C2 toàn diện.',
'', '6. VIỆC CÒN THIẾU',
'Xác minh convention frame_index trong phần mềm ghi/capture hoặc mốc ảnh',
'độc lập, và hỏi bản AVI đã flush đủ. Không tự loại N dòng đầu/cuối.',
'Sau đó mới kiểm tra PPG clock/latency; chung host clock không tự đảm bảo',
'sensor synchronous. Tiếp đó mới khóa E3/cache và cho phép prune ngoài dev.',
'Dự thảo câu hỏi upstream: MCD_AUTHOR_QUESTION_DRAFT.txt, chưa gửi/đăng.',
'Codec-domain protocol riêng tại CROSS_DATASET_PROTOCOL.txt.']
(b/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[:18]))
