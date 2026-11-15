import csv,json,statistics,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
rows=list(csv.DictReader((P/'MCD_VIEW_INVENTORY.csv').open()))
folds=json.loads((P/'MCD_INDEX_SCHEMA.json').read_text())['folds']['folds']
assignment={p:f['fold'] for f in folds for p in f['mcd_val_subjects']}
assert len(assignment)==60 and len(rows)==357
for r in rows:r['training_validation_fold']=assignment[r['person']]
assert all(len({r['training_validation_fold'] for r in rows if r['person']==p})==1 for p in assignment)
with (P/'MCD_VIEW_INVENTORY_WITH_TRAINING_FOLD.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Ba camera MCD: kiểm kê dữ liệu thật','',
'Đã kiểm kê 357 bản ghi thuộc 60 người DEV, gồm 119 phiên quay. Mỗi phiên có ba camera, cùng một file PPG thô: SHA-256 khớp trong cả 119/119 bộ ba. Không đọc D3.',
'', '| Camera | Bản ghi | Kích thước | Codec | FPS theo đồng hồ nguồn: trung vị [min–max] | Bitrate toàn file / thời lượng nguồn: trung vị kbps |', '|---|---:|---|---|---:|---:|']
for camera in sorted({r['camera'] for r in rows}):
 rr=[r for r in rows if r['camera']==camera];fs=[float(r['metadata_effective_fps']) for r in rr];br=[float(r['wholefile_sourceclock_kbps']) for r in rr]
 dims=sorted({r['width']+'×'+r['height'] for r in rr});codec=sorted({r['codec'] for r in rr})
 lines.append(f"| {camera} | {len(rr)} | {', '.join(dims)} | {', '.join(codec)} | {statistics.median(fs):.3f} [{min(fs):.3f}–{max(fs):.3f}] | {statistics.median(br):.3f} |")
lines += ['', 'FPS là số khoảng giữa các khung chia cho thời gian từ timestamp đầu tới cuối; không lấy nhãn FPS của container làm tốc độ cảm biến. Bitrate bao gồm overhead của file và dùng thời lượng nguồn, không phải riêng stream bitrate. Bảng CSV giữ cả FPS mux, số dòng metadata, thời lượng và độ phủ PPG để kiểm tra.',
'', '## Người và fold', '',
'Đã nối mã người với đúng training_index của T8: mỗi fold validation có 12 người, bốn fold còn lại có 48 người. Cả FullHD/USB/Iriun của cùng một người đều thuộc cùng fold. Cột official_fold trong bảng gốc là phân chia do bộ dữ liệu công bố; cột training_validation_fold mới là fold thực tế đội đang dùng. Hai cột không được thay thế cho nhau.',
'', '## Có PPG, nhưng độ trễ camera chưa được xác minh', '',
'Iriun là view điện thoại. PPG và timestamp chung đã có, nên nhận định trước đây rằng đội không có video điện thoại kèm waveform là sai. Tuy nhiên, timestamp được ghi ở máy thu; cùng một đồng hồ không tự chứng minh thời điểm phơi sáng và PPG đã khớp sau độ trễ truyền hình ảnh.',
'', 'Bài của tác giả mô tả đồng bộ tương đối giữa video bằng đồng hồ hiển thị và ước lượng độ lệch PPG bằng tương quan POS; họ cũng nêu giới hạn khi góc nhìn làm POS yếu. Vì vậy inventory này xác nhận dữ liệu và quan hệ phiên quay, chưa xác nhận độ trễ phơi sáng trên từng view. Các chẩn đoán lag đã lưu trước đó cũng không đủ mạnh để suy ra offset mới cho Iriun.',
'', 'Nguồn tác giả: [Gaze into the Heart, mục 2–3](https://arxiv.org/html/2508.17924v1).',
'', 'V5 theo điều kiện “PPG đồng bộ được xác minh” chưa được thả. Không tự tìm offset bằng MAE, không mở D3; bốn lượt padding-fixed đang chạy độc lập. Việc giữ V5 ở đây không có nghĩa Iriun không có nhãn hoặc không thể dùng nghiên cứu.',
'', 'Tái lập: python report_inventory.py. Nguồn đo: MCD_VIEW_INVENTORY.csv; fold: MCD_INDEX_SCHEMA.json; bảng nối: MCD_VIEW_INVENTORY_WITH_TRAINING_FOLD.csv.']
(P/'MCD_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(P/'MCD_FOLD_PARITY.json').write_text(json.dumps(dict(records=357,people=60,fold_people={f['fold']:len(f['mcd_val_subjects']) for f in folds},same_person_same_fold=True,source_inventory_sha256=hashlib.sha256((P/'MCD_VIEW_INVENTORY.csv').read_bytes()).hexdigest(),fold_schema_sha256=hashlib.sha256((P/'MCD_INDEX_SCHEMA.json').read_bytes()).hexdigest()),indent=2))
print('MCD_INVENTORY_REPORT_COMPLETE',len(rows),len(assignment))
