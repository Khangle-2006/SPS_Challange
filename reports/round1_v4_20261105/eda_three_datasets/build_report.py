"""Build the local aggregate EDA from ffprobe inventories and the verified workbook."""
from pathlib import Path
from html import escape
import hashlib
import os
import re
import sys
import numpy as np
import pandas as pd
from openpyxl import load_workbook

out=Path(__file__).resolve().parent
sheet=Path(sys.argv[1])
sha=hashlib.sha256(sheet.read_bytes()).hexdigest()
assert sha=='e52a9b190bf91f8a5d6d476b8374b8904f47328a668bfbe7cd60f3c14fafc49a'
media=pd.read_csv(out/'media.csv')
u=pd.read_csv(out/'ubfc_reference.csv')
a=pd.read_csv(out/'aipex_reference.csv')
clock=pd.read_csv(out/'btc_clock.csv')
rec=pd.read_csv(out.parent/'organiser_download/BTC_rows_reconciled.csv')
verified=pd.read_csv(out.parent/'organiser_download/P100_verified_files.csv')
assert verified.sha256.nunique()==296
btc=media.query("dataset=='BTC'").merge(clock,on='id',validate='one_to_one')
assert len(btc)==len(clock)==296
btc['stem']=btc.file.map(lambda v:Path(v).stem.strip())
def lighting(v):
    return next((s for s in ['Bright','Dim','Med'] if '_'+s+'_' in v),'Không ghi')
btc['light']=btc.stem.map(lighting)
btc['size_px']=btc.width.astype(str)+'×'+btc.height.astype(str)
headers=['record','link','name','sex','age','skin','height','weight','task','spo2','sys','dia','temp','pr','health','ward']
wb=load_workbook(sheet,read_only=True,data_only=True)
rows=[]
for g in ['Normal','Abnormal']:
    for i,values in enumerate(wb[g].iter_rows(min_row=2,max_col=16,values_only=True),2):
        if not(values[0] and values[1]):continue
        row=dict(zip(headers,values))
        row['name_key']=hashlib.sha256(' '.join(str(row.pop('name') or '').casefold().split()).encode()).hexdigest()
        row['health']=bool(row['health']);row['ward']=bool(row['ward'])
        rows.append(dict(row,group=g,sheet_row=i))
b=pd.DataFrame(rows).merge(rec,on=['group','sheet_row'],validate='one_to_one')
assert len(b)==295 and b.verified_drive_id.notna().sum()==293
assert (b.record.str.strip()==b.recording.str.strip()).all()
b['pid']=b.record.str.strip().str.extract(r'^(S_(?:AB|N)\d+)',expand=False)
assert b.pid.notna().all() and b.pid.nunique()==202
b['task']=b.task.str.strip().str.replace(' ','',regex=False)
for field in ['skin','sex']:
    b[field]=b[field].str.strip()
b['light']=b.record.map(lighting)
b['present']=b.verified_drive_id.notna()
labels=['pr','spo2','sys','dia','temp']
numbers=labels+['age','height','weight']
for k in numbers:b[k]=pd.to_numeric(b[k],errors='coerce')
matched=b[b.present].merge(btc,left_on='verified_drive_id',right_on='id',suffixes=('','_video'),validate='one_to_one')
assert len(matched)==293 and (matched.group==matched.group_video).all()
ap=media.query("dataset=='AiPEX'").copy()
ap['roi']=ap.file.str.extract(r'_(forehead|leftcheek|rightcheek)\.avi$')
unit=['group','id']
assert (ap.groupby(unit).roi.nunique()==3).all() and (ap.groupby(unit).frames.nunique()==1).all()
people=ap[ap.roi=='forehead'].merge(a,on=unit,validate='one_to_one')
assert len(people)==88
um=media.query("dataset=='UBFC'").merge(u,on=unit,suffixes=('','_gt'),validate='one_to_one')
assert len(um)==43
assert (um.ppg_n==um.hr_n).all() and (um.t_n==um.hr_n).all()
assert (um.query("group=='UBFC2'").ppg_n==um.query("group=='UBFC2'").frames).all()
assert (matched.stem.str.casefold()==matched.record.str.strip().str.casefold()).all()
assert b[numbers].notna().all().all()
def counts(s):
    return ' · '.join(f'{k} ({int(v)})' for k,v in s.value_counts(dropna=False).items())
def stats(v):
    v=pd.Series(v).dropna()
    return [len(v),v.min(),v.quantile(.25),v.median(),v.mean(),v.quantile(.75),v.max()]
statcols=['n','Min','P25','Trung vị','TB','P75','Max']
def fmt(v):
    if pd.isna(v):return '—'
    if isinstance(v,(float,np.floating)):
        return (f'{v:.2f}'.rstrip('0').rstrip('.')).replace('.',',')
    return escape(str(v))
page=[]
def h(t):page.append('<h2 id="s'+t.split('.')[0]+'">'+escape(t)+'</h2>')
def sub(t):page.append('<h3>'+escape(t)+'</h3>')
def p(t):page.append('<p>'+t+'</p>')
def table(cols,rows):
    page.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(c)+'</th>' for c in cols)+'</tr></thead><tbody>')
    for row in rows:page.append('<tr>'+''.join('<td>'+fmt(v)+'</td>' for v in row)+'</tr>')
    page.append('</tbody></table></div>')

parts=[('BTC · Normal',btc[btc.group=='Normal']),('BTC · Abnormal',btc[btc.group=='Abnormal']),('UBFC1',um[um.group=='UBFC1']),('UBFC2 · dev',um[um.group=='UBFC2']),('AiPEX · India',people[people.group=='India']),('AiPEX · Sierra Leone',people[people.group=='Sierra_Leone'])]
p('Ngày 07/11/2026 · Đo trên bản lưu P100. Mỗi dòng của bảng thời lượng đại diện cho một lần quay. Với AiPEX, ba ROI thuộc cùng lần quay được gộp sau khi kiểm tra số frame khớp trên cả 88 người.')
p('<a href="#s2">Video và nhãn BTC</a> · <a href="#s3">Waveform UBFC</a> · <a href="#s4">ROI AiPEX</a> · <a href="#s5">Độ sáng và so sánh</a> · <a href="#s6">Nguồn và dữ liệu đo</a>')
h('1. Ba bộ dữ liệu trả lời ba câu hỏi khác nhau')
table(['Bộ','Phạm vi','Hình ảnh phát hành','Độ chi tiết nhãn','Có thể đối chiếu trực tiếp'],[
['BTC / SRMC','296 video; 293 ghép dòng; 202 mã người trong bảng','Video MOV/MP4, nhiều cấu hình','Một hàng cho mỗi bản ghi có nhãn','PR, SpO₂, huyết áp tâm thu/tâm trương, nhiệt độ theo bản ghi'],
['UBFC-rPPG','50 video trong mirror; EDA mở 8 + 35','Video webcam toàn mặt','PPG, HR thiết bị và timestamp; UBFC1 có thêm SpO₂','Dạng sóng và HR trên khoảng thời gian đã kiểm tra clock'],
['AiPEX / RPPG Dataset','88 người; 264 AVI','Ba vùng trán và má đã cắt','Một HR trung bình cho mỗi người/bản ghi','HR trung bình theo người, theo điểm thu']])
p('Phụ lục 1, trang 17–18 của tài liệu SP Cup chỉ đích danh UBFC-rPPG và repo AiPEX. Bộ bệnh viện do BTC cung cấp là một nguồn riêng. Tài liệu mô tả nguồn lâm sàng tại Sri Ramachandra Hospital, Tamil Nadu, Ấn Độ.')
sub('Thời lượng phát theo bản ghi, giây')
table(['Nguồn']+statcols+['Tổng phút','Frame'],[[k,*stats(d.seconds),d.seconds.sum()/60,int(d.frames.sum())] for k,d in parts])
p('Mỗi dòng là một nhóm. Ví dụ, với BTC Normal, một nửa số video dài tối đa 30,66 giây. P25 và P75 là hai mốc bao quanh 50% số video ở giữa. Thời lượng lấy từ luồng hình. Với AiPEX, chỉ tính một lần cho ba vùng mặt của cùng người.')
p('BTC có tổng cộng 2 giờ 23 phút 54 giây video. UBFC có khoảng 49 phút 32 giây trong phần mở. AiPEX có khoảng 9 giờ 53 phút phát nếu đếm một lần trên mỗi người; cộng ba vùng mặt sẽ nhân con số ấy lên ba lần.')
h('2. FaceVitals BTC · Normal và Abnormal')
table(['Nhóm','Dòng BTC','Mã người trong bảng','Video trong thư mục','Video khớp nhãn','Mã người có video khớp','Thiếu video','Video dư','GB'],[[g,len(d),d.pid.nunique(),sum(btc.group==g),int(d.present.sum()),d[d.present].pid.nunique(),int((~d.present).sum()),sum(btc.group==g)-d.present.sum(),btc[btc.group==g].bytes.sum()/1e9] for g,d in b.groupby('group',sort=False)])
p('Kiểm bảng Normal và Abnormal, bỏ tab Copy of Normal. Dòng Normal 70 dùng nhầm link Dim của dòng 73. Bản Bright được khôi phục theo đúng tên từ thư mục Normal. Ba file dư là ba task Dim của S_N43. Hai dòng thiếu video là S_AB27_Task1 và S_AB37_Resting. Các file trên P100 có 296 SHA-256 riêng biệt. Phép đếm này chỉ phát hiện file trùng từng byte.')
sub('Video, clock và khả năng cắt cửa sổ')
table(['Nhóm','Codec (số file)','Định dạng pixel','FPS TB min / trung vị / max','Bitrate video min / trung vị / max (Mb/s)','Có audio','Có xoay metadata'],[[g,counts(d.codec),counts(d.pixel),' / '.join(fmt(x) for x in [d.avg_fps.min(),d.avg_fps.median(),d.avg_fps.max()]),' / '.join(fmt(x/1000) for x in [d.video_kbps.min(),d.video_kbps.median(),d.video_kbps.max()]),int(d.audio.sum()),int(d.rotation.ne(0).sum())] for g,d in btc.groupby('group')])
table(['Kích thước mã hóa','Normal','Abnormal'],[[k,int((btc.query("group=='Normal'").size_px==k).sum()),int((btc.query("group=='Abnormal'").size_px==k).sum())] for k in btc.size_px.value_counts().index])
p('Kích thước trên là trước khi áp ma trận xoay. Có cả video 10-bit và 8-bit. Đọc rotation, dải màu và pixel format trước khi lấy RGB. Bitrate ở bảng trên là của riêng luồng hình.')
table(['Nhóm','Hàm truyền màu theo header','Đuôi file','Video có bitrate ≤0,5 Mb/s','10 fps theo header (±0,1)'],[[g,counts(d.transfer),counts(d.file.map(lambda x:Path(x).suffix.lower())),int((d.video_kbps<=500).sum()),int((d.avg_fps.sub(10).abs()<=.1).sum())] for g,d in btc.groupby('group')])
p('20 video khai báo hàm truyền HLG (arib-std-b67). Độ sáng đo trên mã pixel và độ sáng vật lý là hai thang khác nhau. C5 của dự án là điều kiện tái mã hóa ở 10 fps và ≤500 kb/s; nó cần được tách khỏi cấu hình gốc của bản phát hành.')
table(['Nhóm','File đủ 10 s','Đủ 20 s','Đủ 30 s','Số cửa sổ 10 s rời nhau','Có gap PTS >1,5×','Có gap >2×','Gap lớn nhất (ms)','Packet thiếu PTS','PTS trùng'],[[g,*[int((d.seconds>=x).sum()) for x in [10,20,30]],int(np.floor(d.seconds/10).sum()),int(d.gt_1_5median.gt(0).sum()),int(d.gt_2median.gt(0).sum()),d.max_dt_ms.max(),int((d.packets-d.pts).sum()),int(d.duplicates.sum())] for g,d in btc.groupby('group')])
p('Gap đo trên timestamp của mọi video packet, sắp theo thứ tự hiển thị; ngưỡng tính theo trung vị từng file. Khoảng thưa có thể do nhịp mã hóa biến thiên. FPS file không xác định được FPS cảm biến. Số cửa sổ chỉ xét thời lượng, trước các điều kiện chất lượng và nhãn. S_N54_Med_Task1.MOV là trường hợp cần phân biệt rõ: 7.315 frame, 30,43 s phát, trung bình 240,37 fps.')
sub('Nhãn BTC và phân bố')
p('PR là pulse rate (bpm). SpO₂ theo %, BP-SYS và BP-DIA là hai thành phần huyết áp (thông lệ mmHg; tiêu đề bảng chỉ ghi BP). Nhiệt độ theo °F. Bảng có tuổi, giới, skin tone do nguồn ghi, chiều cao cm, cân nặng kg, protocol, bệnh sử và khoa. Cột thứ tư của Abnormal trống tiêu đề, giá trị Female/Male; báo cáo đọc theo cấu trúc tương ứng của Normal. Bảng không cung cấp timestamp đo, PPG thô, dụng cụ đo cho từng dòng hay nhãn HR cho từng cửa sổ 10 s.')
human={'pr':'PR (bpm)','spo2':'SpO₂ (%)','sys':'BP-SYS','dia':'BP-DIA','temp':'T (°F)','age':'Tuổi','height':'Chiều cao (cm)','weight':'Cân nặng (kg)'}
table(['Nhóm','Biến']+statcols+['Thiếu'],[[g,human[c],*stats(d[c]),int(d[c].isna().sum())] for g,d in b.groupby('group',sort=False) for c in numbers])
p('Bảng trên tính theo 295 dòng sổ, kể cả hai dòng thiếu video. Để đối chiếu trực tiếp với mô hình, phạm vi giao dưới đây chỉ tính 293 dòng có cả video và nhãn.')
table(['Nhóm ghép','N','PR trung vị','SpO₂ trung vị','SYS / DIA trung vị','T trung vị °F','PR <60','PR >100','SpO₂ <95','SpO₂ <90','SYS ≤ DIA'],[[g,len(d),d.pr.median(),d.spo2.median(),fmt(d.sys.median())+' / '+fmt(d.dia.median()),d.temp.median(),int((d.pr<60).sum()),int((d.pr>100).sum()),int((d.spo2<95).sum()),int((d.spo2<90).sum()),int((d.sys<=d.dia).sum())] for g,d in matched.groupby('group')])
p('Các ngưỡng chỉ dùng đếm độ phủ nhãn. Normal/Abnormal là nhóm do BTC đặt. Ví dụ, Normal vẫn có SpO₂ 85% và PR 142 bpm. Dùng tên nhóm như một chẩn đoán hoặc thay cho giá trị đo sẽ làm mất thông tin.')
sub('Số video theo từng khoảng PR, SpO₂ và huyết áp tâm thu')
p('Bảng đếm các bản ghi đã ghép được video. Phần trăm tính riêng trong 194 video Normal và 99 video Abnormal. Cận trên của mỗi khoảng thuộc dòng kế tiếp.')
for field,title,edges in [('pr','PR (bpm)',[60,80,100,120]),('spo2','SpO₂ (%)',[90,95,98]),('sys','Huyết áp tâm thu (mmHg)',[100,120,140,160])]:
    intervals=[-np.inf,*edges,np.inf]
    ranges=[f'Dưới {edges[0]}',*[f'{lo} đến dưới {hi}' for lo,hi in zip(edges,edges[1:])],f'Từ {edges[-1]} trở lên']
    rows=[]
    for (lo,hi),name in zip(zip(intervals,intervals[1:]),ranges):
        cells=[name]
        for group in ['Normal','Abnormal']:
            values=matched.loc[matched.group==group,field]
            count=int(((values>=lo)&(values<hi)).sum())
            cells.extend([count,100*count/len(values)])
        rows.append(cells)
    assert sum(r[1] for r in rows)==194 and sum(r[3] for r in rows)==99
    table([title,'Normal (video)','Normal (%)','Abnormal (video)','Abnormal (%)'],rows)

sub('Người, nhãn lặp và điều kiện ghi hình')
table(['Nhóm','Giới theo dòng','Skin tone theo dòng','Protocol theo bảng','Ánh sáng theo tên file (toàn thư mục)'],[[g,counts(d.sex),counts(d.skin),counts(d.task),counts(btc.loc[btc.group==g,'light'])] for g,d in b.groupby('group',sort=False)])
per=b.groupby(['group','pid'])
table(['Nhóm','Số dòng / mã người (số người)','Có nhiều dòng','Nhiều dòng, 5 vitals giống hệt ở mọi dòng','Mã có nhiều tuổi / giới / skin tone','Mã có nhiều cách ghi tên'],[[g,counts(d.groupby('pid').size()),sum(len(z)>1 for _,z in d.groupby('pid')),sum(len(z)>1 and len(z[labels].drop_duplicates())==1 for _,z in d.groupby('pid')),sum((z[['age','sex','skin']].nunique()>1).any() for _,z in d.groupby('pid')),int((d.groupby('pid').name_key.nunique()>1).sum())] for g,d in b.groupby('group',sort=False)])
p('Sau khi chuẩn hóa khoảng trắng, hai mã Normal có nhiều mức skin tone và một mã có hơn một cách ghi tên. Đây là điểm cần làm rõ khi đánh giá theo người hoặc màu da. Chỉ thống kê số lượng; không xuất tên, không tự gộp người trùng tên. Các ô phụ bên phải cột Ward ở một số dòng Abnormal không có tiêu đề thống nhất nên không dùng làm nhãn sinh lý.')
table(['Nhóm','Giới theo mã người¹','Skin tone theo mã người¹','Tuổi trung vị theo mã người¹','Mã có tuổi <18¹','Điền bệnh sử (dòng)','Điền khoa (dòng)'],[[g,counts(d.drop_duplicates('pid').sex),counts(d.drop_duplicates('pid').skin),d.drop_duplicates('pid').age.median(),sum(d.drop_duplicates('pid').age<18),int(d.health.sum()),int(d.ward.sum())] for g,d in b.groupby('group',sort=False)])
p('¹ Lấy dòng đầu của từng mã để tránh người có nhiều task được tính nhiều lần. Các mã người được lấy từ tiền tố của ID do BTC cấp. Các mục bệnh sử và khoa chỉ thống kê độ điền ô, không xuất nội dung. Dấu Bright/Med/Dim từ tên tệp chưa được đo lux. Protocol lấy từ cột BTC, đã gộp “Task 1” và “Task1”. Một số tên tệp dùng Task trong khi bảng ghi Walking. Giữ cả hai nguồn khi đối chiếu.')
table(['Nhóm có video','Protocol BTC','Số video','Số mã người','Thời lượng trung vị (s)','PR trung vị','SpO₂ trung vị'],[[g,t,len(d),d.pid.nunique(),d.seconds.median(),d.pr.median(),d.spo2.median()] for (g,t),d in matched.groupby(['group','task'])])
sub('Độ phủ theo người và điều kiện camera')
person=b.drop_duplicates(['group','pid']).copy()
person['age_band']=pd.cut(person.age,[0,17,34,54,74,200],labels=['<18','18–34','35–54','55–74','≥75'],include_lowest=True)
table(['Tuổi (năm)','Normal · mã người','Abnormal · mã người'],[[str(k),int(((person.group=='Normal')&(person.age_band==k)).sum()),int(((person.group=='Abnormal')&(person.age_band==k)).sum())] for k in person.age_band.cat.categories])
table(['Nhóm','Skin tone do BTC ghi','Mã người¹','File khớp','Resting có video','PR trung vị²','SpO₂ trung vị²'],[[g,k,int(((person.group==g)&(person.skin==k)).sum()),len(d),sum(d.task=='Resting'),d.pr.median(),d.spo2.median()] for (g,k),d in matched.groupby(['group','skin'])])
p('¹ Số mã người lấy mức ở dòng đầu của mỗi mã. ² Trung vị trên bản ghi có video. Nhóm Abnormal chỉ có 4 dòng Fair trong toàn bảng, còn Normal có 57. Khác biệt độ tuổi, hoạt động, thiết bị và phân bố nhãn cùng tồn tại. Một chênh lệch MAE theo nhóm hoặc skin tone cần được đọc cùng độ phủ người và protocol.')
p('Nếu chia train/validation, các task, mức sáng và mọi cửa sổ của cùng mã người phải ở cùng một phía. Với PR một giá trị mỗi dòng, phép so trực tiếp là tổng hợp dự đoán thành một số trên mỗi video rồi đối chiếu PR. Có thể báo thêm kết quả lấy trung bình theo người để cân bằng số lần quay.')
p('Đối chiếu toàn bộ ID với protocol trong sổ có đúng 7 dòng hậu tố Task1 nhưng trường protocol là Walking (5 Normal, 2 Abnormal). Trong 293 dòng có video, tên file ghép khớp ID sau chuẩn hóa hoa thường và khoảng trắng. Đây là khác biệt ngữ nghĩa giữa hai cột, không phải bảy file thất lạc.')
h('3. UBFC-rPPG · hai protocol và hai cấu trúc nhãn')
p('Tác giả dùng webcam Logitech C920, khoảng cách khoảng 1 m trong nhà, ánh sáng tự nhiên và đèn. Thiết bị tham chiếu là pulse oximeter CMS50E. UBFC1 yêu cầu ngồi yên, một số video có chuyển động; UBFC2 dùng trò chơi tính toán có giới hạn thời gian để thay đổi nhịp tim. P100 có đủ 8 và 42 video. EDA mở đủ 8 UBFC1 và 35 người dev của UBFC2. Bảy người test giữ kín. Thư mục UBFC1 là mã bản ghi; quan hệ giữa mã và danh tính người cần xác minh trước khi chia theo người.')
table(['Nhóm','Video · codec · pixel','FPS min / trung vị / max','Mẫu PPG','Mẫu PPG / file min–max','HR thiết bị min–max (bpm)','Median HR của mỗi file min–max','SpO₂ (%)'],[[g,counts(d.codec)+' · '+counts(d.pixel),' / '.join(fmt(x) for x in [d.avg_fps.min(),d.avg_fps.median(),d.avg_fps.max()]),int(d.ppg_n.sum()),f'{int(d.ppg_n.min())}–{int(d.ppg_n.max())}',fmt(d.hr_min.min())+'–'+fmt(d.hr_max.max()),fmt(d.hr_median.min())+'–'+fmt(d.hr_median.max()),'95–99' if g=='UBFC1' else 'Không có trường'] for g,d in um.groupby('group')])
p('UBFC1 dùng gtdump.xmp dạng CSV: thời điểm mili giây, HR, SpO₂, PPG. UBFC2 dùng ground_truth.txt có ba dòng lần lượt là PPG, HR và thời điểm giây. Các kênh có đơn vị và tần suất khác nhau. UBFC1 có 40.273 mẫu nhãn cho 18.538 frame. Ở 35 UBFC2, độ dài ba dòng bằng số frame trong từng bản ghi. Điều này mô tả bản xuất đang có, không xác nhận phần cứng lấy mẫu đồng thời.')
table(['Nhóm','Video 640×480','Trung vị Δt nhãn min–max (ms)','Bản ghi có t không tăng','Số bước t không tăng','Bản ghi có gap >2×','Tổng gap >2×','HR min <35 ở bao nhiêu file'],[[g,len(d),'–'.join(fmt(x) for x in [d.median_dt_ms.min(),d.median_dt_ms.max()]),sum(d.nonincreasing>0),d.nonincreasing.sum(),sum(d.gaps_gt_2median>0),d.gaps_gt_2median.sum(),sum(d.hr_min<35)] for g,d in um.groupby('group')])
p('Bốn bản UBFC2 dev có HR thiết bị xuống 1 bpm. Riêng subject20 có trung vị dòng HR là 7 bpm. Đây là lý do cần phân biệt HR thiết bị và HR giải từ dạng sóng PPG, kiểm tra timestamp/độ tin cậy trước khi chấm. UBFC1 có clock nhãn riêng, 16 ms là trung vị bước thời gian. UBFC2 của mirror dùng FFV1; mô tả 30 fps của thiết kế thu hình và FPS thực tế trong file là hai đại lượng khác nhau.')
p('Theo audit UBFC1 đã lưu, tốc độ cảm biến trung bình theo bản ghi khoảng 62,10–62,18 Hz. Hai bản ghi có đoạn PPG ngắn chạm 127 (5 và 14 mẫu); cả tám bản chỉ có một kênh PPG xuất ra cùng SpO₂ của thiết bị. Phạm vi SpO₂ 95–99% chỉ bao phủ bốn điểm phần trăm. Readme của tác giả ghi hạn chế dùng ảnh mặt subject21 và subject27 trong báo cáo, slide và bài báo.')
ex=pd.read_csv(out/'reference_example.csv')
sub('Tín hiệu PPG trong 5 giây, tóm tắt từng giây')
p('Ví dụ lấy từ giây 10 đến trước giây 15 của hai bản ghi. Mỗi dòng cho biết số mẫu và biên độ PPG trong một giây. PPG dùng thang riêng của từng file; cột HR dùng bpm. <a href="reference_example.csv">Bảng số gốc theo từng thời điểm lấy mẫu</a>.')
ex['second']=np.floor(ex.time).astype(int)
rows=[]
for (group,second),sample in ex.groupby(['group','second']):
    rows.append([group,sample.id.iloc[0],f'{second} đến dưới {second+1}',len(sample),sample.ppg.min(),sample.ppg.mean(),sample.ppg.max(),sample.hr.median()])
assert len(rows)==10 and sum(row[3] for row in rows)==len(ex)
table(['Bộ','Bản ghi','Giây','Số mẫu','PPG min','PPG TB','PPG max','HR trung vị (bpm)'],rows)

h('4. AiPEX · 88 người, 264 video ROI')
p('Mỗi người có ba vùng từ cùng buổi quay: trán, má trái và má phải. Tác giả mô tả vùng lấy trên ảnh gốc lần lượt 60×30, 25×25 và 25×25 pixel. Tất cả file AVI được phát hành có khung mã hóa 200×200, MPEG-4, yuv420p, 15 fps. 264 tệp không tương đương với 264 người hoặc 264 HR độc lập. Hai file ground truth cung cấp mã video và HR trung bình (bpm); khóa ghép là quốc gia + mã video vì video1, video2… được tái sử dụng ở cả hai nơi.')
table(['Điểm thu','Người','File ROI','Frame/lần quay min–max','Phút phát duy nhất','GB (3 ROI)','HR tham chiếu']+statcols[1:],[[g,len(d),len(ap[ap.group==g]),str(int(d.frames.min()))+'–'+str(int(d.frames.max())),d.seconds.sum()/60,ap[ap.group==g].bytes.sum()/1e9,'bpm',*stats(d.hr)[1:]] for g,d in people.groupby('group')])
p('Trong mọi bộ ba, cả số frame và mã người đều khớp. Theo bài Dasari và cộng sự, thiết bị quay là Samsung J7, tốc độ thu trung bình khoảng 25 fps, thời gian thu tối thiểu 120 s, còn thiết bị tiếp xúc cho giá trị trung bình trong 120 s. File công khai phát ở 15 fps và có thời lượng rất khác nhau. Chưa có ánh xạ giữa khoảng 120 s của nhãn và từng đoạn trên clock file. Không thể gán số HR trung bình làm nhãn pha hay nhãn HR 10 s được xác minh.')
table(['Điểm thu','Bản phát <120 s','120–300 s','>300 s','Bản có HR <60','Bản có HR >100'],[[g,sum(d.seconds<120),sum(d.seconds.between(120,300)),sum(d.seconds>300),sum(d.hr<60),sum(d.hr>100)] for g,d in people.groupby('group')])
p('Hai cột gốc trong mỗi CSV chỉ cho phép ghép mã và HR. Quốc gia lấy từ thư mục. Các biến nhân khẩu trong bài hoặc bảng kết quả tác giả cần khóa liên kết phù hợp trước khi ghép theo người. ITA từ phân tích cũ là ước lượng từ pixel (77/88 có giá trị), khác với skin tone do BTC ghi; quốc gia không đại diện cho màu da.')
p('Khoảng HR tham chiếu của toàn AiPEX là 65,72–92,75 bpm. Hai nhóm có số người bằng nhau nhưng dải nhãn khác nhau; với tập này chỉ đánh giá được sai số trong dải HR quan sát. Trái lại, PR của bảng BTC trải từ 40 đến 142 bpm.')
h('5. Điều kiện so sánh và giới hạn của từng nguồn')
photos=pd.read_csv(out/'sample_luma.csv')
samples=media.merge(photos,on=['group','id','file'],validate='one_to_one')
assert len(samples)==603
sub('Độ sáng trên ảnh thực tế')
p('Mỗi video được lấy một frame tại 10%, 50% và 90% thời lượng phát. Chuyển cả khung sang mức xám 8-bit, thu xuống 64×64 bằng area, tính độ sáng trung bình và độ lệch chuẩn không gian trên mỗi thumbnail, rồi trung bình ba lần đo. Chỉ số dưới đây phản ánh nội dung hiển thị, hậu cảnh và xử lý của camera; không đo lux hay sắc tố da. Không lưu ảnh mặt. Các đại lượng của AiPEX tính riêng từng ROI.')
ss=samples.copy()
ss['stratum']=ss.group
mask=ss.dataset=='AiPEX'
ss.loc[mask,'stratum']=ss.loc[mask,'group']+' · '+ss.loc[mask,'file'].str.extract(r'_(forehead|leftcheek|rightcheek)\.avi$')[0]
mask=ss.dataset=='BTC'
ss.loc[mask,'stratum']=ss.loc[mask,'group']+' · '+ss.loc[mask,'file'].map(lighting)
table(['Nguồn / điều kiện','File','Đủ 3 frame','Luma P25','Luma trung vị','Luma P75','Tương phản trung vị','% pixel tối ≤10 (TB)','% pixel sáng ≥245 (TB)'],[[k,len(d),int(d.n.eq(3).sum()),d.luma.quantile(.25),d.luma.median(),d.luma.quantile(.75),d.contrast.median(),100*d.dark_fraction.mean(),100*d.bright_fraction.mean()] for k,d in ss.groupby('stratum')])
lightpairs=ss[ss.dataset=='BTC'].copy()
lightpairs['pid']=lightpairs.file.str.extract(r'^(S_(?:AB|N)\d+)')[0]
lightpairs['task']=lightpairs.file.str.extract(r'_(Task[123])\.',flags=re.I)[0]
lightpairs['light']=lightpairs.file.map(lighting)
lp=lightpairs.pivot_table(index=['pid','task'],columns='light',values='luma').dropna(subset=['Bright','Dim'])
delta=lp.Bright-lp.Dim
p(f'Ghép cùng mã người và cùng task trong Normal được {len(delta)} cặp Bright–Dim của {lp.index.get_level_values(0).nunique()} mã người. Chênh Bright trừ Dim có trung vị {fmt(delta.median())}/255; Bright sáng hơn ở {int((delta>0).sum())}/{len(delta)} cặp. Mỗi cặp có chung người nên không xem tất cả cặp là quan sát độc lập.')
sub('Độ sáng theo từng cặp bản ghi của cùng người')
p('Thang sáng từ 0 đến 255. Số dương ở cột cuối nghĩa là bản Bright sáng hơn bản Dim, số âm nghĩa là bản Dim sáng hơn.')
table(['Mã người','Task','Dim (0–255)','Bright (0–255)','Bright trừ Dim'],[[pid,task,row.Dim,row.Bright,row.Bright-row.Dim] for (pid,task),row in lp.iterrows()])

sub('Dùng nhãn nào cho từng phép đánh giá')
table(['Câu hỏi phân tích','BTC','UBFC','AiPEX'],[
['Ước lượng HR theo video','So PR ở cấp bản ghi; báo tách nhóm và protocol','Có thể tổng hợp HR tham chiếu trên cùng phần thời gian','So HR trung bình mỗi người; ghi rõ giả định clock'],
['Học dạng sóng hoặc chấm theo cửa sổ','Cần thêm waveform/timestamp tham chiếu','Có PPG; kiểm tra timestamp và hỗ trợ cửa sổ','Chỉ có HR trung bình'],
['SpO₂, BP, nhiệt độ','Đủ cột theo dòng; phân tầng mức giá trị và tình trạng thiếu video','SpO₂ chỉ ở UBFC1, khoảng quan sát hẹp; thiếu BP/T','Không thấy trong ground truth phát hành'],
['Kiểm ảnh đầu vào','Kiểm xoay, bitrate, HDR/bit depth, FPS và thời lượng','Có video toàn mặt; bảo toàn clock của bản ghi','ROI đã cắt; không đánh giá phát hiện toàn mặt bằng bộ này'],
['Đếm mẫu','Gộp theo người khi so nhóm hoặc chia tập','Giữ tách UBFC1/2 và phần dev/test','Ba ROI đi cùng nhau; nhóm theo địa điểm + mã người']])
h('6. Dữ liệu đối chiếu và cách tái lập')
p('Media thu bằng ffprobe trên P100, mỗi file một hàng. FPS là avg_frame_rate; bitrate video lấy từ stream, bitrate mux tính từ byte và thời lượng container. Tổng GB dùng 10⁹ byte. Phân vị dùng nội suy tuyến tính. Độ sáng ảnh lấy tại ba thời điểm; lux ngoài hiện trường, chuyển động toàn clip, độ nét vùng mặt và thời điểm đo từng vital của BTC chưa được xác minh.')
p('P100 · BTC: /mnt/samsung_t5/khangle/facevitals/datasets/FaceVitals_BTC_20261107/ · UBFC: /data_pool/khangle/facevitals/datasets/UBFC-rPPG/ · AiPEX: /data_pool/khangle/facevitals/datasets/AiPEX/.')
p('Snapshot PatientDetails_Normal.xlsx · SHA-256 '+sha+'. Bảng phân phối BTC chỉ xuất dữ liệu tổng hợp. Danh mục media và bảng tín hiệu đi kèm dùng mã bản ghi. Các số nhãn mô tả giá trị được phát hành, không xác nhận phương pháp đo lâm sàng.')
p('<a href="media.csv">Danh mục media</a> · <a href="btc_clock.csv">PTS video BTC</a> · <a href="ubfc_reference.csv">Nhãn và clock UBFC</a> · <a href="aipex_reference.csv">HR tham chiếu AiPEX</a> · <a href="../organiser_download/BTC_rows_reconciled.csv">Đối chiếu dòng và file BTC</a>.')
p('<a href="https://2027.ieeeicassp.org/sp-cup/">IEEE ICASSP 2027, tài liệu chính thức và phụ lục</a> · <a href="https://sites.google.com/view/ybenezeth/ubfcrppg">UBFC tác giả</a> · <a href="https://github.com/AiPEX-Lab/rppg_biases">AiPEX tác giả</a> · <a href="https://doi.org/10.1038/s41746-021-00462-z">Bài báo AiPEX</a> · <a href="https://docs.google.com/spreadsheets/d/1aZbks-8KG_mDc2Z7zwiirge0Fk-eexP5/edit">Bảng BTC</a>.')
style="""
:root {
  color-scheme: light;
  --paper: #ffffff; --ink: #18212b; --heading: #101923;
  --stripe: #f1f5f8; --head-bg: #dce8ee; --head-ink: #112535;
  --rule: #aabac4; --accent: #216171; --link: #004d72;
}
@media screen and (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --paper: #171c22; --ink: #f1f4f7; --heading: #ffffff;
    --stripe: #232b34; --head-bg: #324650; --head-ink: #ffffff;
    --rule: #5d6f7b; --accent: #87cbd6; --link: #9fdfff;
  }
}
html, body, main { color: var(--ink); background-color: var(--paper); }
body {
  margin: 36px auto; padding: 0 clamp(14px, 3vw, 32px); max-width: 1160px;
  font: 18px/1.62 "Segoe UI", Arial, sans-serif;
}
h1, h2, h3 { color: var(--heading); line-height: 1.3; }
h1 { font-size: clamp(1.65rem, 4vw, 2.25rem); }
h2 { margin-top: 2em; border-top: 2px solid var(--accent); padding-top: .6em; }
h3 { margin-top: 1.5em; }
p { overflow-wrap: anywhere; }
a, a:visited { color: var(--link); text-underline-offset: .15em; }
table { width: 100%; border-collapse: collapse; font-size: 16px; line-height: 1.5; font-variant-numeric: tabular-nums; }
th, td { text-align: left; vertical-align: top; padding: 9px 11px; border-bottom: 1px solid var(--rule); }
thead th { color: var(--head-ink); background-color: var(--head-bg); }
tbody td { color: var(--ink); background-color: var(--paper); }
tbody tr:nth-child(even) td { background-color: var(--stripe); }
.scroll { overflow-x: auto; margin: 12px 0 20px; }
img { display: block; max-width: 100%; height: auto; background-color: white; }
@media print {
  body { max-width: none; margin: 0; padding: 0; font-size: 11pt; }
  table { font-size: 9pt; }
  .scroll { overflow: visible; }
  tr, img, h3 { break-inside: avoid; }
  h2, h3 { break-after: avoid; }
}
"""
(out/'EDA_FaceVitals.html').write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EDA FaceVitals · BTC, UBFC-rPPG và AiPEX</title><style>'+style+'</style><main><h1>FaceVitals · Dữ liệu nào có thể trả lời câu hỏi nào?</h1>'+''.join(page)+'</main></html>',encoding='utf-8')
print('Media, labels, clocks and subject links verified.',len(media),len(matched),len(people))

