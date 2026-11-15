"""Render the explanatory reading edition from the measured EDA tables."""
from pathlib import Path
from html import escape
import tempfile
import subprocess
from lxml import html
import pymupdf

root=Path(__file__).resolve().parent
source=html.fromstring((root/'EDA_FaceVitals.html').read_text(encoding='utf-8'))
tables=source.xpath('//table')
assert len(tables)==26
md=[]
web=[]
def title(t):
    md.append('# '+t);web.append('<h1>'+escape(t)+'</h1>')
def section(t):
    md.append('## '+t);web.append('<h2>'+escape(t)+'</h2>')
def heading(t):
    md.append('### '+t);web.append('<h3>'+escape(t)+'</h3>')
def para(t):
    md.append(t);web.append('<p>'+escape(t)+'</p>')
def tree(t):
    md.append('```text\n'+t+'\n```');web.append('<pre>'+escape(t)+'</pre>')
def tab(rows):
    lines=[]
    for i,r in enumerate(rows):
        lines.append('| '+' | '.join(c.replace('|',r'\|') for c in r)+' |')
        if not i:lines.append('| '+' | '.join('---' for c in r)+' |')
    md.append('\n'.join(lines))
    web.append('<table><thead><tr>'+''.join('<th>'+escape(c)+'</th>' for c in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(c)+'</td>' for c in row)+'</tr>' for row in rows[1:])+'</tbody></table>')
def link(t,url):
    md.append('['+t+']('+url+')');web.append('<p><a href="'+escape(url,quote=True)+'">'+escape(t)+'</a></p>')

title('Đọc hiểu ba bộ dữ liệu FaceVitals')
para('Khi đưa một video khuôn mặt vào mô hình, ta muốn biết con số mô hình dự đoán có gần số đo bằng thiết bị hay không. Để so sánh đúng, trước hết phải biết thiết bị cho một số cho cả video, hay cho một chuỗi số theo thời gian. Ba bộ dữ liệu trong báo cáo khác nhau chủ yếu ở điểm này.')
heading('Một người, một lần quay và một nhãn là ba thứ khác nhau')
para('Hãy hình dung một người được quay hai lần, lúc nghỉ và lúc làm nhiệm vụ. Ta có một người và hai bản ghi. Nếu chia mỗi bản ghi thành ba đoạn 10 giây, ta có sáu đoạn video. Số lần đo của thiết bị vẫn chỉ bằng số lần nó thực sự đo. Việc cắt video không sinh thêm sáu nhãn tham chiếu độc lập.')
tab([['Bộ dữ liệu','Khi mở một bản ghi, bạn thấy gì?','Có thể đối chiếu điều gì?'],['BTC (bệnh viện)','Một video; một dòng Excel ghi mạch, SpO₂, huyết áp và nhiệt độ.','Một dự đoán tổng hợp của video với số BTC ghi cho lần quay đó.'],['UBFC-rPPG (webcam)','Video toàn mặt và file chứa nhiều số đo liên tiếp, kèm thời điểm.','Biến động tín hiệu và nhịp tim trong một đoạn đã ghép đúng thời gian.'],['AiPEX (điện thoại)','Ba video nhỏ quay cùng lúc ở trán, má trái, má phải; một số nhịp tim trung bình.','Một kết quả cho cả lần quay với một nhãn nhịp tim.']])
para('Ví dụ thực tế: AiPEX có 264 file nhưng chỉ có 88 người. Ngược lại, ở UBFC1, riêng một giây của một bản ghi có thể có tới 62 giá trị tín hiệu. Vì vậy khi gặp chữ “mẫu”, hãy nhìn đơn vị đang đếm: người, lần quay, file, khung hình hay số đo cảm biến.')
para('EDA là bước khảo sát dữ liệu trước khi huấn luyện hoặc chấm mô hình. Phần chính dưới đây hướng dẫn cách đọc file và nhãn. Các bảng chi tiết được đặt ở phụ lục để bạn có thể tra lại từng kết luận. Số liệu đo trên bản lưu P100 ngày 07/11/2026.')

section('1. Bộ của BTC: lần theo một video đến dòng Excel')
para('Bộ do ban tổ chức (BTC) cung cấp đến từ Sri Ramachandra Hospital ở Ấn Độ. Có hai thư mục video mang tên Normal và Abnormal. Một bảng Excel riêng ghi số đo. Tên nhóm là cách BTC phân loại bản ghi; để biết mạch hay huyết áp bao nhiêu, hãy đọc cột số đo.')
tree('FaceVitals_BTC_20261107/\n  Normal/<mã file Drive>/S_N1_Resting.mov\n  Abnormal/<mã file Drive>/<tên bản ghi>.MOV\n\nBảng riêng: PatientDetails_Normal.xlsx\n  Tab Normal   → ID, Video Link, PROTOCOL, PR, SpO2, BP-SYS, BP-DIA, …\n  Tab Abnormal → cùng các trường đo chính')
para('Trong tên S_N1_Resting, S_N1 là mã người trong nhóm Normal, còn Resting nghĩa là lúc nghỉ. Video này nằm ở dòng 2 của tab Normal. Nó dài 30,765 giây và có 922 khung hình. Chia 922 cho 30,765 được khoảng 29,97 hình/giây. Nếu lấy ba đoạn 10 giây để dự đoán mạch, cách dùng nhãn phù hợp là tổng hợp ba dự đoán thành một kết quả rồi đối chiếu với PR của dòng 2. Bảng BTC chỉ cho một PR ở dòng đó.')
heading('Vì sao có 296 video nhưng chỉ 293 cặp để chấm?')
para('Normal có 197 video. Trong đó 194 video có dòng Excel tương ứng; ba video còn lại là ba nhiệm vụ quay trong ánh sáng Dim của S_N43. Abnormal có 101 dòng Excel, nhưng chỉ có file cho 99 dòng. Vậy có 194 + 99 = 293 cặp video–nhãn. Hai bản ghi còn thiếu vẫn thuộc các mã người đã có một video khác, nên 293 cặp này bao phủ đủ 202 mã người trong bảng.')
para('Một trường hợp đã sửa khi đối chiếu: link ở dòng Normal 70 dẫn đến bản Dim, trong khi ID ghi S_N46_Bright_Task2. Trong thư mục có file Bright đúng tên, nên dòng này vẫn ghép được. Các tên Task1/2/3 cho biết số nhiệm vụ. Chỉ mô tả hành động cụ thể khi có ghi rõ trong cột PROTOCOL, chẳng hạn Walking là đi bộ.')
heading('Các con số cho biết gì về hai nhóm?')
para('Trong 194 video Normal ghép nhãn, 105 video có PR từ 80 đến dưới 100 nhịp/phút, tức khoảng 54 trong mỗi 100 video. Ở nhóm Abnormal, 42/99 video có SpO₂ dưới 95%, so với 8/194 ở Normal. Bộ BTC vì vậy cho phép xem mô hình hoạt động thế nào ở các mức SpO₂ thường gặp và ít gặp. Đây là số đếm trên bộ dữ liệu, không dùng để suy tỷ lệ trong dân số.')
para('Điều kiện ghi hình cũng khác: 98/99 video Abnormal có nhãn được ghi lúc nghỉ. Con số tương ứng của Normal là 78/194. Nếu thấy hai nhóm có sai số khác nhau, có thể so riêng các bản Resting trước; nếu gộp tất cả, kết quả còn phụ thuộc sự pha trộn giữa nghỉ và làm nhiệm vụ.')

section('2. UBFC-rPPG: video đi cùng một chuỗi số đo')
para('Người tham gia ngồi trước webcam Logitech C920. Máy đo mạch CMS50E ghi tín hiệu tiếp xúc. UBFC1 chủ yếu yêu cầu ngồi yên; UBFC2 cho người tham gia chơi trò tính toán có giới hạn thời gian. Hai phần dùng cách lưu nhãn khác nhau.')
tree('UBFC-rPPG/\n  DATASET_1/10-gt/\n    vid.avi          video khuôn mặt\n    gtdump.xmp       mỗi dòng là một thời điểm đo\n  DATASET_2/subject1/\n    vid.avi          video khuôn mặt\n    ground_truth.txt ba dòng dài, mỗi dòng là cả một chuỗi')
para('UBFC1: đọc bốn cột của gtdump.xmp theo thứ tự thời điểm (mili giây), nhịp tim, SpO₂ và biên độ PPG. Ví dụ thật từ bản 10-gt, sau khi đổi thời gian sang giây: ở 10,012 giây, nhịp tim ghi 74 bpm, SpO₂ là 97% và PPG bằng 40. Đến 10,028 giây, PPG xuống 38, còn hai con số kia vẫn là 74 và 97. PPG là độ lớn tín hiệu quang ở từng thời điểm. Nhịp tim là tốc độ lặp của các nhịp trong tín hiệu. Con số PPG bằng 40 vì vậy không thể đọc là 40 nhịp/phút.')
para('UBFC2: dòng thứ nhất của ground_truth.txt chứa chuỗi PPG, dòng thứ hai chứa HR thiết bị và dòng thứ ba chứa thời điểm tính bằng giây. Muốn lấy các số đo từ giây 10 đến trước giây 20, trước hết tìm các vị trí thỏa điều kiện ở dòng thời gian, sau đó lấy đúng vị trí ấy ở hai dòng còn lại. UBFC2 không cung cấp cột SpO₂ như UBFC1.')
heading('Vì sao phải nhìn thời gian trước khi chấm nhịp tim?')
para('Tổng 8 bản UBFC1 có 18.538 khung hình nhưng có 40.273 lần cảm biến lấy mẫu. Nếu cứ ghép ảnh thứ 100 với số đo thứ 100, hai bên có thể thuộc hai thời điểm khác nhau. Việc cần làm là đối chiếu theo mốc thời gian của từng nguồn. UBFC2 cũng cần kiểm tra: trong 35 bản phát triển, 8 bản có ít nhất một bước thời gian đứng yên hoặc đi lùi.')
para('Trong 35 bản UBFC2 được mở, bốn bản có giá trị HR thiết bị xuống tới 1 bpm. Ở subject20, trung vị của dòng HR là 7 bpm. Khi dùng bộ này làm chuẩn đánh giá, cần ghi rõ nhịp tham chiếu lấy từ dòng HR thiết bị hay tính từ PPG trên khoảng thời gian hợp lệ. Chọn hai cách khác nhau có thể cho hai chuẩn chấm khác nhau.')
para('P100 lưu đủ 8 video UBFC1 và 42 video UBFC2. Đội chỉ khảo sát 35 video phát triển của phần 2. Bảy video kiểm thử được dành riêng để đánh giá sau cùng. Thống kê UBFC2 trong báo cáo này dùng mẫu số 35.')

section('3. AiPEX: ba góc nhìn nhỏ cùng dùng một nhãn')
para('Đây là bộ “RPPG Dataset” mà phụ lục cuộc thi dẫn tới. Mỗi người đóng góp một lần quay. Để chia sẻ dữ liệu, tác giả xuất riêng ba vùng trên mặt. Cụm “ba vùng” ở đây là ba phần của cùng một khuôn mặt trong cùng khoảng thời gian.')
tree('AiPEX/\n  Data/India/video1_forehead.avi      trán\n  Data/India/video1_leftcheek.avi     má trái\n  Data/India/video1_rightcheek.avi    má phải\n  Data/Sierra_Leone/…                 cũng có mã video1, video2, …\n  Ground_Truth/India_GT.csv           mã video và HR trung bình\n  Ground_Truth/Sierra_Leone_GT.csv    bảng riêng cho Sierra Leone')
para('Ví dụ thật: video1 ở India có 480 khung hình cho mỗi vùng; file phát ở 15 hình/giây. Mỗi vùng vì thế dài 480 ÷ 15 = 32 giây. Cả ba vùng cùng ứng với nhãn 85,413 bpm trong India_GT.csv. Lần quay kéo dài 32 giây theo file phát hành; cộng thành 96 giây sẽ đếm cùng khoảng thời gian ba lần. Khi ghép nhãn, phải dùng cả India và video1 vì bên Sierra Leone cũng có một video1.')
para('Tác giả mô tả cắt vùng trán 60×30 điểm ảnh và mỗi má 25×25 trên ảnh gốc. Các file phát hành đều có kích thước 200×200. Kích thước vùng lấy trên mặt và kích thước khung hình trong file xuất là hai bước khác nhau. Với đầu vào này, ta có thể khảo sát tín hiệu ở trán và má; bài toán tìm khuôn mặt trong video toàn cảnh cần nguồn video khác.')
heading('Một số HR trung bình cho phép kết luận đến đâu?')
para('Có 44 người ở mỗi nước. Toàn bộ 88 nhãn HR nằm từ 65,72 đến 92,75 bpm. Một kết quả tốt trên AiPEX cho biết mô hình làm được gì trong dải này. Muốn đánh giá mạch 120 bpm, phải tìm bản ghi có nhãn ở mức đó, chẳng hạn trong bộ BTC.')
para('Bài báo mô tả camera thu trung bình khoảng 25 hình/giây và số đo tham chiếu được lấy trung bình trong 120 giây. File ROI phát hành lại khai báo 15 hình/giây. Chưa xác minh được đoạn nào của file tương ứng chính xác 120 giây đo. Vì vậy, một nhãn cho cả người có thể dùng để đối chiếu một kết quả tổng hợp, nhưng chưa đủ để biết HR đúng ở riêng giây thứ 10.')

section('4. Cách đọc ký hiệu, đơn vị và chỉ số')
heading('Nhãn sinh lý: thiết bị đã đo đại lượng nào?')
tab([['Ký hiệu trên file','Nghĩa và đơn vị','Ví dụ đọc'],['HR / PR / bpm','HR là nhịp tim; PR là nhịp mạch ngoại vi. Cả hai ghi bằng bpm, tức số nhịp mỗi phút.','BTC ghi PR nên báo cáo gọi là mạch. PR 87 nghĩa là 87 nhịp mạch/phút.'],['PPG / rPPG','PPG là tín hiệu quang thay đổi theo mạch máu. rPPG là cách ước lượng tín hiệu ấy từ xa qua video.','UBFC cung cấp PPG tiếp xúc để đối chiếu với tín hiệu dự đoán từ ảnh mặt.'],['SpO₂ (%)','Độ bão hòa oxy ngoại vi theo máy đo, ghi bằng phần trăm.','Dự đoán 96% so với nhãn 98% chênh 2 điểm phần trăm.'],['BP-SYS / BP-DIA','Huyết áp tâm thu / tâm trương. Đơn vị thông dụng là mmHg (milimét thủy ngân); tiêu đề BTC chỉ ghi BP.','120/72 là hai số đi cùng nhau: 120 tâm thu và 72 tâm trương.'],['T (°F)','Nhiệt độ theo độ Fahrenheit. Đổi sang độ C bằng (°F − 32) × 5/9.','97 °F tương đương khoảng 36,1 °C.'],['a.u.','Đơn vị tương đối của tín hiệu. Mỗi cách xuất tín hiệu có thể dùng một thang riêng.','PPG trung bình 40,82 ở UBFC1 không thể so trực tiếp với 0,11 ở UBFC2 để nói mạch mạnh hơn.']])
heading('Thống kê mô tả: một số đứng ở đâu trong nhóm?')
para('Đọc một dòng thật: 197 video BTC Normal có Min 5, P25 28,86, trung vị 30,66, trung bình 28,16, P75 30,99 và Max 40,96 giây. Min và Max là ngắn nhất và dài nhất. Trung vị là mốc giữa sau khi xếp video theo thời lượng. P25 và P75 là các mốc 25% và 75%: khoảng một nửa số video tập trung từ 28,86 đến 30,99 giây. Trung bình (TB) là tổng thời lượng chia 197; các video rất ngắn kéo số này xuống 28,16 giây.')
para('n hay N là số đối tượng trong phép tính. Dòng Normal có thể mang n = 197 khi hỏi “video dài bao lâu”, n = 194 khi hỏi “bao nhiêu video có nhãn PR”, hoặc n = 103 khi hỏi “có bao nhiêu mã người”. Dấu < là nhỏ hơn, ≤ là nhỏ hơn hoặc bằng, ≥ là lớn hơn hoặc bằng. Ví dụ “80 đến dưới 100” gồm 80 nhưng không gồm 100.')
heading('Metric đánh giá mô hình khác gì số mô tả bộ dữ liệu?')
para('MAE là sai số tuyệt đối trung bình. Ví dụ minh họa: hai video có nhãn 80 và 100 bpm; mô hình dự đoán 84 và 94. Sai số tuyệt đối là 4 và 6; MAE = (4 + 6) ÷ 2 = 5 bpm. Số càng nhỏ thì dự đoán càng sát nhãn. P95 sai số là mốc mà khoảng 95% sai số nằm bằng hoặc dưới nó, dùng để nhìn những trường hợp khó. Đây là ví dụ giải thích cách chấm. Báo cáo EDA này mô tả dữ liệu và chất lượng nhãn, không đo MAE của mô hình.')
para('Cũng cần chọn đơn vị lấy trung bình. Nếu một người quay chín video và một người quay một video, tính MAE theo video khiến người thứ nhất có trọng số gấp chín lần. Muốn hai người có trọng số bằng nhau, trước hết lấy sai số trung bình của từng người, sau đó mới trung bình hai con số ấy.')

section('5. Cách đọc các thông số video trong phụ lục')
tab([['Từ gặp trong bảng','Đọc theo cách này'],['Frame, FPS, Hz','Frame là một ảnh; FPS là số ảnh trong một giây. Hz là số lần mỗi giây, thường dùng cho cảm biến. Camera 30 fps và máy đo khoảng 62 Hz có số mẫu khác nhau dù chạy cùng thời lượng.'],['Timestamp, clock, PTS, Δt, gap','Timestamp là mốc thời gian. Clock là trục thời gian của nguồn. PTS là lúc khung hình được trình chiếu. Δt là khoảng giữa hai mốc liền nhau. Gap là khoảng cách dài bất thường theo ngưỡng ghi trong bảng. 1.000 ms = 1 giây.'],['Pixel, độ phân giải, ROI, RGB','Pixel là một điểm ảnh; 1920×1080 là rộng 1.920 điểm, cao 1.080 điểm. ROI là vùng được chọn trên ảnh, như vùng trán. RGB là ba kênh màu đỏ, lục, lam.'],['MOV, MP4, AVI; codec','MOV, MP4 và AVI là dạng tệp chứa video. Codec là cách mã hóa hình bên trong. Ví dụ file BTC có thể mang đuôi MOV nhưng dùng codec HEVC. H.264 và HEVC là các cách nén; FFV1 nén bảo toàn giá trị lúc mã hóa; rawvideo lưu hình chưa nén.'],['Bitrate, kb/s, Mb/s, GB','Bitrate cho biết trung bình bao nhiêu dữ liệu dùng cho một giây. 0,5 Mb/s bằng 500 kb/s. GB đo dung lượng cả tệp (1 GB = một tỷ byte trong báo cáo).'],['C5','Điều kiện thử của dự án: đưa video về mục tiêu 10 hình/giây với bitrate đo được tối đa 500 kb/s. Hai điều kiện này mô tả bản tái mã hóa.'],['Header, rotation, audio','Header là thông tin file khai báo; rotation là góc phải xoay để hiển thị đúng; audio là luồng âm thanh. Kích thước ghi trong file có thể khác hướng bạn nhìn khi mở.'],['Luma, độ tương phản, lux','Luma trong EDA là mức sáng của ảnh xám từ 0 (đen) đến 255 (trắng). Độ tương phản ở đây là độ phân tán mức sáng giữa các điểm ảnh. Lux đo độ rọi vật lý tại hiện trường.'],['YUV, bgr24, bgr0, 8/10-bit, HLG','Đây là các quy ước lưu và diễn giải màu. YUV tách thành phần sáng và màu; BGR lưu các kênh màu theo thứ tự lam, lục, đỏ. 10-bit có nhiều mức mã hơn 8-bit. HLG là cách mã hóa dải sáng rộng.'],['Dev, test, protocol, ID, SHA-256','Dev là phần dùng phát triển; test là phần để kiểm thử độc lập. Protocol là điều kiện hoặc hoạt động khi quay. ID là mã nhận diện. SHA-256 là dấu kiểm tra nội dung tệp, dùng so bản sao.']])
para('Ví dụ: hai bản ghi cùng dài 30 giây có thể có dung lượng rất khác do độ phân giải, codec hoặc bitrate. Chỉ nhìn kích thước file không đủ kết luận video dài hơn, hình sáng hơn hay mô hình sẽ đo mạch tốt hơn.')

# Each entry binds a verified source table to its reader-facing question and interpretation.
notes=[
('Bộ nào có nhãn cho câu hỏi của mình?','Nếu cần biết tín hiệu thay đổi theo từng giây, hãy bắt đầu với UBFC. Nếu cần đối chiếu huyết áp theo một lần quay, bảng BTC có hai cột SYS và DIA.'),
('Một lần quay thường dài bao nhiêu?','Con số 429,17 giây của India là khoảng 7 phút 9 giây phát. AiPEX có ba ROI đồng thời nên chỉ cộng thời lượng một vùng. BTC Normal và Abnormal có tổng lần lượt 92,45 và 51,46 phút.'),
('Đếm file, dòng nhãn hay người?','Đọc ngang dòng Abnormal: 101 hàng nhãn thuộc 99 mã người. Có video khớp cho 99 hàng và cả 99 mã người đều còn đại diện. Vì hai người có thêm lần quay nên có thể thiếu hai video mà vẫn đủ số mã người.'),
('Video BTC được lưu bằng cách nào?','HEVC (122) ở dòng Normal nghĩa là có 122 file dùng HEVC. Bộ ba FPS được đọc theo thứ tự thấp nhất, trung vị, cao nhất. Mức 240,37 là giá trị cao nhất trong file, chưa xác minh được tốc độ thu của camera.'),
('Kích thước khung hình nào thường gặp?','Ở kích thước 1920×1080 có 117 video Normal và 51 video Abnormal. Mỗi ô là số video. Hệ thống đọc ảnh phải chấp nhận nhiều kích thước thay vì giả định mọi file đều cùng độ phân giải.'),
('Video gốc có sẵn điều kiện nén C5 không?','Cả hai nhóm đều có 0 file ở mỗi tiêu chí 10 fps và bitrate ≤0,5 Mb/s. Cần tạo bản nén nếu muốn thử C5. BT.709, SMPTE 170M và ARIB STD-B67 là mã chỉ cách diễn giải màu hoặc độ sáng được file khai báo.'),
('Có bao nhiêu video đủ dài để cắt đoạn?','Trong 197 file Normal, 195 file đủ 10 giây và chỉ 143 file đủ 30 giây. Có thể cắt 517 đoạn 10 giây rời nhau nếu chỉ xét độ dài. Gap >2× là bước thời gian dài hơn hai lần bước trung vị của chính file đó. Packet ở đây là gói dữ liệu hình được đọc để kiểm tra PTS.'),
('Nhãn nào được ghi và nằm trong khoảng nào?','Trong 194 hàng Normal, SpO₂ nhỏ nhất là 85%, lớn nhất là 100%, trung vị 98%. Cột Thiếu bằng 0 nghĩa là các ô của biến đó đều có số, không xác nhận chất lượng của phép đo. Bảng này gồm đủ 295 dòng, kể cả hai dòng thiếu video.'),
('Nếu yêu cầu cả nhãn lẫn video, kết quả đổi thế nào?','Số dòng Abnormal giảm từ 101 xuống 99; PR trung vị chuyển từ 82 sang 83 bpm. Số 7 ở cột SpO₂ <90 nghĩa là bảy bản ghi có nhãn dưới 90%, không phải 7% hay bảy điểm sai số.'),
('Có bao nhiêu video ở mỗi khoảng mạch?','105/194 = 54,12% video Normal có PR từ 80 đến dưới 100 bpm. Nhóm Abnormal có 41/99 = 41,41% ở cùng khoảng. Phần trăm luôn chia cho số video của chính nhóm đó.'),
('Các mức SpO₂ xuất hiện nhiều đến đâu?','Ở Normal, 138/194 video có SpO₂ từ 98% trở lên, nhưng chỉ một video dưới 90%. Khi báo sai số ở nhóm dưới 90%, một video này cung cấp rất ít thông tin về khả năng tổng quát.'),
('Các mức huyết áp tâm thu xuất hiện nhiều đến đâu?','Mức từ 160 trở lên có 2/194 video Normal và 10/99 video Abnormal. Đây là 12 video có nhãn nằm trong khoảng đó; bảng dùng ngưỡng để đếm, không phân loại bệnh.'),
('Một điều kiện quay có bao nhiêu bản ghi?','Female (116) là 116 hàng Normal, có thể lặp một người. Cột ánh sáng dùng tất cả 197 file Normal nên có mẫu số khác. Bright, Med, Dim lần lượt là sáng, trung bình và tối theo tên file; Dark, Medium Fair, Fair là các mức màu da do nguồn ghi.'),
('Ai đóng góp nhiều hơn một bản ghi?','Ở Normal, “9 (2)” nghĩa là hai mã người, mỗi mã có chín dòng. Hai mã Normal có thông tin màu da thay đổi giữa các dòng. Cần kiểm tra các mã này trước khi coi mỗi người thuộc duy nhất một nhóm màu da.'),
('Khi mỗi mã người chỉ tính một lần thì sao?','Normal có 61 mã Female, dù bảng theo dòng đếm 116 bản ghi Female. Báo cáo lấy dòng đầu tiên của mỗi mã cho các cột theo người. “Điền bệnh sử” chỉ đếm ô có nội dung, không biểu thị số người mắc bệnh.'),
('Đang so lúc nghỉ hay khi làm nhiệm vụ?','Trong phần có video, Abnormal có 98 bản lúc nghỉ và một bản đi bộ. Với Normal, Task2 có 37 video của 25 mã người; trung vị PR là 95 bpm. Một người có thể xuất hiện nhiều lần ở cùng nhiệm vụ.'),
('Hai nhóm có phân bố tuổi giống nhau không?','Normal có 59/103 mã người ở tuổi 18–34. Abnormal có 12/99 mã người cùng khoảng tuổi. Nếu thấy chênh lệch giữa hai nhóm, độ tuổi là một khác biệt có sẵn cần xét cùng hoạt động và nhãn.'),
('Cỡ mẫu có đủ khi tách theo màu da?','Abnormal chỉ có bốn mã Fair và bốn video; mỗi người ảnh hưởng lớn đến trung bình của nhóm nhỏ này. Với Normal Dark, 41 mã được gán nhóm theo dòng đầu nhưng có 65 video mang nhãn Dark theo dòng.'),
('Ở UBFC, một hình ảnh ứng với bao nhiêu số đo?','UBFC1 có 40.273 mẫu PPG trong 8 video. Cột HR thiết bị min–max lấy số nhỏ nhất và lớn nhất trong chuỗi, còn cột median HR lấy trung vị từng file trước rồi mới tìm khoảng. Hai cột trả lời hai câu hỏi khác nhau.'),
('Có bao nhiêu chỗ bất thường trên trục thời gian?','UBFC2 có tổng cộng 117 khoảng cách dài hơn hai lần bước trung vị, rải trên 29 bản ghi. Con số 117 đếm bước thời gian, còn 29 đếm file. Cả hai cùng mô tả mốc thời gian của nhãn.'),
('Trong một giây, PPG và HR khác nhau thế nào?','Ở giây 10–11 của 10-gt, 62 lần đo PPG trải từ 15 đến 83, trung bình 40,82; HR trung vị là 74 bpm. Biên độ dao động vì tín hiệu thay đổi trong một nhịp. Muốn xem từng lần đo, mở reference_example.csv.'),
('Với AiPEX, mỗi nơi thu có bao nhiêu nhãn?','India có 44 người và 132 file ROI, nhưng có 44 nhãn HR. Số 268,47 là tổng phút phát khi đếm một lần trên mỗi người. Nhóm này có HR trung bình 76,21 bpm; mức đó mô tả nhãn, không phải sai số mô hình.'),
('AiPEX bao phủ thời lượng và mức mạch nào?','Có 30/44 bản India phát dài hơn 300 giây (5 phút). Cả hai nhóm đều có 0 bản HR trên 100 bpm. Dữ liệu này không trả lời được mô hình có đo tốt ở mức 120 bpm hay không.'),
('Độ sáng file được đo bằng gì?','Mỗi file lấy ba ảnh ở 10%, 50% và 90% thời lượng. Normal Bright có trung vị mức sáng toàn ảnh 143,34/255. Cột “% pixel tối” đếm phần trăm điểm ảnh có mã ≤10 rồi lấy trung bình. Nền, camera và vùng chọn đều tác động lên chỉ số này.'),
('So cùng người, cùng task, Bright sáng hơn được bao nhiêu?','Với S_N42 ở Task1, chênh lệch là 146,01 − 136,85 = +9,16. Với S_N43 ở Task3, chênh lệch là −10,62. Trên 18 cặp của 6 người, Bright sáng hơn ở 10 cặp, trung vị chênh lệch +1,47 trên thang 255. Tên điều kiện không đủ để dự báo chắc độ sáng toàn khung.'),
('Chọn bộ nào theo mục đích đánh giá?','Để chấm mạch trên BTC, gộp các đoạn về một số mỗi video. Để học tín hiệu theo thời gian, ghép video UBFC với PPG đúng thời điểm. Với AiPEX, giữ ba vùng mặt của một người cùng một phía khi chia dữ liệu.')
]
groups={1:'A. Quy mô và thời lượng',3:'B. File video BTC',8:'C. Nhãn sinh lý của BTC',13:'D. Người và hoạt động trong bộ BTC',19:'E. Tín hiệu UBFC',22:'F. Nhãn AiPEX',24:'G. Ánh sáng và lựa chọn bộ đánh giá'}
assert len(notes)==len(tables)
for i,(t,(question,example)) in enumerate(zip(tables,notes),1):
    if i in groups:section('Phụ lục '+groups[i])
    heading(f'Bảng {i}. '+question)
    para(example)
    tab([[c.text_content() for c in r] for r in t.xpath('.//tr')])
section('Nguồn và phạm vi đối chiếu')
para('Đã đọc thông tin kỹ thuật của 603 file video trên P100: 296 BTC, 43 UBFC thuộc phần mở và 264 file vùng mặt của AiPEX. Phân bố nhãn BTC lấy từ hai tab Normal và Abnormal của PatientDetails_Normal.xlsx. Mục Copy of Normal được bỏ để tránh đếm trùng. Nội dung báo cáo chỉ trình bày mã bản ghi hoặc số đếm tổng hợp.')
para('ITA là một chỉ số ước lượng màu da từ màu điểm ảnh; bảng nhãn gốc AiPEX không có trường ITA. Phân tích trước đây của nhóm ước lượng được chỉ số này cho 77/88 người. Khi so theo nơi thu, India và Sierra Leone chỉ ghi vị trí thu thập; quốc gia, màu da do nguồn ghi và độ sáng của ảnh là ba thuộc tính riêng.')
for label,url in [('Tài liệu cuộc thi và phụ lục','https://2027.ieeeicassp.org/sp-cup/'),('Mô tả UBFC-rPPG của tác giả','https://sites.google.com/view/ybenezeth/ubfcrppg'),('Kho AiPEX và cách đọc nhãn','https://github.com/AiPEX-Lab/rppg_biases'),('Bài báo về thu nhận dữ liệu AiPEX','https://doi.org/10.1038/s41746-021-00462-z'),('Bảng ban tổ chức','https://docs.google.com/spreadsheets/d/1aZbks-8KG_mDc2Z7zwiirge0Fk-eexP5/edit'),('Danh mục video đã đo','media.csv'),('Đối chiếu từng dòng của BTC','../organiser_download/BTC_rows_reconciled.csv'),('Ví dụ UBFC ở từng thời điểm','reference_example.csv')]:link(label,url)

css="""@page{size:A4;margin:15mm 15mm 16mm;@bottom-right{content:counter(page);font:9pt Arial;color:#222}}
body{font:12pt/1.5 'Segoe UI',Arial,sans-serif;color:#101010;background:white;margin:0}
h1{font-size:25pt;line-height:1.18}h2{font-size:18pt;line-height:1.2;break-before:page}h3{font-size:13pt;line-height:1.28;margin:13pt 0 4pt}h1,h2,h3{break-after:avoid}p{margin:5pt 0 8pt;orphans:2;widows:2}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:10pt/1.45 Consolas,monospace;background:#f3f5f6;border-left:2pt solid #4a5860;padding:7pt;break-inside:avoid}
table{width:100%;border-collapse:collapse;font:9.6pt/1.32 'Segoe UI',Arial,sans-serif;margin:5pt 0 10pt}th,td{border:.55pt solid #a0a6a9;padding:3.5pt;vertical-align:top;text-align:left}th{background:#e9edef}thead{display:table-header-group}tr{break-inside:avoid}a{color:#12374b}"""
tmp=Path(tempfile.mkdtemp(prefix='eda-guide-'))
inp=tmp/'guide.html'
inp.write_text('<!doctype html><meta charset="utf-8"><base href="'+root.as_uri()+'/"><title>FaceVitals · Hướng dẫn đọc ba bộ dữ liệu</title><style>'+css+'</style>'+''.join(web),encoding='utf-8')
pdf=tmp/'guide.pdf'
subprocess.run([r'C:\Program Files\Google\Chrome\Application\chrome.exe','--headless','--disable-gpu','--no-first-run','--user-data-dir='+str(tmp/'profile'),'--no-pdf-header-footer','--print-to-pdf='+str(pdf),inp.as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
with pymupdf.open(pdf) as doc:
    from PIL import Image
    preview=Image.new('RGB',(240*5,345*((len(doc)+4)//5)),'#cfcfcf')
    flat=' '.join(p.get_text() for p in doc)
    for n in range(1,27): assert f'Bảng {n}.' in flat
    for i,page in enumerate(doc):
        words=page.get_text('words')
        assert words and all(w[0]>=0 and w[2]<=page.rect.width+1 for w in words)
        pix=page.get_pixmap(matrix=pymupdf.Matrix(.40,.40));preview.paste(Image.frombytes('RGB',(pix.width,pix.height),pix.samples),(i%5*240,i//5*345))
    print('Pages and word counts:',[(i+1,len(p.get_text('words'))) for i,p in enumerate(doc)])
    preview.save(Path(tempfile.gettempdir())/'facevitals_guide_qa.png')
for stem in ['EDA_FaceVitals','EDA_FaceVitals_GiaiThich']:
    (root/(stem+'.pdf')).write_bytes(pdf.read_bytes())
    (root/(stem+'.md')).write_text('\n\n'.join(md)+'\n',encoding='utf-8')

