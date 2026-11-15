from pathlib import Path
import re, html, shutil
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, PageBreak, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

root=Path(__file__).resolve().parent
src=root/'SP_Cup_Data_preprocessing_readable_20261108.md'
text=src.read_text(encoding='utf-8').replace('Ví dụ 10.012 mili giây = 10,012 giây. Dấu chấm ở số mili giây này phân cách hàng nghìn; dấu phẩy ở số giây là phần thập phân.', 'Ví dụ: 10012 mili giây ÷ 1000 = 10,012 giây.')
src.write_text(text,encoding='utf-8')
for name,file in [('Body','arial.ttf'),('BodyBold','arialbd.ttf'),('Mono','consola.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
styles={
 'p':ParagraphStyle('p',fontName='Body',fontSize=11,leading=16,spaceAfter=8,textColor=colors.HexColor('#111111'),allowWidows=0,allowOrphans=0),
 'h1':ParagraphStyle('h1',fontName='BodyBold',fontSize=22,leading=27,spaceAfter=15),
 'h2':ParagraphStyle('h2',fontName='BodyBold',fontSize=17,leading=22,spaceAfter=12,keepWithNext=True),
 'h3':ParagraphStyle('h3',fontName='BodyBold',fontSize=12.5,leading=17,spaceBefore=9,spaceAfter=7,keepWithNext=True),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=9.2,leading=12.5),
 'mono':ParagraphStyle('mono',fontName='Mono',fontSize=8.8,leading=13,backColor=colors.HexColor('#f2f4f5'),spaceBefore=4,spaceAfter=10),
}
def fmt(x):
    x=html.escape(x).replace('—','-').replace('→','->').replace('←','<-').replace('₂','<sub>2</sub>')
    x=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2" color="#005577">\1</a>',x)
    x=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',x)
    x=re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',x)
    x=re.sub(r'`([^`]+)`',r'<font name="Mono">\1</font>',x)
    return x

story=[]; out=[]; lines=text.splitlines(); i=0
while i<len(lines):
    line=lines[i]
    if not line.strip(): i+=1;continue
    if line.startswith('```'):
        block=[];i+=1
        while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
        raw='\n'.join(block)
        story.append(KeepTogether([Preformatted(raw,styles['mono'],maxLineLength=84)]))
        out.append('<pre>'+html.escape(raw)+'</pre>')
    elif line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            cells=[c.strip() for c in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r'[:\- ]+',c) for c in cells):rows.append(cells)
            i+=1
        i-=1
        count=len(rows[0]); weights=[max(9,len(c)) for c in rows[0]]
        # Avoid tiny columns in mixed text tables.
        if count==3:weights=[1.1,1.2,2.4]
        elif count==2:weights=[1,1.4]
        elif count==4:weights=[1.4,1,1,1.5]
        elif count==5:weights=[1.5,1,1,1,1]
        elif count==6:weights=[1.5,1,1,1,1,1]
        widths=[481.89*w/sum(weights) for w in weights]
        data=[[Paragraph(('<b>'+fmt(c)+'</b>') if r==0 else fmt(c),styles['cell']) for c in row] for r,row in enumerate(rows)]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8edf0')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#a4adb4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.extend([KeepTogether([t]) if len(rows)<10 else t,Spacer(1,10)])
        out.append('<table>'+''.join('<tr>'+''.join(('<th>' if j==0 else '<td>')+fmt(c)+('</th>' if j==0 else '</td>') for c in row)+'</tr>' for j,row in enumerate(rows))+'</table>')
    elif line.startswith('#'):
        depth=len(line)-len(line.lstrip('#'));val=line[depth:].strip()
        story.append(Paragraph(fmt(val),styles['h'+str(min(depth,3))]))
        if depth>1:out.append(f'<h{depth}>'+fmt(val)+f'</h{depth}>')
    else:
        vals=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','- ')):vals.append(lines[i]);i+=1
        i-=1;val=' '.join(vals)
        story.append(Paragraph(fmt(val),styles['p']));out.append('<p>'+fmt(val)+'</p>')
    i+=1
def footer(c,d):
    c.setFont('Body',9);c.setFillColor(colors.HexColor('#555555'));c.drawRightString(538,27,str(d.page))
pdf=root/'SP_Cup_Data_preprocessing.pdf'
SimpleDocTemplate(str(pdf),pagesize=(595.28,841.89),rightMargin=56.69,leftMargin=56.69,topMargin=45,bottomMargin=45,title='SP Cup - Cấu trúc và cách đọc dữ liệu',author='FaceVitals').build(story,onFirstPage=footer,onLaterPages=footer)
(root/'SP_Cup_Data_preprocessing.md').write_text(text,encoding='utf-8')
(root/'SP_Cup_Data_preprocessing_readable_20261108.html').write_text('<!doctype html><meta charset="utf-8">'+''.join(out),encoding='utf-8')
print({'pdf_bytes':pdf.stat().st_size,'markdown_chars':len(text),'html_chars':sum(map(len,out))})
