"""Read only A/N cells of the predeclared218 dev rows; never expose other rows."""
import json,zipfile,xml.etree.ElementTree as ET,re,hashlib
from pathlib import Path
C=Path(__file__).resolve().parent;P=C.parent/'inference_repair_20261113'
N={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};REL='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 jobs=[j for j in json.loads((P/'MANIFEST.json').read_text()) if j['dataset']=='BTC'];assert len(jobs)==218
 selected={g:{int(j['metadata']['sheet_row']):j for j in jobs if j['metadata']['group']==g} for g in ['Normal','Abnormal']}
 p=C/'source_workbook_opaque.xlsx';z=zipfile.ZipFile(p);rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'));links={x.attrib['Id']:x.attrib['Target'] for x in rels};wb=ET.fromstring(z.read('xl/workbook.xml'));sheets={s.attrib['name']:s.attrib[REL] for s in wb.find('m:sheets',N)};cells={};wanted_strings=set()
 for group in selected:
  target=links[sheets[group]];target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
  with z.open(target) as stream:
   for event,e in ET.iterparse(stream,events=['end']):
    if e.tag!=f"{{{N['m']}}}row":continue
    row=int(e.attrib['r'])
    if row in selected[group] or row==1:
     for cell in e.findall('m:c',N):
      col=re.sub(r'\d','',cell.attrib['r'])
      if col not in (['N'] if row==1 else ['A','N']):continue
      v=cell.find('m:v',N);text=v.text if v is not None else ''.join(t.text or '' for t in cell.findall('m:is/m:t',N));typ=cell.attrib.get('t','n');cells[(group,row,col)]=(typ,text)
      if typ=='s':wanted_strings.add(int(text))
    # Nonselected rows are discarded without inspecting cell values.
    e.clear()
 strings={}
 if wanted_strings:
  with z.open('xl/sharedStrings.xml') as stream:
   index=0
   for event,e in ET.iterparse(stream,events=['end']):
    if e.tag!=f"{{{N['m']}}}si":continue
    if index in wanted_strings:strings[index]=''.join(t.text or '' for t in e.findall('.//m:t',N))
    e.clear();index+=1
 def value(key):
  typ,text=cells.get(key,('n',None));return strings[int(text)] if typ=='s' else text
 rows=[];issues=[]
 for group,rs in selected.items():
  for row,j in rs.items():
   ident=value((group,row,'A'));pr=value((group,row,'N'));ident_norm=re.sub(r'\s+','',str(ident));prnum=float(pr) if pr not in [None,''] else None
   checks={'recording':ident_norm==re.sub(r'\s+','',j['recording']),'PR':prnum==j['metadata']['pr_bpm']}
   r=dict(recording=j['recording'],group=group,row=row,ID_cell=f'A{row}',PR_cell=f'N{row}',source_recording=ident,source_PR=prnum,expected_PR=j['metadata']['pr_bpm'],checks=checks,reference_time=None);rows.append(r)
   if not all(checks.values()):issues.append(r)
 receipt=dict(snapshot_sha256=sha(p),frozen_sha256='e52a9b190bf91f8a5d6d476b8374b8904f47328a668bfbe7cd60f3c14fafc49a',same_frozen_file=False,selected_dev_rows=218,holdout_rows_accessed=0,PR_headers={g:value((g,1,'N')) for g in selected},inconsistencies=issues,interpretation='Current separately versioned export. Hash differs; never substitutes frozen original workbook or changes reference labels. Selected-row agreement cannot verify byte identity of the missing original workbook.')
 (C/'CURRENT_ANNOTATION_SELECTED.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');(C/'CURRENT_ANNOTATION_AUDIT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
