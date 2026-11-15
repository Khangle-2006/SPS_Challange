"""Descriptive analysis only: no fitting, media decoding, or holdout access."""
import ast,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr,spearmanr
P=Path(__file__).resolve().parent;B=P.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
node=next(n for n in ast.parse((B/'evaluate.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='summarize')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(B/'evaluate.py'),'exec'))
manifest=read(B.parent/'round1_v4_20261105/btc_zeroshot/dev_manifest.json')
assert manifest['split_sha256']=='97944f717b7fc166bb8a0e91c791c7ad6b7e451cffed564f40232dde3ec8e221'
lumas={r['recording']:r for r in read(P/'ROI_LUMA.json')['rows']}
metadata={}
for v in manifest['videos']:
 m=v['media'];fs=float(m['avg_fps']);pixels=int(m['width'])*int(m['height'])
 rate=float(m['video_kbps']) if m.get('video_kbps') not in (None,'','N/A') else float(m['mux_kbps'])
 metadata[v['recording']]=dict(recording=v['recording'],person=v['person'],group=v['group'],task=v['task'],codec=m['codec'],width=int(m['width']),height=int(m['height']),source_fps=fs,source_kbps=rate,bpp=rate*1000/(pixels*fs),bitrate_basis='video stream' if m.get('video_kbps') not in (None,'','N/A') else 'whole file proxy',luma=lumas[v['recording']]['luma'])
methods=['constant','POS_old_argmax','calibrated_median_POS_CHROM_V3']
rows=list(csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open(encoding='utf-8-sig')))
for r in rows:r['prediction']=float(r['prediction']);r['truth']=float(r['truth'])
results=[];edges_all={};correlations=[]
for cond in ['native','c5']:
 lookup={m:{r['recording']:r for r in rows if r['condition']==cond and r['method']==m} for m in methods}
 keys=set.intersection(*(set(x) for x in lookup.values()));groups={'all':keys}
 for field in ['bpp','luma']:
  edges=np.quantile([metadata[k][field] for k in keys if metadata[k][field] is not None],[.25,.5,.75]);edges_all[cond+'_'+field]=edges.tolist()
  for k in keys:
   x=metadata[k][field];label=field+' unknown' if x is None else field+' Q'+str(1+np.searchsorted(edges,x))
   groups.setdefault(label,set()).add(k)
 for k in keys:groups.setdefault('codec '+metadata[k]['codec'],set()).add(k)
 for label,ks in groups.items():
  base=[lookup['constant'][k] for k in sorted(ks)]
  for method in methods:
   rr=[lookup[method][k] for k in sorted(ks)]
   a=summarize(rr);d=summarize(rr,base)
   results.append(dict(condition=cond,stratum=label,method=method,people=a['people'],videos=a['videos'],MAE=a['MAE'],bias=a['bias'],MAE_CI_low=a['CI95'][0][0],MAE_CI_high=a['CI95'][0][1],delta_constant=d['MAE'],delta_CI_low=d['CI95'][0][0],delta_CI_high=d['CI95'][0][1]))
 for method in methods:
  kk=sorted(keys);x=[metadata[k]['bpp'] for k in kk];e=[abs(lookup[method][k]['prediction']-lookup[method][k]['truth']) for k in kk]
  rho=float(spearmanr(x,e).statistic)
  correlations.append(dict(condition=cond,method=method,videos=len(kk),people=len({metadata[k]['person'] for k in kk}),Spearman_abs_error_vs_source_bpp=rho))
def csvout(name,rr):
 with (P/name).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
csvout('COMPRESSION_STRATA.csv',results);csvout('COMPRESSION_SPEARMAN.csv',correlations)
csvout('COMPRESSION_METADATA.csv',list(metadata.values()))
walking=[v for v in metadata.values() if v['task'].lower()=='walking'];csvout('WALKING_COMPRESSION.csv',walking)
(P/'STRATIFICATION_RECEIPT.json').write_text(json.dumps(dict(quantile_edges=edges_all,holdout_evaluations=0,refitting=False,luma_definition=read(P/'ROI_LUMA.json')['definition'],C5_bpp_definition='source-native bpp; not reencoded C5 bpp'),indent=2),encoding='utf8')
def table(headers,rr):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(x) for x in r)+' |' for r in rr])
lines=['# Nén và sai số trên BTC dev','',
 'Đây là phân tích mô tả trên đầu ra đã lưu, không train hoặc chọn lại phương pháp. Bpp = bitrate luồng hình (bit/giây) / (rộng × cao × fps nguồn): số bit trung bình cho mỗi pixel mỗi khung. Bpp không đo trực tiếp chất lượng ảnh và không so ngang hoàn hảo giữa HEVC/H.264. Codec, độ phân giải, ánh sáng, chuyển động và nhóm người cùng thay đổi; tương quan không chứng minh nén gây lỗi.', '',
 'Luma là trung bình 0,2126R + 0,7152G + 0,0722B của ba ROI da hợp lệ sau chuyển SDR 8-bit. Đây là mức sáng tương đối của pixel đã chọn, không phải lux hay luma toàn ảnh. Pixel bão hòa/tối bị mask loại nên có thiên lệch chọn vùng. Nhóm Q1–Q4 được chia theo metadata trên giao video, không dùng sai số. C5 cũng phân tầng theo bpp của nguồn native.', '',
 'MAE và bias tính trong từng người trước rồi trung bình người. Bias = dự đoán − PR. CI bootstrap ghép cặp theo người 10.000 lần, seed 0; mỗi ô có cùng video giữa ba phương pháp. Người có nhiều video có thể thuộc nhiều nhóm. Các CI phân tầng là mô tả, không hiệu chỉnh đa kiểm định.', '', '## Tương quan cấp video', '',table(['Điều kiện','Phương pháp','Video/người','Spearman |lỗi| với bpp'],[[r['condition'],r['method'],f"{r['videos']}/{r['people']}",f"{r['Spearman_abs_error_vs_source_bpp']:.3f}"] for r in correlations]),'',
 'Hệ số âm nghĩa là bpp lớn đi kèm sai số nhỏ. Đây là Spearman cấp video theo yêu cầu; không gán p-value độc lập vì một người có thể có nhiều video.', '', '## Các video Walking', '', table(['Video','Codec','Bpp','Luma SDR','FPS'],[[v['recording'],v['codec'],f"{v['bpp']:.5f}",f"{v['luma']:.2f}" if v['luma'] is not None else 'thiếu',f"{v['source_fps']:.3f}"] for v in walking]),'']
for cond in ['native','c5']:
 lines+=['## '+cond,'',table(['Nhóm','Phương pháp','Người/video','MAE','Bias','Δ hằng số','CI95% Δ'],[[r['stratum'],r['method'],f"{r['people']}/{r['videos']}",f"{r['MAE']:.2f}",f"{r['bias']:+.2f}",f"{r['delta_constant']:+.2f}",f"[{r['delta_CI_low']:.2f}; {r['delta_CI_high']:.2f}]"] for r in results if r['condition']==cond]),'']
(P/'COMPRESSION_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps(dict(strata_rows=len(results),walking_videos=len(walking),correlations=correlations)))
