"""Saved predictions only: common videos and exactly matched windows."""
import json,hashlib,collections
from pathlib import Path
P=Path(__file__).resolve().parent;N=P.parent/'final_priorities_20261111'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (P/'CODEC_COMMON.json').exists()
    rows=[];gops=[];bindings={}
    for p in sorted((N/'classical_screen/predictions').glob('*/*/c5prime_*.json')):
        a=read(p);bindings[str(p)]=sha(p);rows.extend(r for r in a['rows'] if r['method'] in ['POS','CHROM'])
        ep=Path(a['evidence']['path']);assert sha(ep)==a['evidence']['sha256']
        g=read(ep);types=[v['pict_type'] for v in g['frames']];runs=[];k=0
        for t in types+['I']:
            if t=='B':k+=1
            elif k:runs.append(k);k=0
        gops.append(dict(dataset=a['rows'][0]['dataset'],recording=a['rows'][0]['recording'],condition=a['rows'][0]['condition'],frames=len(types),types=dict(collections.Counter(types)),consecutive_B_runs=dict(collections.Counter(runs)),prefix=''.join(types[:80]),gop_sha256=sha(ep)))
    table=[]
    for ds in ['BTC','UBFC','MCD']:
        for method in ['POS','CHROM']:
            look={c:{(r['recording'],r['start']):r for r in rows if r['dataset']==ds and r['method']==method and r['condition']=='c5prime_'+c and r['prediction'] is not None} for c in ['default','bf0']}
            vids=set(r[0] for r in look['default'])&set(r[0] for r in look['bf0']);common=set(look['default'])&set(look['bf0'])
            for cohort in ['common_video','common_window']:
                for c in look:
                    rr=[r for k,r in look[c].items() if k in common] if cohort=='common_window' else [r for k,r in look[c].items() if k[0] in vids]
                    hit=sum(148<=r['prediction']<=152 for r in rr)
                    table.append(dict(dataset=ds,method=method,cohort=cohort,condition=c,people=len({r['person'] for r in rr}),videos=len({r['recording'] for r in rr}),windows=len(rr),hits=hit,fraction=hit/len(rr) if rr else None))
    out=dict(table=table,gop=gops,input_hashes=bindings,protocol_sha256=sha(P/'PROTOCOL.md'),note='Frequency counts only; no MAE/model selection. Exact common windows primary; common video sensitivity has unequal valid window counts. Encoder toggle not isolated causal proof.')
    (P/'CODEC_COMMON.json').write_text(json.dumps(out,indent=2));lines=['# Cụm 148–152 bpm trên cùng dữ liệu','','|Bộ|Phương pháp|Giao|Encoder|Người/video/cửa sổ|Số cửa sổ 148–152|Tỉ lệ|','|---|---|---|---|---:|---:|---:|']
    for r in table:lines.append(f"|{r['dataset']}|{r['method']}|{r['cohort']}|{r['condition']}|{r['people']}/{r['videos']}/{r['windows']}|{r['hits']}|{100*r['fraction']:.2f}%|" if r['fraction'] is not None else f"|{r['dataset']}|{r['method']}|{r['cohort']}|{r['condition']}|0|0|NA|")
    lines+=['','Đây là tỉ lệ dự đoán, không phải bảng MAE. Hai encoder còn thay đổi các hành vi nén khác; không đủ để khẳng định một cơ chế vật lý. C5′ chưa xác minh tương đương MATLAB VideoWriter. Chi tiết pict_type và chuỗi B liên tiếp được lưu trong JSON.']
    (P/'CODEC_COMMON.md').write_text('\n'.join(lines));print(json.dumps([r for r in table if r['cohort']=='common_window']))
if __name__=='__main__':main()
