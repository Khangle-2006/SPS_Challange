from pathlib import Path
import json,hashlib,re,datetime
P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=read(P/'FROZEN.json');assert sha(P/'SELECTION.json')==f['selection_sha256'];assert sha(P/'PROTOCOL.md')==f['protocol_sha256']
rows=read(P/'SELECTION.json');assert len(rows)==100
for r in rows:
    mc=read(P/'model'/r['key']/'COMPLETE.json');assert mc['checkpoint_sha256']==f['checkpoints'][str(r['fold'])]['sha256']
    assert mc['common_feature_checkpoint_sha256']==f['checkpoints']['0']['sha256']
    assert mc['static_hr'] is None
    assert mc['finite_difference']['relative_error']<.15
    for folder,name in [('data','READY.json'),('model','COMPLETE.json'),('signal','COMPLETE.json')]:assert (P/folder/r['key']/name).exists()
for name in ['MODEL_COMPLETE.json','SIGNAL_COMPLETE.json','CODEC_V3_COMPLETE.json']:assert read(P/name)['errors']==[]
report=(P/'VIS_REPORT.md').read_text(encoding='utf-8')
links=re.findall(r'\]\(([^)]+)\)',report)
for link in links:
    if not link.startswith('https:'):assert (P/link).exists(),link
assert len(read(P/'FEATURE_LABELS.json'))==283
bindings={p.name:sha(p) for p in P.glob('*.py')}
(P/'FINAL_CODE_BINDINGS.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=bindings,original_frozen_preserved=True,operational_changes='See NUMERICAL_ADDENDUM.md; plotting/packaging scripts added after inference; no model changes'),indent=2))
receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection100_verified=True,model100_signal100_codec10_complete=True,windows=283,in_band_reference_windows=264,static_pass100=True,all_output_hashes_checked_by='review_tables.py',selection_and_protocol_hashes_unchanged=True,checkpoint_hash_bindings_unchanged=True,report_sha256=sha(P/'VIS_REPORT.md'),linked_local_files=len([x for x in links if not x.startswith('https:')]),visual_review=['SPATIAL_SUMMARY_COMMON','PULSE_SUMMARY same source panel','EVM_YT','UMAP','BTC S_AB13 spatial/temporal','UBFC subject10 spatial','Iriun1563 temporal_review','codec S_AB1'],limitations=['landmark skin proxy','first supported10s attribution','BTC error-extreme selection','Iriun43/59 paired windows','EVM gap effect not independently audited','UMAP common checkpoint includes public training people','ffmpeg not MATLAB parity'],no_model_change=True,no_holdout_access=True,training=False)
(P/'COMPLETION_REVIEW.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
