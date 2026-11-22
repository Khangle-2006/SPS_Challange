"""One-time audited operational recovery; no experiment recipe changes."""
import json,sys,hashlib,time,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=R/'round1_execution_20261110';V5=E/'v5_multiview_20261112'
sys.path.insert(0,str(V5));import runtime
size=lambda p:sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.exists() else 0
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not any(Path('/proc/'+str(i)).exists() for i in [2155580,2155581,2155582,2155583,2162569,2143956])
assert json.loads((P/'W2B_DISPOSITION.json').read_text())['state']=='closed_after_pair'
assert (R/'w2_btc_20261116/STOPPED_BY_USER_20261116.json').exists()
assert (V5/'TRAIN_COMPLETE.json').exists() and (V5/'DISPATCH_V3_COMPLETE.json').exists()
assert sha(P/'constraints_reader_v2.py')=='6d8f14f6062b1a0c38c61b40617401886321a1bc08a4b5263e69b132c10d4a26'
done=list((P/'constraints').glob('*/*/COMPLETE.json'));assert len(done)==483
verified={}
for f in done:
 a=json.loads(f.read_text());assert sha(f.parent/'meta.npz')==a['meta_sha256'];verified[str(f.relative_to(P))]=dict(receipt_sha256=sha(f),meta_sha256=a['meta_sha256'])
partials=list((P/'constraints').glob('*/*/PRESERVED_RAM_PARTIAL.json'));assert len(partials)==4
for f in partials:assert not (f.parent/'COMPLETE.json').exists() and not (f.parent/'meta.npz').exists()
b=runtime.budget(True)
extras=[R/'w2_btc_20261116',R/'reader_v1_20261116',R/'reader_v2_20261116',R/'reader_label_audit_20261116',R/'reader_candidates_freeze_20261116']
extra=sum(size(d) for d in extras)+sum(f.stat().st_size for f in R.glob('READER_CANDIDATES_FROZEN_20261116*') if f.is_file())
ramold=sum(size(d) for d in Path('/dev/shm').glob('reader_C*'))
reservation=420_000_000+200_000_000+2_000_000_000+50_000_000
scope=b['scope_actual_bytes']+extra+ramold+reservation
project=b['project_actual_bytes']+ramold+reservation
assert scope<84_000_000_000 and project<500_000_000_000,(scope,project)
receipt=dict(passed=True,measured_epoch=time.time(),budget=b,scope_conservative_bytes=scope,project_conservative_bytes=project,old_RAM_partials_preserved_bytes=ramold,extra_actual_bytes=extra,new_scratch_cap=2_000_000_000,new_scratch_stop=1_900_000_000,protected_reserve=420_000_000,followup_cap=200_000_000,additional_reserve=50_000_000,released_unspent_reservations=dict(V5=b['remaining_reserved_bytes'],W2_old_envelope=1_000_000_000,reader_old_envelope=100_000_000),release_basis='V5 terminal, W2 stopped, W2b closed; all actual files counted, no deletion; no future training authorized',verified_complete=verified,recipe_unchanged=True)
archive=P/'C_ATTEMPT2_RAM_GUARD_20261117';assert not archive.exists();archive.mkdir()
for f in partials:
 src=f.parent;dest=archive/'partials'/src.parent.name/src.name
 assert src.resolve().is_relative_to((P/'constraints').resolve()) and dest.resolve().is_relative_to(archive.resolve())
 dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(dest))
for pattern in ['C_ERROR_SHARD*.json','C_STATUS_SHARD*.json','C_SHARD*.log']:
 for f in P.glob(pattern):shutil.move(str(f),str(archive/f.name))
(P/'PREFLIGHT_C_RECOVERY_V3.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='verified_complete'}))
