import json
from pathlib import Path
P=Path('/data_pool/khangle/facevitals/STORAGE_POLICY.json')
policy=json.loads(P.read_text())
out=Path('/mnt/samsung_t5/khangle/facevitals/mapping_audit_20261030')
dev=json.loads((out/'development_plan.json').read_text())
assert policy['combined_server_ceiling']==500_000_000_000
policy.update(updated='2026-10-30',mcd_acquisition='development60 only; audit30 first; keep existing raw',
    protected_mcd_people=dev['development_subjects']+['1020'],
    mcd_raw_deletion_enabled=False,mcd_cache_training_eligible=False,
    raw_prune_gate='E3/frontend frozen AND frame/capture mapping verified AND caches validated; retain protected people',
    mmpd_subset_request={'people':10,'files':60,'bytes_from_manifest':45097301294,'access':'pending','download_enabled':False},
    mapping_report=str(out/'aggregate.json'),
    ubfc2_storage='FFV1 lossless conversion42/42 complete; all-frame RGB/PTS checks passed, original/new SHA recorded')
tmp=P.with_suffix('.json.tmp');tmp.write_text(json.dumps(policy,indent=2));tmp.replace(P)
(out/'storage_policy.json').write_text(json.dumps(policy,indent=2))
print('Updated policy; no data deleted')
