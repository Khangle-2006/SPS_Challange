import json,shlex,sys
from pathlib import Path
r=Path(sys.argv[1]);remote='/mnt/samsung_t5/khangle/facevitals/reports/round1_final_20261105/revision_02';v3='/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102'
text='These commands document reproduction; they were not run again. Existing output paths refuse replacement. Choose a new destination and recheck storage before generating additional artifacts. Frozen inputs remain read-only.\n\n'
text+='## Frozen inputs and CPU reports on P100\n\n```sh\n'
text+=f'PY=/data_pool/khangle/facevitals/envs/hr/bin/python\nR={remote}\nV3={v3}\nNEW={remote}/reproduce_new\nmkdir "$NEW"\ncp "$R/DECISIONS.md" "$NEW/DECISIONS.md"\n'
text+='"$PY" "$R/code/prepare_frozen.py" --code "$V3" --controls /mnt/samsung_t5/khangle/facevitals/t8_oval_20261031/runs --joint /mnt/samsung_t5/khangle/facevitals/t8_polarity_v3_20261102 --index /mnt/samsung_t5/khangle/facevitals/t8_oval_20261031/training_index.json --output "$NEW/frozen_data" --classical_code /data_pool/khangle/facevitals/code\n'
for command in ['freeze_report.py --root "$NEW"','lag_check.py --root "$NEW" --pos-source /data_pool/khangle/facevitals/code/hr_baseline.py','mcd_c5_report.py --root "$NEW"','hybrid.py --root "$NEW" --code "$V3" --controls /mnt/samsung_t5/khangle/facevitals/t8_oval_20261031/runs --joint /mnt/samsung_t5/khangle/facevitals/t8_polarity_v3_20261102']:
    script,args=command.split(' ',1);text+=f'"$PY" "$R/code/{script}" {args}\n'
text+='ln -s "$R/packaging_inputs" "$NEW/packaging_inputs"\n"$PY" "$R/code/final_checks.py" "$NEW"\n"$PY" "$R/code/cadence_receipt.py" "$NEW"\n"$PY" "$R/code/source_codec.py" "$NEW" "$R/packaging_inputs/training_index.json"\n"$PY" "$R/code/benchmark72.py" "$NEW" "$V3"\n```\n\nThe benchmark is synthetic forward timing only. No72 training command was executed because storage preflight failed. The header-derived storage estimate and all1540rows are in BRANCH72_STORAGE_PREFLIGHT.json.\n\n'
text+='## Exact end-to-end commands actually executed\n\nFor reproduction, first replace only --out with a new directory; the published output must remain immutable. The commands below deliberately show the original recorded invocation.\n\n'
for ds,host in [('MCD','P100'),('UBFC2','A100')]:
    launch=json.loads((r/('receipts_'+host)/('packaging_'+ds)/'attempt2/LAUNCH.json').read_text())
    text+=f'### {host}: {ds}\n\n```sh\nPYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '+shlex.join(launch['command'])+'\n```\n\n'
text+='The fresh environment construction is implemented in code/create_venv.py; pass a new venv path, then the recorded dependency environments from FRESH_VENV.json in order. It exposes pinned installedpackages and systemdist-packages instead of downloading fresh dependencies. All source, checkpoint and frontend hashes are in the bundle/receipts.\n\n'
text+='## Local final reports\n\nAfter placing verified small receipts under a new revision directory with the same structure, use:\n\n```powershell\n'
py='C:/Users/Windows/AppData/Local/Programs/Python/Python311/python.exe'
text+=f"$py = '{py}'\n$revision = '{r.resolve().as_posix()}'\n"
text+='& $py "$revision/code/test_contracts.py"\n& $py "$revision/code/schema_report.py" $revision\n& $py "$revision/code/final_reports.py" $revision\n```\n\nThese scripts require output reports not to exist. schema_report renders the E2E receipts; final_reports renders branch72_report, optional_mcdnative_report, branch_choice, all_branch_metrics and parent SUMMARY. No model inference or statistic changes occur during rendering.\n'
p=r/'REPRODUCE.md';assert not p.exists();p.write_text(text,encoding='utf-8')
