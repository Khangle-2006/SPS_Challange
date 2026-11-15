import json
from pathlib import Path
from statistics import mean

root = Path('reports/t8_polarity_v3_20261102/final_20261105')
s = json.loads((root/'SUMMARY.json').read_text())
x = json.loads((root/'SUPPLEMENT.json').read_text())
lines = ['T8 V3 corrected polarity - COMPLETE development OOF',
         'Training finished 2026-11-05 09:57:12 Asia/Saigon, about12h38m after launch.',
         '15 corrected joint +15 original UBFC controls,5folds,3seeds,finalepoch30.',
         '35 UBFC development people and MCD dev60; D3 and7UBFCtest closed.',
         'Original unharmonized joint excluded due independent label diagnostic; all artifacts retained.',
         'C5 target10fps, measuredwholefile<=500kbps; sensor clock reported separately below.', '',
         'MAE bpm, mean per-person across all3seeds. Delta=correctedjoint-control.',
         'PairedCI resamples persons10000times, carrying all3seeds together.']
for c,r in s['conditions'].items():
    u,j,p = r['ubfc_only'],r['joint'],r['paired']
    lines += [f"{c}: control={u['MAE']:.6f}, joint={j['MAE']:.6f}, delta={p['mean_joint_minus_control']:.6f}, CI95={p['CI95']}",
              f"  Mean subjectP95 control={u['subject_macro_P95']:.6f}, joint={j['subject_macro_P95']:.6f}",
              f"  Pooled windowP95 per seed control={u['window_P95_by_seed']}, joint={j['window_P95_by_seed']}",
              f"  Errors>20 per seed control={u['gt20_by_seed']}, joint={j['gt20_by_seed']}",
              f"  Counts perseed control={u['counts_by_seed'][0]}, joint={j['counts_by_seed'][0]}",
              f"  Eligible-window coverage control={u['coverage_by_seed']}, joint={j['coverage_by_seed']}"]
lines += ['',f"S mean control={s['S_mean']['ubfc_only']:.6f}, joint={s['S_mean']['joint']:.6f}",
          f"S by seed {s['S_by_seed']}",
          'S=mean(UBFC native,UBFC C5,MCD native); MCD C5 is diagnostic only. No missing-value/135 penalty.',
          f"Descriptive joint/control gates {x['descriptive_gates']}",
          'These do not evaluate72-versus36, which has not run.72 remains held.', '',
          'Predeclared expectations: MCD native better PASS; UBFC C5 nonworse FAIL.',
          'The corrected joint intervention improves MCD but leaves C5 degradation and clean UBFC point-estimate degradation.',
          'This comparison alone cannot attribute all changes to polarity or prove doublecompression/another bug.',
          'Joint replaces half UBFC draws by MCD, so UBFC exposure changes despite matched steps.',
          'Do not retune labels, offsets, seeds, checkpoints or cohorts based on these outcomes.',
          'Next diagnostic priority is existing encoding/source provenance for possible doublecompression and remaining timing/frontend/domain/optimization issues; no new grid.',
          'POS25.5419 and oldcrop10.14 are not directly ranked here; cohort/window/protocol matching pending.', '',
          'MCD before/after strata, mean of3seed person-macroMAEs. Before/after are recording conditions, not polarity versions.']
for condition in ('MCD_native','MCD_f10_b500'):
    for stage in ('before','after'):
        values={arm:mean(v['macro_MAE'] for k,v in x['before_after'].items() if k.endswith('_'+condition+'_'+stage) and '_'+arm+'_' in k) for arm in ('ubfc_only','joint')}
        lines.append(f'{condition} {stage}: {values}')
lines += ['',f"Actual C5 effective sensorFPS and bitrate ranges {x['encoding_summary']}",
          'MuxFPS=10, model/reference grid30Hz; effective sensorFPS is derived from selected frame source timestamps.',
          f"Dataset trainMSE, means over15runs at epoch1/10/20/30 {x['dataset_loss_epoch_means']}",
          'Loss scales alone do not identify cause or establish synchronization. MCD reduction is much smaller than UBFC.', '',
          'Validation passed all30checkpoint/status/history/provenance, frozenmanifest, paired initialization/pools/steps/control draws.',
          'All60 fold/seed/condition pairs passed oldreferenceSHA==newrawcopySHA, exact global sign/time/support/HR/failure parity, prediction/scorer hashes.',
          'FullOOF rescored with unchanged canonical scorer;24 independent fold-weighted macro/count cross-checks passed.',
          'Aggregator first validation attempt rejected JSONfloat versus CSVtext for window_start_s; numeric conversion only fixed receipt comparison. Raw/canonical CSV fields remained exactly compared. No hash mismatch waived.',
          'Detailed seed/fold/person/P95/>20/beforeafter/loss/encoding data in adjacent JSON artifacts.',
          'Remote full reference/prediction/metrics artifacts at /mnt/samsung_t5/khangle/facevitals/t8_polarity_v3_20261102/aggregate_20261105.']
(root/'REPORT.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('\n'.join(lines[:8]))
