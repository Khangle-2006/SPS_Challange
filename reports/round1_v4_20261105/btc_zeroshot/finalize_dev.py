"""Apply fixed dev gate only after all45 checkpoint evaluations; never open holdout."""
import csv,json
from pathlib import Path
from collections import defaultdict
from summarize_dev import boot,save,sha
HERE=Path(__file__).resolve().parent
def threshold_gate(mae,constant_mae):
    return mae>=5 or constant_mae-mae<2
def main():
    summary=json.loads((HERE/'SUMMARY.json').read_text())
    assert summary['state']=='complete_dev','Full dev predictions required; no partial selection'
    assert summary['checkpoint_counts']=={'ubfc_only':15,'v3_joint':15,'v4_repaired':15}
    manifest=json.loads((HERE/'dev_manifest.json').read_text());byrec={v['recording']:v for v in manifest['videos']}
    allrows=list(csv.DictReader((HERE/'VIDEO_PREDICTIONS.csv').open()))
    for r in allrows:r['prediction']=float(r['prediction']) if r['prediction'] else None;r['truth']=float(r['truth'])
    methods=['POS','CHROM','ubfc_only_median15','v3_joint_median15','v4_repaired_median15','median_POS_CHROM_v4_repaired_median15']
    methods+=['calibrated_'+m for m in methods]
    # Native Resting is the user's primary no-training gate; C5 is reported separately.
    grouped={m:{r['recording']:r for r in allrows if r['method']==m and r['condition']=='native' and r['scheme']=='10s'
        and byrec[r['recording']]['task'].casefold()=='resting' and r['prediction'] is not None} for m in methods+['constant']}
    common=set.intersection(*(set(v) for v in grouped.values()))
    assert len({byrec[k]['person'] for k in common})>=2,'Insufficient paired Resting people; cannot select'
    paired={m:[v[k] for k in sorted(common)] for m,v in grouped.items()}
    metrics={m:boot(r) for m,r in paired.items()}
    ranking=sorted(methods,key=lambda m:(metrics[m]['MAE'],methods.index(m)))
    best,runner=ranking[:2];delta=boot(paired[best],paired[runner]);constant=metrics['constant']['MAE']
    need=threshold_gate(metrics[best]['MAE'],constant)
    distinguish=delta['CI95'][0][1]<0
    decision=dict(state='fine_tune_gate_met' if need else 'zeroshot_sufficient',best_method=best,runner_up=runner,
        primary='native / Resting / 10s nonoverlap, common accepted videos across all fixed candidates',
        common_videos=len(common),common_people=metrics[best]['people'],candidate_metrics=metrics,
        best_minus_runner=delta,distinguishable_from_runner=distinguish,
        constant_improvement_bpm=constant-metrics[best]['MAE'],fine_tune_gate_met=need,
        candidate_configuration_chosen=False,holdout_released=False,holdout_evaluations=0,
        reason='Review paired coverage, static controls and rule uncertainty before final configuration. This script never trains or opens holdout.',
        summary_sha256=sha(HERE/'SUMMARY.json'),video_predictions_sha256=sha(HERE/'VIDEO_PREDICTIONS.csv'),split_sha256=manifest['split_sha256'])
    save(HERE/'DEV_DECISION.json',decision)
if __name__=='__main__':main()
