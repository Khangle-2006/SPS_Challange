"""Render final reports from frozen receipts only; no model fitting or selection search."""
import csv,json,sys,hashlib,shlex
from pathlib import Path
from metrics import table,CONDITIONS
r=Path(sys.argv[1]);s=json.loads((r/'ALL_BRANCH_BOOTSTRAP.json').read_text());lag=json.loads((r/'LAG_SUMMARY.json').read_text())
b72=json.loads((r/'BRANCH72_STORAGE_PREFLIGHT.json').read_text());benchmark=json.loads((r/'BRANCH72_BENCHMARK.json').read_text())
def put(name,text):
    p=r/name;assert not p.exists(),p;p.write_text(text,encoding='utf-8')
def fmt(x):return f"{x['value']:.3f} [{x['CI95'][0]:.3f}, {x['CI95'][1]:.3f}]"
def readings():
    rows=['| Contrast | Condition | Delta MAE [95% CI] | Reading |','|---|---|---:|---|']
    for k in ('v3_joint minus ubfc_only','hybrid_ubfc_only minus ubfc_only','hybrid_v3_joint minus v3_joint'):
        for c,v in s['paired'][k].items():
            reading=v.get('reading','better' if v['CI95'][1]<0 else 'worse' if v['CI95'][0]>0 else 'no detectable difference')
            rows.append(f"| {k} | {c} | {v['delta']:.3f} [{v['CI95'][0]:.3f}, {v['CI95'][1]:.3f}] | {reading} |")
    return '\n'.join(rows)
def gates():
    rows=['| Candidate | S decrease | Clean MAE increase | C5 P95 change (bpm) | S / clean / P95 | All |','|---|---:|---:|---:|---|---|']
    for b,g in s['gates'].items():
        rows.append(f"| {b} | {g['S_relative_reduction']*100:.2f}% | {g['UBFC_clean_relative_increase']*100:.2f}% | {g['UBFC_C5_P95_delta']:+.3f} | {' / '.join('PASS' if g[k] else 'FAIL' for k in ['pass_S','pass_clean','pass_P95'])} | {'PASS' if g['pass_all'] else 'FAIL'} |")
    for b in ('v3_joint_72','v3_joint_mcdnative'):rows.append(f'| {b} | NA | NA | NA | NOT RUN | NOT EVALUATED |')
    return '\n'.join(rows)
scores='\n'.join(f"| {b} | {fmt(v['S'])} |" for b,v in s['branches'].items())
scores='| Branch | S [95% CI] |\n|---|---:|\n'+scores
unrun='\n'.join(f'| {b} | {c} | NA | NA | NA | NA |' for b in ('v3_joint_72','v3_joint_mcdnative') for c in CONDITIONS)
choice='# Branch choice\n\nRecommend **'+s['recommended_branch']+'**: no alternative passes all three preregistered gates. This is a development recommendation pending the listed submission-readiness items.\n\n'+table(s,['ubfc_only','v3_joint','hybrid_ubfc_only','hybrid_v3_joint'])+'\n'+unrun+'\n\n'+scores+'\n\n'+gates()+'\n\nMain risk: V3 remains worse than UBFC-only on UBFC C5 by paired95%CI, and the lower-S hybrid increases C5 tail error. MCD-C5 is excluded from S and branch choice.\n'
put('branch_choice.md',choice)
put('branch72_report.md',f'''# 72x72 branch: NOT RUN — storage guard

The revised request authorizes this experiment, superseding the earlier hold. Preflight ran while task1 was active. No cache, checkpoint or GPU training was launched. Completed runs0/15; fold-complete seeds0/3. No scientific gate can be evaluated.

WholeT8 was {b72['T8_existing_bytes']:,}bytes, cap65,000,000,000bytes and reservedheadroom100,000,000bytes. Available {b72['available_bytes']:,}bytes. The same frozen float16 layout at72 needs **{b72['new72_RGB_payload_bytes']:,} additional RGBbytes**, before metadata/checkpoints. This comes from all{len(b72['rows'])} existing devcache headers. Raw datasets and existing outputs were preserved. Upscaling36 or substituting T3 affine was not used. Approval for an experiment did not waive the earlier storagecap. A new storage allocation or approved equivalent streaming design remains necessary.

| Input | Parameters | CPU median ms/10s clip |\n|---|---:|---:|
'''+''.join(f"| {v['size']}x{v['size']} | {v['parameters']:,} | {v['median_ms_per_clip']:.3f} |\n" for v in benchmark['measurements'])+'''
Benchmark: one CPU thread,2 warmups,5 forward passes, synthetic inputs and untrainedweights, excludes ROI/decoder/I/O. Parameter count changes because the existing architecture's dense head depends on image size; architecture definitions are otherwise unchanged. This is not a trained72 quality result.

The18November23:59 Asia/Saigon cutoff remains. No partial-fold score is substituted. Optional v3_joint_mcdnative was skipped because72 is not running and its prerequisites are unmet; it must not consume resources intended for72.
''')
put('branch72_per_person.csv','person,dataset,condition,branch,fold,seed,MAE,signed_bias,P95_abs_error,coverage,status\n')
put('BRANCH72_CHECKPOINT_MANIFEST.json',json.dumps(dict(state='not_run_storage_guard',completed_runs=0,fold_complete_seeds=0,checkpoints=[],scientific_gate='not_evaluated'),indent=2))
put('optional_mcdnative_report.md','# Optional branch: SKIPPED\n\nTask6 is conditional on task5 running with spare authorized GPU capacity. Task5 is storage-blocked; no optional training or outcome selection was performed.\n')
rows=[]
for b,cc in s['branches'].items():
    for c in CONDITIONS:
        d=dict(branch=b,condition=c,persons=cc[c]['persons'])
        for metric in ('MAE','P95_abs_error','signed_bias','coverage'):
            v=cc[c][metric];d.update({metric:v['value'],metric+'_CI_low':v['CI95'][0],metric+'_CI_high':v['CI95'][1]})
        rows.append(d)
with (r/'all_branch_metrics.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
schema=(r/'schema_check.md').read_text()
cohort=json.loads((r/'COHORT_COUNTS.json').read_text())
cohort={k:({a:b for a,b in v.items() if a!='recording_ids'} if isinstance(v,dict) else v) for k,v in cohort.items()}
for v in cohort['C5'].values():v['selected_frame_source_clock_fps_range']=v.pop('actual_sensor_fps_range')
cadence=json.loads((r/'SENSOR_CADENCE_CLARIFICATION.json').read_text())
cohort['native_acquisition_cadence']=cadence['summary']
summary=('# LABEL PHASE ISSUE\n\n' if lag['LABEL_PHASE_ISSUE'] else '')+'# Round1 final development evaluation — 5November2026\n\n'
summary+='Recommend **v3_joint** under the fixed gate. Tasks1–4 and the two task7 end-to-end attempts are documented below;72 is unrun because of storage and the optional branch is skipped. Preserve freeze19November and planned submission29November2026. No submission was sent.\n\n'
summary+='## Cohort and protocol\n\n'+json.dumps(cohort,indent=2).join(['```json\n','\n```\n\n'])
summary+='All30 finalepoch checkpoints are preserved (15V3,15UBFC-only); all five person-level splits pass disjointness assertions. D3 and the7UBFCtest people were not opened.10s windows, oval36, canonical0.7–3Hz, seeds2027/2028/2029. Average seeds within each person before macroaggregation. Bootstrap10000, seed0, pairedpeople. P95 is the mean person-level within-window absolute-errorP95 after seed averaging, not pooled-windowP95; alternative P95personMAE and medianMAE remain in the JSON receipts. Coverage is accepted/reference-eligible; raw candidate/eligible/accepted counts remain in the CSVs. CIs below are pointwise, not multiplicity-adjusted.\n\n'
summary+='C5 target/containerFPS is10; measured whole-filebitrate is below500kbps for every recording. Historical encoder effective_sensor_fps describes selectedframes on the sourceclock, not the originalacquisitionrate. Nativecadence above is measured from originalsource timestamps; hardwareclockaccuracy is not independentlyverified. SENSOR_CADENCE_CLARIFICATION.json corrects the ambiguous COHORT_COUNTS key without rewriting that receipt.\n\n'
summary+='## All results\n\n'+table(s)+'\n'+unrun+'\n\n'+scores+'\n\nMCD-C5 is reported separately and excluded fromS. All known values reproduced within0.05bpm; no discrepancies.\n\n'
summary+='## Fixed readings\n\n'+readings()+'\n\nMCD-C5 V3-minus-control CI is entirely below0: **better**, supporting the requested UBFC-specific pattern.119/119 MCDsources are MPEG-4 and C5 isH.264, confirming doublecompression. This does not establish its causal contribution. No additional native-to-C5 reference-eligible window loss was found; all branches use the verified common support mask.\n\n'
summary+='## Lag\n\n'+json.dumps(lag,indent=2).join(['```json\n','\n```\n\n'])
summary+='FullHD medianabsolute lag3frames does not exceed the fixed>3 trigger; LABEL PHASE ISSUE is not raised.42.02% of recordings nevertheless exceed3frames. This borderline result does not prove synchronization. The globalcanonical signs and lag convention were fixed before calculation; no labeloffset was changed.\n\n'
summary+='## Gates and branch choice\n\n'+gates()+'\n\nNeither hybrid passes C5P95; hybridV3 reduces S13.57% but increases P95 by4.388bpm. No passing alternative: retainV3. MCD-C5 was not used for selection.\n\n'
summary+='## Failed, blocked, skipped and human items\n\n- 72:0/15runs,0fold-complete seeds;140.513GBadditional RGBcache alone exceeds1.750GBheadroom. Preserve all artifacts; resource decision required. See revision_02/branch72_report.md.\n- OptionalMCDnative:skipped because72 is not running.\n- Initial packagingvenv probe failed on missing packaging; separate attempt2 exposes installedsystem dependencies. This is a fresh interpreter environment, not an isolated dependency rebuild. Any later packagingfailure is retained in schema_check.md and its FAILURE receipt.\n- No historic joint results with unharmonizedpolarity enter the comparison; no oldresult was deleted or rewritten. Earlier QA45-run request was superseded before starting.\n- Organizerpilotdata remains unavailable; supervisorCOI form needs a human; official details markedUNKNOWN require clarification. Acquisitionlux/distance/Fitzpatrick and detailedMATtyping are not inferred.\n\n'
summary+='## Submission readiness\n\n'+schema+'\n\n## Reproduction\n\n'+(r/'REPRODUCE.md').read_text()+'\n\n## Created files\n\nThe exhaustive LOCAL_FILES.csv and REMOTE_FILES.csv include paths,bytecounts andSHA256 (venvmetadata is listed without binary hashing where noted). All new artifacts stay under reports/round1_final_20261105. Remote large MAT/cache files remain on their originatingserver; local smallreports are hash-verified. See revision_02/FILE_LIST.md for every createdpath.\n'
out=r.parent/'SUMMARY.md';assert not out.exists();out.write_text(summary,encoding='utf-8')
print('FINAL_REPORTS_WRITTEN')
