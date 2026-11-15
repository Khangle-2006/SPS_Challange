"""Report-only revision using saved results. No inference, fitting or holdout access."""
import csv, hashlib, json
from pathlib import Path

P=Path(__file__).resolve().parent
B=P.parent/'round1_bayes_20261108'
OUT=P/'report_revision_20261111'
OUT.mkdir(exist_ok=True)
sources=[B/'revision_20261109/IEEE_REPORT_DRAFT.tex',B/'BTC_COMMON_METRICS.csv',
         P/'C5_REPORT.md',P/'sweep/fold2/NATIVE_STOP_GATE.json',
         P/'sweep/fold3/STATIC_EPOCH8.json',P/'FASTTRACK_STOP_GATE.json',
         P/'OCTAVE_PARITY.json',P/'process_round1.m',P/'GATES.md']
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
old=sources[0].read_text(encoding='utf-8')
metrics=list(csv.DictReader(sources[1].open(encoding='utf-8')))
native={x['method']:x for x in metrics if x['condition']=='native' and x['group']=='all'}
gate=json.loads(sources[3].read_text())
rows={x['method']:x for x in gate['rows'] if x['group']=='all'}
stop=json.loads(sources[5].read_text())
parity=json.loads(sources[6].read_text())
assert stop['state']=='stopped_user_native_gate' and stop['cancelled_folds']==[0,1,4]
assert not gate['passed_native_gate'] and rows['fasttrack']['people']==32
assert rows['fasttrack']['delta_calibrated_CI'][0]>0
assert parity['passed'] and len(parity['results'])==5
assert all(r['max_abs_difference_bpm']==0 for r in parity['results'])

# Keep the historical draft. Replace stale status paragraphs in a new version only.
text=old.replace('\\usepackage{amsmath,booktabs,graphicx,url}',
                 '\\usepackage{amsmath,booktabs,url}')
fig_start=text.index('\\begin{figure}')
fig_end=text.index('\\end{figure}',fig_start)+len('\\end{figure}')
text=text[:fig_start]+r'''Participant-mean agreement is distinct from window-level agreement. The existing Bland--Altman statistics do not yet include the selected calibrated estimator; a complete, matched agreement table remains a reporting requirement. No missing limits of agreement are fabricated.
'''+text[fig_end:]
text=text.replace('Official C5-prime results remain pending because MATLAB is unavailable.',
                 'The user-authorized ffmpeg C5-prime proxy has now been evaluated; direct MATLAB equivalence remains unverified.')
start=text.index('\\section{Outstanding Experiments and Decision Gates}')
end=text.index('\\section{Limitations and Reproducibility}',start)
new=r'''\section{Completed Public and Compression Checks}
Canonical POS was repaired and reproduces 4,304 stored reference predictions exactly. Its original MCD comparison has MAE 4.67015 beats/min. Median blind SNR remains $-0.24956$ dB, failing the previously declared positive-SNR interpretation gate. We therefore do not use the Bayes experiment to establish a physiological or causal explanation.
For C5-prime, fixed frame skipping targets 10 fps and libx264 uses a 500 kbps maximum rate and 1000 kb buffer. One variant retains default GOP/B-frame settings; a second disables B-frames. These are proxies, not MATLAB VideoWriter validation. Completed processing of 750 variants does not imply all candidate windows pass validity checks.
\begin{table}[t]\centering\scriptsize
\caption{Public POS MAE on an identical window intersection across all four conditions. UBFC: 35 people/180 windows; MCD: 60 people/1,929 windows. Numbers differ slightly from other cohorts.}
\begin{tabular}{lrr}\toprule Condition & UBFC & MCD\\\midrule
Native & 1.441 & 4.669\\
Old C5 & 25.730 & 18.669\\
C5-prime default & 23.323 & 22.020\\
C5-prime no B-frames & 3.152 & 7.242\\
\bottomrule\end{tabular}\end{table}
Disabling B-frames substantially changes measured error in this paired comparison. It also changes encoder decisions, so this result does not isolate a single artifact mechanism. BTC accepted cohorts differ across conditions: old C5 has 217 videos/157 people, default C5-prime 161/109, and no-B C5-prime 158/106. These aggregate errors must not be treated as a paired cross-condition effect. Full method-by-condition tables and coverage records are retained separately.

\section{Adaptation Experiments and Stopping Decision}
AdaBN recalculated training-fold BatchNorm running statistics without changing learned parameters. On the same 218 videos/158 people, MAE changed from 19.06018 to 19.23028 beats/min, compared with 12.07068 for the constant and 11.37385 for calibration. All five static controls passed. Recalibration alone was insufficient; this does not prove that all model features lack pulse information.
The five-run plan was amended before training to prioritize run 4 (fold 3, SiNC) and run 3 (fold 2, weak labels), with runs 1, 2 and 5 conditional on the priority results. These are different participant folds and different recipes, not a matched causal ablation. Augmentation, fold-specific initialization, seed 2027 and final epoch 10 were fixed. Native validation no longer waited for C5-prime evaluation.
Run 4 was rejected at epoch 8: a static image produced 61.5234 beats/min despite an output standard deviation of only $2.425\times10^{-9}$. This is a failed abstention control, not a physiological measurement. Training loss changed from 3.00516 to 2.96747. No native MAE was measured for this rejected run, and none is imputed.
Run 3 completed ten epochs and passed all ten static tests. Its video prediction SD was 12.6106 beats/min, exceeding the preregistered minimum of 8, but participant Spearman correlation decreased from $-0.1090$ before adaptation to $-0.1354$ afterward. It failed the joint weak-label acceptance gate.
\begin{table}[t]\centering\scriptsize
\caption{Run 3 native validation only: identical 42 videos/32 people. This is not five-fold BTC performance.}
\begin{tabular}{lrr}\toprule Method & MAE & Bias\\\midrule
Training-fold constant & 9.682 & +1.214\\
Saved calibration & 9.417 & -1.074\\
Weak-label adaptation & 18.150 & -14.095\\
\bottomrule\end{tabular}\end{table}
Weak-label adaptation was worse than calibration by 8.73293 beats/min (paired 95\% CI [4.09618, 13.28093]). Neither priority run satisfied the validity and strict native MAE below 12 stopping gate. Training stopped at 22:18 ICT on November 10; runs 1, 2 and 5 were cancelled without starting. A full five-fold learned result, sliding-window experiment and downstream learned-model stacking were not produced.
The retained BTC development candidate is the saved calibrated ensemble, with MAE 11.37385 versus 12.07068 for the constant. Its paired difference CI [$-1.51016$, 0.14547] includes zero. The global target (MAE at most 10, improvement at least 1, and paired CI excluding zero) was not achieved. This is the end of the authorized sweep, not proof that every possible learning approach has been exhausted.
\textbf{Holdout remains locked, zero evaluations.} A final chosen-configuration evaluation is reserved for November 26--27. Missing holdout results are not zeros or passes.
'''
text=text[:start]+new+text[end:]
text=text.replace('The supplied MATLAB wrapper is a skeleton and has not been executed; end-to-end packaging remains incomplete.',
'''The toolbox-free POS wrapper now runs from precomputed, hash-bound ROI traces and agrees with Python to 0 beats/min on five real ROI inputs under GNU Octave 8.4 in traditional mode. This is not execution in real MATLAB, raw-video end-to-end validation, or packaging of the selected calibrated estimator. The latter needs its exact feature pipeline, fitted configuration and deployment parity documented before release.''')
text=text.replace('\\author{\\IEEEauthorblockN{Team authors --- to be completed}}',
'''\\author{\\IEEEauthorblockN{Team authors --- to be completed}}
% REPORT REVISION 2026-11-11: source-only draft, not compiled or submission-ready.''')
(OUT/'IEEE_REPORT_DRAFT.tex').write_text(text,encoding='utf-8')
checks=dict(preregistered_stop_consistent=True,weak_result_saved_only=True,
            old_native_table_matches_saved_metrics=all(f"{float(native[m]['MAE']):.2f}" in text for m in ['constant','POS_old_argmax','CHROM_old','V3_median15','V4_original_padding_median15','calibrated_median_POS_CHROM_V3','Bayes_none']),
            octave_parity_five=True,no_holdout_access=True,no_inference_or_refit=True,
            latex_compiled=False,layout_reviewed=False,submission_ready=False)
assert checks['old_native_table_matches_saved_metrics']
receipt=dict(input_sha256={str(f):sha(f) for f in sources},checks=checks,
             output_sha256=sha(OUT/'IEEE_REPORT_DRAFT.tex'))
(OUT/'REPORT_REVIEW_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(checks))
