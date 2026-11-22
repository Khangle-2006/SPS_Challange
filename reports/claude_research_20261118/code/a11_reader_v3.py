"""reader_v3 candidate for BTC native: capture-class-aware posterior reader on the FROZEN oval trace (same evidence as reader_v1/v2).
Evidence   : Z(f) = mean of the standardised log CHROM and POS spectra, 10 s windows every 5 s, frozen readout operator (unchanged since 16 Nov).
Class      : read from the video file itself: 'low' if width x height < 1.8 megapixels (in dev: exactly the 848x480-type .mp4 files), else 'high'.
Model F    : prediction = posterior median of prior(f) * exp(beta_class * Z(f)) + offset(class, Task); prior = person-weighted KDE (6 bpm) of
             the training labels; beta per class from a fixed grid by person-weighted training MAE; offset = person-weighted mean training
             residual of the cell; classes or cells with fewer than 12 training videos fall back to the pooled beta / to no offset.
Protocol   : 218 dev videos / 158 people, 5 person-disjoint folds of VIDEOS.json, person-first MAE, paired bootstrap over people (10,000, seed 0).
Controls   : class flag shuffled x20, Task flag shuffled x20, labels shuffled between people x10 (everything refitted each time).
Outputs    : reader_v3/OOF_PREDICTIONS.csv, reader_v3/METRICS.json, reader_v3/reader_model_dev218_v3.json (fitted on all dev; for ONE holdout run).
The holdout is not read anywhere in this script."""
import csv, hashlib, json, sys
import numpy as np
sys.path.insert(0, '/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118/code')
from lib_btc import *
import reader_models as rm
RV = W / 'reader_v3'; V = videos(); M = meta(); rec = [v['recording'] for v in V]
lab = np.array([v['pr_bpm'] for v in V]); fold = np.array([v['fold'] for v in V]); per = np.array([v['person'] for v in V]); prot = np.array([v['task'] for v in V]); grp = np.array([v['group'] for v in V])
task = np.char.startswith(prot, 'Task'); low = np.array([int(M[r]['width']) * int(M[r]['height']) < 1800000 for r in rec]).astype(int)
Z = np.zeros((len(rec), len(GRID))); nwin = np.zeros(len(rec), int)
for i, r in enumerate(rec):
    b = load_blocks(r); a = avg_spectra(b['t'], b['valid'], b['rgb_mean'])
    if a is not None: Z[i] = (zlog(a[0]) + zlog(a[1])) / 2; nwin[i] = a[2]
rows = list(csv.DictReader((NUM / 'tables/round1_bayes_20261108/VIDEO_PREDICTIONS.csv').open()))
def saved(method): d = {r['recording']: float(r['prediction']) for r in rows if r['method'] == method and r['condition'] == 'native'}; return np.array([d[r] for r in rec])
const = saved('constant'); cal = saved('calibrated_median_POS_CHROM_V3')
P = {}; FITS = {}
for mode in ['v1', 'v2', 'F']: P[mode], FITS[mode] = rm.crossval(Z, lab, per, fold, low, task, mode)
def line(name, p, refs):
    s = summarize(p, lab, per); e = p - lab; d = {}
    txt = f"{name:34s} MAE {s['MAE']:6.3f} [{s['CI'][0]:.2f};{s['CI'][1]:.2f}]  RMSE {s['RMSE']:5.2f}  r {s['r']:+.2f}  bias {s['bias']:+.2f}"
    for rn, rp in refs:
        q = summarize(p, lab, per, rp); txt += f" | vs {rn} {q['delta']:+.3f} [{q['dCI'][0]:+.2f};{q['dCI'][1]:+.2f}]"; d[rn] = [q['delta']] + q['dCI']
    print(txt, flush=True); return dict(**s, deltas=d)
MET = {}
print(f"videos {len(rec)} people {len(set(per))} | low class {int(low.sum())} | no supported window: {int((nwin == 0).sum())}")
MET['constant'] = line('constant (train-fold median)', const, []); MET['calibrated'] = line('calibrated median (old candidate)', cal, [('constant', const)])
MET['v1'] = line('reader_v1 (frozen 16 Nov)', P['v1'], [('constant', const), ('calibrated', cal)])
MET['v2'] = line('reader_v2 (frozen 16 Nov)', P['v2'], [('constant', const), ('calibrated', cal), ('v1', P['v1'])])
MET['v3'] = line('reader_v3 = F (this script)', P['F'], [('constant', const), ('calibrated', cal), ('v1', P['v1']), ('v2', P['v2'])])
print('fold fits of F (beta by class {0 high, 1 low}; offset by cell {0 high rest, 1 high Task, 2 low rest, 3 low Task}):')
for f, x in enumerate(FITS['F']): print('   fold', f, x)
S = lambda p, l=lab: summarize(p, l, per)['MAE']
rng = np.random.default_rng(7)
a = [S(rm.crossval(Z, lab, per, fold, rng.permutation(low), task, 'F')[0]) for _ in range(20)]
b = [S(rm.crossval(Z, lab, per, fold, low, rng.permutation(task), 'F')[0]) for _ in range(20)]
people = sorted(set(per)); c = []
for _ in range(10):
    pm = dict(zip(people, rng.permutation(people))); l2 = lab.copy()
    for p in people:
        src = np.flatnonzero(per == pm[p]); dst = np.flatnonzero(per == p); l2[dst] = lab[src[np.arange(len(dst)) % len(src)]]
    c.append(S(rm.crossval(Z, l2, per, fold, low, task, 'F')[0], l2))
print(f"controls for F: class flag shuffled x20: min {min(a):.3f} mean {np.mean(a):.3f} | Task flag shuffled x20: min {min(b):.3f} mean {np.mean(b):.3f} | labels shuffled between people x10: mean {np.mean(c):.3f} min {min(c):.3f}")
MET['controls'] = dict(class_shuffle=a, task_shuffle=b, label_shuffle=c)
print('per-video MAE by stratum (diagnostic):        n   const    cal     v1     v2     v3')
strata = {'high rest': (low == 0) & (prot == 'Resting'), 'high Task': (low == 0) & task, 'high Walking': (low == 0) & (prot == 'Walking'), 'low not Task': (low == 1) & ~task, 'low Task': (low == 1) & task,
          'Normal': grp == 'Normal', 'Abnormal': grp == 'Abnormal'}
MET['strata'] = {}
for k, s in strata.items():
    v = [float(np.abs(q - lab)[s].mean()) for q in (const, cal, P['v1'], P['v2'], P['F'])]; MET['strata'][k] = dict(n=int(s.sum()), people=len(set(per[s])), const=v[0], cal=v[1], v1=v[2], v2=v[3], v3=v[4])
    print(f"   {k:14s} {int(s.sum()):3d} videos {len(set(per[s])):3d} people  " + '  '.join(f"{x:5.2f}" for x in v))
RV.mkdir(exist_ok=True)
with (RV / 'OOF_PREDICTIONS.csv').open('w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['recording', 'person', 'fold', 'group', 'protocol', 'low_class', 'truth_bpm', 'constant', 'calibrated', 'reader_v1', 'reader_v2', 'reader_v3'])
    for i, r in enumerate(rec): w.writerow([r, per[i], fold[i], grp[i], prot[i], low[i], lab[i], const[i], cal[i], P['v1'][i], P['v2'][i], P['F'][i]])
m = rm.fit(Z, lab, per, low, task, 'F')
model = dict(name='reader_v3', fitted_on='all 218 dev videos of VIDEOS.json (no holdout)', grid_bpm=GRID.tolist(), prior=m['prior'].tolist(), beta_by_class={'high': m['beta'][0], 'low': m['beta'][1]},
             offset_by_cell={'high_not_task': m['off'][0], 'high_task': m['off'][1], 'low_not_task': m['off'][2], 'low_task': m['off'][3]}, class_rule='low if width*height < 1800000 pixels else high',
             task_rule='Task1/Task2/Task3 in the protocol field', evidence='mean of standardised log CHROM and POS window-averaged spectra, frozen oval frontend and readout operator', prior_bandwidth_bpm=6.0, hop_s=5.0, min_cell=rm.MIN_N)
(RV / 'reader_model_dev218_v3.json').write_text(json.dumps(model, indent=1))
print('model on all dev: beta', model['beta_by_class'], 'offsets', {k: round(v, 2) for k, v in model['offset_by_cell'].items()})
(RV / 'METRICS.json').write_text(json.dumps(MET, indent=1))
print('sha256 model', hashlib.sha256((RV / 'reader_model_dev218_v3.json').read_bytes()).hexdigest())
