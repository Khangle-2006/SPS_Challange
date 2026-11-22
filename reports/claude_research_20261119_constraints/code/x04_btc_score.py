"""x04 - BTC dev (218 videos / 158 people): reader_v3 under every condition of out/<tag>_btc (built by x03), with the gates of PREREG.md.
  refit  : reader fitted on the 4 training folds of the SAME condition (primary estimate)
  frozen : reader fitted on the native training folds, applied to the condition's held-out fold (secondary)
  class  : 'low' if the video the pipeline sees has < 1.8 MP -> every video is 'low' under C4 (and under C5 down-scaled / face-crop modes)
  oa     : occlusion-aware evidence (rows flagged by colour excluded); reported for native (cost of false alarms) and C7
Person-first MAE, paired bootstrap over people (10,000, seed 0). No holdout access.
Usage: $PY code/x04_btc_score.py [tag=x01]  -> out/<tag>_btc_scores.csv, out/<tag>_btc_pred.npz"""
import sys, csv, json
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
sys.path.insert(0, str(OLD / 'code'))
from lib_btc import videos, meta, summarize, zlog, GRID, NUM
import reader_models as rm
tag = sys.argv[1] if len(sys.argv) > 1 else 'x01'
V = videos(); M = meta(); rec = np.array([v['recording'] for v in V]); lab = np.array([v['pr_bpm'] for v in V]); fold = np.array([v['fold'] for v in V]); per = np.array([v['person'] for v in V])
task = np.char.startswith(np.array([v['task'] for v in V]), 'Task'); low = np.array([int(M[r]['width']) * int(M[r]['height']) < 1800000 for r in rec]).astype(int)
rows = list(csv.DictReader((NUM / 'tables/round1_bayes_20261108/VIDEO_PREDICTIONS.csv').open()))
cd = {r['recording']: float(r['prediction']) for r in rows if r['method'] == 'constant' and r['condition'] == 'native'}; const = np.array([cd[r] for r in rec])
def ev(C, P):
    Z = np.zeros((len(rec), len(GRID))); ok = np.isfinite(C).all(1) & np.isfinite(P).all(1); Z[ok] = (zlog(C[ok]) + zlog(P[ok])) / 2; return Z, ok
def load(t, c):
    d = dict(np.load(W / 'out' / (t + '_btc') / (c + '.npz'))); assert (d['rec'] == rec).all(); return d
Dn = load('x01', 'native'); Zn, okn = ev(Dn['C'], Dn['P']); hn = Dn['have']; pnat = np.full(len(lab), np.nan); pnat[hn] = rm.crossval(Zn[hn], lab[hn], per[hn], fold[hn], low[hn], task[hn], 'F')[0]
def frozen(Z, lowc):
    out = np.zeros(len(lab))
    for f in sorted(set(fold)):
        tr = np.flatnonzero(fold != f); te = np.flatnonzero(fold == f); m = rm.fit(Zn[tr], lab[tr], per[tr], low[tr], task[tr], 'F'); out[te] = rm.predict(m, Z[te], lowc[te], task[te])
    return out
conds = sorted(p.stem for p in (W / 'out' / (tag + '_btc')).glob('*.npz')); conds = (['native'] if 'native' in conds else []) + [c for c in conds if c != 'native']
s0 = summarize(const[hn], lab[hn], per[hn]); sn = summarize(pnat[hn], lab[hn], per[hn], const[hn]); print('videos with a native re-extraction:', int(hn.sum()), '/ 218')
print(f"constant {s0['MAE']:.3f} RMSE {s0['RMSE']:.2f} | native re-extracted reader_v3 {sn['MAE']:.3f} [{sn['CI'][0]:.2f};{sn['CI'][1]:.2f}] RMSE {sn['RMSE']:.2f} r {sn['r']:.2f} vs const {sn['delta']:+.2f} [{sn['dCI'][0]:+.2f};{sn['dCI'][1]:+.2f}] | videos with evidence {int(okn.sum())}/218")
out = []; PRED = {'native_refit': pnat, 'const': const}
def cv(Z, lowc, sub, Ztrain=None, lowtrain=None):
    """reader fitted on the training folds (restricted to `sub`), predictions for the held-out fold; Ztrain = native evidence -> 'frozen'."""
    p = np.full(len(lab), np.nan); Zt = Z if Ztrain is None else Ztrain; lt = lowc if lowtrain is None else lowtrain
    for f in sorted(set(fold)):
        tr = np.flatnonzero((fold != f) & sub); te = np.flatnonzero((fold == f) & sub)
        if len(te): m = rm.fit(Zt[tr], lab[tr], per[tr], lt[tr], task[tr], 'F'); p[te] = rm.predict(m, Z[te], lowc[te], task[te])
    return p
for c in conds:
    d = load(tag, c); sub = d['have'] & Dn['have']; n = int(sub.sum())
    if n < 60: print(c, 'only', n, 'videos - skipped'); continue
    lowsee = np.ones(len(rec), int) if (c.startswith('c4') or c in ('c5_vga', 'c5_face')) else low; pn = pnat if n == len(rec) else cv(Zn, low, sub)      # native reader on the same videos
    for var in (['std', 'oa'] if (c.startswith('c7') or c == 'native' or 'c7' in c) else ['std']):
        for cls, lowc in ([('seen', lowsee)] + ([('source', low)] if (lowsee != low).any() else [])):
            Z, ok = ev(d['C'], d['P']) if var == 'std' else ev(d['Co'], d['Po'])
            p = cv(Z, lowc, sub); pf = cv(Z, lowc, sub, Zn, low); S = lambda a_, b_: summarize(a_[sub], lab[sub], per[sub], b_[sub])
            a = S(p, const); b = S(p, pn); f = S(pf, pn); fc = S(pf, const)
            cov = float(ok[okn & sub].mean()); wcov = float((d['n'] if var == 'std' else d['no'])[sub].sum() / max(Dn['n'][sub].sum(), 1))
            g = ('a' if cov >= .9 else '-') + ('b' if a['dCI'][0] <= 0 else '-') + ('c' if a['dCI'][1] < 0 else '-')
            name = c + ('' if var == 'std' else '+oa') + ('' if cls == 'seen' else ' [class from source]')
            print(f"{name:30s} n={n:3d}/{a['people']:3d} refit {a['MAE']:6.3f} [{a['CI'][0]:.2f};{a['CI'][1]:.2f}] RMSE {a['RMSE']:5.2f} r {a['r']:+.2f} | vs const {a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}] | vs native {b['delta']:+.2f} [{b['dCI'][0]:+.2f};{b['dCI'][1]:+.2f}]"
                  f" | frozen {f['MAE']:6.3f} vs native {f['delta']:+.2f} [{f['dCI'][0]:+.2f};{f['dCI'][1]:+.2f}] vs const {fc['delta']:+.2f} [{fc['dCI'][0]:+.2f};{fc['dCI'][1]:+.2f}]"
                  f" | cover video {cov:.3f} window {wcov:.3f} valid {np.nanmean(d['vf'][sub]):.3f} rows>0 {int((d['rows'][sub] > 0).sum())} | gates {g}", flush=True)
            out.append(dict(condition=c, variant=var, cls=cls, videos=n, people=a['people'], MAE=a['MAE'], lo=a['CI'][0], hi=a['CI'][1], RMSE=a['RMSE'], r=a['r'], d_const=a['delta'], dc_lo=a['dCI'][0], dc_hi=a['dCI'][1], d_native=b['delta'], dn_lo=b['dCI'][0], dn_hi=b['dCI'][1],
                            frozen_MAE=f['MAE'], fz_d_native=f['delta'], fz_lo=f['dCI'][0], fz_hi=f['dCI'][1], fz_d_const=fc['delta'], fzc_lo=fc['dCI'][0], fzc_hi=fc['dCI'][1], cover_video=cov, cover_window=wcov, valid=float(np.nanmean(d['vf'][sub])), rows_flagged=int((d['rows'][sub] > 0).sum()), gates=g))
            PRED[f'{c}|{var}|{cls}|refit'] = p; PRED[f'{c}|{var}|{cls}|frozen'] = pf
with (W / 'out' / (tag + '_btc_scores.csv')).open('w', newline='') as fh:
    w = csv.DictWriter(fh, list(out[0])); w.writeheader(); w.writerows(out)
np.savez_compressed(W / 'out' / (tag + '_btc_pred.npz'), rec=rec, lab=lab, per=per, fold=fold, task=task, low=low, **PRED)
