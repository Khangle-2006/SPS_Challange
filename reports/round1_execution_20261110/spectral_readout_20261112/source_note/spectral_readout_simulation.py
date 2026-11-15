#!/usr/bin/env python3
"""Mechanistic experiment, not a BTC benchmark or a trained rPPG network.

Question: can a tiny target-domain readout use secondary spectral/color evidence
that a scalar argmax+affine calibrator discards? Synthetic RGB traces only; no
image codec, ISP, pigmentation model, learned V5 outputs, or real video here.
Generator/model settings were fixed before the first run. After that run we added
the identical KDE-prior-only control and label-permuted candidate diagnostic;
no generator, feature, regularizer, or optimizer settings changed. Exploratory,
not preregistered. Negative cases are intentional.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp
from scipy.ndimage import maximum_filter1d
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 20261112
SUBJECTS = 160
VIDEOS_PER_SUBJECT = 2
WINDOWS = 3
FS = 30
SECONDS = 20
PATCHES = 4
GRID = np.arange(42., 181.)
SIGMA_LABEL = 5.
RIDGE = .02
SCENARIOS = ['recoverable_color', 'indistinguishable_pair',
             'pulse_absent', 'reference_outside_video']
FEATURE_NAMES = ['log_POS_power', 'log_CHROM_power', 'log_G_power',
                 'C_rr', 'C_gg', 'C_bb', 'Re_C_rg', 'Re_C_rb', 'Re_C_gb',
                 'green_spatial_coherence']


def pos(rgb):
    r, g, b = np.moveaxis(rgb, -1, 0)
    x, y = g-b, g+b-2*r
    return x + x.std(-1, keepdims=True)/np.maximum(y.std(-1, keepdims=True), 1e-12)*y


def chrom(rgb):
    r, g, b = np.moveaxis(rgb, -1, 0)
    x, y = 3*r-2*g, 1.5*r+g-1.5*b
    return x - x.std(-1, keepdims=True)/np.maximum(y.std(-1, keepdims=True), 1e-12)*y


def extract(rgb):
    """Inputs are normalized AC/DC RGB, shape windows,patches,time,3."""
    n = rgb.shape[-2]
    hann = np.hanning(n)
    # nfft=1800 gives interpolated 1bpm grid; true 20s resolution is ~3bpm.
    fr = np.fft.rfftfreq(1800, 1/FS)*60
    idx = np.flatnonzero((fr >= 42-1e-9) & (fr <= 180+1e-9))
    z = np.fft.rfft(rgb*hann[None, None, :, None], n=1800, axis=2)[:, :, idx]
    roi = rgb.mean(1)
    signals = np.stack([pos(roi), chrom(roi), roi[:, :, 1]], -1)
    zz = np.fft.rfft(signals*hann[None, :, None], n=1800, axis=1)[:, idx]
    power = np.abs(zz)**2
    rel = np.log1p(power / np.maximum(np.median(power, axis=1, keepdims=True), 1e-12))
    csd = np.einsum('wkfc,wkfd->wfcd', z, z.conj()).real/PATCHES
    csd /= np.maximum(np.trace(csd, axis1=-2, axis2=-1)[..., None, None], 1e-12)
    geom = np.stack([csd[:, :, 0, 0], csd[:, :, 1, 1], csd[:, :, 2, 2],
                     csd[:, :, 0, 1], csd[:, :, 0, 2], csd[:, :, 1, 2]], -1)
    green = z[..., 1]
    coh = np.abs(green.sum(1))**2 / np.maximum(PATCHES*(np.abs(green)**2).sum(1), 1e-12)
    features = np.concatenate([rel, geom, coh[..., None]], -1)
    scalar = float(np.median(GRID[np.argmax(power, axis=1)]))
    agg = rel.mean((0, 2))
    ispeak = agg == maximum_filter1d(agg, size=7)
    order = np.flatnonzero(ispeak)[np.argsort(agg[ispeak])[::-1]]
    candidates = GRID[order[:3]]
    return features, scalar, candidates


def generate(scenario, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(FS*SECONDS)/FS
    rows = []
    for subject in range(SUBJECTS):
        base = float(np.clip(rng.normal(88, 16), 53, 133))
        # Shared person-dependent camera/color response; held out as a group.
        channel = rng.lognormal(0, .13, 3)
        pulse_color = np.array([.45, 1., .20])*channel
        artifact_color = np.array([.15, .60, .25])*channel
        for video in range(VIDEOS_PER_SUBJECT):
            h = np.clip(base + rng.normal(0, 4) + rng.normal(0, 3, WINDOWS), 48, 150)
            y = float(h[rng.integers(WINDOWS)] + rng.normal(0, SIGMA_LABEL))
            art_h = rng.uniform(45, 174, WINDOWS)
            pulse_amp = .0015*rng.lognormal(0, .18)
            art_amp = pulse_amp*rng.uniform(1.8, 3.2)
            qart = artifact_color
            if scenario == 'indistinguishable_pair':
                # Two exchangeable cardiac-like sources: no feature can identify
                # which is physiological, although the pair still constrains HR.
                a = np.clip(rng.normal(88, 16), 53, 133)
                b = np.clip(rng.normal(88, 16), 53, 133)
                track_a = np.clip(a + rng.normal(0, 3, WINDOWS), 48, 150)
                track_b = np.clip(b + rng.normal(0, 3, WINDOWS), 48, 150)
                true_track = track_a if rng.random() < .5 else track_b
                h, art_h = track_a, track_b
                y = float(true_track[rng.integers(WINDOWS)] + rng.normal(0, SIGMA_LABEL))
                art_amp = pulse_amp
                qart = pulse_color
            if scenario == 'reference_outside_video':
                # Label state is statistically independent of all observed RGB.
                y = float(rng.normal(88, np.sqrt(16**2+SIGMA_LABEL**2)))
            pulse = np.sin(2*np.pi*h[:, None]*t[None, :]/60 + rng.uniform(0, 2*np.pi, (WINDOWS, 1)))
            artifact = np.sin(2*np.pi*art_h[:, None]*t[None, :]/60 + rng.uniform(0, 2*np.pi, (WINDOWS, 1)))
            if scenario == 'pulse_absent':
                pulse *= 0
            gains = rng.uniform(.75, 1.25, (WINDOWS, PATCHES, 1, 1))
            rgb = (pulse_amp*gains*pulse[:, None, :, None]*pulse_color
                   + art_amp*artifact[:, None, :, None]*qart
                   + rng.normal(0, .004, (WINDOWS, PATCHES, len(t), 3)))
            # Add a smooth achromatic illumination component outside the HR band.
            rgb += .008*np.sin(2*np.pi*.17*t)[None, None, :, None]
            features, scalar, candidates = extract(rgb)
            rows.append((subject, y, features, scalar, candidates))
    return rows


def kde_prior(y):
    p = np.exp(-.5*((GRID[None, :]-y[:, None])/6.)**2).mean(0) + 1e-5
    return p/p.sum()


def median_grid(p):
    return GRID[np.argmax(np.cumsum(p, axis=-1) >= .5, axis=-1)]


def fit_ranker(x, y, x_eval, features):
    x, x_eval = x[..., features], x_eval[..., features]
    mean = x.mean((0, 1, 2), keepdims=True)
    std = np.maximum(x.std((0, 1, 2), keepdims=True), 1e-6)
    x, x_eval = (x-mean)/std, (x_eval-mean)/std
    logprior = np.log(kde_prior(y))
    kernel = np.exp(-.5*((y[:, None]-GRID)/SIGMA_LABEL)**2)/(SIGMA_LABEL*np.sqrt(2*np.pi))

    def objective(beta):
        score = np.einsum('nwfd,d->nwf', x, beta, optimize=False)+logprior
        q = softmax(score, axis=-1)
        qw = (q*kernel[:, None, :]).sum(-1)
        likelihood = np.maximum(qw.mean(-1), 1e-15)
        loss = -np.log(likelihood).mean() + .5*RIDGE*(beta@beta)
        dscore = q*(kernel[:, None, :]-qw[:, :, None])
        grad_i = np.einsum('nwf,nwfd->nd', dscore, x, optimize=False)/WINDOWS
        grad = -(grad_i/likelihood[:, None]).mean(0)+RIDGE*beta
        return loss, grad

    fit = minimize(objective, np.zeros(x.shape[-1]), method='L-BFGS-B', jac=True,
                   bounds=[(-5.,5.)]*x.shape[-1], options={'maxiter':200, 'ftol':1e-10})
    q = softmax(np.einsum('nwfd,d->nwf', x_eval, fit.x, optimize=False)+logprior, axis=-1).mean(1)
    # Predict the reference-device observation distribution, not only latent HR.
    obs = np.exp(-.5*((GRID[:, None]-GRID[None, :])/SIGMA_LABEL)**2)
    obs /= obs.sum(1, keepdims=True)
    pred = median_grid(q@obs)
    return pred, {'success':bool(fit.success),'message':str(fit.message),
                  'iterations':int(fit.nit),'coefficients':fit.x.tolist()}


def evaluate(rows, seed, shuffle_training_labels=False):
    subjects = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    x = np.stack([r[2] for r in rows])
    scalar = np.array([r[3] for r in rows])
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(SUBJECTS), 5)
    methods = ['constant', 'prior_distribution', 'scalar_affine', 'spectral_power_readout', 'spectral_color_readout']
    pred = {m:np.zeros(len(y)) for m in methods}
    fits = []
    for fold, heldout in enumerate(folds):
        te = np.isin(subjects, heldout)
        tr = ~te
        yy = y[tr].copy()
        if shuffle_training_labels:
            # Permute whole two-video person blocks; retain within-person labels.
            yy = yy.reshape(-1, VIDEOS_PER_SUBJECT)[rng.permutation(SUBJECTS-len(heldout))].reshape(-1)
        pred['constant'][te] = np.median(yy)
        obs = np.exp(-.5*((GRID[:, None]-GRID[None, :])/SIGMA_LABEL)**2)
        obs /= obs.sum(1, keepdims=True)
        pred['prior_distribution'][te] = median_grid(kde_prior(yy)@obs)
        design = np.column_stack([np.ones(tr.sum()), scalar[tr]])
        coef = np.linalg.lstsq(design, yy, rcond=None)[0]
        pred['scalar_affine'][te] = coef[0]+coef[1]*scalar[te]
        foldfits = {'fold':fold}
        for name, ff in [('spectral_power_readout',list(range(3))),
                         ('spectral_color_readout',list(range(len(FEATURE_NAMES))))]:
            pred[name][te], foldfits[name] = fit_ranker(x[tr], yy, x[te], ff)
        fits.append(foldfits)
    summary = {}
    for name, yy in pred.items():
        errors = np.abs(yy-y).reshape(SUBJECTS, VIDEOS_PER_SUBJECT).mean(1)
        rho = float(spearmanr(yy, y).statistic) if np.std(yy)>1e-12 else None
        summary[name] = {'person_mae':float(errors.mean()), 'spearman_video':rho}
    # Paired bootstrap of synthetic people: uncertainty conditional on this simulator.
    idx = rng.integers(0, SUBJECTS, (3000, SUBJECTS))
    d = (np.abs(pred['spectral_color_readout']-y)-np.abs(pred['scalar_affine']-y)).reshape(SUBJECTS, VIDEOS_PER_SUBJECT).mean(1)
    summary['color_minus_scalar'] = {'delta':float(d.mean()), 'ci95':np.quantile(d[idx].mean(1), [.025,.975]).tolist()}
    oracle = np.array([np.min(np.abs(r[4]-r[1])) for r in rows])
    summary['top3_oracle_diagnostic_only'] = {'person_mae':float(oracle.mean()),
                                             'within5':float((oracle<=5).mean())}
    chance = []
    for _ in range(100):
        shuffled = y.reshape(SUBJECTS, VIDEOS_PER_SUBJECT)[rng.permutation(SUBJECTS)].reshape(-1)
        chance.append(np.mean([np.min(np.abs(r[4]-label)) for r, label in zip(rows, shuffled)]))
    summary['top3_oracle_after_subject_label_permutation'] = {'mean_person_mae':float(np.mean(chance)),
                                                            'repetitions':100}
    return summary, fits


def phase_attenuation():
    half_range = .2
    f = np.array([1.,1.5,2.,3.])
    # E[sin(2pi f(t+delta))|t] for independent uniform timing jitter.
    a = np.sinc(2*f*half_range)
    return {'uniform_lag_seconds':[-half_range,half_range],
            'frequencies_hz':f.tolist(),'conditional_mean_signed_gain':a.tolist(),
            'assumption':'Unobserved independent uniform timing jitter, squared waveform loss; not a measured MCD offset distribution.'}


def main():
    out = Path(__file__).resolve().parent
    result = {'kind':'Synthetic mechanism study; NOT BTC results',
              'settings':{'seed':SEED,'subjects':SUBJECTS,'videos_per_subject':VIDEOS_PER_SUBJECT,
                          'windows':WINDOWS,'fs':FS,'seconds_per_window':SECONDS,'patches':PATCHES,
                          'label_sigma':SIGMA_LABEL,'ridge':RIDGE,'features':FEATURE_NAMES},
              'phase_jitter':phase_attenuation(),'scenarios':{}}
    for k, name in enumerate(SCENARIOS):
        rows = generate(name, SEED+100*k)
        summary, fits = evaluate(rows, SEED+2000+k)
        result['scenarios'][name] = {'metrics':summary,'fits':fits}
        print(name, json.dumps(summary), flush=True)
        if name == 'recoverable_color':
            summary, fits = evaluate(rows, SEED+2000+k, shuffle_training_labels=True)
            result['scenarios']['recoverable_color_shuffled_train_labels'] = {'metrics':summary,'fits':fits}
            print('shuffled_training_labels',json.dumps(summary),flush=True)
        (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    names = list(result['scenarios'])
    methods = ['constant','scalar_affine','spectral_power_readout','spectral_color_readout']
    labels = ['Train median','Argmax + affine','Spectra: 3 weights','Spectra + color: 10 weights']
    fig, ax = plt.subplots(figsize=(12,6))
    xx = np.arange(len(names))
    for j,(m,label) in enumerate(zip(methods,labels)):
        ax.bar(xx+(j-1.5)*.19,[result['scenarios'][n]['metrics'][m]['person_mae'] for n in names],width=.18,label=label)
    ax.set_xticks(xx, ['Distinct color\nsecondary pulse','Shuffled labels','Exchangeable\npulse/artifact','Pulse absent','Label outside\nvideo'])
    ax.set_ylabel('Synthetic held-out-person MAE (bpm)')
    ax.set_ylim(0,18.5)
    ax.set_title('Testing when a spectral readout can recover secondary pulse evidence\nSynthetic RGB traces; no codec, V5, or BTC videos')
    ax.legend(ncol=2,fontsize=9,loc='upper right')
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'spectral_readout_simulation.png', dpi=160)
    print('all_optimizers_successful',all(f[m]['success'] for s in result['scenarios'].values() for f in s['fits'] for m in ['spectral_power_readout','spectral_color_readout']),flush=True)


if __name__=='__main__':
    main()
