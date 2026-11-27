"""Reproduce the existing nested person-fold fits; no search beyond the frozen recipe."""
import numpy as np
from reader_v4 import MODELS, evidence
from . import calibrate, conditional_reader as anchor, fallback_reader as fallback, router
from .router_fit import fit_router


def fit_models(data, outer_folds=(0, 1, 2, 3, 4, None), progress=None):
    y, people, fold, low, task = [data[k] for k in ('truth', 'person', 'fold', 'low', 'task')]
    for person in np.unique(people):
        if len(np.unique(fold[people == person])) != 1:
            raise ValueError('A person occurs in multiple folds')
    if set(fold) != set(range(5)):
        raise ValueError('The frozen recipe needs all five person folds')
    native = router.sources(np.stack([data['C'], data['P']], axis=1), data)
    shifted = router.sources(np.stack([data['sC'], data['sP']], axis=1), data)
    ztrain, ztest = router.evidence(native), router.evidence(shifted)
    zn = evidence(data['C'], data['P'], data)
    zs = evidence(data['sC'], data['sP'], data)
    reader_cache, anchor_cache, result = {}, {}, {}

    def readers(excluded):
        key = tuple(sorted(excluded))
        if key not in reader_cache:
            tr = ~np.isin(fold, key)
            reader_cache[key] = [calibrate.fit('median_offset', ztrain[tr, j], y[tr],
                people[tr], low[tr], task[tr], fold[tr]) for j in range(6)]
        return reader_cache[key]

    def anchor_model(rows):
        key = tuple(np.unique(fold[rows]))
        if key not in anchor_cache:
            anchor_cache[key] = anchor.fit('shift10_median', zn[rows], zs[rows],
                y[rows], people[rows], low[rows], task[rows])
        return anchor_cache[key]

    def apply_anchor(model, rows):
        return anchor.predict(model, zs[rows], low[rows], task[rows])

    def pulse(rows, pred):
        return fallback.local_pulse_features(pred, data['sC'][rows], data['sP'][rows],
                                             {m: data[m][rows] for m in MODELS})

    for outer in outer_folds:
        suffix = 'deploy' if outer is None else str(outer)
        if progress:
            progress('Fitting ' + suffix)
        excluded = [] if outer is None else [outer]
        mask = ~np.isin(fold, excluded)
        inner = np.empty((len(y), 6))
        for f in np.unique(fold[mask]):
            ids = fold == f
            inner[ids] = np.column_stack([calibrate.predict(m, ztest[ids, j], low[ids], task[ids])
                                         for j, m in enumerate(readers([*excluded, f]))])
        err = abs(inner[mask] - y[mask, None])
        routing = fit_router(router.features(shifted[mask], inner[mask], low[mask], task[mask]),
                             err[:, 1:] - err[:, :1], people[mask])
        result['router_' + suffix + '.json'] = dict(router=routing, readers=readers(excluded))

        rows = np.flatnonzero(mask)
        second_level = np.empty(len(y))
        for g in np.unique(fold[rows]):
            tr = rows[fold[rows] != g]
            te = rows[fold[rows] == g]
            if set(people[tr]) & set(people[te]):
                raise ValueError('Person leakage in fallback inner folds')
            second_level[te] = apply_anchor(anchor_model(tr), te)
        jl = rows[low[rows].astype(bool)]
        if not len(jl):
            raise ValueError('No low-bitrate train rows for fallback')
        p = second_level[jl]
        extra = pulse(jl, p)
        # The shipped fold files contain signal only; deploy also preserves capture.
        methods = ('capture', 'signal') if outer is None else ('signal',)
        fitted = {method: fallback.fit(method, data['capture'][jl], data['trace'][jl],
                                      extra, y[jl] - p, people[jl]) for method in methods}
        result['fallback_' + suffix + '.json'] = dict(anchor=anchor_model(rows), fallback=fitted)
    return result
