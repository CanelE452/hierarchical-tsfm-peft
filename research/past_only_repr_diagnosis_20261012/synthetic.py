"""Declared six-channel lagged systems; synthetic feasibility, not real-data evidence."""
import numpy as np

from common import H, L, ahash, select_origins

CONDITIONS = ('unrelated', 'fixed', 'changed')
SYNTHETIC_SEEDS = tuple(range(92610, 92620))
LENGTH, BURN_IN = 4096, 512
INITIAL_SOURCES, INITIAL_LAGS = (0, 1, 2), (4, 8, 16)
CHANGED_SOURCES, CHANGED_LAGS = (1, 2, 0), (16, 4, 8)


def make_record(condition, seed, noise=True):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    fit_end, tune_end = int(.6*LENGTH), int(.8*LENGTH)
    change = tune_end + (LENGTH-tune_end)//2
    rng = np.random.default_rng(seed)
    innovations = rng.standard_normal((BURN_IN+LENGTH, 6)) if noise else np.zeros((BURN_IN+LENGTH, 6))
    if not noise:
        innovations[:, :3] = rng.standard_normal((BURN_IN+LENGTH, 3))
    full = np.zeros_like(innovations)
    source_at, lag_at = [], []
    for t in range(1, len(full)):
        full[t, :3] = .3*full[t-1, :3] + innovations[t, :3]
        switched = condition == 'changed' and t >= BURN_IN+change
        sources = CHANGED_SOURCES if switched else INITIAL_SOURCES
        lags = CHANGED_LAGS if switched else INITIAL_LAGS
        for receiver, (source, lag) in enumerate(zip(sources, lags)):
            cross = 0. if condition == 'unrelated' else .6
            full[t, receiver+3] = .3*full[t-1, receiver+3] + cross*(full[t-lag, source] if t >= lag else 0.) + innovations[t, receiver+3]
        if t >= BURN_IN:
            source_at.append(sources)
            lag_at.append(lags)
    physical = full[BURN_IN:].astype(np.float64)
    if not np.isfinite(physical).all():
        raise ValueError('Synthetic non-finite values')
    mean, std = physical[:fit_end].mean(0), physical[:fit_end].std(0)
    std = np.where(std > 1e-12, std, 1.)
    x = (physical-mean)/std
    origins = {'fit': select_origins(np.arange(L, fit_end-H+1), 256),
               'tune': select_origins(np.arange(fit_end, tune_end-H+1), 64),
               'val': np.arange(tune_end, LENGTH-H+1)}
    metadata = {'condition': condition, 'seed': seed, 'effective_length': LENGTH, 'burn_in': BURN_IN,
                'split_bounds': {'fit': [0, fit_end], 'tune': [fit_end, tune_end], 'val': [tune_end, LENGTH]},
                'change_timestamp_effective': change if condition == 'changed' else None,
                'change_timestamp_with_burn_in': BURN_IN+change if condition == 'changed' else None,
                'ar_coefficient': .3, 'cross_coefficient': 0. if condition == 'unrelated' else .6,
                'initial_sources': INITIAL_SOURCES, 'initial_lags': INITIAL_LAGS,
                'changed_sources': CHANGED_SOURCES, 'changed_lags': CHANGED_LAGS,
                'noise_distribution': 'independent standard normal; source and receiver innovations',
                'source_to_receiver': [[s, r+3] for r, s in enumerate(INITIAL_SOURCES)],
                'feedback_cycles': False, 'finite': True, 'triangular_ar_eigenvalues': [.3]*6,
                'physical_hash': ahash(physical), 'innovation_hash': ahash(innovations),
                'interpretation': '10 fixed independent noise records per condition; no neural models or real-data generalization claim'}
    data = {'dataset': f'synthetic_{condition}', 'x': x, 'physical': physical, 'filled': physical,
            'mask': np.ones(physical.shape, dtype=bool), 'mean': mean, 'std': std, 'origins': origins,
            'fit_bounds': (0, fit_end), 'tune_bounds': (fit_end, tune_end), 'val_bounds': (tune_end, LENGTH),
            'columns': ['u1', 'u2', 'u3', 'v1', 'v2', 'v3'], 'times': np.arange(LENGTH, dtype=np.int64),
            'interval_seconds': 1, 'source': 'synthetic.py generator', 'source_hash': metadata['physical_hash'],
            'normalization_fit_scope': 'FIT_ONLY', 'metadata': metadata}
    return data, {'physical': physical, 'innovations_including_burn_in': innovations,
                  'source_for_receiver': np.asarray(source_at), 'lag_for_receiver': np.asarray(lag_at)}


def noiseless_alignment():
    checks = []
    for condition in CONDITIONS:
        data, arrays = make_record(condition, SYNTHETIC_SEEDS[0], noise=False)
        x, change = arrays['physical'], data['metadata']['change_timestamp_effective']
        cross = data['metadata']['cross_coefficient']
        maximum = 0.
        for t in range(max(INITIAL_LAGS), LENGTH):
            sources = CHANGED_SOURCES if change is not None and t >= change else INITIAL_SOURCES
            lags = CHANGED_LAGS if change is not None and t >= change else INITIAL_LAGS
            for receiver, (source, lag) in enumerate(zip(sources, lags)):
                actual = x[t, receiver+3]-.3*x[t-1, receiver+3]
                expected = cross*x[t-lag, source]
                maximum = max(maximum, abs(actual-expected))
        checks.append({'check': f'{condition}_generation_equation', 'maximum_absolute_error': maximum,
                       'passed': maximum < 1e-12})
    origin = 1000
    for lag in INITIAL_LAGS + CHANGED_LAGS:
        for horizon_zero_based in range(lag):
            feature_lag = lag-horizon_zero_based-1
            source_timestamp = origin+horizon_zero_based-lag
            checks.append({'check': 'known_source_target_alignment', 'source_lag': lag,
                           'horizon': horizon_zero_based+1, 'feature_lag_from_last_observed': feature_lag,
                           'source_timestamp': source_timestamp,
                           'passed': source_timestamp == origin-1-feature_lag and 0 <= feature_lag <= 32})
    if not all(row['passed'] for row in checks):
        raise ValueError('Declared synthetic lag direction/alignment failed')
    return {'status': 'PASS', 'checks': checks,
            'note': 'Equation and timestamp alignment only; noisy prediction improvement must be tested separately'}
