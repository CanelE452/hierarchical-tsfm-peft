"""Small independent mathematical fixtures; no scientific assets/models/GPU."""
from itertools import combinations
import inspect

import numpy as np
import pytest

from research.smallk_failure_followup_20261014 import assess_channel_budget as cb


def tiny_values():
    y = np.array([[[np.sqrt(1.5), 1.], [np.sqrt(1.5), np.nan]],
                  [[0., np.nan], [0., np.nan]]], np.float32)
    observed = np.isfinite(y)
    raw = np.nan_to_num(y).astype(np.float32)
    return {'raw_forecast': raw, 'target': y, 'observed': observed,
            'raw_last': np.zeros((2, 2), np.float32),
            'history': np.zeros((2, 128, 2), np.float32)}


def full_fixture():
    rng = np.random.default_rng(92601)
    result = {}
    for split in ('fit', 'tune', 'val'):
        history = rng.normal(size=(3, 512, 5)).astype(np.float32)
        raw = np.repeat(history[:, -1:, :] + np.array([.1, .3, -.1, .2, -.2], np.float32), 48, axis=1)
        target = raw + rng.normal(0, .1, raw.shape).astype(np.float32)
        observed = rng.random(raw.shape) > .3
        observed[:, :, 4] = False
        target[~observed] = np.nan
        result[split] = {'history': history, 'history_observed': np.ones_like(history, bool),
            'raw_last': history[:, -1].copy(), 'raw_forecast': raw, 'target': target,
            'observed': observed, 'origins': np.arange(3, dtype=np.int64) * 60}
    return result


def test_stable_ties_zero_and_negative_scores_use_physical_index():
    assert cb.stable_topk([0, 0, 0], 2).tolist() == [[0, 1]]
    assert cb.stable_topk([-.5, -1., -.5], 2).tolist() == [[0, 2]]
    with pytest.raises(ValueError, match='FINITE'):
        cb.stable_topk([np.nan, 1], 1)
    with pytest.raises(ValueError, match='VALID_K'):
        cb.stable_topk([0, 1], 3)


def test_future_oracle_uses_global_channel_denominator_not_query_mean():
    values = tiny_values()
    evidence = cb.losses_and_gains(values)
    assert evidence['full_split_observed_count'].tolist() == [4, 1]
    assert cb.stable_topk(evidence['gain_macro_query'], 1)[0].tolist() == [1]
    naive_query_mean = evidence['gain_sse_query'][0] / values['observed'][0].sum(0)
    assert int(np.argmax(naive_query_mean)) == 0
    methods, _, _ = cb.simulate_split(values, 1, np.array([0], np.int64), np.ones(2, np.float32))
    assert np.isclose(methods['FUTURE_ORACLE_TOPK']['normalized']['mse'], .375)
    assert np.isclose(methods['FIT_FIXED_TOPK']['normalized']['mse'], .5)


def test_oracle_matches_exhaustive_restricted_menu_and_decomposes_gain():
    rng = np.random.default_rng(92601)
    n, h, c, k = 3, 4, 5, 2
    y = rng.normal(size=(n, h, c)).astype(np.float32)
    observed = rng.random(y.shape) > .25
    y[~observed] = np.nan
    values = {'target': y, 'observed': observed,
        'raw_last': rng.normal(size=(n, c)).astype(np.float32),
        'raw_forecast': rng.normal(size=(n, h, c)).astype(np.float32),
        'history': rng.normal(size=(n, 128, c)).astype(np.float32)}
    evidence = cb.losses_and_gains(values)
    indices = cb.stable_topk(evidence['gain_macro_query'], k)
    for query in range(n):
        optimum = max(sum(evidence['gain_macro_query'][query, index] for index in selected)
                      for selected in combinations(range(c), k))
        assert np.isclose(evidence['gain_macro_query'][query, indices[query]].sum(), optimum)
    methods, _, _ = cb.simulate_split(values, k, np.array([0, 1], np.int64), np.ones(c, np.float32))
    oracle = methods['FUTURE_ORACLE_TOPK']
    assert np.isclose(methods['LAST']['normalized']['mse'] - oracle['normalized']['mse'], oracle['macro_MSE_gain_vs_LAST'])


def test_exact_K_oracle_can_be_worse_than_LAST_when_all_gains_negative():
    values = {'target': np.zeros((2, 3, 3), np.float32), 'observed': np.ones((2, 3, 3), bool),
        'raw_forecast': np.ones((2, 3, 3), np.float32), 'raw_last': np.zeros((2, 3), np.float32),
        'history': np.zeros((2, 128, 3), np.float32)}
    methods, _, _ = cb.simulate_split(values, 1, np.array([0], np.int64), np.ones(3, np.float32))
    assert methods['LAST']['normalized']['mse'] == 0
    assert methods['FUTURE_ORACLE_TOPK']['normalized']['mse'] > 0
    assert methods['FUTURE_ORACLE_TOPK']['indices'].tolist() == [[0], [0]]


def test_FIT_fixed_and_past_variance_ids_ignore_evaluation_future_targets():
    values = full_fixture()
    mean, std = np.zeros(5, np.float32), np.ones(5, np.float32)
    before, arrays_before = cb.assess(values, mean, std, 2)
    changed = {split: dict(payload) for split, payload in values.items()}
    changed['val']['target'] = values['val']['target'] + np.array([10, -20, 30, -40, 0], np.float32)
    after, arrays_after = cb.assess(changed, mean, std, 2)
    assert before['fixed_FIT_indices'] == after['fixed_FIT_indices']
    for method in ('FIT_FIXED_TOPK', 'PAST_VARIANCE_TOPK'):
        assert np.array_equal(arrays_before['val'][method + '_indices'], arrays_after['val'][method + '_indices'])
    assert not np.array_equal(arrays_before['val']['FUTURE_ORACLE_TOPK_indices'], arrays_after['val']['FUTURE_ORACLE_TOPK_indices'])


def test_past_variance_uses_exact_last128_and_never_targets():
    history = np.zeros((1, 512, 3), np.float32)
    history[0, :384, 0] = np.arange(384)
    history[0, -128:, 1] = np.arange(128)
    assert cb.stable_topk(cb.past_variance_scores(history), 1).tolist() == [[1]]
    assert tuple(inspect.signature(cb.past_variance_scores).parameters) == ('history',)


def test_channel_macro_mask_and_zero_observed_channel_are_not_pooled():
    values = tiny_values()
    prediction = np.zeros_like(values['raw_forecast'])
    score = cb.observed_score(prediction, values['target'], values['observed'])
    assert np.isclose(score['mse'], .875)
    pooled = float(np.mean(np.nan_to_num(values['target'])[values['observed']] ** 2))
    assert not np.isclose(score['mse'], pooled)
    values = full_fixture()
    result, arrays = cb.assess(values, np.zeros(5, np.float32), np.ones(5, np.float32), 2)
    assert result['fit_channel_normalized_MSE_gain'][4] == 0
    assert result['splits']['val']['methods']['LAST']['normalized']['per_channel_mse'][4] is None
    assert arrays['fit']['LAST_indices'].shape == (3, 0)


def test_methods_predeclared_and_not_deployment_or_universal_bound():
    assert cb.METHODS == ('LAST', 'FIT_FIXED_TOPK', 'PAST_VARIANCE_TOPK', 'FUTURE_ORACLE_TOPK')
    assert cb.DECLARATION['actual_K_channel_F_queries'] is False
    assert cb.DECLARATION['future_oracle_deployable'] is False
    assert cb.DECLARATION['universal_forecasting_bound'] is False
    assert cb.DECLARATION['strict_K_only_compression'] is False
    source = inspect.getsource(cb)
    assert 'torch.load(' not in source
    assert 'import_old_engine(' not in source
    assert 'FrozenBolt(' not in source
