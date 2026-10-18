import copy
import pytest
from research.b2_revin_20261018_v1.contracts import needs_extension, common_normalization


def group():
    return [dict(spec=dict(dataset='jena', seed=seed, revin=enabled),
                 curves=[dict(epoch=e, astop_mse=v) for e, v in ((0, 3.), (.25, 2.), (1, 1.), (3, .9))])
            for enabled in (False, True) for seed in (92601, 92602)]


def test_extension_checks_whole_group_before_early_positive():
    values = group()
    values[-1]['curves'].pop()
    with pytest.raises(ValueError):
        needs_extension(values, 3)


def test_extension_requires_both_seeds_and_normalizations():
    values = group()
    values[-1] = copy.deepcopy(values[0])
    with pytest.raises(ValueError):
        needs_extension(values, 3)


def test_extension_strict_best_and_tie_does_not_trigger():
    values = group()
    assert needs_extension(values, 3)
    for value in values:
        value['curves'][-1]['astop_mse'] = 1.
    assert not needs_extension(values, 3)


def test_normalization_is_two_seed_astop_mean_tie_off():
    values = group()
    assert common_normalization(values)[0] is False
    values[2]['curves'][-1]['astop_mse'] = .5
    assert common_normalization(values)[0] is True
