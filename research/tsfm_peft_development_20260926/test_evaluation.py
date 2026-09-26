import numpy as np
import pytest

from evaluate_development import paired_block_effect
from run import score_arrays


def test_channel_macro_with_unequal_observation_counts():
    target = np.zeros((2, 2, 2))
    pred = np.broadcast_to([1.0, 3.0], target.shape)
    mask = np.ones_like(target, dtype=bool)
    mask[0, :, 0] = False
    result = score_arrays(pred, target, mask)
    assert result['channel_mse'] == [1.0, 9.0]
    assert result['mse'] == 5.0
    assert result['target_counts'] == [2, 4]


def test_effect_averages_seed_losses_and_preserves_paired_seeds():
    target = np.zeros((14, 2, 2))
    mask = np.ones_like(target, dtype=bool)
    predictions = [np.full_like(target, 0.5), np.ones_like(target)]
    references = [np.full_like(target, 2), np.full_like(target, 2)]
    result = paired_block_effect(predictions, references, target, mask)
    assert result['gain_pct'] == 84.375
    assert result['paired_seed_gain_pct'] == [93.75, 75.0]
    assert result['conditional_block95_pct'] == [84.375, 84.375]
    assert result['period_halves_gain_pct'] == [84.375, 84.375]


def test_sparse_time_blocks_do_not_silently_drop_channels():
    target = np.zeros((14, 2, 2))
    mask = np.ones_like(target, dtype=bool)
    mask[1:, :, 0] = False
    with pytest.raises(ValueError, match='no targets'):
        paired_block_effect([target], [np.ones_like(target)], target, mask)


def test_zero_loss_reference_has_no_percentage_effect():
    target = np.zeros((14, 2, 2))
    mask = np.ones_like(target, dtype=bool)
    with pytest.raises(ValueError, match='zero-loss'):
        paired_block_effect([target], [target], target, mask)
