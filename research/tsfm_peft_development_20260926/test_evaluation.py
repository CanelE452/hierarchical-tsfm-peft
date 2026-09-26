import numpy as np
import pytest
import torch

from evaluate_development import paired_block_effect
from model import macro_loss
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


@pytest.mark.parametrize('missing', [False, True])
def test_global_count_microbatch_formula_matches_full_batch_gradient(missing):
    target = torch.zeros((4, 3, 3), dtype=torch.float64)
    mask = torch.ones_like(target, dtype=torch.bool)
    if missing:
        mask[1:, :, 0] = False
        mask[:, 0, 1] = False
        mask[:, :, 2] = False
    values = torch.tensor([10.0, 1.0, 2.0], dtype=torch.float64).expand_as(target)
    full = values.clone().requires_grad_()
    micro = values.clone().requires_grad_()
    reference = macro_loss(full, target, mask)
    reference.backward()
    count = mask.sum(dim=(0, 1))
    valid = count > 0
    micro_value = 0.0
    for i in range(len(micro)):
        total = ((micro[i:i + 1] - target[i:i + 1]).square() * mask[i:i + 1]).sum(dim=(0, 1))
        loss = (total[valid] / count[valid]).mean()
        micro_value += loss.item()
        loss.backward()
    assert micro_value == pytest.approx(reference.item(), abs=1e-12)
    torch.testing.assert_close(micro.grad, full.grad, rtol=1e-12, atol=1e-12)
    old = torch.stack([macro_loss(full[i:i + 1], target[i:i + 1], mask[i:i + 1]) for i in range(len(full))]).mean()
    if missing:
        assert reference.item() == 50.5
        assert old.item() == 13.375
    else:
        assert old.item() == reference.item()
