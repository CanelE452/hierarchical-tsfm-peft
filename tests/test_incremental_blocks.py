import importlib
import importlib.util

import numpy as np
import pytest
import torch
from torch import nn

from research.informs_broad_validation_20261015_v1.training import macro_loss


BLOCK_MODULE = 'research.incremental_latent_adaptation_20261019_v1.blocks'
STAGE_MODULE = 'research.incremental_latent_adaptation_20261019_v1.c_stage'


def modules():
    assert importlib.util.find_spec(BLOCK_MODULE) is not None, 'incremental blocks are missing'
    assert importlib.util.find_spec(STAGE_MODULE) is not None, 'incremental C stage is missing'
    return importlib.import_module(BLOCK_MODULE), importlib.import_module(STAGE_MODULE)


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.length, self.horizon, self.channels, self.kmax = 64, 3, 3, 12
        self.front = nn.Linear(3, 3)
        self.register_buffer('center', torch.zeros(3))
        self.E = nn.Parameter(torch.randn(3, 12) * .1)
        self.D = nn.Parameter(torch.randn(7, 12, 3) * .1)


class Foundation(nn.Module):
    def __init__(self):
        super().__init__()
        self.streams = self.physical_streams = self.calls = 0

    def forward(self, z):
        self.calls += 1
        self.streams += len(z) * z.shape[-1]
        self.physical_streams += len(z) * z.shape[-1]
        return z[:, -1:] + .2 * z[:, -3:]


def data():
    rng = np.random.default_rng(72)
    x = rng.normal(size=(33, 64, 3)).astype(np.float32)
    g = rng.normal(size=(33, 3, 3)).astype(np.float32) * .1
    y = g + .2 * x[:, -3:]
    mask = np.ones_like(y, dtype=bool)
    mask[::3, :, 1] = False
    return dict(s_model=Parent(), x=x, g=g, y=y, mask=mask,
                origins=np.arange(100, 133),
                split={'fit_indices': np.arange(14), 'select_indices': np.arange(17, 33)})


def test_zero_prefix_skips_F_and_blocks_are_separate_parameters():
    blocks, _ = modules()
    parent = Parent()
    model = blocks.LatentBlocks(parent, 'JOINT_FULL')
    x = torch.randn(2, 64, 3)
    f = Foundation()
    assert torch.equal(model(x, f), torch.zeros(2, 3, 3))
    assert torch.equal(model.correction_for_prefix(x, 0, None), torch.zeros(2, 3, 3))
    assert f.calls == 1
    assert model.E_blocks[0].data_ptr() != model.E_blocks[1].data_ptr()
    assert model.D_blocks[0].data_ptr() != model.D_blocks[1].data_ptr()
    assert not any(p.requires_grad for p in model.front.parameters())
    torch.testing.assert_close(torch.cat(tuple(model.E_blocks), dim=1), parent.E)


def test_concatenation_matches_block_calls_and_later_block_preserves_prefix():
    blocks, _ = modules()
    model = blocks.LatentBlocks(Parent(), 'SEQ_BLOCKS')
    with torch.no_grad():
        model.D_blocks[0].normal_(std=.02)
        model.D_blocks[1].normal_(std=.02)
    x, f = torch.randn(3, 64, 3), Foundation()
    first = model.correction_for_prefix(x, 6, f).detach().clone()
    full = model.correction_for_prefix(x, 12, f)
    assert f.calls == 2
    separate = model.correction_for_prefix(x, 12, f, block_calls=True)
    torch.testing.assert_close(full, separate, atol=1e-7, rtol=1e-6)
    model.set_stage(1, correction_only=True)
    frozen = blocks.frozen_state_hash(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    model(x, f).square().mean().backward()
    optimizer.step()
    assert blocks.frozen_state_hash(model) == frozen
    assert torch.equal(first, model.correction_for_prefix(x, 6, f))


def test_joint_prefix_objective_uses_one_F_pass_and_both_losses():
    blocks, _ = modules()
    model = blocks.LatentBlocks(Parent(), 'JOINT_PREFIX')
    with torch.no_grad():
        for value in model.D_blocks:
            value.normal_(std=.02)
    x, g, y = torch.randn(4, 64, 3), torch.randn(4, 3, 3), torch.randn(4, 3, 3)
    mask = torch.rand_like(y) > .2
    counts = mask.sum((0, 1))
    f = Foundation()
    actual = model.training_loss(x, g, y, mask, f, counts, 4, 4)
    assert f.calls == 1
    half = model.correction_for_prefix(x, 6, f)
    full = model.correction_for_prefix(x, 12, f)
    expected = .5 * macro_loss(g+half, y, mask, counts, 4, 4) + .5 * macro_loss(g+full, y, mask, counts, 4, 4)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(value.grad.abs().sum() > 0 for value in model.E_blocks)


def test_prefix_exact_tie_chooses_G_only():
    _, stage = modules()
    chosen = stage.select_prefix([{'K': 12, 'M_select_mse': 1.},
                                  {'K': 6, 'M_select_mse': 1.},
                                  {'K': 0, 'M_select_mse': 1.}])
    assert chosen['K'] == 0


def test_stage_switch_clears_old_gradients_before_stage2_clipping():
    blocks, _ = modules()
    model = blocks.LatentBlocks(Parent(), 'SEQ_BLOCKS')
    x = torch.randn(3, 64, 3)
    model(x, Foundation()).sum().backward()
    assert model.D_blocks[0].grad is not None
    model.set_stage(1, correction_only=True)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.E_blocks[0].requires_grad and model.E_blocks[1].requires_grad


@pytest.mark.parametrize('method', ['SEQ_BLOCKS', 'JOINT_FULL', 'JOINT_PREFIX'])
def test_c_stage_uses_Mselect_then_fresh_all_Mcal_and_reloads(tmp_path, method):
    blocks, stage = modules()
    value = data()
    f = Foundation()
    result = stage.run({'device': 'cpu'}, value, 'jena', 92601, method, 3, f,
                       tmp_path, None, {'parent_sha': 'synthetic_A60'})
    assert result['status'] == 'COMPLETE'
    assert result['new_V_predictions_or_scores'] == 0
    assert result['selected_prefix'] in (0, 6, 12)
    assert len(result['prefix_selection_rows']) == 3
    assert all(part['fresh_initialization'] for part in result['refits'])
    assert all(part['n_fit'] == len(value['x']) for part in result['refits'])
    assert len(result['searches']) == (2 if method == 'SEQ_BLOCKS' else 1)
    if method == 'SEQ_BLOCKS':
        assert result['frozen_stage1_checks']['parameter_buffer_hash_unchanged']
        assert result['frozen_stage1_checks']['optimizer_checkpoint_hash_unchanged']
        assert len(result['stage1_caches']) == 2
        assert result['searches'][0]['sampled_indices_sha256'] == result['searches'][1]['sampled_indices_sha256']
    restored = blocks.restore(value['s_model'], result['state_path'])
    x = torch.as_tensor(value['x'][:2])
    assert restored.prefix_k == result['selected_prefix']
    assert restored(x, f).shape == (2, 3, 3)


def test_partial_C_returns_checkpoint_without_refit(tmp_path):
    _, stage = modules()
    result = stage.run({'device': 'cpu'}, data(), 'jena', 92601, 'SEQ_BLOCKS', 3,
                       Foundation(), tmp_path, 0, {'parent_sha': 'synthetic_A60'})
    assert result['status'] == 'PARTIAL_CHECKPOINTED'
    assert result['refits'] == []


def test_seq_resume_keeps_selection_and_original_cache_cost_receipts(tmp_path):
    blocks, stage = modules()
    value = data()
    args = ({'device': 'cpu'}, value, 'jena', 92601, 'SEQ_BLOCKS', 3,
            Foundation(), tmp_path)
    stage.run(*args, deadline=0, contract={'parent_sha': 'synthetic_A60'})
    first = stage.run(*args, contract={'parent_sha': 'synthetic_A60'})
    from pathlib import Path
    seal = Path(tmp_path/'SELECTION_SEAL.json').read_bytes()
    receipt = Path(tmp_path/'search1/STAGE1_CACHE.json').read_bytes()
    restored = blocks.restore(value['s_model'], first['state_path'])
    before = restored(torch.as_tensor(value['x'][:2]), Foundation())
    resumed = stage.run(*args, contract={'parent_sha': 'synthetic_A60'})
    assert resumed['status'] == 'COMPLETE'
    assert Path(tmp_path/'SELECTION_SEAL.json').read_bytes() == seal
    assert Path(tmp_path/'search1/STAGE1_CACHE.json').read_bytes() == receipt
    after = blocks.restore(value['s_model'], resumed['state_path'])(torch.as_tensor(value['x'][:2]), Foundation())
    assert torch.equal(before, after)
