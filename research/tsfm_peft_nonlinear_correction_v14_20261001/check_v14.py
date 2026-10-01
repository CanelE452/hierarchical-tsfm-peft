"""One bounded CPU synthetic session: four shapes and eight optimizer updates."""
import argparse
import copy
import io

import torch
from torch import nn

from runtime_v14 import HERE, DATASETS, Job, budget_snapshot, configure, masked_macro_loss, save_json, source_receipt
from model_v14 import CorrectionModel, state_digest
from train_v14 import assert_equal, optimizer_for


class SyntheticParent(nn.Module):
    def __init__(self, direct=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.2 if direct else .3))
        self.calls = 0
        self.chunk_rows = None

    def forward(self, x):
        self.calls += 1
        return x[:, -1:, :] + self.scale * x[:, -48:, :].tanh()


def basis_for(channels, latent):
    basis = torch.zeros(channels, latent)
    for group in range(latent):
        members = torch.arange(group, channels, latent)
        basis[members, group] = len(members) ** -.5
    return basis


def check_mask():
    prediction = torch.tensor([[[1., 5., 2.], [3., 7., 4.]],
                               [[2., 6., 3.], [4., 8., 5.]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[:, :, 2] = False
    mask[0, 1, 1] = False
    target[~mask] = float('nan')
    loss = masked_macro_loss(prediction, target, mask)
    expected = (torch.tensor([1., 3., 2., 4.]).square().mean()
                + torch.tensor([5., 6., 8.]).square().mean()) / 2
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=0)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[~mask]) == 0
    assert torch.count_nonzero(prediction.grad[mask]) == int(mask.sum())
    return {'manual_mse': float(expected), 'observed_zero_retained': True, 'missing_gradient_zero': True}


def check_shape(dataset, job):
    channels, latent, _ = DATASETS[dataset]
    generator = torch.Generator().manual_seed(92601 + channels)
    basis = basis_for(channels, latent)
    initial = {'residual.down.weight': torch.randn(32, 512, generator=generator) * .02,
               'residual.up.weight': torch.zeros(48, 32)}
    x = torch.randn(2, 512, channels, generator=generator)
    target = torch.randn(2, 48, channels, generator=generator)
    mask = torch.rand(2, 48, channels, generator=generator) > .15
    model = CorrectionModel(dataset, 'level_gelu', 92601, basis, initial, SyntheticParent())
    linear = CorrectionModel(dataset, 'level_linear', 92601, basis, initial, SyntheticParent())
    direct = CorrectionModel(dataset, 'direct_gelu', 92601, basis, initial, SyntheticParent(True))
    for other in (linear, direct):
        assert_equal(model.adapter_state(), other.adapter_state())
    assert sum(value.numel() for value in model.parameters() if value.requires_grad) == 17920
    frozen_before = state_digest(model.parent.state_dict())
    with torch.no_grad():
        parent = model.parent_prediction(x)
        before_calls = model.parent.calls
        online = model(x)
        assert model.parent.calls == before_calls + 1
        torch.testing.assert_close(online, parent, atol=0, rtol=0)
        torch.testing.assert_close(linear(x), online, atol=0, rtol=0)
        torch.testing.assert_close(direct(x), direct.parent_prediction(x), atol=0, rtol=0)
        before_calls = model.parent.calls
        torch.testing.assert_close(model.forecast_from_parent(x, parent), online, atol=0, rtol=0)
        assert model.parent.calls == before_calls
        model.chunk_rows = 3
        assert model.parent.chunk_rows == 3
        model.chunk_rows = None
    optimizer, scheduler = optimizer_for(model, {'lr': .001})
    gradients = []
    for step in range(2):
        job.check_limits()
        model.train()
        assert not model.parent.training
        optimizer.zero_grad(set_to_none=True)
        loss = masked_macro_loss(model.forecast_from_parent(x, parent), target, mask)
        loss.backward()
        down, up = model.residual.down.weight.grad, model.residual.up.weight.grad
        assert torch.isfinite(down).all() and torch.isfinite(up).all()
        assert torch.count_nonzero(up) > 0
        assert (torch.count_nonzero(down) == 0) if step == 0 else (torch.count_nonzero(down) > 0)
        assert all(value.grad is None for value in model.parent.parameters())
        gradients.append({'step': step + 1, 'down_norm': float(down.norm()), 'up_norm': float(up.norm())})
        torch.nn.utils.clip_grad_norm_([value for value in model.parameters() if value.requires_grad], 1.)
        optimizer.step()
        job.updates += 1
        scheduler.step(float(loss))
        job.heartbeat({'dataset': dataset, 'synthetic_step': step + 1}, updates=job.updates)
    assert state_digest(model.parent.state_dict()) == frozen_before
    with torch.no_grad():
        trained = model.forecast_from_parent(x, parent)
        torch.testing.assert_close(model(x), trained, atol=0, rtol=0)
        centered = (x - x[:, -1:, :]).transpose(1, 2)
        expected = parent + model.residual.up(torch.nn.functional.gelu(model.residual.down(centered), approximate='none')).transpose(1, 2)
        torch.testing.assert_close(trained, expected, atol=0, rtol=0)
        linear.restore_adapter(model.adapter_state())
        expected_linear = parent + linear.residual.up(linear.residual.down(centered)).transpose(1, 2)
        torch.testing.assert_close(linear(x), expected_linear, atol=0, rtol=0)
        direct.restore_adapter(model.adapter_state())
        torch.testing.assert_close(direct(x) - direct.parent_prediction(x), trained - parent, atol=1e-6, rtol=1e-5)
    saved_head = model.adapter_state()
    checkpoint = {'state': saved_head, 'optimizer': copy.deepcopy(optimizer.state_dict()),
                  'scheduler': copy.deepcopy(scheduler.state_dict()), 'rng': torch.get_rng_state()}
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    stream.seek(0)
    saved = torch.load(stream, map_location='cpu', weights_only=False)
    restored = CorrectionModel(dataset, 'level_gelu', 92601, basis, initial)
    assert not hasattr(restored, 'parent')
    restored.restore_adapter(saved['state'])
    optimizer2, scheduler2 = optimizer_for(restored, {'lr': .001})
    optimizer2.load_state_dict(saved['optimizer'])
    scheduler2.load_state_dict(saved['scheduler'])
    assert_equal(optimizer.state_dict(), optimizer2.state_dict())
    assert_equal(scheduler.state_dict(), scheduler2.state_dict())
    torch.set_rng_state(saved['rng'])
    replay_random = torch.rand(8)
    torch.set_rng_state(saved['rng'])
    assert torch.equal(replay_random, torch.rand(8))
    with torch.no_grad():
        torch.testing.assert_close(restored.forecast_from_parent(x, parent), trained, atol=0, rtol=0)
    return {'dataset': dataset, 'C': channels, 'K': latent, 'updates': 2,
            'initial_equal_parent_all_families': True, 'no_double_level': True,
            'exact_gelu_and_identity_formulas': True, 'online_parent_called_once': True,
            'online_cache_parity': True, 'chunk_forwarded': True, 'gradient_steps': gradients,
            'frozen_parent_sha256': frozen_before, 'head_only_restore': True,
            'trainable_parameters': 17920, 'model_optimizer_scheduler_rng_roundtrip': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, default=90)
    args = parser.parse_args()
    with Job('cpu_check', args.job, reserve_s=args.reserve_s, synthetic=True,
             metadata={'purpose': 'Synthetic shapes/mask/formula/cache/freeze/gradients/restart; no real parents/data'}) as job:
        configure()
        report = {'schema': 'v14_synthetic_check_1', 'status': 'passed', 'ledger_id': job.id,
                  'source': source_receipt(), 'scope': 'Self-check, not independent reproduction; CPU synthetic only',
                  'mask': check_mask(), 'datasets': [check_shape(dataset, job) for dataset in DATASETS],
                  'updates': job.updates, 'real_model_runs': 0, 'real_fits': 0,
                  'budget_during_check': budget_snapshot()}
        assert job.updates == 8
        save_json(HERE / (args.job + '.json'), report)
        job.heartbeat({'checks': 'passed', 'updates': job.updates})


if __name__ == '__main__':
    main()
