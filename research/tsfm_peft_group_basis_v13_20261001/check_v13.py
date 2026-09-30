"""One accounted CPU synthetic session; no real data or pretrained backbone."""
import argparse
import copy
import io
from types import SimpleNamespace

import torch
from torch import nn

from runtime_v13 import (HERE, DATASETS, Job, budget_snapshot, configure,
                        masked_macro_loss, save_json, source_receipt)
from model_v13 import GroupModel, state_digest
from train_v13 import assert_equal, optimizer_for


class SyntheticBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.3))
        self.chronos_config = SimpleNamespace(quantiles=[.5])

    def forward(self, context):
        point = context[:, -1:] + self.scale * context[:, -48:].tanh()
        return SimpleNamespace(quantile_preds=point[:, None, :])


def basis_for(channels, latent):
    basis = torch.zeros(channels, latent)
    for group in range(latent):
        members = torch.arange(group, channels, latent)
        basis[members, group] = len(members) ** -.5
    return basis


def frozen_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()
            if not name.startswith('residual.')}


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
    return {'masked_channel_macro_mse': float(loss), 'manual_mse': float(expected),
            'observed_zero_targets_retained': True, 'missing_gradient_zero': True}


def check_shape(dataset, job):
    channels, latent, _ = DATASETS[dataset]
    generator = torch.Generator().manual_seed(92601 + channels)
    basis = basis_for(channels, latent)
    initial = {'residual.down.weight': torch.randn(32, 512, generator=generator) * .02,
               'residual.up.weight': torch.zeros(48, 32)}
    x = torch.randn(2, 512, channels, generator=generator)
    target = torch.randn(2, 48, channels, generator=generator)
    mask = torch.rand(2, 48, channels, generator=generator) > .15
    model = GroupModel(dataset, 'group_res', 92601, basis, initial, SyntheticBackbone())
    raw = GroupModel(dataset, 'group_raw', 92601, basis, initial, SyntheticBackbone())
    assert model.adapter_state().keys() == raw.adapter_state().keys()
    assert sum(value.numel() for value in model.parameters() if value.requires_grad) == 17920
    frozen_before = state_digest(frozen_state(model))
    with torch.no_grad():
        main = model.main_prediction(x)
        expected = main + (x - (x @ basis) @ basis.T)[:, -1:, :].expand(-1, 48, -1)
        online = model(x)
        torch.testing.assert_close(online, expected, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(model.forecast_from_main(x, main), online, atol=0, rtol=0)
        torch.testing.assert_close(raw(x), online, atol=0, rtol=0)
        model.chunk_rows = 3
        torch.testing.assert_close(model(x), online, atol=1e-5, rtol=1e-4)
        model.chunk_rows = None
    optimizer, scheduler = optimizer_for(model, {'lr': .001})
    gradients = []
    for step in range(2):
        job.check_limits()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = masked_macro_loss(model.forecast_from_main(x, main), target, mask)
        loss.backward()
        down, up = model.residual.down.weight.grad, model.residual.up.weight.grad
        assert torch.isfinite(down).all() and torch.isfinite(up).all()
        assert torch.count_nonzero(up) > 0
        assert (torch.count_nonzero(down) == 0) if step == 0 else (torch.count_nonzero(down) > 0)
        assert all(value.grad is None for value in model.parameters() if not value.requires_grad)
        gradients.append({'step': step + 1, 'down_norm': float(down.norm()), 'up_norm': float(up.norm())})
        torch.nn.utils.clip_grad_norm_([value for value in model.parameters() if value.requires_grad], 1.)
        optimizer.step()
        job.updates += 1
        scheduler.step(float(loss))
        job.heartbeat({'dataset': dataset, 'synthetic_step': step + 1}, updates=job.updates)
    assert frozen_before == state_digest(frozen_state(model))
    assert torch.count_nonzero(model.residual.up.weight) > 0
    with torch.no_grad():
        trained = model.forecast_from_main(x, main)
        torch.testing.assert_close(model(x), trained, atol=0, rtol=0)
        raw.restore_adapter(model.adapter_state())
        raw_expected = main + (x - (x @ basis) @ basis.T)[:, -1:, :].expand(-1, 48, -1)
        raw_expected = raw_expected + raw.residual(x - x[:, -1:, :])
        torch.testing.assert_close(raw(x), raw_expected, atol=1e-5, rtol=1e-4)
    checkpoint = {'state': model.adapter_state(), 'optimizer': copy.deepcopy(optimizer.state_dict()),
                  'scheduler': copy.deepcopy(scheduler.state_dict()), 'rng': torch.get_rng_state()}
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    stream.seek(0)
    saved = torch.load(stream, map_location='cpu', weights_only=False)
    restored = GroupModel(dataset, 'group_res', 92601, basis, initial, SyntheticBackbone())
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
        torch.testing.assert_close(restored.forecast_from_main(x, main), trained, atol=0, rtol=0)
    return {'dataset': dataset, 'C': channels, 'K': latent, 'updates': 2,
            'initial_formula_and_res_raw_parity': True, 'online_cache_chunk_parity': True,
            'raw_trained_formula': True, 'gradient_steps': gradients, 'frozen_sha256': frozen_before,
            'frozen_parameters_no_gradient': True, 'trainable_parameters': 17920,
            'model_optimizer_scheduler_rng_roundtrip': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, default=90)
    args = parser.parse_args()
    with Job('cpu_check', args.job, reserve_s=args.reserve_s, synthetic=True,
             metadata={'purpose': 'Synthetic shape/mask/cache/freeze/gradient/restart checks; no real F0/data'}) as job:
        configure()
        report = {'schema': 'v13_synthetic_check_1', 'status': 'passed', 'ledger_id': job.id,
                  'source': source_receipt(), 'scope': 'Self-check, not independent reproduction; CPU synthetic only',
                  'mask': check_mask(), 'datasets': [check_shape(dataset, job) for dataset in DATASETS],
                  'updates': job.updates, 'real_model_runs': 0, 'real_fits': 0,
                  'budget_during_check': budget_snapshot()}
        assert job.updates == 8
        save_json(HERE / (args.job + '.json'), report)
        job.heartbeat({'checks': 'passed', 'updates': job.updates})


if __name__ == '__main__':
    main()
