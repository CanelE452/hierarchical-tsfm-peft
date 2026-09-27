"""Synthetic CPU checks: one session, ten optimizer updates total; no real data."""
import argparse
import copy
import io
from types import SimpleNamespace

from runtime_v4 import HERE, Job, save_json


def run_checks(job):
    import numpy as np
    import torch
    from torch import nn
    from model_v4 import V3_MODEL, build_with_backbone, initial_edg, macro_loss, tensor_hashes
    from train_v4 import assert_nested_equal, optimizer_for, restore_rng, rng_state

    torch.set_num_threads(4)
    torch.manual_seed(92704)
    np.random.seed(92704)
    steps = 0
    records = {}

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.8))
            self.chronos_config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9])

        def forward(self, context, target=None, target_mask=None):
            value = (context[:, -48:] * self.scale + context.mean(-1, keepdim=True) * 0.2)
            return SimpleNamespace(quantile_preds=torch.stack([value - 0.1, value, value + 0.1], dim=1))

    basis = torch.linalg.qr(torch.randn(32, 8)).Q
    before_rng = torch.get_rng_state().clone()
    initial = initial_edg(basis, 92601)
    assert torch.equal(before_rng, torch.get_rng_state())
    initial_hashes = tensor_hashes(initial)
    x, y = torch.randn(2, 512, 32), torch.randn(2, 48, 32)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0, :10, 0] = False
    modified = y.clone()
    modified[~mask] = 100000
    prediction = torch.randn_like(y)
    assert torch.equal(macro_loss(prediction, y, mask), macro_loss(prediction, modified, mask))
    expected = (((prediction-y).square()*mask).sum((0, 1))/mask.sum((0, 1))).mean()
    assert torch.equal(macro_loss(prediction, y, mask), expected)

    for family in ('tsfm_res', 'linear_res'):
        for mode in ('current', 'fixed_ed'):
            name = family + '_' + mode
            spec = {'id': 'synthetic_' + name, 'dataset': 'synthetic', 'family': family,
                    'arm': 'residual', 'ed_mode': mode, 'seed': 92601, 'latent': 8,
                    'normalization': 'context', 'lr': 1e-3, 't_lr': 1e-3, 'epochs': 120}
            before_rng = torch.get_rng_state().clone()
            model = (V3_MODEL.LinearResidual(spec, initial) if family == 'linear_res'
                     else build_with_backbone(spec, basis, initial, FakeBackbone()))
            assert torch.equal(before_rng, torch.get_rng_state())
            assert tensor_hashes({k: model.adapter_state()[k] for k in initial}) == initial_hashes
            opt, scheduler = optimizer_for(model, spec)
            initial_model = model.adapter_state()
            if family == 'linear_res':
                assert not hasattr(model, 'backbone')
                assert {id(p) for p in model.temporal.parameters()}.isdisjoint({id(p) for p in model.residual.parameters()})
            main, correction = model.components(x)
            assert correction.count_nonzero().item() == 0 and main.shape == y.shape
            if family == 'tsfm_res':
                raw_spec = dict(spec, family='tsfm_raw', arm='raw_bypass')
                compress_spec = dict(spec, family='compress', arm='compress')
                raw_model = build_with_backbone(raw_spec, basis, initial, FakeBackbone())
                compress_model = build_with_backbone(compress_spec, basis, initial, FakeBackbone())
                with torch.no_grad():
                    assert torch.equal(model(x), raw_model(x))
                    assert torch.equal(model(x), compress_model(x))
                del raw_model, compress_model
            gradients = []
            frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
            for update in range(2):
                job.check_limits()
                opt.zero_grad(set_to_none=True)
                loss = macro_loss(model(x), y, mask)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                down = model.residual.down.weight.grad.abs().max().item()
                assert (down == 0) if update == 0 else (down > 0)
                if mode == 'current':
                    assert model.encoder.weight.grad is not None and model.encoder.weight.grad.abs().max() > 0
                    assert model.decoder.weight.grad is not None and model.decoder.weight.grad.abs().max() > 0
                else:
                    assert model.encoder.weight.grad is None and model.decoder.weight.grad is None
                if family == 'linear_res':
                    assert model.temporal.weight.grad.abs().max() > 0 and model.temporal.bias.grad.abs().max() > 0
                gradients.append({'g_down_max_abs': down, 'loss': float(loss.detach())})
                opt.step()
                steps += 1
                scheduler.step(float(loss.detach()))
                job.heartbeat(name + f' synthetic update {update+1}', steps)
            final = model.adapter_state()
            assert final['residual.down.weight'].ne(initial_model['residual.down.weight']).any()
            assert final['residual.up.weight'].ne(initial_model['residual.up.weight']).any()
            for key in ('encoder.weight', 'decoder.weight'):
                assert torch.equal(final[key], initial_model[key]) == (mode == 'fixed_ed')
            for key, parameter in model.named_parameters():
                if key in frozen:
                    assert torch.equal(parameter, frozen[key]) and parameter.grad is None
            records[name] = {'first_two_updates': gradients, 'freeze_and_actual_updates': True,
                             'initial_shared_hashes': initial_hashes}
            if family == 'linear_res' and mode == 'current':
                restart_spec, restart_model, restart_opt, restart_scheduler = spec, model, opt, scheduler

    # Two additional optimizer calls verify resumed continuation against uninterrupted continuation.
    saved = {'model': restart_model.adapter_state(), 'optimizer': copy.deepcopy(restart_opt.state_dict()),
             'scheduler': copy.deepcopy(restart_scheduler.state_dict()), 'rng': rng_state()}
    stream = io.BytesIO()
    torch.save(saved, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=False)
    restored = V3_MODEL.LinearResidual(restart_spec, initial)
    restored.restore_adapter(saved['model'])
    ropt, rscheduler = optimizer_for(restored, restart_spec)
    ropt.load_state_dict(saved['optimizer'])
    rscheduler.load_state_dict(saved['scheduler'])
    assert_nested_equal(restart_model.adapter_state(), restored.adapter_state())
    assert_nested_equal(restart_opt.state_dict(), ropt.state_dict())
    assert_nested_equal(restart_scheduler.state_dict(), rscheduler.state_dict())
    with torch.no_grad():
        assert torch.equal(restart_model(x), restored(x))
    for model, opt, scheduler in ((restart_model, restart_opt, restart_scheduler), (restored, ropt, rscheduler)):
        restore_rng(saved['rng'])
        assert_nested_equal(rng_state(), saved['rng'])
        opt.zero_grad(set_to_none=True)
        value = macro_loss(model(x), y, mask)
        value.backward()
        opt.step()
        steps += 1
        scheduler.step(float(value.detach()))
        job.heartbeat('restart continuation check', steps)
    assert_nested_equal(restart_model.adapter_state(), restored.adapter_state())
    assert_nested_equal(restart_opt.state_dict(), ropt.state_dict())
    assert_nested_equal(restart_scheduler.state_dict(), rscheduler.state_dict())
    constant = torch.full((2, 8, 512), 3.0, requires_grad=True)
    normalized, loc, scale = restored.normalize_context(constant)
    (restored.temporal(normalized) * scale + loc).sum().backward()
    assert torch.isfinite(constant.grad).all()
    assert steps == 10
    return {'status': 'pass', 'actual_optimizer_updates': steps, 'data_kind': 'synthetic_only',
            'models': records, 'mask_objective': True, 'restart_prediction_and_continuation': True,
            'initial_res_raw_compress_prediction_equal': True, 'constant_context_gradient_finite': True,
            'real_data_read': False, 'backbone': 'CPU synthetic fake only'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--reserve-seconds', type=float, default=180)
    args = parser.parse_args()
    target = HERE / (args.name + '.json')
    if target.exists():
        raise RuntimeError('Preserve prior check result; use a new name')
    with Job(args.name, category='cpu_check', reserve_seconds=args.reserve_seconds, synthetic=True) as job:
        try:
            result = run_checks(job)
            save_json(target, result)
        except BaseException:
            import traceback
            save_json(target, {'status': 'failed', 'error': traceback.format_exc(),
                               'actual_optimizer_updates': 'see ledger heartbeat'})
            raise


if __name__ == '__main__':
    main()
