"""One CPU-only synthetic optimizer session, at most ten updates in total."""
import argparse
import copy
import io
from types import SimpleNamespace

from runtime_v5 import HERE, Job, save_json, source_receipt


def run_checks(job):
    import numpy as np
    import torch
    from torch import nn
    from model_v5 import (build_with_backbone, initial_state, macro_loss,
                          nested_basis, tensor_hashes)
    from train_v5 import assert_nested_equal, optimizer_for, rng_state, restore_rng

    torch.set_num_threads(4)
    torch.manual_seed(92805)
    np.random.seed(92805)
    updates = 0
    records = {}

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.8))
            self.chronos_config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9])

        def forward(self, context):
            value = context[:, -48:] * self.scale + context.mean(-1, keepdim=True) * 0.2
            return SimpleNamespace(quantile_preds=torch.stack([value-0.1, value, value+0.1], dim=1))

    base_basis = torch.linalg.qr(torch.randn(32, 8)).Q
    expanded_basis = torch.as_tensor(nested_basis(np.random.randn(80, 32), base_basis.numpy(), 16))
    assert torch.equal(expanded_basis[:, :8], base_basis)
    before_rng = torch.get_rng_state().clone()
    base_initial = initial_state(base_basis, 92601, 'level')
    assert torch.equal(before_rng, torch.get_rng_state())
    x, y = torch.randn(2, 512, 32), torch.randn(2, 48, 32)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0, :10, 0] = False
    changed_target = y.clone()
    changed_target[~mask] = float('nan')
    prediction = torch.randn_like(y)
    assert torch.equal(macro_loss(prediction, y, mask), macro_loss(prediction, changed_target, mask))
    direct = (((prediction-y).square()*mask).sum((0, 1))/mask.sum((0, 1))).mean()
    assert torch.equal(macro_loss(prediction, y, mask), direct)

    # The selected LEVEL candidate gets the optimizer checks; unselected
    # registry entries receive forward-only structural checks below.
    method = 'level'
    for variant in ('res', 'raw'):
        for mode in ('fixed_ed', 'current'):
            name = method + '_' + variant + '_' + mode
            basis = base_basis
            initial = initial_state(basis, 92601, method, base_initial, base_latent=8)
            spec = {'id': 'synthetic_' + name, 'dataset': 'synthetic', 'method': method,
                    'variant': variant, 'ed_mode': mode, 'seed': 92601, 'channels': 32,
                    'latent': basis.shape[1], 'lr': 1e-3, 'epochs': 120}
            before_rng = torch.get_rng_state().clone()
            model = build_with_backbone(spec, basis, initial, FakeBackbone())
            assert torch.equal(before_rng, torch.get_rng_state())
            assert tensor_hashes(model.adapter_state()) == tensor_hashes(initial)
            raw = build_with_backbone(dict(spec, variant='raw' if variant == 'res' else 'res'), basis, initial, FakeBackbone())
            with torch.no_grad():
                assert torch.equal(model(x), raw(x))
                parts = model.parts(x)
                expected_initial = parts['main'] + parts['residual_history'][:, -1:, :].expand(-1, 48, -1) if method == 'level' else parts['main']
                assert torch.equal(model(x), expected_initial)
                assert parts['learned'].count_nonzero().item() == 0
                if method == 'level':
                    assert not parts['persistence'].requires_grad
                    assert torch.equal(raw.parts(x)['persistence'], parts['persistence'])
                    if variant == 'res':
                        assert torch.equal(raw.parts(x)['correction_input'], x-x[:, -1:, :])
            del raw
            opt, scheduler = optimizer_for(model, spec)
            frozen = {key: p.detach().clone() for key, p in model.named_parameters() if not p.requires_grad}
            gradients = []
            update_count = 2
            for update in range(update_count):
                job.check_limits()
                opt.zero_grad(set_to_none=True)
                loss = macro_loss(model(x), y, mask)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                if method == 'full':
                    assert model.residual.linear.weight.grad.abs().max() > 0
                    g_gradient = float(model.residual.linear.weight.grad.abs().max())
                else:
                    down = float(model.residual.down.weight.grad.abs().max())
                    assert (down == 0) if update == 0 else (down > 0)
                    assert model.residual.up.weight.grad.abs().max() > 0
                    g_gradient = down
                if mode == 'current':
                    assert model.encoder.weight.grad is not None and model.encoder.weight.grad.abs().max() > 0
                    assert model.decoder.weight.grad is not None and model.decoder.weight.grad.abs().max() > 0
                else:
                    assert model.encoder.weight.grad is None and model.decoder.weight.grad is None
                gradients.append({'loss': float(loss.detach()), 'g_gradient_max': g_gradient})
                opt.step()
                updates += 1
                scheduler.step(float(loss.detach()))
                job.heartbeat(name + f' update {update+1}', updates)
            trained = model.adapter_state()
            for key in ('encoder.weight', 'decoder.weight'):
                assert torch.equal(trained[key], initial[key]) == (mode == 'fixed_ed')
            if method == 'full':
                assert trained['residual.linear.weight'].ne(initial['residual.linear.weight']).any()
            else:
                assert trained['residual.up.weight'].ne(initial['residual.up.weight']).any()
                if method == 'level':
                    assert trained['residual.down.weight'].ne(initial['residual.down.weight']).any()
            for key, parameter in model.named_parameters():
                if key in frozen:
                    assert torch.equal(parameter, frozen[key]) and parameter.grad is None
            with torch.no_grad():
                if mode == 'fixed_ed' and variant == 'res':
                    correction = model.components(x)[1]
                    projection = basis @ basis.T
                    assert float((correction @ projection).abs().max()) < 1e-5
            clone = build_with_backbone(spec, basis, initial, FakeBackbone())
            clone.restore_adapter(trained)
            with torch.no_grad():
                assert torch.equal(model(x), clone(x))
            records[name] = {'initial_res_raw_equivalence': True, 'initial_function_verified': True,
                             'gradient_records': gradients, 'frozen_and_updated_parameters': True,
                             'checkpoint_prediction_replay': True,
                             'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
            if variant == 'res' and mode == 'current':
                restart_spec, restart_basis, restart_initial = spec, basis, initial
                restart_model, restart_opt, restart_scheduler = model, opt, scheduler

    for method in ('full', 'k'):
        basis = expanded_basis if method == 'k' else base_basis
        initial = initial_state(basis, 92601, method, base_initial, base_latent=8)
        spec = {'id': 'synthetic_forward_' + method, 'dataset': 'synthetic', 'method': method,
                'variant': 'res', 'ed_mode': 'fixed_ed', 'seed': 92601, 'channels': 32,
                'latent': basis.shape[1], 'lr': 1e-3, 'epochs': 120}
        model = build_with_backbone(spec, basis, initial, FakeBackbone())
        raw = build_with_backbone(dict(spec, variant='raw'), basis, initial, FakeBackbone())
        with torch.no_grad():
            assert torch.equal(model(x), raw(x))
            assert torch.equal(model(x), model.parts(x)['main'])
        records[method + '_forward_only'] = {'initial_res_raw_equivalence': True,
                                            'initial_compress_equivalence': True, 'optimizer_updates': 0}

    saved = {'model': restart_model.adapter_state(), 'optimizer': copy.deepcopy(restart_opt.state_dict()),
             'scheduler': copy.deepcopy(restart_scheduler.state_dict()), 'rng': rng_state()}
    stream = io.BytesIO()
    torch.save(saved, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=False)
    restored = build_with_backbone(restart_spec, restart_basis, restart_initial, FakeBackbone())
    restored.restore_adapter(saved['model'])
    restored_opt, restored_scheduler = optimizer_for(restored, restart_spec)
    restored_opt.load_state_dict(saved['optimizer'])
    restored_scheduler.load_state_dict(saved['scheduler'])
    assert_nested_equal(restart_model.adapter_state(), restored.adapter_state())
    for model, opt, scheduler in ((restart_model, restart_opt, restart_scheduler),
                                  (restored, restored_opt, restored_scheduler)):
        restore_rng(saved['rng'])
        assert_nested_equal(rng_state(), saved['rng'])
        opt.zero_grad(set_to_none=True)
        loss = macro_loss(model(x), y, mask)
        loss.backward()
        opt.step()
        updates += 1
        scheduler.step(float(loss.detach()))
        job.heartbeat('matched uninterrupted/restarted continuation', updates)
    assert_nested_equal(restart_model.adapter_state(), restored.adapter_state())
    assert_nested_equal(restart_opt.state_dict(), restored_opt.state_dict())
    assert_nested_equal(restart_scheduler.state_dict(), restored_scheduler.state_dict())
    assert updates == 10
    return {'status': 'pass', 'actual_optimizer_updates': updates, 'data_kind': 'synthetic_only',
            'models': records, 'mask_objective': True, 'nested_basis_keeps_original_axes': True,
            'restart_continuation_exact': True, 'real_data_read': False,
            'backbone': 'CPU synthetic fake only', 'source': source_receipt()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--reserve-seconds', type=float, default=180)
    args = parser.parse_args()
    target = HERE / (args.name + '.json')
    if target.exists():
        raise RuntimeError('Preserve prior check result; use a new session name')
    with Job(args.name, category='cpu_check', reserve_seconds=args.reserve_seconds, synthetic=True) as job:
        try:
            save_json(target, run_checks(job))
        except BaseException:
            import traceback
            save_json(target, {'status': 'failed', 'error': traceback.format_exc(),
                               'actual_optimizer_updates': 'see ledger heartbeat'})
            raise


if __name__ == '__main__':
    main()
