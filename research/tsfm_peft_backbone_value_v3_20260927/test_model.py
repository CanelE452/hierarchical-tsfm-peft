import copy
import io
import math
import time
import traceback

import torch

from model import LinearResidual, macro_loss


def synthetic_initial(channels, latent, seed=19):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    basis, _ = torch.linalg.qr(torch.randn(channels, latent, generator=generator))
    down = torch.empty(32, 512).uniform_(-1 / math.sqrt(512), 1 / math.sqrt(512), generator=generator)
    return {
        'encoder.weight': basis.T.clone(),
        'decoder.weight': basis.clone(),
        'residual.down.weight': down,
        'residual.up.weight': torch.zeros(48, 32),
    }


def synthetic_batch(channels, seed=71):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(2, 512, channels, generator=generator)
    target = 0.4 * x[:, -48:] + torch.randn(2, 48, channels, generator=generator) * 0.1
    mask = torch.rand(2, 48, channels, generator=generator) > 0.15
    return x, target, mask


def spec_for(mode='current', normalization='context', arm='residual'):
    return {
        'dataset': 'electricity' if mode == 'current' else 'bull',
        'seed': 92601, 'latent': 8 if mode == 'current' else 4,
        'ed_mode': mode, 'arm': arm, 'normalization': normalization,
    }


def optimizer_for_test(model):
    temporal = list(model.temporal.parameters())
    others = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith('temporal.')]
    optimizer = torch.optim.AdamW([
        {'params': others, 'lr': 1e-3, 'name': 'ed_g'},
        {'params': temporal, 'lr': 1e-4, 'name': 'temporal'},
    ], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
    return optimizer, scheduler


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def run_checks():
    started = time.perf_counter()
    torch.set_num_threads(4)
    result = {
        'status': 'running', 'data': 'synthetic only', 'device': 'cpu',
        'optimizer_updates': 0, 'optimizer_update_limit': 10, 'checks': [],
    }

    def step(optimizer):
        if result['optimizer_updates'] >= 10:
            raise RuntimeError('synthetic session update budget exhausted')
        optimizer.step()
        result['optimizer_updates'] += 1

    def check(name, callback):
        try:
            details = callback()
            result['checks'].append({'name': name, 'status': 'pass', 'details': details})
        except Exception:
            result['checks'].append({'name': name, 'status': 'failed', 'error': traceback.format_exc()})

    def structure_and_normalization():
        initial = synthetic_initial(32, 8)
        rng_before = torch.get_rng_state().clone()
        model = LinearResidual(spec_for(), initial)
        assert torch.equal(rng_before, torch.get_rng_state())
        again = LinearResidual(dict(spec_for(), temporal_lr=1e-3), initial)
        assert_nested_equal(model.adapter_state(), again.adapter_state())
        assert not any(name.startswith('backbone') for name, _ in model.named_parameters())
        assert sum(p.numel() for p in model.parameters()) == 43_056
        pointers = [p.data_ptr() for p in model.parameters()]
        assert len(set(pointers)) == len(pointers)
        assert torch.count_nonzero(model.temporal.weight) > 0
        assert torch.count_nonzero(model.temporal.bias) > 0
        assert torch.count_nonzero(model.residual.up.weight) == 0
        x, _, _ = synthetic_batch(32)
        main, correction = model.components(x)
        assert main.shape == (2, 48, 32)
        assert torch.count_nonzero(correction) == 0
        assert torch.equal(model(x), main)
        z = model.encoder(x).transpose(1, 2)
        loc = z.mean(-1, keepdim=True)
        scale = (z - loc).square().mean(-1, keepdim=True).sqrt()
        expected = (model.temporal((z - loc) / scale) * scale + loc).transpose(1, 2)
        assert torch.allclose(model.temporal_point(z.transpose(1, 2)), expected, atol=1e-6, rtol=1e-6)
        missing = torch.tensor([[float('nan'), float('nan')], [2.0, float('nan')]])
        normalized, missing_loc, missing_scale = model.normalize_context(missing)
        assert torch.equal(missing_loc, torch.tensor([[0.0], [2.0]]))
        assert torch.equal(missing_scale, torch.tensor([[1.0], [1e-5]]))
        assert torch.equal(normalized, torch.zeros_like(missing))
        raw = LinearResidual(spec_for(normalization='raw', arm='raw_bypass'), initial)
        z_raw = raw.encoder(x)
        assert torch.equal(raw.temporal_point(z_raw), raw.temporal(z_raw.transpose(1, 2)).transpose(1, 2))
        payload = model.checkpoint_payload()
        assert payload['model_config']['normalization_detach'] is False
        return {'parameters': 43_056, 'state_keys': sorted(payload['state']), 'caller_rng_unchanged': True}

    def constant_gradient():
        model = LinearResidual(spec_for(), synthetic_initial(32, 8))
        latent = torch.full((2, 512, 8), 2.0, requires_grad=True)
        output = model.temporal_point(latent)
        assert torch.isfinite(output).all(), 'nonfinite constant-input output'
        output.square().mean().backward()
        assert torch.isfinite(latent.grad).all(), 'nonfinite constant-input latent gradient'
        assert all(torch.isfinite(p.grad).all() for p in model.temporal.parameters()), 'nonfinite T gradient'
        return {'constant_output_and_backward_finite': True}

    def normalization_forward_parity():
        model = LinearResidual(spec_for(), synthetic_initial(32, 8))
        generator = torch.Generator(device='cpu').manual_seed(731)
        normal = torch.randn(2, 8, 512, generator=generator)
        partial_missing = normal.clone()
        partial_missing[:, :, ::7] = float('nan')
        cases = {
            'random': normal,
            'zero': torch.zeros_like(normal),
            'constant': torch.full_like(normal, 2.0),
            'small_scale': normal * 1e-8,
            'all_nan': torch.full_like(normal, float('nan')),
            'partial_nan': partial_missing,
        }
        for name, latent in cases.items():
            loc = torch.nan_to_num(torch.nanmean(latent, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((latent - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, 1e-5, scale)
            normalized = torch.where(torch.isnan(latent), 0.0, (latent - loc) / scale)
            actual_normalized, actual_loc, actual_scale = model.normalize_context(latent)
            assert torch.equal(actual_loc, loc), f'loc changed: {name}'
            assert torch.equal(actual_scale, scale), f'scale changed: {name}'
            assert torch.equal(actual_normalized, normalized), f'normalized input changed: {name}'
            expected = (model.temporal(normalized) * scale + loc).transpose(1, 2)
            actual = model.temporal_point(latent.transpose(1, 2))
            assert torch.equal(actual, expected), f'T output changed: {name}'
        return {'exact_forward_parity_cases': list(cases), 'all_nan_backward_claimed': False}

    def masked_loss():
        pred = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]], requires_grad=True)
        target = torch.zeros_like(pred)
        mask = torch.tensor([[[True, False, False], [True, True, False]]])
        loss = macro_loss(pred, target, mask)
        assert loss.item() == ((1 + 9) / 2 + 16) / 2
        loss.backward()
        assert torch.count_nonzero(pred.grad[~mask]) == 0
        changed = target.clone()
        changed[~mask] = 10_000
        assert torch.equal(loss, macro_loss(pred, changed, mask))
        try:
            macro_loss(pred, target, mask.transpose(1, 2))
        except ValueError:
            pass
        else:
            raise AssertionError('mask axis mismatch was accepted')
        return {'equal_channel_weighting': True, 'masked_targets_have_zero_gradient': True}

    def updates(mode):
        channels = 32 if mode == 'current' else 16
        spec = spec_for(mode)
        initial = synthetic_initial(channels, spec['latent'])
        model = LinearResidual(spec, initial)
        original = model.adapter_state()
        optimizer, _ = optimizer_for_test(model)
        x, target, mask = synthetic_batch(channels)
        down_gradients = []
        for index in range(2):
            optimizer.zero_grad(set_to_none=True)
            macro_loss(model(x), target, mask).backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None, f'missing gradient: {name}'
                    assert torch.isfinite(parameter.grad).all(), f'nonfinite gradient: {name}'
            assert torch.count_nonzero(model.temporal.weight.grad) > 0
            assert torch.count_nonzero(model.temporal.bias.grad) > 0
            down_gradients.append(int(torch.count_nonzero(model.residual.down.weight.grad)))
            if index == 0:
                assert down_gradients[-1] == 0
            else:
                assert down_gradients[-1] > 0
            if mode == 'fixed_ed':
                assert model.encoder.weight.grad is None
                assert model.decoder.weight.grad is None
            step(optimizer)
        final = model.adapter_state()
        differences = {name: int(torch.count_nonzero(final[name] - original[name])) for name in original}
        for name, count in differences.items():
            if mode == 'fixed_ed' and name.startswith(('encoder.', 'decoder.')):
                assert count == 0, f'frozen state changed: {name}'
            else:
                assert count > 0, f'trainable state did not change: {name}'
        return {'changed_scalars': differences, 'G_down_nonzero_gradient_counts': down_gradients}

    def restart():
        spec = spec_for('fixed_ed')
        initial = synthetic_initial(16, 4)
        batches = [synthetic_batch(16, seed=100 + index) for index in range(2)]

        def update(model, optimizer, scheduler, index):
            optimizer.zero_grad(set_to_none=True)
            x, target, mask = batches[index]
            macro_loss(model(x), target, mask).backward()
            step(optimizer)
            scheduler.step(float(index + 1))

        uninterrupted = LinearResidual(spec, initial)
        optimizer_a, scheduler_a = optimizer_for_test(uninterrupted)
        update(uninterrupted, optimizer_a, scheduler_a, 0)
        update(uninterrupted, optimizer_a, scheduler_a, 1)

        interrupted = LinearResidual(spec, initial)
        optimizer_b, scheduler_b = optimizer_for_test(interrupted)
        update(interrupted, optimizer_b, scheduler_b, 0)
        payload = interrupted.checkpoint_payload()
        payload.update(optimizer=copy.deepcopy(optimizer_b.state_dict()),
                       scheduler=copy.deepcopy(scheduler_b.state_dict()),
                       rng=torch.get_rng_state().clone(), next_step=1, stale=0)
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location='cpu', weights_only=False)
        resumed = LinearResidual(saved['model_config']['spec'], initial)
        resumed.restore_adapter(saved['state'])
        optimizer_c, scheduler_c = optimizer_for_test(resumed)
        optimizer_c.load_state_dict(saved['optimizer'])
        scheduler_c.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['rng'])
        assert torch.equal(interrupted(batches[0][0]), resumed(batches[0][0]))
        update(resumed, optimizer_c, scheduler_c, saved['next_step'])
        assert_nested_equal(uninterrupted.adapter_state(), resumed.adapter_state())
        assert_nested_equal(optimizer_a.state_dict(), optimizer_c.state_dict())
        assert_nested_equal(scheduler_a.state_dict(), scheduler_c.state_dict())
        assert optimizer_c.param_groups[1]['lr'] == 0.1 * optimizer_c.param_groups[0]['lr']
        incomplete = resumed.adapter_state()
        del incomplete['temporal.bias']
        try:
            resumed.restore_adapter(incomplete)
        except RuntimeError:
            pass
        else:
            raise AssertionError('missing T checkpoint tensor was accepted')
        return {'serialized_restore_exact': True, 'optimizer_scheduler_restart_exact': True,
                'checkpoint_includes_T': True, 'checkpoint_bytes': buffer.getbuffer().nbytes}

    check('structure_and_normalization', structure_and_normalization)
    check('constant_input_gradient', constant_gradient)
    check('normalization_forward_parity', normalization_forward_parity)
    check('masked_loss', masked_loss)
    check('current_updates', lambda: updates('current'))
    check('fixed_ed_updates', lambda: updates('fixed_ed'))
    check('checkpoint_and_restart', restart)
    result['status'] = 'pass' if all(item['status'] == 'pass' for item in result['checks']) else 'failed'
    result['elapsed_s'] = time.perf_counter() - started
    return result
