"""One ledger-reserved CPU synthetic session, at most ten optimizer updates."""
import argparse
import io
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from model_v6 import FixedAdapter, initial_edg, macro_loss, state_digest, tensor_hashes
from runtime_v6 import HERE, Job, save_json, source_receipt
from train_v6 import assert_nested_equal, configure, optimizer_for


class SyntheticBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.8))
        self.chronos_config = SimpleNamespace(quantiles=[0.5])

    def forward(self, context):
        prediction = self.scale * context[:, -48:]
        return SimpleNamespace(quantile_preds=prediction[:, None, :])


def spec_for(family):
    return {'id': 'synthetic_' + family, 'dataset': 'robin', 'family': family,
            'arm': 'raw_bypass' if family == 'level_raw' else 'residual',
            'ed_mode': 'fixed_ed', 'channels': 17, 'latent': 5, 'seed': 92601,
            'lr': 1e-3, 't_lr': 1e-3, 'normalization': 'context', 'epochs': 120}


def run_checks(job):
    torch.manual_seed(9262026)
    basis = torch.linalg.qr(torch.randn(17, 5)).Q
    initial = initial_edg(basis, 92601)
    x = torch.randn(2, 512, 17)
    target = 0.4 * x[:, -48:] + 0.2 * torch.randn(2, 48, 17)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, :4, 0] = False
    target[~mask] = float('nan')
    models = {}
    optimizers = {}
    report = {}
    updates = 0
    for family in ('level_res', 'old_res', 'level_raw', 'level_linear_res'):
        spec = spec_for(family)
        model = FixedAdapter(spec, basis, initial, None if family == 'level_linear_res' else SyntheticBackbone())
        models[family] = model
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        before = model.adapter_state()
        assert torch.count_nonzero(model.residual.up.weight) == 0
        assert model(x).shape == target.shape and torch.isfinite(model(x)).all()
        opt, scheduler = optimizer_for(model, spec)
        gradients = []
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            loss = macro_loss(model(x), target, mask)
            assert torch.isfinite(loss)
            loss.backward()
            grads = {name: None if p.grad is None else float(p.grad.abs().max())
                     for name, p in model.named_parameters() if p.requires_grad}
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters() if p.requires_grad)
            assert grads['residual.up.weight'] > 0
            if step == 0:
                assert grads['residual.down.weight'] == 0
            else:
                assert grads['residual.down.weight'] > 0
            if family == 'level_linear_res':
                assert grads['temporal.weight'] > 0 and grads['temporal.bias'] > 0
            gradients.append(grads)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            scheduler.step(float(loss.detach()))
            updates += 1
            job.heartbeat(f'{family} synthetic update {step + 1}', updates)
        optimizers[family] = (opt, scheduler)
        for name, value in frozen.items():
            p = dict(model.named_parameters())[name]
            assert p.grad is None and torch.equal(p, value)
        after = model.adapter_state()
        assert not torch.equal(before['residual.down.weight'], after['residual.down.weight'])
        assert not torch.equal(before['residual.up.weight'], after['residual.up.weight'])
        if family == 'level_linear_res':
            assert not hasattr(model, 'backbone')
            assert set(after) == {'encoder.weight', 'decoder.weight', 'residual.down.weight',
                                  'residual.up.weight', 'temporal.weight', 'temporal.bias'}
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 42544
            const = torch.ones(2, 5, 512, requires_grad=True)
            normalized, loc, scale = model.normalize_context(const)
            assert torch.equal(normalized, torch.zeros_like(normalized))
            assert torch.all(scale == 1e-5)
            model.temporal_point(const.transpose(1, 2)).sum().backward()
            assert torch.isfinite(const.grad).all()
            missing = torch.full((1, 5, 512), float('nan'))
            normalized, loc, scale = model.normalize_context(missing)
            assert torch.equal(normalized, torch.zeros_like(normalized))
            assert torch.equal(loc, torch.zeros_like(loc)) and torch.equal(scale, torch.ones_like(scale))
        else:
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 17920
        report[family] = {'finite_loss_gradient': True, 'frozen_parameters_unchanged': True,
                          'both_G_layers_updated': True, 'initial_hashes': tensor_hashes(before),
                          'first_two_gradients': gradients, 'optimizer_updates': 2}

    fresh_res = FixedAdapter(spec_for('level_res'), basis, initial, SyntheticBackbone())
    fresh_raw = FixedAdapter(spec_for('level_raw'), basis, initial, SyntheticBackbone())
    fresh_old = FixedAdapter(spec_for('old_res'), basis, initial, SyntheticBackbone())
    parts = fresh_res.parts(x)
    torch.testing.assert_close(fresh_res(x), fresh_raw(x), atol=0, rtol=0)
    torch.testing.assert_close(fresh_old(x), parts['main'], atol=0, rtol=0)
    torch.testing.assert_close(fresh_res(x), parts['main'] + parts['persistence'], atol=0, rtol=0)
    assert fresh_res.residual.down.weight.data_ptr() != fresh_raw.residual.down.weight.data_ptr()
    learned = models['level_res']
    parts = learned.parts(x)
    correction = parts['persistence'] + parts['learned']
    torch.testing.assert_close(correction @ basis, torch.zeros(2, 48, 5), atol=3e-6, rtol=0)
    offset = torch.randn(1, 1, 17)
    offset = offset - (offset @ basis) @ basis.T
    torch.testing.assert_close(learned(x + offset), learned(x) + offset, atol=4e-6, rtol=1e-5)
    weights = learned.residual.up.weight @ learned.residual.down.weight
    history = parts['residual_history'].transpose(1, 2)
    explicit = history @ weights.T + history[:, :, -1:] * (1 - weights.sum(-1))[None, None, :]
    torch.testing.assert_close(correction, explicit.transpose(1, 2), atol=2e-6, rtol=1e-5)

    model = models['old_res']
    spec = spec_for('old_res')
    opt, scheduler = optimizers['old_res']
    archive = io.BytesIO()
    torch.save({'state': model.adapter_state(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                'rng': torch.get_rng_state()}, archive)
    archive.seek(0)
    saved = torch.load(archive, map_location='cpu', weights_only=False)
    restored = FixedAdapter(spec, basis, initial, SyntheticBackbone())
    restored.restore_adapter(saved['state'])
    restored_opt, restored_scheduler = optimizer_for(restored, spec)
    restored_opt.load_state_dict(saved['optimizer'])
    restored_scheduler.load_state_dict(saved['scheduler'])
    assert_nested_equal(opt.state_dict(), restored_opt.state_dict())
    assert_nested_equal(scheduler.state_dict(), restored_scheduler.state_dict())
    torch.set_rng_state(saved['rng'])
    assert torch.equal(torch.get_rng_state(), saved['rng'])
    torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)
    for instance, optimizer in ((model, opt), (restored, restored_opt)):
        optimizer.zero_grad(set_to_none=True)
        macro_loss(instance(x), target, mask).backward()
        optimizer.step()
        updates += 1
        job.heartbeat('synthetic restart continuation comparison', updates)
    assert state_digest(model.adapter_state()) == state_digest(restored.adapter_state())
    assert_nested_equal(opt.state_dict(), restored_opt.state_dict())
    assert updates == 10
    return {'status': 'PASS', 'synthetic_only': True, 'real_data': False, 'backbone': 'synthetic scalar fake',
            'optimizer_updates': updates, 'checks': report,
            'initial_level_raw_correspondence': True, 'initial_old_compress_correspondence': True,
            'fixed_PCA_complement_correction': True, 'complement_offset_equivariance': True,
            'explicit_shared_linear_identity': True, 'restart_equal_after_one_update_each': True,
            'real_LoRA_check_scope': 'deferred to reserved real fits; not covered by this synthetic result',
            'not_evidence_of_real_data_convergence': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', default='synthetic01.json')
    args = parser.parse_args()
    configure()
    target = HERE / args.output
    if target.exists():
        raise RuntimeError('Preserve earlier check results; use a new filename')
    with Job(args.job, category='cpu_check', reserve_seconds=120, synthetic=True) as job:
        result = run_checks(job)
        result['source'] = source_receipt()
        save_json(target, result)


if __name__ == '__main__':
    main()
