"""One reserved CPU synthetic session; ten optimizer updates across all checks."""
import argparse
import io
from types import SimpleNamespace

import torch
from torch import nn

from model_v7 import FixedAdapter, initial_direct, initial_edg, macro_loss, state_digest, tensor_hashes
from runtime_v7 import HERE, Job, save_json, source_receipt
from train_v7 import assert_nested_equal, configure, optimizer_for, rng_state, restore_rng


class SyntheticBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.8))
        self.chronos_config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9])

    def forward(self, context):
        point = self.scale * context[:, -48:]
        return SimpleNamespace(quantile_preds=torch.stack((point - 1, point, point + 1), dim=1))


class AuthorSharedNLinearReference(nn.Module):
    """Shared branch transcribed from cure-lab/LTSF-Linear/models/NLinear.py."""
    def __init__(self):
        super().__init__()
        self.Linear = nn.Linear(512, 48)

    def forward(self, x):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x


def spec_for(family, dataset='jena'):
    direct = family == 'direct_nlinear'
    return {'id': 'synthetic_' + dataset + '_' + family, 'dataset': dataset, 'family': family,
            'arm': 'direct' if direct else 'raw' if family == 'level_raw' else 'residual',
            'ed_mode': 'none' if direct else 'fixed_ed', 'channels': 17 if dataset == 'robin' else 21,
            'latent': 0 if direct else 6, 'seed': 92601, 'lr': 1e-3,
            'normalization': 'train_standardization_only' if direct else 'context', 'epochs': 120}


def run_checks(job):
    torch.manual_seed(9272026)
    basis = torch.linalg.qr(torch.randn(21, 6)).Q
    initial = initial_edg(basis, 92601)
    direct_initial = initial_direct(92601)
    x = torch.randn(2, 512, 21)
    target = 0.4 * x[:, -48:] + 0.2 * torch.randn(2, 48, 21)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, :4, 0] = False
    target[~mask] = float('nan')
    models, optimizers, report = {}, {}, {}
    updates = 0
    for family in ('level_res', 'old_res', 'level_raw', 'direct_nlinear'):
        spec = spec_for(family)
        direct = family == 'direct_nlinear'
        model = FixedAdapter(spec, basis, direct_initial if direct else initial,
                             None if direct else SyntheticBackbone())
        models[family] = model
        before = model.adapter_state()
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        if direct:
            assert not any(hasattr(model, name) for name in ('backbone', 'encoder', 'decoder', 'residual'))
            assert sum(p.numel() for p in model.parameters()) == 24624
            reference = AuthorSharedNLinearReference()
            reference.Linear.load_state_dict({'weight': before['temporal.weight'], 'bias': before['temporal.bias']})
            torch.testing.assert_close(model(x), reference(x), atol=0, rtol=0)
            const = torch.ones(2, 512, 21)
            expected = const[:, -1:, :] + model.temporal.bias[None, :, None]
            torch.testing.assert_close(model(const), expected.expand(2, 48, 21), atol=0, rtol=0)
            robin_basis = torch.linalg.qr(torch.randn(17, 5)).Q
            robin = FixedAdapter(spec_for(family, 'robin'), robin_basis, direct_initial)
            torch.testing.assert_close(robin(x[:, :, :17]), reference(x[:, :, :17]), atol=0, rtol=0)
            assert tensor_hashes(robin.adapter_state()) == tensor_hashes(before)
            assert model.temporal.weight.data_ptr() != robin.temporal.weight.data_ptr()
        else:
            assert torch.count_nonzero(model.residual.up.weight) == 0
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 17920
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
            if direct:
                assert grads['temporal.weight'] > 0 and grads['temporal.bias'] > 0
            else:
                assert grads['residual.up.weight'] > 0
                assert (grads['residual.down.weight'] == 0) == (step == 0)
            gradients.append(grads)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            scheduler.step(float(loss.detach()))
            updates += 1
            job.heartbeat(f'{family} CPU synthetic update {step + 1}', updates)
        optimizers[family] = (opt, scheduler)
        for name, value in frozen.items():
            parameter = dict(model.named_parameters())[name]
            assert parameter.grad is None and torch.equal(parameter, value)
        after = model.adapter_state()
        keys = ('temporal.weight', 'temporal.bias') if direct else ('residual.down.weight', 'residual.up.weight')
        assert all(not torch.equal(before[key], after[key]) for key in keys)
        report[family] = {'finite_loss_gradient': True, 'frozen_parameters_unchanged': True,
                          'required_parameters_updated': list(keys), 'initial_hashes': tensor_hashes(before),
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
    torch.testing.assert_close((parts['persistence'] + parts['learned']) @ basis,
                               torch.zeros(2, 48, 6), atol=5e-6, rtol=0)
    probe = models['direct_nlinear'](x)
    masked_error = (probe - torch.nan_to_num(target)).square()[mask]
    counts = mask.sum((0, 1))
    expected_loss = torch.where(mask, (probe - torch.nan_to_num(target)).square(), 0).sum((0, 1)) / counts
    torch.testing.assert_close(macro_loss(probe, target, mask), expected_loss.mean(), atol=0, rtol=0)
    assert torch.isfinite(masked_error).all()
    try:
        macro_loss(probe, target, torch.zeros_like(mask))
    except ValueError:
        pass
    else:
        raise AssertionError('All-missing target batch must fail')

    model = models['direct_nlinear']
    opt, scheduler = optimizers['direct_nlinear']
    archive = io.BytesIO()
    torch.save({'state': model.adapter_state(), 'optimizer': opt.state_dict(),
                'scheduler': scheduler.state_dict(), 'rng': rng_state()}, archive)
    archive.seek(0)
    saved = torch.load(archive, map_location='cpu', weights_only=False)
    restored = FixedAdapter(spec_for('direct_nlinear'), basis, direct_initial)
    restored.restore_adapter(saved['state'])
    restored_opt, restored_scheduler = optimizer_for(restored, spec_for('direct_nlinear'))
    restored_opt.load_state_dict(saved['optimizer'])
    restored_scheduler.load_state_dict(saved['scheduler'])
    assert_nested_equal(opt.state_dict(), restored_opt.state_dict())
    assert_nested_equal(scheduler.state_dict(), restored_scheduler.state_dict())
    restore_rng(saved['rng'])
    assert_nested_equal(rng_state(), saved['rng'])
    torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)
    for instance, optimizer in ((model, opt), (restored, restored_opt)):
        optimizer.zero_grad(set_to_none=True)
        macro_loss(instance(x), target, mask).backward()
        optimizer.step()
        updates += 1
        job.heartbeat('CPU synthetic restart continuation comparison', updates)
    assert state_digest(model.adapter_state()) == state_digest(restored.adapter_state())
    assert_nested_equal(opt.state_dict(), restored_opt.state_dict())
    assert updates == 10
    return {'status': 'PASS', 'synthetic_only': True, 'real_data': False,
            'backbone': 'synthetic scalar fake; actual Bolt/LoRA not loaded', 'optimizer_updates': updates,
            'checks': report, 'direct_author_shared_forward_parity': True,
            'reference_source': 'https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py',
            'reference_scope': 'Transcribed shared forward with identical W/bias; not an official training reproduction',
            'initial_level_raw_correspondence': True, 'initial_old_compress_correspondence': True,
            'fixed_PCA_complement_correction': True, 'masked_macro_loss_checked': True,
            'restart_equal_after_one_update_each': True,
            'real_LoRA_check_scope': 'deferred to reserved real fits; not covered by this synthetic result',
            'not_evidence_of_real_data_convergence': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', default='synthetic_models01.json')
    args = parser.parse_args()
    target = HERE / args.output
    if target.exists():
        raise RuntimeError('Preserve earlier check results; use a new filename')
    configure()
    with Job(args.job, category='cpu_check', reserve_seconds=120, synthetic=True) as job:
        result = run_checks(job)
        result['source'] = source_receipt()
        save_json(target, result)


if __name__ == '__main__':
    main()
