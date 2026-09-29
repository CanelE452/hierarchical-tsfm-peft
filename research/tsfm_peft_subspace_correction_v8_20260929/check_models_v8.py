"""One reserved CPU synthetic session; ten optimizer updates in total."""
import argparse
import importlib.util
import io
from pathlib import Path
import tempfile
import traceback
from types import SimpleNamespace
from unittest.mock import patch

from runtime_v8 import CACHE, HERE, ROOT, Job, digest, save_json, source_receipt


def run_checks(job):
    import torch
    from torch import nn
    import model_v8
    from model_v8 import FixedAdapter, DATASETS, initial_edg, initial_raw, macro_loss, state_digest, tensor_hashes
    from train_v8 import assert_nested_equal, optimizer_for, rng_state, restore_rng

    torch.set_num_threads(4)
    torch.manual_seed(9292026)
    legacy_path = ROOT / 'research/tsfm_peft_practical_controls_v7_20260928/model_v7.py'
    legacy_spec = importlib.util.spec_from_file_location('_v8_synthetic_legacy_model', legacy_path)
    legacy = importlib.util.module_from_spec(legacy_spec)
    legacy_spec.loader.exec_module(legacy)

    class SyntheticBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.8))
            self.chronos_config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9])

        def forward(self, context):
            point = self.scale * context[:, -48:]
            return SimpleNamespace(quantile_preds=torch.stack((point - 1, point, point + 1), dim=1))

    def spec_for(dataset, family):
        channels, latent = DATASETS[dataset]
        return {'id': 'synthetic_' + dataset + '_' + family, 'dataset': dataset, 'family': family,
                'arm': family, 'ed_mode': 'fixed_ed', 'channels': channels, 'latent': latent,
                'seed': 92601, 'lr': 1e-3, 'normalization': 'context', 'epochs': 120}

    def fake_build(spec, basis, initial, device='cpu'):
        assert device == 'cpu'
        return FixedAdapter(spec, basis, initial, SyntheticBackbone())

    updates, report, artifacts = 0, {}, {}
    for dataset, (channels, latent) in DATASETS.items():
        basis = torch.linalg.qr(torch.randn(channels, latent)).Q
        initial = initial_edg(basis, 92601)
        prior = legacy.initial_edg(basis, 92601)
        for key in ('encoder.weight', 'decoder.weight'):
            assert torch.equal(initial[key], prior[key])
        for layer in ('down', 'up'):
            assert torch.equal(initial[f'residual_q.{layer}.weight'], prior[f'residual.{layer}.weight'])
        assert initial['residual_q.down.weight'].data_ptr() != initial['retained.down.weight'].data_ptr()
        assert not torch.equal(initial['residual_q.down.weight'], initial['retained.down.weight'])
        raw_initial = initial_raw(initial)
        assert torch.equal(raw_initial['residual.down.weight'][:32], initial['residual_q.down.weight'])
        assert torch.equal(raw_initial['residual.down.weight'][32:], initial['retained.down.weight'])
        assert torch.count_nonzero(raw_initial['residual.up.weight']) == 0
        x = torch.randn(2, 512, channels)
        target = 0.4 * x[:, -48:] + 0.2 * torch.randn(2, 48, channels)
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[:, :4, 0] = False
        target[~mask] = float('nan')
        fresh = {family: fake_build(spec_for(dataset, family), basis, initial) for family in ('pq_split', 'raw64')}
        torch.testing.assert_close(fresh['pq_split'](x), fresh['raw64'](x), atol=0, rtol=0)
        parts = fresh['pq_split'].parts(x)
        torch.testing.assert_close(fresh['pq_split'](x), parts['main'] + parts['persistence'], atol=0, rtol=0)
        with torch.no_grad():
            z = fresh['pq_split'].encoder(x)
            torch.testing.assert_close(fresh['pq_split'].backbone_point(z), 0.8 * z[:, -48:], atol=0, rtol=0)
        for family, model in fresh.items():
            spec = spec_for(dataset, family)
            model.train()
            assert model.backbone.training is False
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 35840
            assert all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters())
            before = model.adapter_state()
            frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
            opt, scheduler = optimizer_for(model, spec)
            assert {id(p) for group in opt.param_groups for p in group['params']} == {id(p) for p in model.parameters() if p.requires_grad}
            heads = ('residual_q', 'retained') if family == 'pq_split' else ('residual',)
            gradients = []
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                output = model(x)
                assert output.shape == target.shape and torch.isfinite(output).all()
                loss = macro_loss(output, target, mask)
                assert torch.isfinite(loss)
                loss.backward()
                grads = {name: None if parameter.grad is None else float(parameter.grad.abs().max())
                         for name, parameter in model.named_parameters() if parameter.requires_grad}
                assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                           for parameter in model.parameters() if parameter.requires_grad)
                for head in heads:
                    assert grads[head + '.up.weight'] > 0
                    assert (grads[head + '.down.weight'] == 0) == (step == 0)
                gradients.append(grads)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                scheduler.step(float(loss.detach()))
                updates += 1
                job.heartbeat(dataset + ' ' + family + ' CPU synthetic update ' + str(step + 1), updates)
            for name, value in frozen.items():
                parameter = dict(model.named_parameters())[name]
                assert parameter.grad is None and torch.equal(parameter, value)
            after = model.adapter_state()
            keys = [head + '.' + layer + '.weight' for head in heads for layer in ('down', 'up')]
            assert all(not torch.equal(before[key], after[key]) for key in keys)
            output = model(x)
            safe_target = torch.where(mask, target, torch.zeros_like(target))
            expected = torch.where(mask, (output - safe_target).square(), 0).sum((0, 1)) / mask.sum((0, 1))
            torch.testing.assert_close(macro_loss(output, target, mask), expected.mean(), atol=0, rtol=0)
            try:
                macro_loss(output, target, torch.zeros_like(mask))
            except ValueError:
                pass
            else:
                raise AssertionError('All-missing target batch must fail')
            state_bad = dict(after)
            state_bad.pop(keys[0])
            try:
                model.restore_adapter(state_bad)
            except ValueError:
                pass
            else:
                raise AssertionError('Incomplete adapter state was accepted')
            report[dataset + '_' + family] = {'trainable_parameters': 35840, 'optimizer_updates': 2,
                                             'finite_loss_gradient': True, 'first_two_gradients': gradients,
                                             'frozen_parameters_unchanged': True, 'required_parameters_updated': keys,
                                             'initial_hashes': tensor_hashes(before)}
            artifacts[(dataset, family)] = (spec, basis, initial, model, opt, scheduler, x, target, mask)
        learned = fresh['pq_split'].parts(x)
        projector = basis @ basis.T
        torch.testing.assert_close(learned['learned_q'] @ basis, torch.zeros(2, 48, latent), atol=1e-5, rtol=0)
        torch.testing.assert_close(learned['learned_p'] @ (torch.eye(channels) - projector),
                                   torch.zeros(2, 48, channels), atol=1e-5, rtol=0)
        torch.testing.assert_close((fresh['pq_split'](x) @ projector),
                                   learned['main'] + learned['learned_p'], atol=1e-5, rtol=1e-5)
        tied = fake_build(spec_for(dataset, 'pq_split'), basis, initial)
        tied_raw = fake_build(spec_for(dataset, 'raw64'), basis, initial)
        with torch.no_grad():
            tied.residual_q.up.weight.normal_(0, .01)
            tied.retained.load_state_dict(tied.residual_q.state_dict())
            tied_raw.residual.down.weight.copy_(torch.cat([tied.residual_q.down.weight] * 2, dim=0))
            tied_raw.residual.up.weight.copy_(torch.cat([tied.residual_q.up.weight * .5] * 2, dim=1))
        torch.testing.assert_close(tied(x), tied_raw(x), atol=1e-5, rtol=1e-5)

    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='synthetic_v8_', dir=CACHE) as directory:
        directory = Path(directory)
        for key, (spec, basis, initial, model, opt, scheduler, x, target, mask) in artifacts.items():
            initial_path = directory / ('_'.join(key) + '_initial.pt')
            checkpoint_path = directory / ('_'.join(key) + '_selected.pt')
            torch.save({'state': initial, 'basis': basis, 'tensor_hashes': tensor_hashes(initial)}, initial_path)
            payload = {'spec': spec, 'basis': basis, 'state': model.adapter_state(),
                       'initial_path': str(initial_path), 'initial_sha256': digest(initial_path),
                       'model_config': model.model_config(), 'epoch': 0, 'steps': 2}
            torch.save(payload, checkpoint_path)
            with patch.object(model_v8, 'build_model', fake_build):
                restored = model_v8.restore_model(checkpoint_path, device='cpu')
                torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)
                payload['model_config'] = dict(payload['model_config'], horizon=49)
                torch.save(payload, directory / 'invalid_config.pt')
                try:
                    model_v8.restore_model(directory / 'invalid_config.pt', device='cpu')
                except ValueError:
                    pass
                else:
                    raise AssertionError('Changed checkpoint configuration was accepted')

    spec, basis, initial, model, opt, scheduler, x, target, mask = artifacts[('jena', 'pq_split')]
    archive = io.BytesIO()
    torch.save({'state': model.adapter_state(), 'optimizer': opt.state_dict(),
                'scheduler': scheduler.state_dict(), 'rng': rng_state()}, archive)
    archive.seek(0)
    saved = torch.load(archive, map_location='cpu', weights_only=False)
    restored = fake_build(spec, basis, initial)
    restored.restore_adapter(saved['state'])
    restored_opt, restored_scheduler = optimizer_for(restored, spec)
    restored_opt.load_state_dict(saved['optimizer'])
    restored_scheduler.load_state_dict(saved['scheduler'])
    assert_nested_equal(opt.state_dict(), restored_opt.state_dict())
    assert_nested_equal(scheduler.state_dict(), restored_scheduler.state_dict())
    restore_rng(saved['rng'])
    assert_nested_equal(rng_state(), saved['rng'])
    for instance, optimizer, schedule in ((model, opt, scheduler), (restored, restored_opt, restored_scheduler)):
        restore_rng(saved['rng'])
        optimizer.zero_grad(set_to_none=True)
        loss = macro_loss(instance(x), target, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(instance.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        schedule.step(float(loss.detach()))
        updates += 1
        job.heartbeat('CPU restart continuation comparison', updates)
    assert state_digest(model.adapter_state()) == state_digest(restored.adapter_state())
    assert_nested_equal(opt.state_dict(), restored_opt.state_dict())
    assert_nested_equal(scheduler.state_dict(), restored_scheduler.state_dict())
    assert updates == 10
    return {'status': 'PASS', 'synthetic_only': True, 'real_data': False,
            'backbone': 'synthetic scalar fake; actual Chronos not loaded', 'optimizer_updates': updates,
            'checks': report, 'legacy_Q_initialization_equal': True, 'legacy_source': {'path': str(legacy_path), 'sha256': digest(legacy_path)},
            'shared_initial_function_and_raw_concat_checked': True,
            'P_Q_output_constraints_and_nonzero_tied_map_RAW_parity': True,
            'masked_macro_loss_checked': True, 'all_family_shape_checkpoint_roundtrips': True,
            'restore_model_scope': 'actual serialization/configuration checks with CPU fake factory replacing Chronos construction',
            'restart_equal_after_one_update_each': True, 'not_evidence_of_real_data_convergence': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', default='synthetic_models01.json')
    args = parser.parse_args()
    target = HERE / args.output
    if target.exists():
        raise RuntimeError('Preserve earlier check results; use a new filename')
    with Job(args.job, category='cpu_check', reserve_seconds=120, synthetic=True) as job:
        try:
            result = run_checks(job)
            result['source'] = source_receipt()
            save_json(target, result)
        except BaseException:
            save_json(target, {'status': 'FAIL', 'synthetic_only': True, 'source': source_receipt(),
                               'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
