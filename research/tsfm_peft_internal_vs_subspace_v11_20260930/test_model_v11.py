"""Bounded CPU synthetic model checks: six optimizer updates, no real data.

The differentiable stub does not establish actual Chronos or LoRA gradients.
Run only inside the root runner's pre-reserved synthetic-check session.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
from torch import nn

MODULE_PATH = Path(__file__).with_name('model_v11.py')
_spec = importlib.util.spec_from_file_location('v11_model_under_test', MODULE_PATH)
model_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model_api)
_runtime_spec = importlib.util.spec_from_file_location('v11_runtime_for_synthetic_checks', MODULE_PATH.with_name('runtime_v11.py'))
runtime = importlib.util.module_from_spec(_runtime_spec)
_runtime_spec.loader.exec_module(runtime)


class SyntheticBolt(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(712)
            self.temporal = nn.Linear(512, 64, bias=True)
        self.chronos_config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9])

    def forward(self, context):
        point = self.temporal(context) + torch.tanh(context[:, -1:])
        return SimpleNamespace(quantile_preds=torch.stack((point - .2, point, point + .2), dim=1))


def _assert_close(left, right, atol=1e-5, rtol=1e-4):
    if not torch.allclose(left, right, atol=atol, rtol=rtol):
        raise AssertionError(f'Output mismatch: maxabs={(left-right).abs().max().item()}')


def _step(model, optimizer, x, target, mask, session_job=None):
    optimizer.zero_grad(set_to_none=True)
    loss = model_api.macro_loss(model(x), target, mask)
    if not torch.isfinite(loss):
        raise AssertionError('Synthetic loss is nonfinite')
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if not gradients or any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError('Missing/nonfinite synthetic gradient')
    torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1)
    optimizer.step()
    if session_job is not None:
        session_job.heartbeat('synthetic optimizer update completed', updates=session_job.updates + 1)
    return float(loss.detach())


def run_checks(session_job=None):
    torch.set_num_threads(4)
    receipt = {'data_kind': 'synthetic', 'device': 'cpu', 'optimizer_updates': 0,
               'actual_chronos_or_lora_verified': False, 'checks': []}
    scratch = runtime.CACHE / 'checks'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='synthetic_model_', dir=scratch) as temporary:
        temp = Path(temporary)
        for dataset, (channels, latent) in model_api.DATASETS.items():
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(24601 + channels)
                basis = torch.linalg.qr(torch.randn(channels, latent), mode='reduced').Q
                x = torch.randn(2, 512, channels)
                target = torch.randn(2, 48, channels)
                mask = torch.ones_like(target, dtype=torch.bool)
                mask[0, 0, 0] = False
                target[0, 0, 0] = float('nan')
            seed = 92601
            direct_spec = {'dataset': dataset, 'family': 'direct_nlinear', 'seed': seed}
            direct = model_api.build_model(direct_spec, basis)
            source_path = temp / (dataset + '_synthetic_direct.pt')
            torch.save(model_api.make_checkpoint(direct), source_path)
            sources = {'direct_checkpoint': str(source_path), 'direct_sha256': model_api.file_hash(source_path)}
            learned_spec = {'dataset': dataset, 'family': 'b_learned_u', 'seed': seed}
            initial = model_api.generate_initial(learned_spec, basis)
            learned = model_api.build_model(learned_spec, basis, initial=initial,
                                           sources=sources, backbone=SyntheticBolt())
            fixed = model_api.build_model({**learned_spec, 'family': 'b_fixed_u'}, basis,
                                         sources=sources, backbone=SyntheticBolt())
            _assert_close(learned.U, basis, atol=1e-6, rtol=0)
            _assert_close(learned.U.T @ learned.U, torch.eye(latent), atol=1e-5, rtol=0)
            with torch.no_grad():
                _assert_close(learned(x), fixed(x))
                parts = learned.mixture_features(x)
                explicit = parts['latent'] @ parts['U'].T + parts['S'] - parts['S'] @ parts['U'] @ parts['U'].T
                _assert_close(learned(x), explicit)
            if any(parameter.requires_grad for parameter in learned.backbone.parameters()):
                raise AssertionError('Synthetic backbone was not frozen')
            if any(parameter.requires_grad for parameter in learned.direct.parameters()):
                raise AssertionError('Synthetic S was not frozen')
            frozen_before = learned.frozen_state_digest()
            u_before = learned.U.detach().clone()
            optimizer = torch.optim.AdamW([parameter for parameter in learned.parameters() if parameter.requires_grad],
                                          lr=1e-3, weight_decay=0)
            _step(learned, optimizer, x, target, mask, session_job)
            receipt['optimizer_updates'] += 1
            if torch.equal(u_before, learned.U):
                raise AssertionError('Raw U changed without changing the constrained U')
            if frozen_before != learned.frozen_state_digest():
                raise AssertionError('Frozen S/backbone/buffer state changed')
            _assert_close(learned.U.T @ learned.U, torch.eye(latent), atol=1e-5, rtol=0)

            saved = model_api.make_checkpoint(learned, extra={'optimizer': copy.deepcopy(optimizer.state_dict()),
                                                            'rng': torch.get_rng_state().clone()})
            checkpoint_path = temp / (dataset + '_synthetic_learned.pt')
            torch.save(saved, checkpoint_path)
            reread = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            required = {'directions.parametrizations.weight.original', 'directions.parametrizations.weight.0.base'}
            if not required.issubset(reread['state']):
                raise AssertionError('Orthogonal restart state is incomplete')
            restored = model_api.build_model(reread['spec'], reread['basis'], sources=reread['sources'],
                                            backbone=SyntheticBolt())
            restored.restore_adapter(reread['state'])
            with torch.no_grad():
                _assert_close(learned(x), restored(x), atol=1e-6, rtol=0)
            if dataset == 'robin':
                restored_optimizer = torch.optim.AdamW([parameter for parameter in restored.parameters() if parameter.requires_grad],
                                                       lr=1e-3, weight_decay=0)
                restored_optimizer.load_state_dict(reread['optimizer'])
                _step(learned, optimizer, x, target, mask, session_job)
                _step(restored, restored_optimizer, x, target, mask, session_job)
                receipt['optimizer_updates'] += 2
                for name, value in learned.adapter_state().items():
                    _assert_close(value, restored.adapter_state()[name], atol=1e-6, rtol=0)

            with torch.no_grad():
                learned.set_gamma(torch.zeros(latent))
                _assert_close(learned(x), learned.direct_prediction(x), atol=0, rtol=0)
                zero_deployment = model_api.deployment_copy(learned)
                if hasattr(zero_deployment, 'backbone'):
                    raise AssertionError('Gamma0 deployment retained the unused TSFM')
                _assert_close(zero_deployment(x), learned.direct_prediction(x), atol=0, rtol=0)
                gamma = torch.linspace(0, 1, latent)
                learned.set_gamma(gamma)
                full = learned(x)
                deployed = model_api.deployment_copy(learned)
                if hasattr(deployed, 'directions'):
                    raise AssertionError('Deployment retained the training parametrization')
                _assert_close(deployed(x), full)
                deployed.chunk_rows = latent
                _assert_close(deployed(x), full)
                learned.set_gamma(torch.ones(latent))
            receipt['checks'].append({'dataset': dataset, 'shape': [channels, latent],
                                      'orthogonality_maxabs': float((learned.U.T @ learned.U - torch.eye(latent)).abs().max()),
                                      'frozen_state_unchanged': frozen_before == learned.frozen_state_digest(),
                                      'restart_and_export': 'pass'})

        direct_initial = model_api.generate_initial(direct_spec, basis)
        fresh_direct = model_api.build_model(direct_spec, basis, initial=direct_initial)
        if sum(parameter.numel() for parameter in fresh_direct.parameters() if parameter.requires_grad) != 24624:
            raise AssertionError('DIRECT trainable parameter count differs')
        before = fresh_direct.trainable_state()
        direct_optimizer = torch.optim.AdamW(fresh_direct.parameters(), lr=1e-3, weight_decay=0)
        _step(fresh_direct, direct_optimizer, x, target, mask, session_job)
        receipt['optimizer_updates'] += 1
        if model_api.state_digest(before) == model_api.state_digest(fresh_direct.trainable_state()):
            raise AssertionError('DIRECT parameters did not update')
    if receipt['optimizer_updates'] != 6:
        raise AssertionError('Synthetic optimizer update accounting differs from the reserved six')
    receipt['status'] = 'passed'
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with runtime.Job('v11 synthetic model checks', category='cpu_check', reserve_seconds=180,
                     synthetic=True, metadata={'planned_optimizer_updates': 6, 'actual_chronos_or_lora': False}) as job:
        result = run_checks(job)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)
