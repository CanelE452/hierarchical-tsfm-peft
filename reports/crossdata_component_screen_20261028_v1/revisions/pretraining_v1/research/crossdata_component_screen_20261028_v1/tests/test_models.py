"""Synthetic CPU contracts for unchanged cross-data component interventions."""
import copy
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch

from research.crossdata_component_screen_20261028_v1 import models

BROAD = Path(__file__).resolve().parents[2] / 'informs_broad_validation_20261015_v1'
sys.path.insert(0, str(BROAD))
import fit_base
import training


DATASETS = (('pems32', 32, 8, 12), ('beijing', 11, 6, 48))
COUNTS = {
    'pems32': dict(OURS=37472, TIME_ONLY=35360, A5_LAG0=35936, E_FIXED=37216),
    'beijing': dict(OURS=34587, TIME_ONLY=33840, A5_LAG0=34191, E_FIXED=34521),
}


class CPUFoundation(torch.nn.Module):
    initial_hash = 'SYNTHETIC_CPU_ONLY'
    device = 'cpu'

    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        self.scale = torch.nn.Parameter(torch.tensor(1.1), requires_grad=False)
        self.streams = self.row_calls = self.physical_streams = self.physical_calls = 0

    def forward(self, z):
        assert z.device.type == 'cpu'
        self.streams += z.shape[0] * z.shape[-1]
        return self.scale * (.7 * z[:, -self.horizon:] + .3 * z.mean(1, keepdim=True))

    def assert_frozen_unchanged(self):
        assert self.scale.grad is None and self.scale.item() == torch.tensor(1.1).item()


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    yield
    torch.set_num_threads(threads)


def task(dataset):
    name, channels, _, horizon = dataset
    rng = np.random.default_rng(4107)
    physical = rng.normal(size=(760, channels)).astype(np.float32)
    physical[576:, 0] += 3 * physical[576:, 1]
    return dict(physical=physical, observed=np.ones(physical.shape, bool), L=512, H=horizon,
                dataset='synthetic-' + name, columns=[f'c{i}' for i in range(channels)])


def spec(dataset, method='OURS'):
    return dict(run_id='synthetic-component', method_id=method, family='B2', latent_k=dataset[2],
                seed=92601, device='cpu', lr=.001, microbatch=4, effective_batch=16)


def stage(tmp_path, dataset, method, *, fit_end=576, stop=None, suffix='stage', epochs=(.25, 1.)):
    source, _ = fit_base._source_contract()
    contract = dict(source_sha=source, config_sha='synthetic', data_sha='synthetic', model_sha='SYNTHETIC_CPU_ONLY')
    return fit_base._stage(spec(dataset, method), task(dataset), fit_end, stop, list(epochs),
                           tmp_path/suffix/'cache', tmp_path/suffix/'public', CPUFoundation(dataset[3]),
                           time.time()+45, contract)


@pytest.mark.parametrize('dataset', DATASETS)
@pytest.mark.parametrize('method', models.METHODS)
def test_real_factory_same_seed_state_rng_shapes_and_active_parameters(dataset, method):
    data, foundation = task(dataset), CPUFoundation(dataset[3])
    reference, norm = fit_base.create_model(spec(dataset), data, 576, foundation)
    original_state = copy.deepcopy(reference.state_dict())
    rng = torch.get_rng_state().clone()
    job_spec = spec(dataset, method)
    unchanged_spec = copy.deepcopy(job_spec)
    with models.patch_core(fit_base, training, method) as audits:
        actual, actual_norm = fit_base.create_model(job_spec, data, 576, foundation)
        assert audits == []
        assert actual.method_id == method
        assert torch.equal(torch.get_rng_state(), rng)
        for name, value in actual.state_dict().items():
            expected = original_state['D'][0] if name == 'D0' else original_state[name]
            assert torch.equal(value, expected), name
        expected_names = set(original_state)
        if method == 'TIME_ONLY':
            expected_names = {name for name in expected_names if not name.startswith('front.mixer.feature.')}
        if method == 'A5_LAG0':
            expected_names = expected_names - {'D'} | {'D0'}
        assert set(actual.state_dict()) == expected_names
        assert np.array_equal(norm['basis'], actual_norm['basis'])
        assert np.array_equal(norm['mean'], actual_norm['mean'])
        assert np.array_equal(norm['std'], actual_norm['std'])
        assert sum(p.numel() for p in actual.parameters() if p.requires_grad) == COUNTS[dataset[0]][method]
        assert actual.E.requires_grad == (method != 'E_FIXED')
        x = torch.randn(2, 512, dataset[1])
        forecast = fit_base.predict_model(actual, x, foundation, 'B2')
        assert forecast.shape == (2, dataset[3], dataset[1])
        torch.testing.assert_close(forecast, reference(x, foundation), rtol=1e-6, atol=1e-6)
    assert job_spec == unchanged_spec


@pytest.mark.parametrize('dataset', DATASETS)
def test_time_branch_is_exact_nonzero_retained_branch(dataset):
    with models.patch_core(fit_base, training, 'TIME_ONLY'):
        model, _ = fit_base.create_model(spec(dataset, 'TIME_ONLY'), task(dataset), 576, CPUFoundation(dataset[3]))
    mixer = model.front.mixer
    with torch.no_grad():
        mixer.temporal.output.weight.fill_(.003)
        mixer.temporal.output.bias.fill_(.01)
    x = torch.randn(2, 512, dataset[1])
    expected = x + mixer.temporal(mixer.temporal_norm(x).transpose(1, 2)).transpose(1, 2)
    assert torch.equal(model.front(x), expected)
    assert not torch.equal(expected, x)
    assert not hasattr(mixer, 'feature')
    assert mixer.temporal_norm.normalized_shape == (512, dataset[1])


@pytest.mark.parametrize('dataset', DATASETS)
def test_lag0_exact_formula_retains_anchor_and_removes_lag_parameters(dataset):
    with models.patch_core(fit_base, training, 'A5_LAG0'):
        model, _ = fit_base.create_model(spec(dataset, 'A5_LAG0'), task(dataset), 576, CPUFoundation(dataset[3]))
    x = torch.randn(2, 512, dataset[1])
    z, future = torch.randn(2, 512, dataset[2]), torch.randn(2, dataset[3], dataset[2])
    with torch.no_grad():
        model.D0.add_(.03)
    expected = x[:, -1:] + (future - z[:, -1:]) @ model.D0
    assert torch.equal(model.decode(x, z, future), expected)
    shifted = x.clone()
    shifted[:, -1] += 2
    torch.testing.assert_close(model.decode(shifted, z, future) - expected, torch.full_like(expected, 2))
    assert not hasattr(model, 'D')


@pytest.mark.parametrize('dataset', DATASETS)
@pytest.mark.parametrize('method', models.METHODS)
def test_gradient_reaches_adapters_through_frozen_foundation(dataset, method):
    foundation = CPUFoundation(dataset[3])
    with models.patch_core(fit_base, training, method):
        model, _ = fit_base.create_model(spec(dataset, method), task(dataset), 576, foundation)
    x = torch.randn(2, 512, dataset[1], requires_grad=True)
    model(x, foundation).square().mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert model.front.mixer.temporal.output.weight.grad.abs().sum() > 0
    if method != 'TIME_ONLY':
        assert model.front.mixer.feature.output.weight.grad.abs().sum() > 0
    assert (model.D0 if method == 'A5_LAG0' else model.D).grad.abs().sum() > 0
    if method == 'E_FIXED':
        assert model.E.grad is None
    else:
        assert model.E.grad.abs().sum() > 0
    foundation.assert_frozen_unchanged()


@pytest.mark.parametrize('dataset', DATASETS)
@pytest.mark.parametrize('method', models.METHODS)
def test_actual_core_training_updates_retained_parameters_and_audits_E(tmp_path, dataset, method):
    with models.patch_core(fit_base, training, method) as audits:
        initial, _ = fit_base.create_model(spec(dataset, method), task(dataset), 576, CPUFoundation(dataset[3]))
        state = copy.deepcopy(initial.state_dict())
        result = stage(tmp_path, dataset, method)
    assert result['summary']['updates'] > 0
    assert result['summary']['trainable_parameters'] == COUNTS[dataset[0]][method]
    assert not torch.equal(state['D0' if method == 'A5_LAG0' else 'D'], result['model']['D0' if method == 'A5_LAG0' else 'D'])
    assert any(not torch.equal(value, result['model'][name]) for name, value in state.items() if name.startswith('front.'))
    if method == 'E_FIXED':
        assert torch.equal(state['E'], result['model']['E'])
        assert len(audits) == 1 and audits[0]['status'] == 'PASS'
        assert audits[0]['optimizer_parameter_group_checks'] == 2
        assert audits[0]['frozen_parameters'] == dataset[1] * dataset[2]
        assert audits[0]['initial_E_sha256'] == audits[0]['live_E_sha256'] == audits[0]['selected_E_sha256']
    else:
        assert audits == [] and not torch.equal(state['E'], result['model']['E'])


@pytest.mark.parametrize('dataset', DATASETS)
def test_E_is_separately_fitted_and_frozen_in_search_and_fresh_refit(tmp_path, dataset):
    with models.patch_core(fit_base, training, 'E_FIXED') as audits:
        search = stage(tmp_path, dataset, 'E_FIXED', stop=(576, 672), suffix='search', epochs=(.25,))
        refit = stage(tmp_path, dataset, 'E_FIXED', fit_end=672, suffix='refit', epochs=(.25,))
    assert [row['phase'] for row in audits] == ['search', 'refit']
    assert [row['fit_end'] for row in audits] == [576, 672]
    assert not torch.equal(search['model']['E'], refit['model']['E'])
    assert all(row['initial_E_sha256'] == row['selected_E_sha256'] for row in audits)


@pytest.mark.parametrize('fault', ['optimizer', 'live_E', 'selected_E'])
def test_E_audit_rejects_real_optimizer_inclusion_or_mutation(tmp_path, monkeypatch, fault):
    def broken_body(*args):
        model = args[-2]
        if fault == 'optimizer':
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            return training.train_segment(model, None, None, None, optimizer, 1, None)
        if fault == 'live_E':
            with torch.no_grad():
                model.E.add_(1)
        state = training.cpu_state(model)
        if fault == 'selected_E':
            state['E'].add_(1)
        return dict(model=state, summary={})
    monkeypatch.setattr(fit_base, '_stage_body', broken_body)
    with models.patch_core(fit_base, training, 'E_FIXED'):
        with pytest.raises(ValueError, match='included in optimizer|E differs'):
            stage(tmp_path, DATASETS[0], 'E_FIXED')


@pytest.mark.parametrize('method', models.METHODS)
def test_patch_cleanup_after_failure_wrong_method_and_nested_scope(method):
    before = fit_base.create_model, fit_base.predict_model, fit_base._stage_body, training.train_segment
    with pytest.raises(RuntimeError, match='synthetic failure'):
        with models.patch_core(fit_base, training, method):
            with pytest.raises(RuntimeError, match='one model patch'):
                with models.patch_core(fit_base, training, 'OURS'):
                    pass
            wrong = 'E_FIXED' if method == 'OURS' else 'OURS'
            with pytest.raises(ValueError, match='active B2 intervention'):
                fit_base.create_model(spec(DATASETS[0], wrong), task(DATASETS[0]), 576, CPUFoundation(12))
            raise RuntimeError('synthetic failure')
    assert (fit_base.create_model, fit_base.predict_model, fit_base._stage_body, training.train_segment) == before
    assert models._active is False
    assert models.branches._original_factory is None and models.decoder._original_factory is None


@pytest.mark.parametrize('legacy', ['branch', 'decoder', 'fixed_e'])
def test_refuses_active_legacy_patch_without_disturbing_it(legacy):
    scope = {'branch': lambda: models.branches.patch_core(fit_base),
             'decoder': lambda: models.decoder.patch_core(fit_base),
             'fixed_e': lambda: models.fixed_e.patch_core(fit_base, training)}[legacy]
    with scope():
        patched = fit_base.create_model
        with pytest.raises(RuntimeError, match='one model patch'):
            with models.patch_core(fit_base, training, 'OURS'):
                pass
        assert fit_base.create_model is patched


@pytest.mark.parametrize('deterministic', [False, True])
def test_evaluation_scope_restores_mixed_modes_and_gradient_setting(deterministic):
    with models.patch_core(fit_base, training, 'OURS'):
        model, _ = fit_base.create_model(spec(DATASETS[0]), task(DATASETS[0]), 576, CPUFoundation(12))
    model.train()
    model.front.mixer.temporal.eval()
    before = {id(module): module.training for module in model.modules()}
    with pytest.raises(RuntimeError, match='synthetic eval'):
        with models.evaluation_scope(model, deterministic):
            assert not torch.is_grad_enabled()
            assert not any(module.training for module in model.modules())
            raise RuntimeError('synthetic eval')
    assert torch.is_grad_enabled()
    assert {id(module): module.training for module in model.modules()} == before


def test_unknown_method_does_not_install_hooks():
    original = fit_base.create_model
    with pytest.raises(ValueError, match='Unknown'):
        with models.patch_core(fit_base, training, 'NEW_NETWORK'):
            pass
    assert fit_base.create_model is original and models._active is False
