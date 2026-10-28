"""Synthetic tests: branch semantics, selection chronology, and job boundaries."""
import copy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from research.jena_p_branch_20261027_v1 import models, runtime, run_one, controller

BROAD = runtime.REPO / 'research/informs_broad_validation_20261015_v1'
sys.path.insert(0, str(BROAD))
import networks


def job(method='P_TIME_ONLY', extended=False):
    return dict(job_id=f'JPB_{method}_jena_K6_s92601' + ('_ext10' if extended else ''),
                method_id=method, dataset='jena', K=6, seed=92601, extended=extended)


@pytest.mark.parametrize('method,branch_count', [('P_TIME_ONLY', 33312), ('P_CHANNEL_ONLY', 1397)])
def test_exact_nonzero_branch_equation_and_no_new_rng(method, branch_count):
    torch.manual_seed(92601)
    original = networks._AlternatingMixer(512, 21, 32)
    with torch.no_grad():
        original.temporal.output.weight.fill_(.003)
        original.temporal.output.bias.fill_(.01)
        original.feature.output.weight.fill_(.007)
        original.feature.output.bias.fill_(-.02)
    x = torch.randn(2, 512, 21)
    rng = torch.get_rng_state().clone()
    retained = models.SingleBranchMixer(original, method)
    assert torch.equal(torch.get_rng_state(), rng)
    assert sum(p.numel() for p in retained.parameters()) == branch_count
    if method == 'P_TIME_ONLY':
        expected = x + original.temporal(original.temporal_norm(x).transpose(1, 2)).transpose(1, 2)
        assert retained.temporal is original.temporal
    else:
        expected = x + original.feature(original.feature_norm(x))
        assert retained.feature is original.feature
    assert not torch.equal(expected, x)
    assert torch.equal(retained(x), expected)
    assert not torch.equal(retained(x), original(x))


@pytest.mark.parametrize('method,expected_count', [('P_TIME_ONLY', 34320), ('P_CHANNEL_ONLY', 2405)])
def test_full_adapter_common_parameters_identity_and_input_gradient(method, expected_count):
    torch.manual_seed(92601)
    basis = np.eye(21, 6, dtype=np.float32)
    full = networks.ForecastAdapter('B2', 512, 48, 21, 6, np.zeros(21, np.float32), basis)
    E, D = full.E.detach().clone(), full.D.detach().clone()
    rng = torch.get_rng_state().clone()
    full.front.mixer = models.SingleBranchMixer(full.front.mixer, method)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(full.E, E) and torch.equal(full.D, D)
    assert sum(p.numel() for p in full.parameters() if p.requires_grad) == expected_count
    x = torch.randn(2, 512, 21, requires_grad=True)
    assert torch.equal(full.front(x), x)

    class FrozenProxy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.1), requires_grad=False)

        def forward(self, z):
            return self.scale * z[:, -48:]

    foundation = FrozenProxy()
    output = full(x, foundation)
    output.square().mean().backward()
    assert output.shape == (2, 48, 21)
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert full.E.grad is not None and full.E.grad.abs().sum() > 0
    branch = full.front.mixer.temporal if method == 'P_TIME_ONLY' else full.front.mixer.feature
    assert branch.output.weight.grad is not None and branch.output.weight.grad.abs().sum() > 0
    assert foundation.scale.grad is None


def test_patch_preserves_factory_rng_and_shared_state_then_restores():
    def factory(spec, task, endpoint, foundation):
        torch.manual_seed(spec['seed'])
        model = networks.ForecastAdapter('B2', 512, 48, 21, 6, np.zeros(21, np.float32), np.eye(21, 6, dtype=np.float32))
        return model, {'fit_end': endpoint}

    fit = SimpleNamespace(create_model=factory, predict_model=lambda *args: None)
    spec = dict(method_id='P_TIME_ONLY', seed=92601, family='B2')
    reference, _ = factory(spec, None, 10482, None)
    rng = torch.get_rng_state().clone()
    with models.patch_core(fit):
        actual, normalization = fit.create_model(spec, None, 10482, None)
        assert torch.equal(torch.get_rng_state(), rng)
        assert normalization['fit_end'] == 10482
        for name, tensor in actual.state_dict().items():
            assert torch.equal(tensor, reference.state_dict()[name])
    assert fit.create_model is factory


@pytest.mark.parametrize('changes', [dict(seed=92600), dict(K=3), dict(dataset='ETTh1'),
    dict(job_id='wrong'), dict(extended=1), dict(method_id='A1')])
def test_job_rejects_other_axes(changes):
    with pytest.raises(ValueError):
        run_one.validate_job(dict(job(), **changes))


def test_schedule_selection_requires_complete_curve_and_earlier_tie():
    assert run_one.schedule(job()) == [.25, 1., 3.]
    assert run_one.schedule(job(extended=True)) == [.25, 1., 3., 6., 10.]
    result = dict(curves=[dict(epoch=e, astop_mse=m) for e, m in [(0., 2.), (.25, 1.), (1., 1.), (3., 3.)]],
                  summary=dict(status='SEGMENT_COMPLETE', selected_epoch=.25, selected_astop_mse=1.))
    assert run_one.choose_epoch(result, [.25, 1., 3.]) == (.25, 1.)
    result['summary']['selected_epoch'] = 1.
    with pytest.raises(ValueError):
        run_one.choose_epoch(result, [.25, 1., 3.])


def test_extension_whole_group_budget_uses_selection_only(monkeypatch, tmp_path):
    instance = controller.Controller.__new__(controller.Controller)
    instance.jobs = [dict(job(), seed=s, job_id=f'JPB_P_TIME_ONLY_jena_K6_s{s}') for s in range(92601, 92611)]
    instance.extensions = [dict(j, extended=True, job_id=j['job_id']+'_ext10') for j in instance.jobs]
    instance.state = dict(run_statuses={j['job_id']: 'COMPLETE' for j in instance.jobs})
    instance.extension_estimate, instance.margin = 2700., 120.
    monkeypatch.setattr(runtime, 'REPORT', tmp_path)
    monkeypatch.setattr(runtime, 'read', lambda path: {'row': {'selected_epoch': 3., 'MSE': float('nan')}})
    monkeypatch.setattr(controller.time, 'time', lambda: 1000.)
    instance.deadline = 28119.
    assert instance.extension_decision('P_TIME_ONLY')['status'] == 'NOT_RUN_BUDGET'
    instance.deadline = 28120.
    assert instance.extension_decision('P_TIME_ONLY')['status'] == 'ADMITTED'


def test_safe_access_rejects_missing_tail_proof():
    access = dict(new_T_numeric_values_parsed=0, members=[dict(member=n, tail_numeric_values_parsed=0)
                  for n in ('x.npy','finite.npy','times.npy','mean.npy','std.npy','columns.npy')])
    runtime.assert_protected_T(dict(task_access=access))
    access['members'][0]['tail_numeric_values_parsed'] = 1
    with pytest.raises(ValueError):
        runtime.assert_protected_T(dict(task_access=access))


def test_selection_is_sealed_before_fresh_refit_and_V(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'REPORT', tmp_path / 'reports')
    monkeypatch.setattr(runtime, 'CACHE', tmp_path / 'cache')
    public, cache = runtime.REPORT / 'attempt', runtime.CACHE / 'attempt'
    public.mkdir(parents=True)
    cache.mkdir(parents=True)
    events = []
    selected_job = job()
    spec = run_one.build_spec(selected_job)
    toy = torch.nn.Linear(2, 2)
    norm = dict(mean=np.zeros(2, np.float32), std=np.ones(2, np.float32))
    bounds = dict(A_fit=(0, 10482), A_stop=(10482, 12579), M_cal=(12579, 14675),
                  V_select=(14675, 16772), T_final=(16772, 20965))
    task = dict(physical=np.zeros((16772, 2), np.float32), split_ranges=bounds, L=512, H=48,
                columns=['a', 'b'], array_sha256={}, data_sha256='data', dataset='jena', _campaign_access={})

    def stage(specification, data, endpoint, stop, epochs, *unused):
        if stop is not None:
            events.append(('search', endpoint, stop))
            result = dict(curves=[dict(epoch=e, astop_mse=m) for e, m in [(0., 2.), (.25, 1.), (1., 1.), (3., 3.)]],
                          summary=dict(status='SEGMENT_COMPLETE', selected_epoch=.25, selected_astop_mse=1.))
        else:
            assert runtime.read(public / 'SELECTION.json')['V_used'] is False
            events.append(('refit', endpoint, stop))
            assert epochs == [.25]
            result = dict(summary=dict(status='SEGMENT_COMPLETE'))
        result['summary'].update(seconds=1., exposures=1, updates=1)
        result.update(model=toy.state_dict(), normalization=norm)
        return result

    def predict(model, normalization, data, foundation, specification, path, modules, **kwargs):
        seal = runtime.read(public / 'SELECTION.json')
        assert seal['V_used'] is False and seal['selected_epoch'] == .25
        events.append(('V', bounds['V_select']))
        path.write_bytes(b'fake V arrays')
        return dict(MSE=1., MAE=1., prediction_path=str(path), prediction_sha256=runtime.sha(path))

    fake_fit = SimpleNamespace(_source_contract=lambda: ('core', {'fit.py': 'hash'}), _stage=stage,
        create_model=lambda *a: (toy, norm), _torch_new=lambda path, value: path.write_bytes(b'fake bundle'))
    fake_training = SimpleNamespace(fingerprint=lambda value: 'fingerprint', cpu_state=lambda model: model.state_dict())
    monkeypatch.setattr(run_one, 'predict_saved', predict)
    result = run_one.run_stages(selected_job, task, SimpleNamespace(initial_hash='f'),
        dict(fit=fake_fit, training=fake_training), spec, cache, public, 1000.)
    assert events == [('search', 10482, bounds['A_stop']), ('refit', 12579, None), ('V', bounds['V_select'])]
    assert result['selection_sealed_before_V']
    seal = public / 'SELECTION.json'
    digest = runtime.sha(seal)
    seal.write_text('{}')
    with pytest.raises(ValueError):
        run_one.check_selection(seal, digest, selected_job)
