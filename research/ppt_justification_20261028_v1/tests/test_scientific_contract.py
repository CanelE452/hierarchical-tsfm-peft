"""CPU-only structural, optimizer, selection, and family-budget tests."""
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.ppt_justification_20261028_v1 import models, runtime, run_one, controller, prepare

BROAD = runtime.REPO / 'research/informs_broad_validation_20261015_v1'
sys.path.insert(0, str(BROAD))
import networks
import fit_base
import training

torch.set_num_threads(2)


def original(channels=21, k=6, seed=92601):
    torch.manual_seed(seed)
    return networks.ForecastAdapter('B2', 512, 48, channels, k,
                                    np.zeros(channels, np.float32), np.eye(channels, k, dtype=np.float32))


def job(method='NO_F_LINEAR', dataset='jena', extended=False):
    k = runtime.KS[dataset]
    return dict(job_id=f'PJ_{method}_{dataset}_K{k}_s92601' + ('_ext10' if extended else ''),
                method_id=method, dataset=dataset, K=k, seed=92601, extended=extended)


@pytest.mark.parametrize('channels,k', [(21, 6), (26, 10)])
def test_no_foundation_head_is_registered_live_and_shared(channels, k):
    base = original(channels, k)
    initial = copy.deepcopy(base.state_dict())
    model = models.DirectNoFoundation(base)
    assert sum(p.numel() for p in model.head.parameters()) == 24624
    assert all(torch.equal(value, model.adapter.state_dict()[name]) for name, value in initial.items())
    assert torch.equal(model.head.weight[:, -1], torch.ones(48))
    assert torch.count_nonzero(model.head.weight[:, :-1]) == 0
    assert torch.count_nonzero(model.head.bias) == 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
    included = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in included for p in model.head.parameters())
    foundation = runtime.NoFoundation('cpu')
    x, target = torch.randn(2, 512, channels), torch.randn(2, 48, channels)
    first = model(x, foundation)
    assert torch.equal(first, x[:, -1:].expand(-1, 48, -1))
    before = model.head.weight.detach().clone()
    (first-target).square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, model.head.weight)
    optimizer.zero_grad()
    model(x, foundation).square().mean().backward()
    assert model.adapter.E.grad.abs().sum() > 0
    assert model.adapter.front.mixer.temporal.output.weight.grad.abs().sum() > 0
    foundation.assert_frozen_unchanged()
    assert foundation.initial_hash == 'NO_FOUNDATION' and foundation.streams == 0


@pytest.mark.parametrize('method,channels,k', [('NO_ANCHOR', 7, 4), ('NO_ANCHOR', 26, 10), ('A5_LAG0', 7, 4)])
def test_decoder_interventions_preserve_common_initial_tensors_and_rng(method, channels, k):
    base = original(channels, k)
    expected_count = sum(p.numel() for p in base.parameters()) - (6*k*channels if method == 'A5_LAG0' else 0)
    rng = torch.get_rng_state().clone()
    variant = models.DecoderIntervention(base, method)
    assert torch.equal(rng, torch.get_rng_state())
    assert variant.front is base.front and variant.E is base.E
    assert sum(p.numel() for p in variant.parameters()) == expected_count
    assert not any(name in ('b', 'bias') for name, _ in variant.named_parameters())
    if method == 'NO_ANCHOR':
        assert variant.D is base.D
    else:
        assert torch.equal(variant.D0, base.D[0])


def test_no_anchor_removes_only_direct_addition_with_nonzero_seven_lags():
    base = original(7, 4)
    with torch.no_grad():
        base.D[1:].normal_(0, .05)
        base.front.mixer.temporal.output.weight.fill_(.001)
    variant = models.DecoderIntervention(base, 'NO_ANCHOR')
    x = torch.randn(2, 512, 7)
    z = base.encode(x)
    future = torch.randn(2, 48, 4)
    assert torch.allclose(variant.decode(x, z, future) + x[:, -1:], base.decode(x, z, future), atol=1e-7)
    joined = torch.cat((z, future), 1)
    features = torch.cat([joined[:, 512-lag:560-lag]-z[:, -1-lag, None] for lag in models.LAGS], -1)
    assert torch.equal(variant.decode(x, z, future), features @ base.D.reshape(28, 7))


def test_a5_keeps_anchor_and_only_lag_zero():
    base = original(7, 4)
    variant = models.DecoderIntervention(base, 'A5_LAG0')
    x = torch.randn(2, 512, 7)
    z, future = base.encode(x), torch.randn(2, 48, 4)
    assert torch.allclose(base.decode(x, z, future), variant.decode(x, z, future), atol=1e-7)
    with torch.no_grad():
        base.D[1:].fill_(.25)
    expected = x[:, -1:] + (future-z[:, -1:]) @ base.D[0]
    assert torch.equal(variant.decode(x, z, future), expected)
    assert not torch.allclose(base.decode(x, z, future), expected)


def test_no_p_linear_factory_uses_original_initialization_then_removes_p():
    factory = lambda spec, task, fit_end, foundation: (original(21, 6, spec['seed']), {})
    fit = SimpleNamespace(create_model=factory, predict_model=lambda *a: None)
    expected = original()
    with models.patch_core(fit):
        model, _ = fit.create_model(dict(method_id='NO_F_LINEAR_NO_P', seed=92601), {}, 100, runtime.NoFoundation('cpu'))
    assert isinstance(model.adapter.front, torch.nn.Identity)
    assert torch.equal(model.adapter.E, expected.E) and torch.equal(model.adapter.D, expected.D)
    assert sum(p.numel() for p in model.parameters()) == 24624 + 21*6*8
    assert fit.create_model is factory


def test_real_core_optimizer_cpu_stage_trains_no_f_without_calling_foundation(tmp_path):
    task = dict(physical=np.random.default_rng(45).normal(size=(672, 7)).astype(np.float32),
                observed=np.ones((672, 7), bool), L=512, H=48, dataset='synthetic', columns=list('abcdefg'))
    spec = dict(run_id='cpu-only-synthetic', method_id='NO_F_LINEAR', family='B2', latent_k=4,
                seed=92601, device='cpu', lr=.001, microbatch=4, effective_batch=16)
    foundation = runtime.NoFoundation('cpu')
    source_sha, _ = fit_base._source_contract()
    contract = dict(source_sha=source_sha, config_sha='synthetic', data_sha='synthetic', model_sha='NO_FOUNDATION')
    with models.patch_core(fit_base):
        stage = fit_base._stage(spec, task, 576, (576, 672), [.25, 1., 3.],
                               tmp_path/'cache', tmp_path/'public', foundation, time.time()+30, contract)
    assert stage['summary']['status'] == 'SEGMENT_COMPLETE'
    assert stage['summary']['updates'] > 0
    assert stage['summary']['F_total_logical_streams'] == 0
    assert stage['summary']['contract']['model_sha'] == 'NO_FOUNDATION'
    assert stage['summary']['trainable_parameters'] == 34023 + 24624
    assert 'head.weight' in stage['model'] and 'head.bias' in stage['model']


@pytest.mark.parametrize('changes', [dict(seed=92606), dict(K=3), dict(dataset='ETTh1'),
                                    dict(job_id='wrong'), dict(extended=1), dict(method_id='OURS')])
def test_job_rejects_unregistered_axes(changes):
    with pytest.raises(ValueError):
        run_one.validate_job(dict(job(), **changes))


def test_all_registered_jobs_and_schedules():
    base = prepare.jobs_for(prepare.FAMILIES)
    optional = prepare.jobs_for(prepare.OPTIONAL)
    extensions = prepare.jobs_for(prepare.FAMILIES+prepare.OPTIONAL, True)
    assert (len(base), len(optional), len(extensions)) == (20, 10, 30)
    assert len({j['job_id'] for j in base+optional+extensions}) == 60
    for selected in base+optional+extensions:
        run_one.validate_job(selected)
        assert run_one.schedule(selected) == ([.25, 1., 3., 6., 10.] if selected['extended'] else [.25, 1., 3.])


def test_extension_requires_three_cap_selections_and_whole_family_budget(tmp_path, monkeypatch):
    instance = controller.Controller.__new__(controller.Controller)
    instance.jobs = prepare.jobs_for(['NO_ANCHOR:ETTh1'])
    instance.extensions = prepare.jobs_for(['NO_ANCHOR:ETTh1'], True)
    instance.state = dict(run_statuses={j['job_id']: 'COMPLETE' for j in instance.jobs})
    instance.base_estimate, instance.extension_estimate, instance.margin = 1080., 1800., 120.
    monkeypatch.setattr(runtime, 'REPORT', tmp_path)
    selected = {j['job_id']: 3. if index < 3 else .25 for index, j in enumerate(instance.jobs)}
    monkeypatch.setattr(runtime, 'read', lambda path: {'row': {'selected_epoch': selected[path.parent.name], 'MSE': float('nan')}})
    monkeypatch.setattr(controller.time, 'time', lambda: 1000.)
    instance.deadline = 10119.
    assert instance.extension_decision('NO_ANCHOR:ETTh1')['status'] == 'NOT_RUN_BUDGET'
    instance.deadline = 10120.
    assert instance.extension_decision('NO_ANCHOR:ETTh1')['status'] == 'ADMITTED'
    selected[instance.jobs[2]['job_id']] = .25
    assert instance.extension_decision('NO_ANCHOR:ETTh1')['status'] == 'NOT_REQUIRED'


def test_reestimate_no_f_uses_elapsed_time_and_same_family_only(tmp_path, monkeypatch):
    instance = controller.Controller.__new__(controller.Controller)
    instance.jobs = prepare.jobs_for(['NO_F_LINEAR:jena', 'NO_F_LINEAR_NO_P:jena'])
    instance.base_estimate, instance.extension_estimate = 1080., 1800.
    monkeypatch.setattr(runtime, 'REPORT', tmp_path)
    for selected, duration in ((instance.jobs[0], 100.), (instance.jobs[5], 1000.)):
        path = tmp_path/'jobs'/selected['job_id']/'RESULT.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(elapsed_seconds=duration, row={'MSE': 'not consulted'})))
    assert instance.job_estimate(instance.jobs[1]) == 135.
    assert instance.job_estimate(dict(instance.jobs[1], extended=True)) == 450.


def test_protected_T_evidence_for_each_loader():
    access = dict(new_T_numeric_values_parsed=0, members=[dict(member=n, tail_numeric_values_parsed=0)
                  for n in ('x.npy','finite.npy','times.npy','mean.npy','std.npy','columns.npy')])
    runtime.assert_protected_T(dict(job=job(), task_access=access))
    access['members'][0]['tail_numeric_values_parsed'] = 1
    with pytest.raises(ValueError):
        runtime.assert_protected_T(dict(job=job(), task_access=access))
    for dataset in ('ETTh1', 'appliances'):
        selected = dict(job=job('NO_ANCHOR', dataset), task_access=dict(T_numeric_values_parsed=0, prefix_sha256='safe'))
        runtime.assert_protected_T(selected)
        selected['task_access']['T_numeric_values_parsed'] = 1
        with pytest.raises(ValueError):
            runtime.assert_protected_T(selected)


def test_astop_selection_requires_complete_schedule_and_earlier_tie():
    search = dict(curves=[dict(epoch=e, astop_mse=m) for e, m in [(0.,2.),(.25,1.),(1.,1.),(3.,3.)]],
                  summary=dict(status='SEGMENT_COMPLETE', selected_epoch=.25, selected_astop_mse=1.))
    assert run_one.choose_epoch(search, [.25, 1., 3.]) == (.25, 1.)
    search['summary']['selected_epoch'] = 1.
    with pytest.raises(ValueError):
        run_one.choose_epoch(search, [.25, 1., 3.])


def test_selection_is_sealed_before_fresh_refit_and_V(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'REPORT', tmp_path/'reports')
    monkeypatch.setattr(runtime, 'CACHE', tmp_path/'cache')
    public, cache = runtime.REPORT/'attempt', runtime.CACHE/'attempt'
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
                columns=['a','b'], array_sha256={}, data_sha256='data', dataset='jena', _campaign_access={})

    def stage(specification, data, endpoint, stop, epochs, *unused):
        if stop is not None:
            events.append(('search', endpoint, stop))
            result = dict(curves=[dict(epoch=e, astop_mse=m) for e,m in [(0.,2.),(.25,1.),(1.,1.),(3.,3.)]],
                          summary=dict(status='SEGMENT_COMPLETE', selected_epoch=.25, selected_astop_mse=1.))
        else:
            assert runtime.read(public/'SELECTION.json')['V_used'] is False
            events.append(('refit', endpoint, stop))
            assert epochs == [.25]
            result = dict(summary=dict(status='SEGMENT_COMPLETE'))
        result['summary'].update(seconds=1., exposures=1, updates=1)
        result.update(model=toy.state_dict(), normalization=norm)
        return result

    def predict(model, normalization, data, foundation, specification, path, modules, **kwargs):
        assert runtime.read(public/'SELECTION.json')['selected_epoch'] == .25
        events.append(('V', bounds['V_select']))
        path.write_bytes(b'synthetic V archive')
        return dict(MSE=1., MAE=1., prediction_path=str(path), prediction_sha256=runtime.sha(path))

    fake_fit = SimpleNamespace(_source_contract=lambda: ('core', {'fit.py':'hash'}), _stage=stage,
                              create_model=lambda *a: (toy,norm), _torch_new=lambda path,value: path.write_bytes(b'bundle'))
    fake_training = SimpleNamespace(fingerprint=lambda value: 'fingerprint', cpu_state=lambda model: model.state_dict())
    monkeypatch.setattr(run_one, 'predict_saved', predict)
    result = run_one.run_stages(selected_job, task, runtime.NoFoundation('cpu'),
                               dict(fit=fake_fit, training=fake_training), spec, cache, public, 1000.)
    assert events == [('search',10482,bounds['A_stop']),('refit',12579,None),('V',bounds['V_select'])]
    assert result['selection_sealed_before_V'] and result['contract']['model_sha'] == 'NO_FOUNDATION'
