import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch
from torch import nn


def impl():
    try:
        return importlib.import_module('research.adapter_improvements_20261018_v1.training_jobs')
    except ModuleNotFoundError:
        pytest.fail('Registered training factory is not implemented')


@pytest.fixture
def synthetic(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    campaign = repo / 'research/informs_broad_validation_20261015_v1'
    if str(campaign) not in sys.path:
        sys.path.insert(0, str(campaign))
    modules = {key: importlib.import_module(name) for key, name in
               [('fit', 'fit_base'), ('training', 'training'), ('data', 'data_contract')]}
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    data = rng.normal(size=(160, 3)).astype(np.float32)
    mask = np.ones_like(data, dtype=bool)
    mask[::7, 1] = False
    task = dict(L=40, H=3, physical=data, observed=mask, columns=['a', 'b', 'c'],
                data_sha256=modules['data'].array_sha256(data),
                split_ranges=dict(A_fit=(0, 80), A_stop=(80, 96), M_cal=(96, 112),
                                  V_select=(112, 128), T_final=(128, 160)))

    class Frozen(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.1), requires_grad=False)
            self.streams = self.row_calls = 0
            self.initial_hash = 'synthetic_frozen_F'
            self.horizon = task['H']
            self.row_microbatch = 128
            self.checkpoint_inputs = False

        def forward(self, x):
            self.streams += len(x) * x.shape[-1]
            self.row_calls += 1
            return self.weight * x[:, -1:].expand(-1, task['H'], -1)

        def assert_frozen_unchanged(self):
            assert self.weight.item() == torch.tensor(1.1).item()
            assert self.weight.grad is None

    f = Frozen()
    ctx = dict(repo=repo, asset=tmp_path, cache=tmp_path/'cache', report=tmp_path/'report',
               modules=modules, task=task, foundation=f, deadline=time.time()+120,
               job_deadline=time.time()+120, plan={})
    return ctx, modules, task, f


def spec(family='P_SPLIT', init_kind='PCA'):
    return dict(run_id='registered_synthetic', family=family, init_kind=init_kind, latent_k=2,
                seed=92601, device='cpu', epochs=[.25, 1., 3.], lr=.001,
                effective_batch=16, microbatch=4, F_row_microbatch=128, checkpoint_inputs=False)


def test_factory_uses_internal_b0_preserves_external_identity_and_restores_on_exception(synthetic):
    ctx, modules, task, f = synthetic
    m = impl()
    original = modules['fit'].create_model
    source = modules['fit']._source_contract
    verify = modules['training'].verify_contract
    splits = modules['data'].split_ranges
    try:
        with m.model_factory(ctx) as create:
            model, norm = create(spec(), task, 80, f)
            b0, b0norm = original(dict(spec(), family='B0'), task, 80, f)
            x = torch.randn(2, 40, 3)
            assert torch.equal(model(x, f), b0(x, f))
            assert sum(p.numel() for p in model.parameters()) == 1152 + 6 + 42
            assert norm['family'] == 'P_SPLIT'
            np.testing.assert_array_equal(norm['basis'], b0norm['basis'])
            assert modules['fit']._source_contract()[0] != source()[0]
            raise RuntimeError('deliberate scope exit')
    except RuntimeError as exc:
        assert str(exc) == 'deliberate scope exit'
    assert modules['fit'].create_model is original
    assert modules['fit']._source_contract is source
    assert modules['training'].verify_contract is verify
    assert modules['data'].split_ranges is splits


def test_dca_factory_uses_saved_fp32_basis_pinv_and_rejects_wrong_scope(synthetic):
    ctx, modules, task, f = synthetic
    m = impl()
    calls = []
    basis = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 2)))[0].astype(np.float32)
    def loader(local_ctx, actual_task, fit_end, k, norm):
        calls.append((fit_end, k))
        return basis, dict(status='COMPLETE', fit_scope={'fit_end': fit_end}, cache_sha256='a'*64)
    ctx['initialization_loader'] = loader
    with m.model_factory(ctx) as create:
        model, norm = create(spec('B2', 'DCA'), task, 80, f)
        assert torch.equal(model.E, torch.from_numpy(basis))
        assert torch.equal(model.D[0], torch.linalg.pinv(model.E))
        assert model.D[1:].count_nonzero() == 0
        assert norm['initialization_binding']['cache_sha256'] == 'a'*64
    assert calls == [(80, 2)]
    ctx['initialization_loader'] = lambda *args: (basis, dict(status='COMPLETE', fit_scope={'fit_end':96}, cache_sha256='a'*64))
    with m.model_factory(ctx) as create, pytest.raises(ValueError, match='scope'):
        create(spec('B2', 'DCA'), task, 80, f)


def test_extended_resume_validation_checks_initialization_scope_and_cache(synthetic):
    ctx, modules, _, _ = synthetic
    m = impl()
    contract = dict(source_sha='s', config_sha='c', data_sha='d', model_sha='f', scaler_sha='z',
                    initialization_sha='i', initialization_fit_scope={'fit_end':80}, initialization_cache_sha256='a'*64)
    with m.model_factory(ctx):
        modules['training'].verify_contract(contract, dict(contract))
        for key, changed in [('initialization_fit_scope', {'fit_end':96}),
                             ('initialization_cache_sha256', 'b'*64), ('initialization_sha', 'changed')]:
            with pytest.raises(ValueError, match=key):
                modules['training'].verify_contract(contract, dict(contract, **{key:changed}))


def test_execute_job_saves_each_search_endpoint_and_selected_a60_with_independent_metrics(synthetic):
    ctx, modules, task, f = synthetic
    m = impl()
    job = dict(job_id='synthetic_split_s92601', kind='training', family='P_SPLIT', init_kind='PCA',
               cohort='synthetic', seed=92601, K=2, device='cpu', scientific_identity={'test':'actual_cpu'})
    output = m.execute_job(job, ctx)
    assert output['status'] == 'COMPLETE'
    result = output['result']
    endpoints = result['common_endpoint_rows']
    assert len(endpoints) == 8
    assert {(r['epoch'], r['evaluation']) for r in endpoints} == {
        (epoch, split) for epoch in (0., .25, 1., 3.) for split in ('A_fit', 'A_stop')}
    assert len(result['rows']) == 1 and result['rows'][0]['evaluation'] == 'V_select'
    assert result['rows'][0]['fit'] == 'fresh_A60'
    assert result['rows'][0]['independent_FP64'] == 'PASS'
    all_rows = endpoints + result['rows']
    training_origins = modules['data'].origins(task['split_ranges']['A_fit'],task['L'],task['H'],1)
    for row in all_rows:
        path = ctx['cache'] / row['prediction_binding']['reference'].removeprefix('cache://')
        assert modules['data'].sha256_file(path) == row['prediction_binding']['sha256']
        with np.load(path, allow_pickle=False) as archive:
            expected_origins = modules['data'].origins(task['split_ranges'][row['evaluation']],task['L'],task['H'],task['H'])
            np.testing.assert_array_equal(archive['origins'], expected_origins)
            assert row['n_origins'] == len(expected_origins)
            assert row['evaluation_stride'] == task['H']
            difference = np.where(archive['observed'], archive['pred_normalized'].astype(np.float64)-archive['target_normalized'].astype(np.float64), 0)
            expected = (np.square(difference).sum((0,1))/archive['observed'].sum((0,1))).mean()
        assert row['MSE'] == pytest.approx(expected, rel=1e-14, abs=1e-14)
        assert row['logical_F_rows'] == row['n_origins'] * 2
        if row['stage'] == 'search':
            assert row['exposures'] == int(np.ceil(row['epoch']*len(training_origins)))
    public = json.loads(Path(output['artifact_paths'][0]).read_text(encoding='utf-8'))
    assert public['recipe']['epochs'] == [.25, 1., 3.]
    assert public['recipe']['sampler_initialization_rng_separate']
    assert public['selection_sealed_before_V']
    assert public['initialization_events']
    assert all(event['initialization_seconds'] >= 0 for event in public['initialization_events'])
    bundle = torch.load(ctx['cache'] / result['bundle_binding']['reference'].removeprefix('cache://'), weights_only=False)
    with m.model_factory(ctx) as create:
        model, norm = create(bundle['spec'], task, 96, f)
        model.load_state_dict(bundle['model'], strict=True)
        with np.load(ctx['cache'] / result['rows'][0]['prediction_binding']['reference'].removeprefix('cache://')) as saved:
            x, _, _ = modules['training'].FastWindows(task,norm['mean'],norm['std'],'cpu').get(saved['origins'])
            np.testing.assert_allclose(model(x,f).detach().numpy(), saved['pred_normalized'], atol=2e-6, rtol=2e-6)
        deployment = importlib.import_module('deployment')
        pool = deployment.FoundationPool(None, device='cpu', factory=lambda *args:f)
        route = deployment.Route(dict(family='P_SPLIT', run_id=job['job_id'],
                                      artifact_cache_relative_path=result['bundle_binding']['reference'].removeprefix('cache://'),
                                      artifact_sha256=result['bundle_binding']['sha256']),
                                 task,ctx['cache'],device='cpu',foundation_pool=pool)
        with np.load(ctx['cache'] / result['rows'][0]['prediction_binding']['reference'].removeprefix('cache://')) as saved:
            windows = np.stack([task['physical'][o-task['L']:o] for o in saved['origins']])
            observed = np.stack([task['observed'][o-task['L']:o] for o in saved['origins']])
            physical_prediction = route.predict_physical(windows, observed)
            expected = saved['pred_normalized'].astype(np.float64)*norm['std'].astype(np.float64)+norm['mean'].astype(np.float64)
            np.testing.assert_allclose(physical_prediction, expected, atol=3e-6, rtol=3e-6)
        route.close()
        pool.close()


def test_saved_predictions_are_immutable_on_resume_and_tampering_is_rejected(synthetic):
    ctx, _, _, _ = synthetic
    m = impl()
    job = dict(job_id='immutable_cpu', kind='training', family='P_LOCAL_CONTROL', init_kind='PCA',
               cohort='synthetic', seed=92601, K=2, device='cpu')
    result = m.execute_job(job, ctx)['result']
    path = ctx['cache']/result['rows'][0]['prediction_binding']['reference'].removeprefix('cache://')
    original = path.read_bytes()
    m.execute_job(job, ctx)
    assert path.read_bytes() == original
    with np.load(path) as archive:
        values = {key:archive[key] for key in archive.files}
    values['pred_normalized'][0,0,0] += 1
    with path.open('wb') as handle:
        np.savez(handle, **values)
    with pytest.raises(ValueError, match='Existing saved prediction'):
        m.execute_job(job, ctx)


def test_profile_pair_uses_actual_task_dimensions_and_reserves_complete_pair(synthetic):
    ctx, modules, task, _ = synthetic
    m = impl()
    jobs = [dict(job_id='profile_'+family, family=family, init_kind='PCA', cohort='synthetic',
                 seed=92601, K=2, device='cpu') for family in ('P_SPLIT', 'P_LOCAL_CONTROL')]
    result = m.profile_pair(jobs, ctx, warmup_updates=1, measured_updates=1)
    assert result['dimensions'] == dict(C=3, L=40, H=3, K=2)
    assert result['arms'] == ['P_SPLIT', 'P_LOCAL_CONTROL']
    assert result['estimated_complete_pair_seconds'] > 0
    assert result['includes'] == ['search_CP0_quarter_1_3', 'fresh_A60_3epoch_worst_case',
                                   'all_endpoint_predictions', 'selected_A60_V', 'verification_margin']
    assert result['RNG_restored'] and result['disposable_models']
    counts = {name:len(modules['data'].origins(task['split_ranges'][name],task['L'],task['H'],task['H']))
              for name in ('A_fit','A_stop','V_select')}
    assert counts['A_fit'] == 13
    expected_batches = (4*int(np.ceil(counts['A_stop']/4))+
                        4*(int(np.ceil(counts['A_fit']/8))+int(np.ceil(counts['A_stop']/8)))+
                        int(np.ceil(counts['V_select']/8)))
    assert expected_batches == 21
    assert all(arm['estimated_prediction_batches'] == expected_batches for arm in result['measurements'])
    assert result['evaluation_stride'] == task['H']
    assert result['evaluation_origin_counts'] == counts


def test_unregistered_recipe_override_is_rejected_before_model_load(synthetic):
    ctx, _, _, _ = synthetic
    job = dict(job_id='bad_recipe', kind='training', family='P_SPLIT', init_kind='PCA',
               cohort='synthetic', seed=92601, K=2, device='cpu', lr=.02)
    with pytest.raises(ValueError, match='recipe'):
        impl().execute_job(job, ctx)
