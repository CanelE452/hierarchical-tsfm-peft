"""CPU fixture tests; no model, data archive or accelerator is loaded."""
import copy
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.b2_revin_20261018_v1 import contracts
from research.k_tau_rule_20261023_v1 import runner


@pytest.fixture
def setup(tmp_path):
    repo = tmp_path
    cache = repo / '.cache' / runner.NAME
    report = repo / 'reports' / runner.NAME
    cache.mkdir(parents=True)
    report.mkdir(parents=True)
    job = dict(job_id='B2_jena_K6_OFF_s92610', cohort='jena', family='B2',
               revin=False, init_kind='PCA', K=6, seed=92610, device='cuda')
    calls = []

    def spec(j, ctx):
        assert j['seed'] in (92601, 92602)
        return dict(contracts.RECIPE, run_id=j['job_id'], dataset=j['cohort'],
                    family=j['family'], revin=j['revin'], init_kind='PCA',
                    latent_k=j['K'], seed=j['seed'], device='cuda',
                    scientific_identity={}, core_dtype='float32')

    def search(j, ctx, epochs):
        calls.append('search')
        directory = ctx['cache'] / 'runs' / j['job_id'] / 'cap3'
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / 'checkpoint_000003.pt'
        checkpoint.write_bytes(b'fixture checkpoint; never deserialized')
        result = dict(status='COMPLETE', cap=3, V_used=False, selection_split='A_stop',
            spec=dict(helper._spec(j, ctx), epochs=list(epochs)),
            contract=dict(data_sha='data', model_sha='F', config_sha='config', source_sha='source'),
            curves=[dict(epoch=e, astop_mse=m) for e, m in ((0., 2.), (.25, 1.), (1., 1.), (3., 1.5))],
            selected_epoch=.25, summary=dict(selected_epoch=.25, selected_astop_mse=1., seconds=2.),
            checkpoint_binding=contracts.binding(checkpoint, ctx))
        contracts.sealed_json(directory / 'SEARCH_RESULT.json', result)
        return result

    def selection(j, ctx, cap):
        raise AssertionError('Original four-arm validator must be scoped out')

    def refit(j, ctx, result, cap):
        decision = helper._selection_decision(j, ctx, cap)
        calls.append('V')
        bundle = ctx['cache'] / 'bundle.pt'
        bundle.write_bytes(b'fixture bundle')
        prediction = ctx['cache'] / 'prediction.npz'
        prediction.write_bytes(b'fixture arrays; never loaded')
        return dict(status='COMPLETE', rows=[dict(MSE=.4, MAE=.3, independent_FP64='PASS',
            metric_accumulation_dtype='float64', prediction_binding=contracts.binding(prediction, ctx),
            search_seconds=2., refit_seconds=3.)], bundle_binding=contracts.binding(bundle, ctx),
            selection_decision_binding=decision, selection_sealed_before_V=True)

    helper = SimpleNamespace(_spec=spec, _selection_decision=selection, search=search, _refit_predict=refit)
    foundation = SimpleNamespace(initial_hash='F', assert_frozen_unchanged=lambda: calls.append('frozen'))
    ctx = dict(repo=repo, cache=cache, report=report, asset=repo, model_dir=repo,
        deadline=time.time() + 100, protocol_sha256='p' * 64,
        scientific_source_hashes={}, modules={}, foundation=foundation, helper=helper,
        task=dict(dataset='jena', data_sha256='data', L=512, H=48,
                  columns=[str(i) for i in range(21)],
                  split_ranges=dict(A_fit=[0, 5000], A_stop=[5000, 6000], V_select=[7000, 8000])))
    return ctx, job, helper, calls


def test_new_seed_preserves_scientific_spec_and_seals_before_v(setup):
    ctx, job, helper, calls = setup
    original_spec, original_selection = helper._spec, helper._selection_decision
    expected = original_spec(dict(job, seed=92601), ctx)
    expected['seed'] = job['seed']
    result = runner.run_one(ctx, job)
    stored = contracts.read(ctx['cache'] / 'runs' / job['job_id'] / 'cap3' / 'SEARCH_RESULT.json')
    assert stored['spec'] == dict(expected, epochs=[.25, 1., 3.])
    assert result['rows'][0]['MSE'] == .4
    assert result['rows'][0]['bundle_binding'] == result['bundle_binding']
    assert calls == ['search', 'V', 'frozen']
    assert helper._spec is original_spec
    assert helper._selection_decision is original_selection
    seal = contracts.read(ctx['report'] / 'jobs' / job['job_id'] / 'SELECTION.json')
    assert seal['selected_epoch'] == .25
    assert seal['V_used'] is False


def test_exception_restores_functions_and_does_not_retry(setup):
    ctx, job, helper, calls = setup
    original_spec, original_selection = helper._spec, helper._selection_decision

    def fail(*args, **kwargs):
        calls.append('failure')
        raise RuntimeError('fixture interruption')

    helper.search = fail
    with pytest.raises(RuntimeError, match='fixture interruption'):
        runner.run_one(ctx, job)
    assert calls.count('failure') == 1
    assert 'V' not in calls
    assert helper._spec is original_spec
    assert helper._selection_decision is original_selection


def test_wrong_seal_denied_before_v(setup):
    ctx, job, helper, calls = setup
    original_refit = helper._refit_predict

    def tamper(j, active, result, cap):
        path = active['report'] / 'jobs' / j['job_id'] / 'SELECTION.json'
        value = contracts.read(path)
        value['selected_epoch'] = 3.
        contracts.atomic_json(path, value)
        return original_refit(j, active, result, cap)

    helper._refit_predict = tamper
    with pytest.raises(ValueError, match='seal|SHA|binding'):
        runner.run_one(ctx, job)
    assert 'V' not in calls


def test_non_earliest_tie_and_partial_search_never_reach_v(setup):
    ctx, job, helper, calls = setup
    original = helper.search

    def bad(*args, **kwargs):
        result = copy.deepcopy(original(*args, **kwargs))
        result['selected_epoch'] = 1.
        return result

    helper.search = bad
    with pytest.raises(ValueError, match='tie|minimum'):
        runner.run_one(ctx, job)
    assert 'V' not in calls


@pytest.mark.parametrize('seed', [92600, 92611])
def test_unapproved_seeds_are_rejected(setup, seed):
    ctx, job, helper, calls = setup
    job['seed'] = seed
    with pytest.raises(ValueError, match='seed'):
        runner.run_one(ctx, job)
    assert calls == []


def test_partial_preserves_attempt_and_checkpoint(setup):
    ctx, job, helper, calls = setup
    original = helper.search

    def partial(*args, **kwargs):
        return dict(original(*args, **kwargs), status='PARTIAL_CHECKPOINTED')

    helper.search = partial
    with pytest.raises(runner.IncompleteRun, match='PARTIAL_CHECKPOINTED'):
        runner.run_one(ctx, job)
    assert list((ctx['cache'] / 'runs' / job['job_id']).rglob('checkpoint_*.pt'))
    receipt = contracts.read(ctx['report'] / 'jobs' / job['job_id'] / 'attempt_001' / 'RECEIPT.json')
    assert receipt['status'] == 'PARTIAL_CHECKPOINTED'
    assert calls.count('search') == 1 and 'V' not in calls


@pytest.mark.parametrize('seed', runner.SEEDS)
def test_registration_changes_only_the_approved_seed(setup, seed):
    ctx, job, helper, calls = setup
    job['seed'] = seed
    baseline = helper._spec(dict(job, seed=92601), ctx)
    with runner._registration_scope(helper):
        actual = helper._spec(job, ctx)
    assert actual == dict(baseline, seed=seed)
    assert calls == []


@pytest.mark.parametrize('target', ['checkpoint', 'search', 'foundation'])
def test_bound_scientific_state_tampering_denied_before_v(setup, target):
    ctx, job, helper, calls = setup
    original = helper._refit_predict

    def tamper(j, active, search, cap):
        if target == 'checkpoint':
            contracts.resolve_binding(search['checkpoint_binding'], active).write_bytes(b'changed')
        elif target == 'search':
            path = runner._search_path(active, j)
            stored = contracts.read(path)
            stored['selected_epoch'] = 3.
            contracts.atomic_json(path, stored)
        else:
            active['foundation'].initial_hash = 'changed'
        return original(j, active, search, cap)

    helper._refit_predict = tamper
    with pytest.raises(ValueError):
        runner.run_one(ctx, job)
    assert 'V' not in calls


def test_retry_keeps_identity_and_separate_receipts(setup):
    ctx, job, helper, calls = setup
    # core.load_task materializes tuple bounds; JSON identities must compare
    # equally after loading without changing the actual training task.
    ctx['task']['split_ranges'] = {key: tuple(value)
                                  for key, value in ctx['task']['split_ranges'].items()}
    original_bounds = copy.deepcopy(ctx['task']['split_ranges'])
    original = helper.search

    def partial(*args, **kwargs):
        return dict(original(*args, **kwargs), status='PARTIAL_CHECKPOINTED')

    helper.search = partial
    with pytest.raises(runner.IncompleteRun):
        runner.run_one(ctx, job)
    identity = contracts.sha(ctx['report'] / 'jobs' / job['job_id'] / 'IDENTITY.json')
    helper.search = original
    result = runner.run_one(ctx, job)
    directory = ctx['report'] / 'jobs' / job['job_id']
    assert contracts.sha(directory / 'IDENTITY.json') == identity
    assert contracts.read(directory / 'attempt_001' / 'RECEIPT.json')['status'] == 'PARTIAL_CHECKPOINTED'
    assert contracts.read(directory / 'attempt_002' / 'RECEIPT.json')['status'] == 'COMPLETE'
    assert result['rows'][0]['MSE'] == .4
    runner.run_one(ctx, job)
    assert calls.count('search') == 2 and calls.count('V') == 1
    assert ctx['task']['split_ranges'] == original_bounds
    assert all(isinstance(value, tuple) for value in ctx['task']['split_ranges'].values())
