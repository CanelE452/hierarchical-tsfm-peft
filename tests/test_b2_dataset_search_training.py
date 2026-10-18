import json
from types import SimpleNamespace

import numpy as np
import pytest


def inputs(tmp_path, dataset='example', channels=7):
    from research.b2_dataset_search_20261019_v1 import train
    repo = tmp_path / 'repo'
    (repo / 'reports' / train.NAME).mkdir(parents=True, exist_ok=True)
    (repo / 'reports' / train.NAME / 'PROTOCOL.md').write_text('fixed synthetic protocol', encoding='utf-8')
    prefix = tmp_path / (dataset + '.npz')
    physical = np.arange(3200 * channels, dtype=np.float32).reshape(3200, channels)
    np.savez(prefix, physical=physical, observed=np.ones_like(physical, bool),
             times=np.arange(3200, dtype=np.int64) * 3600_000_000_000,
             columns=np.asarray([f'c{i}' for i in range(channels)]))
    config = dict(dataset=dataset, L=512, H=48, prefix_only_approved=True,
        interval_seconds=3600, prefix_rows=3200, prefix_sha256=train.sha(prefix),
        split_ranges=dict(A_fit=[0, 2000], A_stop=[2000, 2400], M_cal=[2400, 2800],
                          V_select=[2800, 3200], T_final=[3200, 4000]))
    path = tmp_path / (dataset + '.json')
    path.write_text(json.dumps(config), encoding='utf-8')
    return train, repo, path, prefix


def test_trainer_import_is_inert_and_grid_is_registered():
    from research.b2_dataset_search_20261019_v1 import train
    assert train.k_grid(21) == (3, 6, 12)
    assert train.k_grid(7) == (1, 2, 4)
    assert train.k_grid(2) == (1, 2)
    task = dict(dataset='sample', columns=list(range(21)))
    assert {(job['K'], job['seed']) for job in train.jobs(task)} == {
        (k, seed) for k in (3, 6, 12) for seed in (92601, 92602)}


def factory(events, *, fail_at=None):
    class Helper:
        _selection_decision = staticmethod(lambda *args: 'original')
        def search(self, job, ctx, epochs):
            events.append(('search', job['job_id']))
            if job['job_id'] == fail_at:
                return dict(status='PARTIAL_CHECKPOINTED')
            checkpoint = ctx['cache'] / (job['job_id'] + '.pt')
            checkpoint.write_bytes(b'synthetic checkpoint')
            result = dict(status='COMPLETE', cap=3, V_used=False, selected_epoch=1.,
                spec=dict(dataset=job['cohort'], family='B2', revin=False, seed=job['seed'],
                          latent_k=job['K'], epochs=list(epochs)),
                curves=[dict(epoch=e, astop_mse=m) for e, m in [(0., 4.), (.25, 3.), (1., 2.), (3., 2.)]],
                contract=dict(data_sha=ctx['task']['data_sha256'], model_sha='f' * 64),
                checkpoint_binding=train.binding(checkpoint, ctx), summary=dict(selected_astop_mse=2.))
            return result

        def _refit_predict(self, job, ctx, search, cap):
            decision = train.read(ctx['selection_decision']['path'])
            assert len([item for item in events if item[0] == 'search']) == len(decision['matched_arms'])
            events.append(('refit', job['job_id']))
            bundle, prediction = ctx['cache'] / (job['job_id'] + '_bundle.pt'), ctx['cache'] / (job['job_id'] + '_V.npz')
            bundle.write_bytes(b'synthetic bundle')
            prediction.write_bytes(b'synthetic prediction')
            row = dict(run_id=job['job_id'], family='B2', dataset=job['cohort'], revin=False,
                seed=job['seed'], K=job['K'], C=len(ctx['task']['columns']), L=512, H=48,
                cap=3, selected_epoch=1., MSE=.5, MAE=.4, independent_FP64='PASS',
                evaluation='V_select', fit='fresh_A60', data_sha=ctx['task']['data_sha256'],
                model_sha='f' * 64, prediction_binding=train.binding(prediction, ctx))
            return dict(status='COMPLETE', rows=[row], bundle_binding=train.binding(bundle, ctx),
                        data_sha=ctx['task']['data_sha256'], model_sha='f' * 64)
    helper = Helper()
    def runtime(ctx, task, model_dir):
        return helper, dict(ctx, task=task, foundation=SimpleNamespace(assert_frozen_unchanged=lambda: None))
    from research.b2_dataset_search_20261019_v1 import train
    return runtime, helper


def test_all_wave_searches_precede_all_refits_and_all_widths_are_preserved(tmp_path):
    train, repo, config1, prefix1 = inputs(tmp_path, 'first', 7)
    _, _, config2, prefix2 = inputs(tmp_path, 'second', 21)
    events = []
    runtime, helper = factory(events)
    original = helper._selection_decision
    result = train.execute_many([(config1, prefix1), (config2, prefix2)], tmp_path / 'model',
                                repo=repo, runtime_factory=runtime)
    assert result['new_searches'] == result['fresh_refits'] == 12
    assert [kind for kind, job in events] == ['search'] * 12 + ['refit'] * 12
    assert helper._selection_decision is original
    assert [len(item['rows']) for item in result['datasets']] == [6, 6]
    assert {row['K'] for row in result['datasets'][0]['rows']} == {1, 2, 4}
    seal = train.read(result['selection_path'])
    assert len(seal['matched_arms']) == 12
    text = '\n'.join(path.read_text(encoding='utf-8') for path in (repo / 'reports' / train.NAME).rglob('*.json'))
    assert str(tmp_path) not in text


def test_missing_search_blocks_seal_and_never_reaches_v(tmp_path):
    train, repo, config, prefix = inputs(tmp_path)
    task = train.load_task(config, prefix)[1]
    events = []
    runtime, _ = factory(events, fail_at=train.jobs(task)[2]['job_id'])
    with pytest.raises(RuntimeError, match='incomplete'):
        train.execute_many([(config, prefix)], tmp_path / 'model', repo=repo, runtime_factory=runtime)
    assert not any(kind == 'refit' for kind, job in events)
    assert not (repo / 'reports' / train.NAME / 'ALL_SEARCHES_SEALED.json').exists()


def test_wave_plan_prevents_sealing_only_favorable_subset(tmp_path):
    train, repo, config1, prefix1 = inputs(tmp_path, 'first')
    _, _, config2, prefix2 = inputs(tmp_path, 'second')
    contexts = [train.build_context(config, prefix, tmp_path / 'model', repo=repo)
                for config, prefix in [(config1, prefix1), (config2, prefix2)]]
    plan = repo / 'reports' / train.NAME / 'WAVE_PLAN.json'
    train.seal_wave_plan(contexts, plan)
    runtime, _ = factory([])
    state = train.search_dataset(config1, prefix1, tmp_path / 'model', repo=repo,
                                 runtime_factory=runtime, wave_plan=plan)
    with pytest.raises(ValueError, match='wave'):
        train.seal_searches([state], repo / 'reports' / train.NAME / 'ALL_SEARCHES_SEALED.json')


def test_old_namespace_output_and_unregistered_width_are_rejected(tmp_path):
    train, repo, config, prefix = inputs(tmp_path)
    with pytest.raises(ValueError, match='namespace'):
        train.build_context(config, prefix, tmp_path / 'model', repo=repo,
                            cache_dir=repo / '.cache' / 'b2_dataset_extension_20261018_v1')
    task = train.load_task(config, prefix)[1]
    job = train.jobs(task)[0]
    result = dict(status='COMPLETE', cap=3, V_used=False, selected_epoch=3.,
        spec=dict(dataset=job['cohort'], family='B2', revin=False, seed=job['seed'], latent_k=job['K'], epochs=[.25,1.,3.]),
        curves=[dict(epoch=e, astop_mse=m) for e, m in [(0.,4.),(.25,3.),(1.,2.),(3.,2.)]])
    with pytest.raises(ValueError, match='earlier tie'):
        train._verify_search(job, result)


def test_modified_checkpoint_blocks_refit_before_runtime(tmp_path):
    train, repo, config, prefix = inputs(tmp_path)
    events = []
    runtime, _ = factory(events)
    state = train.search_dataset(config, prefix, tmp_path / 'model', repo=repo, runtime_factory=runtime)
    selection = repo / 'reports' / train.NAME / 'ALL_SEARCHES_SEALED.json'
    train.seal_searches([state], selection)
    checkpoint = train.resolve_binding(state['searches'][0]['checkpoint_binding'], state['ctx'])
    checkpoint.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='SHA mismatch'):
        train.refit_dataset(config, prefix, tmp_path / 'model', selection, repo=repo,
                            runtime_factory=lambda *args: pytest.fail('Changed checkpoint must block runtime'))
    assert not any(kind == 'refit' for kind, job in events)


def test_different_frozen_models_cannot_share_wave_seal(tmp_path):
    train, repo, config, prefix = inputs(tmp_path)
    runtime, _ = factory([])
    state = train.search_dataset(config, prefix, tmp_path / 'model', repo=repo, runtime_factory=runtime)
    state['searches'][0]['contract']['model_sha'] = 'a' * 64
    with pytest.raises(ValueError, match='pinned F state'):
        train.seal_searches([state], repo / 'reports' / train.NAME / 'ALL_SEARCHES_SEALED.json')


def test_source_dependencies_include_actual_reused_cores():
    from research.b2_dataset_search_20261019_v1 import train
    sources = train.source_hashes()
    for name in ('train.py', '__init__.py'):
        assert f'research/{train.NAME}/{name}' in sources
    for relative in ('research/b2_dataset_extension_20261018_v1/train.py',
                     'research/b2_dataset_extension_20261018_v1/baselines.py',
                     'research/b2_revin_20261018_v1/training_jobs.py'):
        assert sources[relative] == train.sha(train.REPO / relative)
    assert train.k_grid(4) == (1, 2)


def test_cost_factory_accepts_registered_nonquarter_width_without_loading_f(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    train, repo, config, prefix = inputs(tmp_path)
    ctx = train.build_context(config, prefix, tmp_path / 'model', repo=repo)
    original_context = train.build_context
    monkeypatch.setattr(train, 'build_context', lambda *args, **kwargs: original_context(
        config, prefix, tmp_path / 'model', repo=repo))
    from research.adapter_diagnostics_20261017_v2 import existing_jobs
    from research.b2_revin_20261018_v1 import costs
    monkeypatch.setattr(existing_jobs, '_modules', lambda ctx: {'legacy': SimpleNamespace(F_SHA='f' * 64)})
    route = object()
    monkeypatch.setattr(costs, 'route_loader', lambda *args, **kwargs: lambda: route)
    module = ModuleType('gpu_runtime')
    module.FrozenBolt = lambda *args, **kwargs: pytest.fail('Factory must not load F')
    monkeypatch.setitem(sys.modules, 'gpu_runtime', module)
    bundle, prediction = ctx['cache'] / 'bundle.pt', ctx['cache'] / 'prediction.npz'
    bundle.write_bytes(b'synthetic bundle')
    np.savez(prediction, pred_normalized=np.zeros((1,48,7),np.float32))
    run_id = 'B2_example_K4_OFF_s92601'
    train.sealed_json(ctx['cache'] / 'VERIFIED_ROWS.json', [dict(run_id=run_id,
        family='B2', K=4, seed=92601, data_sha=ctx['task']['data_sha256'],
        prediction_binding=train.binding(prediction, ctx), bundle_binding=train.binding(bundle, ctx))])
    request = dict(config_path=config, prefix_path=prefix, model_dir=tmp_path/'model',
                   dataset='example', K=4, family='B2', run_id=run_id, seed=92601)
    value = train.cost_inputs(request, ctx['report'], ctx['cache'])
    assert value['candidate']['K'] == 4
    assert value['route_loader']() is route
