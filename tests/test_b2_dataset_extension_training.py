import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def fixture_inputs(tmp_path):
    from research.b2_dataset_extension_20261018_v1 import train

    prefix = tmp_path / 'approved_prefix.npz'
    physical = np.arange(3200 * 5, dtype=np.float32).reshape(3200, 5) / 100
    np.savez(prefix, physical=physical, observed=np.ones_like(physical, bool),
             times=np.arange(3200, dtype=np.int64) * 600_000_000_000,
             columns=np.asarray([f'c{i}' for i in range(5)]))
    config = dict(dataset='synthetic', L=512, H=48, interval_seconds=600,
        prefix_only_approved=True, prefix_rows=3200, prefix_sha256=train.sha(prefix),
        split_ranges=dict(A_fit=[0, 2000], A_stop=[2000, 2400], M_cal=[2400, 2800],
                          V_select=[2800, 3200], T_final=[3200, 4000]))
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    return train, config, path, prefix


def test_training_module_is_available_without_loading_gpu():
    from research.b2_dataset_extension_20261018_v1 import train
    assert callable(train.load_task)
    assert callable(train.execute)


def test_approved_prefix_contract_and_fixed_jobs(tmp_path):
    train, config, path, prefix = fixture_inputs(tmp_path)
    loaded, task = train.load_task(path, prefix)
    assert loaded == config
    assert len(task['physical']) == 3200
    assert task['split_ranges']['T_final'] == (3200, 4000)
    assert len(task['columns']) == 5
    assert [(job['K'], job['seed'], job['revin']) for job in train.jobs(task)] == [
        (2, 92601, False), (2, 92602, False)]


@pytest.mark.parametrize('change', ['gap', 'tail', 'hash', 'approval'])
def test_invalid_input_is_rejected_before_training(tmp_path, change):
    train, config, path, prefix = fixture_inputs(tmp_path)
    if change == 'gap':
        config['split_ranges']['A_stop'][0] += 1
    elif change == 'tail':
        config['prefix_rows'] = 4000
    elif change == 'hash':
        config['prefix_sha256'] = '0' * 64
    else:
        config['prefix_only_approved'] = False
    path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError):
        train.load_task(path, prefix)


def test_two_seed_selection_seals_before_any_refit_and_sanitizes_public_paths(tmp_path):
    train, _, path, prefix = fixture_inputs(tmp_path)
    events = []

    class Helper:
        def search(self, job, ctx, epochs):
            events.append(('search', job['seed']))
            checkpoint = ctx['cache'] / f'checkpoint_{job["seed"]}.pt'
            checkpoint.write_bytes(b'synthetic checkpoint')
            return dict(status='COMPLETE', selected_epoch=1., cap=3, V_used=False,
                spec=dict(dataset=job['cohort'], family='B2', revin=False,
                          seed=job['seed'], latent_k=job['K'], epochs=list(epochs)),
                curves=[dict(epoch=0., astop_mse=4.), dict(epoch=.25, astop_mse=3.),
                        dict(epoch=1., astop_mse=2.), dict(epoch=3., astop_mse=2.)],
                checkpoint_binding=train.binding(checkpoint, ctx),
                summary=dict(selected_astop_mse=2.))

        def _refit_predict(self, job, ctx, search, cap):
            selection = train.read(ctx['selection_decision']['path'])
            assert selection['all_searches_complete'] is True
            assert len(selection['matched_arms']) == 2
            assert [kind for kind, _ in events[:2]] == ['search', 'search']
            events.append(('refit', job['seed']))
            return dict(status='COMPLETE', rows=[dict(run_id=job['job_id'],
                seed=job['seed'], selected_epoch=1., MSE=.5, MAE=.4,
                prediction_binding=dict(reference='cache://prediction.npz', sha256='b' * 64))],
                bundle_binding=dict(reference='cache://selected_bundle.pt', sha256='c' * 64))

    helper = Helper()
    def runtime(ctx, task, model_dir):
        return helper, dict(ctx, task=task, foundation=SimpleNamespace(assert_frozen_unchanged=lambda: None))

    repo = tmp_path / 'repo'
    repo.mkdir()
    result = train.execute(path, prefix, tmp_path / 'model', repo=repo, runtime_factory=runtime)
    assert result['status'] == 'COMPLETE'
    assert result['new_searches'] == result['fresh_refits'] == 2
    assert events == [('search', 92601), ('search', 92602), ('refit', 92601), ('refit', 92602)]
    public = repo / 'reports' / train.NAME / 'datasets' / 'synthetic'
    text = '\n'.join(p.read_text(encoding='utf-8') for p in public.rglob('*.json'))
    assert str(tmp_path) not in text


def test_failed_search_never_starts_refit(tmp_path):
    train, _, path, prefix = fixture_inputs(tmp_path)
    class Helper:
        def search(self, job, ctx, epochs):
            return dict(status='PARTIAL_CHECKPOINTED')
        def _refit_predict(self, *args):
            pytest.fail('Failed search must not reach any fresh refit or V')
    def runtime(ctx, task, model_dir):
        return Helper(), dict(ctx, foundation=SimpleNamespace(assert_frozen_unchanged=lambda: None))
    repo = tmp_path / 'repo'
    repo.mkdir()
    with pytest.raises(RuntimeError, match='incomplete'):
        train.execute(path, prefix, tmp_path / 'model', repo=repo, runtime_factory=runtime)


def test_selection_scope_is_restored_after_refit_exception(tmp_path):
    train, _, path, prefix = fixture_inputs(tmp_path)
    original = lambda *args: 'original'
    helper = SimpleNamespace(_selection_decision=original)
    with pytest.raises(RuntimeError, match='intentional'):
        with train.selection_scope(helper):
            assert helper._selection_decision is not original
            raise RuntimeError('intentional')
    assert helper._selection_decision is original


def test_later_tied_checkpoint_is_rejected():
    from research.b2_dataset_extension_20261018_v1 import train
    job = dict(cohort='synthetic', seed=92601, K=2)
    result = dict(status='COMPLETE', cap=3, V_used=False, selected_epoch=3.,
        spec=dict(dataset='synthetic', seed=92601, latent_k=2, family='B2', revin=False,
                  epochs=[.25, 1., 3.]),
        curves=[dict(epoch=epoch, astop_mse=value) for epoch, value in
                [(0., 4.), (.25, 3.), (1., 2.), (3., 2.)]])
    with pytest.raises(ValueError, match='earlier tie'):
        train._verify_search(job, result)


def test_prefix_header_rejected_before_hashing_protected_tail(tmp_path, monkeypatch):
    train, config, path, prefix = fixture_inputs(tmp_path)
    with np.load(prefix) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    values['physical'] = np.concatenate((values['physical'], values['physical'][:1]))
    np.savez(prefix, **values)
    monkeypatch.setattr(train, 'sha', lambda path: pytest.fail('Tail archive must not be hashed'))
    with pytest.raises(ValueError, match='header'):
        train.load_task(path, prefix)


def test_group_seal_includes_all_four_actual_searches(tmp_path):
    train, _, path, prefix = fixture_inputs(tmp_path)
    repo = tmp_path / 'repo'
    repo.mkdir()
    states = []
    for dataset in ('household', 'sml'):
        config = train.read(path)
        config['dataset'] = dataset
        config_path = tmp_path / (dataset + '.json')
        config_path.write_text(json.dumps(config), encoding='utf-8')
        ctx = train.build_context(config_path, prefix, tmp_path / 'model', repo=repo)
        searches = []
        for job in train.jobs(ctx['task']):
            checkpoint = ctx['cache'] / (job['job_id'] + '.pt')
            checkpoint.write_bytes(b'synthetic checkpoint')
            searches.append(dict(status='COMPLETE', cap=3, V_used=False, selected_epoch=1.,
                spec=dict(dataset=dataset, seed=job['seed'], latent_k=2, family='B2',
                          revin=False, epochs=[.25, 1., 3.]),
                checkpoint_binding=train.binding(checkpoint, ctx),
                curves=[dict(epoch=epoch, astop_mse=value) for epoch, value in
                        [(0., 4.), (.25, 3.), (1., 2.), (3., 2.)]]))
        train.sealed_json(ctx['report'] / 'TRAIN_IDENTITY.json', dict(dataset=dataset))
        states.append(dict(ctx=ctx, jobs=train.jobs(ctx['task']), searches=searches))
    selection = repo / 'reports' / train.NAME / 'ALL_SEARCHES_SEALED.json'
    result = train.seal_searches(states, selection)
    assert len(result['matched_arms']) == 4
    assert set(result['datasets']) == {'household', 'sml'}
    for state in states:
        state['ctx']['selection_decision'] = dict(path=selection, sha256=train.sha(selection))
        for job in state['jobs']:
            assert train._selection_decision(job, state['ctx'], 3)['sha256'] == train.sha(selection)
