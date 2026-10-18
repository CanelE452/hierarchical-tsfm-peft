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
        return importlib.import_module('research.b2_revin_20261018_v1.training_jobs')
    except ModuleNotFoundError:
        pytest.fail('B2 RevIN search/refit controller is missing')


@pytest.fixture
def synthetic(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    campaign = repo / 'research/informs_broad_validation_20261015_v1'
    if str(campaign) not in sys.path:
        sys.path.insert(0, str(campaign))
    modules = {key: importlib.import_module(name) for key, name in
               [('fit', 'fit_base'), ('training', 'training'), ('data', 'data_contract')]}
    torch.set_num_threads(2)
    values = np.random.default_rng(42).normal(size=(160, 3)).astype(np.float32)
    observed = np.ones_like(values, dtype=bool)
    observed[::7, 1] = False
    task = dict(L=40, H=3, physical=values, observed=observed, columns=['a', 'b', 'c'],
                data_sha256=modules['data'].array_sha256(values),
                units='synthetic', split_ranges=dict(A_fit=(0, 80), A_stop=(80, 96),
                M_cal=(96, 112), V_select=(112, 128), T_final=(128, 160)))

    class Frozen(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.1), requires_grad=False)
            self.streams = 0
            self.initial_hash = 'synthetic_frozen_F'

        def forward(self, x):
            self.streams += len(x) * x.shape[-1]
            return self.weight * x[:, -1:].expand(-1, 3, -1)

        def assert_frozen_unchanged(self):
            assert self.weight.item() == torch.tensor(1.1).item()
            assert self.weight.grad is None

    ctx = dict(repo=repo, cache=tmp_path / 'cache', report=tmp_path / 'report',
               modules=modules, task=task, foundation=Frozen(), deadline=time.time() + 120)
    return ctx


def job(name='synthetic_on', revin=True):
    return dict(job_id=name, family='B2', cohort='synthetic', seed=92601,
                K=2, device='cpu', revin=revin)


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal_state(left, right)
    else:
        assert a == b


def decision(ctx, item, cap):
    path = ctx['cache'] / f'decision_cap{cap}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(cohort=item['cohort'], cap=cap, selection_split='A_stop',
        V_used=False, matched_arms=[item['job_id'], 'paired_second_seed',
        'paired_off_first_seed', 'paired_off_second_seed'], status='SEALED')), encoding='utf-8')
    import hashlib
    ctx['selection_decision'] = dict(path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_search_never_requires_v_and_does_not_refit_before_group_decision(synthetic):
    ctx = synthetic
    ctx['task'] = dict(ctx['task'], physical=ctx['task']['physical'][:96],
                       observed=ctx['task']['observed'][:96])
    result = impl().search(job(), ctx, [.25, 1., 3.])
    assert result['status'] == 'COMPLETE'
    assert [row['epoch'] for row in result['curves']] == [0., .25, 1., 3.]
    assert result['summary']['exposures'] == 114
    assert result['summary']['updates'] == 8
    assert not list(ctx['cache'].rglob('selected_bundle.pt'))
    assert not list(ctx['cache'].rglob('*.npz'))


def test_three_then_six_restarts_exact_model_optimizer_sampler_and_rng(synthetic):
    ctx = synthetic
    m = impl()
    first = m.search(job('resumed'), ctx, [.25, 1., 3.])
    continued = m.search(job('resumed'), ctx, [.25, 1., 3., 4., 5., 6.], first['parent_checkpoint'])
    direct = m.search(job('uninterrupted'), ctx, [.25, 1., 3., 4., 5., 6.])
    resumed = torch.load(continued['checkpoint_path'], map_location='cpu', weights_only=False)
    uninterrupted = torch.load(direct['checkpoint_path'], map_location='cpu', weights_only=False)
    for key in ('model', 'optimizer', 'sampler', 'torch_rng', 'cuda_rng', 'updates'):
        equal_state(resumed[key], uninterrupted[key])
    assert resumed['sampler']['exposures'] == 228
    assert resumed['updates'] == 17
    assert resumed['extra']['completed_epochs'] == [0., .25, 1., 3., 4., 5., 6.]
    proof = json.loads(Path(continued['migration_proof_path']).read_text(encoding='utf-8'))
    assert proof['state_preserved']
    assert proof['parent_contract'] != proof['branch_contract']
    assert proof['parent_checkpoint_sha256'] == first['parent_checkpoint']['sha256']


def test_continuation_rejects_changed_parent_contract_and_missing_optimizer(synthetic):
    ctx = synthetic
    m = impl()
    first = m.search(job('guarded'), ctx, [.25, 1., 3.])
    parent = dict(first['parent_checkpoint'])
    parent['contract'] = dict(parent['contract'], data_sha='wrong')
    with pytest.raises(ValueError, match='parent contract'):
        m.search(job('guarded'), ctx, [.25, 1., 3., 4., 5., 6.], parent)
    checkpoint = torch.load(first['checkpoint_path'], map_location='cpu', weights_only=False)
    checkpoint.pop('optimizer')
    bad = ctx['cache'] / 'missing_optimizer.pt'
    torch.save(checkpoint, bad)
    import hashlib
    parent = dict(first['parent_checkpoint'], path=bad,
                  sha256=hashlib.sha256(bad.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='optimizer'):
        m.search(job('guarded2'), ctx, [.25, 1., 3., 4., 5., 6.], parent)


def test_refit_v_requires_byte_bound_group_decision_before_resources():
    with pytest.raises(ValueError, match='selection decision'):
        impl()._refit_predict(job(), {}, {}, 3)


def test_fresh_a60_refit_saves_fp64_verified_v_and_a_separate_bundle(synthetic):
    ctx = synthetic
    m = impl()
    item = job('fresh_on')
    search = m.search(item, ctx, [.25, 1., 3.])
    decision(ctx, item, 3)
    result = m._refit_predict(item, ctx, search, 3)
    assert result['status'] == 'COMPLETE'
    assert result['selection_sealed_before_V']
    assert result['fit_receipt']['normalization_fit_end'] == 96
    assert result['fit_receipt']['selected_epoch'] == search['selected_epoch']
    row = result['rows'][0]
    assert row['fit'] == 'fresh_A60'
    assert row['independent_FP64'] == 'PASS'
    assert row['evaluation'] == 'V_select'
    path = ctx['cache'] / row['prediction_binding']['reference'].removeprefix('cache://')
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['origins'], [112, 115, 118, 121, 124])
        errors = np.where(archive['observed'], archive['pred_normalized'].astype(np.float64)
                         - archive['target_normalized'].astype(np.float64), 0.)
        mse = (np.square(errors).sum((0, 1)) / archive['observed'].sum((0, 1))).mean()
    assert row['MSE'] == pytest.approx(mse, rel=1e-14, abs=1e-14)
    bundle = torch.load(ctx['cache'] / result['bundle_binding']['reference'].removeprefix('cache://'),
                        map_location='cpu', weights_only=False)
    assert bundle['normalization']['fit_end'] == 96
    assert bundle['spec']['revin'] is True
    assert bundle['selected_search_epoch'] == search['selected_epoch']


def test_unmatched_revin_and_recipe_changes_are_rejected_before_training(synthetic):
    m = impl()
    with pytest.raises(ValueError, match='recipe'):
        m._spec(dict(job(), microbatch=2), synthetic)
    with pytest.raises(ValueError, match='bool'):
        m._spec(dict(job(), revin='ON'), synthetic)


def test_v_guard_rejects_an_incomplete_group_even_with_a_valid_sha(synthetic):
    ctx = synthetic
    item = job('incomplete')
    decision(ctx, item, 3)
    path = ctx['selection_decision']['path']
    value = json.loads(path.read_text(encoding='utf-8'))
    value['matched_arms'] = [item['job_id']]
    path.write_text(json.dumps(value), encoding='utf-8')
    import hashlib
    ctx['selection_decision']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='four'):
        impl()._refit_predict(item, ctx, {}, 3)


def test_cached_six_epoch_search_cannot_hide_a_changed_migration_parent(synthetic):
    ctx = synthetic
    m = impl()
    item = job('cached_parent')
    first = m.search(item, ctx)
    m.search(item, ctx, [.25, 1., 3., 4., 5., 6.], first['parent_checkpoint'])
    parent = dict(first['parent_checkpoint'], contract=dict(first['parent_checkpoint']['contract'], data_sha='bad'))
    with pytest.raises(ValueError, match='parent'):
        m.search(item, ctx, [.25, 1., 3., 4., 5., 6.], parent)


def test_refit_rejects_changed_selection_and_reuses_exact_prediction_bytes(synthetic):
    ctx = synthetic
    m = impl()
    item = job('immutable')
    selected = m.search(item, ctx)
    decision(ctx, item, 3)
    first = m._refit_predict(item, ctx, selected, 3)
    archive = ctx['cache'] / first['rows'][0]['prediction_binding']['reference'].removeprefix('cache://')
    before = archive.read_bytes()
    assert m._refit_predict(item, ctx, selected, 3) == first
    assert archive.read_bytes() == before
    changed = dict(selected, selected_epoch=3. if selected['selected_epoch'] != 3. else 1.)
    with pytest.raises(ValueError, match='search'):
        m._refit_predict(item, ctx, changed, 3)


def test_v_interruption_recovers_the_original_fresh_a60_bundle(synthetic, monkeypatch):
    ctx = synthetic
    m = impl()
    item = job('interrupted')
    selected = m.search(item, ctx)
    decision(ctx, item, 3)
    original = m.previous._predict_saved

    def interrupted(*args, **kwargs):
        raise OSError('Synthetic V write interruption')

    monkeypatch.setattr(m.previous, '_predict_saved', interrupted)
    with pytest.raises(OSError, match='interruption'):
        m._refit_predict(item, ctx, selected, 3)
    artifact = ctx['cache'] / 'runs' / item['job_id'] / 'cap3' / 'selected_bundle.pt'
    before = artifact.read_bytes()
    monkeypatch.setattr(m.previous, '_predict_saved', original)
    result = m._refit_predict(item, ctx, selected, 3)
    assert result['status'] == 'COMPLETE'
    assert artifact.read_bytes() == before


def test_cached_branch_verifies_the_original_import_checkpoint_bytes(synthetic):
    ctx = synthetic
    m = impl()
    item = job('import_bytes')
    first = m.search(item, ctx)
    m.search(item, ctx, [.25, 1., 3., 4., 5., 6.], first['parent_checkpoint'])
    imported = ctx['cache'] / 'runs' / item['job_id'] / 'cap6' / 'search' / 'checkpoint_000000.pt'
    checkpoint = torch.load(imported, map_location='cpu', weights_only=False)
    next(iter(checkpoint['model'].values())).add_(1)
    torch.save(checkpoint, imported)
    with pytest.raises(ValueError, match='SHA'):
        m.search(item, ctx, [.25, 1., 3., 4., 5., 6.], first['parent_checkpoint'])
