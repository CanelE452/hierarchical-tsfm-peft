import importlib
from pathlib import Path
import time

import numpy as np
import pytest
import torch


def impl():
    try:
        return importlib.import_module('research.adapter_improvements_20261018_v1.lift_jobs')
    except ModuleNotFoundError:
        pytest.fail('Conditional paired LIFT execution is not implemented')


def sample_split(n=7, seed=4):
    m = impl()
    rng = np.random.default_rng(seed)
    context = rng.normal(size=(n, 32, 5)).astype(np.float32)
    anchor = context[:, -1:]
    base = anchor + rng.normal(size=(n, 8, 5)).astype(np.float32)
    target = base + .2
    mask = rng.random(target.shape) > .15
    return m.CachedBaseSplit(base, target, mask, anchor, np.arange(n, dtype=np.int64),
                             lambda indices: torch.from_numpy(context[indices]), 'synthetic')


def test_output_gain_matches_independent_augmented_svd_global_counts_and_fresh_scaling():
    m = impl()
    rng = np.random.default_rng(71)
    base = rng.normal(size=(6, 5, 3))
    anchor = rng.normal(size=(6, 1, 3))
    target = rng.normal(size=base.shape)
    mask = rng.random(base.shape) > .3
    mask[:, :2, 1] = False
    lam, gamma = .03, 10.
    state = m.fit_output_gain(base, anchor, target, mask, lam, gamma)
    g = m.horizon_basis(5)
    expected = np.empty((3, 2))
    nc = mask.sum((0, 1))
    delta = base-anchor
    rms = np.sqrt(np.mean(delta*delta, axis=(0, 1)))
    for c in range(3):
        z = delta[..., c]/rms[c]
        design = np.stack([z, z*g], -1)[mask[..., c]]
        y = (target-anchor)[..., c][mask[..., c]]
        aug = np.vstack([design, np.sqrt(lam*nc[c])*np.diag([1., np.sqrt(gamma)])])
        expected[c] = np.linalg.lstsq(aug, np.r_[y, 0., 0.], rcond=None)[0]
    np.testing.assert_allclose(state['theta'], expected, rtol=1e-12, atol=1e-12)
    prediction = m.predict_output_gain(base, anchor, state)
    data = np.mean(np.where(mask, (prediction-target)**2, 0).sum((0, 1))/nc)
    penalty = lam*np.mean(expected[:, 0]**2+gamma*expected[:, 1]**2)
    assert state['diagnostics']['objective'] == pytest.approx(data+penalty, abs=1e-12)
    refit = m.fit_output_gain(3*base, 3*anchor, target, mask, lam, gamma)
    np.testing.assert_allclose(refit['rms'], 3*rms)
    assert len(m.output_gain_grid()) == 15


def test_output_gain_identity_zero_features_and_empty_channel():
    m = impl()
    base = np.ones((4, 6, 2))
    anchor = np.zeros((4, 1, 2))
    state = dict(rms=[2., 3.], theta=[[2., 0.], [3., 0.]], H=6, C=2)
    np.testing.assert_array_equal(m.predict_output_gain(base, anchor, state), base)
    fit = m.fit_output_gain(base, base[:, :1], base, np.ones_like(base, bool), .1, 1.)
    assert fit['zero_rms_channels'] == [0, 1]
    assert np.isfinite(fit['diagnostics']['objective'])
    mask = np.ones_like(base, bool)
    mask[..., 1] = False
    with pytest.raises(ValueError, match='N_c=0'):
        m.fit_output_gain(base, anchor, base, mask, .1, 1.)


class TinyRefiner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(5))
    def forward(self, x, prediction):
        return prediction+self.bias


def test_masked_macro_loss_uses_global_counts_and_batch_weight():
    m = impl()
    split = sample_split()
    indices = np.array([0, 3, 6])
    _, base, target, mask = split.batch(indices, 'cpu')
    loss = m.masked_macro_loss(base, target, mask, split.counts, len(split), len(indices))
    sse = np.where(mask.numpy(), (base.numpy()-target.numpy())**2, 0).sum((0, 1))
    expected = np.mean(sse/split.counts)*len(split)/len(indices)
    assert float(loss) == pytest.approx(expected, rel=2e-7)


def test_refiner_checkpoint_resume_exact_and_cp0_earliest_selection(tmp_path):
    m = impl()
    fit, stop = sample_split(9), sample_split(4, 12)
    identity = {'scientific_identity': 'synthetic'}
    uninterrupted = m.train_refiner(TinyRefiner, fit, stop, tmp_path/'full.pt', identity,
                                   device='cpu', seed=92601, max_epochs=2, patience=5, batch_size=4, check=lambda: None)
    calls = 0
    def deadline():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError('synthetic deadline')
    partial = m.train_refiner(TinyRefiner, fit, stop, tmp_path/'resume.pt', identity,
                             device='cpu', seed=92601, max_epochs=2, patience=5, batch_size=4, check=deadline)
    assert partial['status'] == 'PARTIAL_CHECKPOINTED'
    resumed = m.train_refiner(TinyRefiner, fit, stop, tmp_path/'resume.pt', identity,
                             device='cpu', seed=92601, max_epochs=2, patience=5, batch_size=4, check=lambda: None)
    assert resumed['curves'] == uninterrupted['curves']
    for name, value in resumed['model_state'].items():
        torch.testing.assert_close(value, uninterrupted['model_state'][name], rtol=0, atol=0)
    assert resumed['updates'] == 6
    # All endpoints tied exactly: CP0 must remain selected.
    tied = sample_split(4)
    tied.target[:] = tied.base
    result = m.train_refiner(TinyRefiner, tied, tied, tmp_path/'ties.pt', identity,
                            device='cpu', seed=92601, max_epochs=3, patience=2, batch_size=4, check=lambda: None)
    assert result['selected_epoch'] == 0 and len(result['curves']) == 3


def test_profile_admission_binds_full_schedule_and_skips_both_at_reserve():
    m = impl()
    profile = dict(update_seconds=.25, evaluation_seconds=.4, gain_grid_seconds=2.,
                   prepare_seconds=1., prediction_seconds=.8, fit_origins=1000, refit_origins=1200)
    estimate = m.estimate_pair(profile)
    expected_updates = 100*(8+10)
    assert estimate['worst_train_updates'] == expected_updates
    assert estimate['max_search_epochs'] == estimate['max_refit_epochs'] == 100
    assert estimate['estimated_seconds'] >= expected_updates*.25+100*.4
    assert not m.admit_pair(estimate, dict(optional_seconds=100., remaining_seconds=1000., reserve_seconds=900.))['admitted']
    assert not m.admit_pair(dict(estimated_seconds=3601.), dict(optional_seconds=7200., remaining_seconds=8000., reserve_seconds=0.))['admitted']
    assert m.admit_pair(estimate, dict(optional_seconds=3600., remaining_seconds=8000., reserve_seconds=4320.))['admitted']


def test_cached_base_matmul_is_fp32_original_d_and_anchor():
    m = impl()
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(3, 8, 14)).astype(np.float32)
    anchor = rng.normal(size=(3, 1, 5)).astype(np.float32)
    d = torch.from_numpy(rng.normal(size=(7, 2, 5)).astype(np.float32))
    actual = m.cached_base_prediction(phi, anchor, d)
    expected = torch.from_numpy(anchor)+torch.from_numpy(phi)@d.reshape(14, 5)
    np.testing.assert_array_equal(actual, expected.numpy())
    assert actual.dtype == np.float32


def test_pair_seals_selection_before_validation_and_fresh_refits(tmp_path):
    m = impl()
    fit, stop, refit, validation = sample_split(7), sample_split(4, 8), sample_split(9, 10), sample_split(4, 22)
    events = []
    def seal(value):
        assert len(value['gain_trials']) == 15
        events.append('sealed')
    def load_v():
        assert events == ['sealed']
        events.append('loaded_V')
        return validation
    result = m.run_pair(fit, stop, refit, load_v, seal, TinyRefiner, tmp_path,
                        {'scientific_identity': 'synthetic'}, device='cpu', seed=92601,
                        max_epochs=1, patience=5, batch_size=4, check=lambda: None)
    assert result['status'] == 'COMPLETE'
    assert events == ['sealed', 'loaded_V']
    assert len(result['rows']) == 2
    assert result['new_F_calls'] == 0
    rms = np.sqrt(np.mean((refit.base.astype(np.float64)-refit.anchor)**2, axis=(0, 1)))
    np.testing.assert_allclose(result['gain_state']['rms'], rms)


def test_execute_pair_saves_rescorable_bindings_with_full_recipe_and_no_f(tmp_path, monkeypatch):
    m = impl()
    from contextlib import nullcontext
    from research.adapter_improvements_20261018_v1.contracts import binding, read, resolve_binding
    rng = np.random.default_rng(13)
    task = dict(L=512, H=48, physical=np.zeros((600, 21)), columns=[str(i) for i in range(21)])
    def split(n, name):
        x = rng.normal(size=(n, 512, 21)).astype(np.float32)
        anchor = x[:, -1:]
        base = anchor+rng.normal(size=(n, 48, 21)).astype(np.float32)
        return m.CachedBaseSplit(base, base.copy(), np.ones_like(base, bool), anchor,
                                 np.arange(n, dtype=np.int64), lambda ix: torch.from_numpy(x[ix]), name)
    fit, stop, refit, validation = [split(n, name) for n, name in ((3, 'fit'), (2, 'stop'), (4, 'refit'), (2, 'V'))]
    ctx = dict(repo=Path(__file__).resolve().parents[1], cache=tmp_path/'cache', report=tmp_path/'report',
               deadline=time.time()+8000, lift_dependencies_complete=True,
               lift_budget=dict(optional_seconds=3600, remaining_seconds=8000, reserve_seconds=4320))
    checkpoint = tmp_path/'cache'/'base.pt'
    checkpoint.parent.mkdir(parents=True)
    torch.save({'D': torch.ones(7, 6, 21)}, checkpoint)
    evidence = dict(checkpoints={'A_fit': binding(checkpoint, ctx), 'A60': binding(checkpoint, ctx)})
    loaded_v = []
    def load_v():
        assert (ctx['cache']/'lift'/'synthetic_pair'/'SELECTION_SEAL.json').exists()
        loaded_v.append(True)
        return validation
    monkeypatch.setattr(m, '_prepare_inputs', lambda job, context, check: (fit, stop, refit, load_v, evidence, task, {'blas_controller': nullcontext()}))
    class CheapRefiner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(21))
        def forward(self, x, base):
            return base+self.bias
    monkeypatch.setattr(m.lift, 'make_official_refiner', lambda *args, **kwargs: CheapRefiner())
    result = m.execute_job(dict(job_id='synthetic_pair', seed=92601, cohort='jena', scientific_identity='synthetic'), ctx)
    assert result['status'] == 'COMPLETE'
    public = read(result['artifact_paths'][0])
    assert public['recipe']['max_epochs'] == 100
    assert public['new_F_calls'] == 0 and len(public['rows']) == 2
    for row in public['rows']:
        path = resolve_binding(row['prediction_binding'], ctx)
        with np.load(path, allow_pickle=False) as arrays:
            metrics = m.decoder_jobs.score_arrays(arrays['pred_normalized'], arrays['target_normalized'], arrays['observed'])
            np.testing.assert_array_equal(arrays['base_pred_normalized'], validation.base)
        assert metrics['MSE'] == row['MSE']
        assert row['saved_array_independent_FP64'] == 'PASS'
        route = m.load_calibrator(row['method'], row['refiner_bundle'], ctx)
        context, base, _, _ = validation.batch(np.arange(len(validation)), 'cpu')
        with torch.no_grad():
            replay = route(context, base).numpy()
        with np.load(path, allow_pickle=False) as arrays:
            np.testing.assert_allclose(replay, arrays['pred_normalized'], rtol=1e-13, atol=1e-13)
    assert all('reference' in item and '://' in item['reference'] for item in public['private_artifacts'])
    assert 'gain_state' not in public
    skip_ctx = dict(ctx, lift_budget=dict(optional_seconds=0., remaining_seconds=8000., reserve_seconds=4320.))
    monkeypatch.setattr(m, 'train_refiner', lambda *args, **kwargs: pytest.fail('Skipped pair started actual training'))
    skipped = m.execute_job(dict(job_id='synthetic_skip', seed=92601, cohort='jena', scientific_identity='synthetic'), skip_ctx)
    assert skipped['status'] == 'SKIPPED_BUDGET' and skipped['rows'] == []
    assert loaded_v == [True]


def test_official_complex_adam_training_and_profile_use_external_predictions_only(tmp_path):
    m = impl()
    source = Path(__file__).resolve().parents[1]/'.cache/adapter_improvements_20261018_v1/official'
    if not (source/'LIFT/models/LIFT.py').exists():
        pytest.skip('Pinned official private source is required for this replay check')
    fit, stop, refit = sample_split(5), sample_split(3, 12), sample_split(7, 14)
    def factory():
        return m.lift.make_official_refiner(source, length=32, horizon=8, channels=5, seed=92601)
    initial = factory().state_dict()
    trained = m.train_refiner(factory, fit, stop, tmp_path/'official.pt', {'kind': 'synthetic'},
                              device='cpu', seed=92601, max_epochs=1, patience=5, batch_size=4)
    assert trained['status'] == 'COMPLETE'
    complex_names = [key for key, value in initial.items() if value.is_complex()]
    assert complex_names and all(torch.isfinite(trained['model_state'][key]).all() for key in complex_names)
    assert any(not torch.equal(trained['model_state'][key], initial[key]) for key in complex_names)
    profile = m._profile(factory, fit, stop, refit, 'cpu', lambda: None)
    assert profile['timed_updates'] == 3 and profile['warmup_updates'] == 1
    assert profile['logical_new_F_calls'] == 0
    assert profile['parameter_inventory']['complex_elements'] > 0
