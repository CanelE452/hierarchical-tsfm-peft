import importlib
from pathlib import Path
import time

import numpy as np
import pytest


def impl():
    return importlib.import_module('research.adapter_improvements_20261018_v1.decoder_jobs')


def test_family_selects_40_before_validation_and_gain_refits_its_own_base(tmp_path):
    m = impl()
    rng = np.random.default_rng(18)
    def arrays(n):
        phi = rng.normal(size=(n, 4, 3)).astype(np.float32)
        return dict(phi=phi, target=rng.normal(size=(n, 4, 2)).astype(np.float32),
                    mask=np.ones((n, 4, 2), bool), anchor=np.zeros((n, 1, 2), np.float32))
    fit, stop, refit, validation = (arrays(n) for n in (21, 7, 28, 7))
    events = []
    def seal(value):
        assert len(value['selection_trials']) == 40
        events.append('sealed')
    def load_v():
        assert events == ['sealed']
        events.append('loaded_V')
        return validation
    result = m.run_family(fit, stop, refit, load_v, seal, lambda: None)
    assert events == ['sealed', 'loaded_V']
    assert len(result['rows']) == 4
    decoder = importlib.import_module('research.adapter_improvements_20261018_v1.decoder')
    gain = result['states']['D_GAIN']
    shared = decoder.fit_head(refit['phi'], refit['target'].astype(np.float64)-refit['anchor'],
                              refit['mask'], 'D_SHARED', gain['lambda_value'])
    np.testing.assert_array_equal(gain['coefficients'], shared['coefficients'])
    expected_rms = np.sqrt(np.mean(refit['phi'].astype(np.float64)**2, axis=(0, 1)))
    np.testing.assert_array_equal(gain['rms'], expected_rms)
    assert result['new_F_calls'] == 0


def test_sufficient_statistics_reconstruct_metrics_and_bands_without_dropping_channels():
    m = impl()
    pred = np.arange(2*6*2).reshape(2, 6, 2)/13
    target = np.zeros_like(pred)
    mask = np.ones_like(pred, bool)
    mask[:, :2, 0] = False
    value = m.score_arrays(pred, target, mask)
    assert value['bands'][0]['status'] == 'UNDEFINED'
    assert value['bands'][0]['MSE'] is None
    sse = np.array(value['horizon_sse'])
    counts = np.array(value['horizon_counts'])
    assert value['MSE'] == pytest.approx(np.mean(sse.sum(0)/counts.sum(0)))
    np.testing.assert_allclose(np.sum([b['channel_sse'] for b in value['bands']], axis=0), sse.sum(0))
    mask[..., 1] = False
    with pytest.raises(ValueError, match='BLOCKED_DATA_CONTRACT'):
        m.score_arrays(pred, target, mask)


def test_cache_shape_dtype_hash_and_origin_binding_are_required(tmp_path):
    m = impl()
    from research.adapter_improvements_20261018_v1.contracts import atomic_json, sha
    values = dict(phi=np.zeros((3, 4, 2), np.float32), target=np.zeros((3, 4, 1), np.float32),
                  mask=np.ones((3, 4, 1), bool), anchor=np.zeros((3, 1, 1), np.float32))
    path = tmp_path/'features.npz'
    np.savez(path, **values)
    key = dict(origins_sha='actual-bound-origin-sha', channel_order=['a'])
    atomic_json(path.with_suffix('.json'), dict(key=key, sha256=sha(path)))
    assert m.load_cache(path, key, (3, 4, 2, 1))['mask'].dtype == bool
    with pytest.raises(ValueError, match='key binding'):
        m.load_cache(path, dict(key, origins_sha='other'), (3, 4, 2, 1))
    values['target'] = values['target'].astype(np.float64)
    np.savez(path, **values)
    atomic_json(path.with_suffix('.json'), dict(key=key, sha256=sha(path)))
    with pytest.raises(ValueError, match='dtype'):
        m.load_cache(path, key, (3, 4, 2, 1))
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='bytes'):
        m.load_cache(path, key, (3, 4, 2, 1))


def test_budget_reserves_twenty_percent_and_never_loads_validation_after_timeout():
    check = impl().budget_checker(dict(deadline=time.time()+1000, job_deadline=time.time()-1), time.time())
    with pytest.raises(TimeoutError):
        check()


def test_shared_family_clock_preserves_verification_reserve_and_second_seed_bound(monkeypatch):
    m = impl()
    family_started = 1000.
    ctx = dict(decoder_family_started_unix=family_started, deadline=5000.,
               job_deadline=family_started+1800.)
    monkeypatch.setattr(m.time, 'time', lambda: family_started+1441.)
    for seed_started in (family_started, family_started+1400.):
        with pytest.raises(TimeoutError):
            m.budget_checker(ctx, seed_started)()
        m.budget_checker(ctx, seed_started, verification=True)()
    monkeypatch.setattr(m.time, 'time', lambda: family_started+1800.)
    with pytest.raises(TimeoutError):
        m.budget_checker(ctx, family_started+1400., verification=True)()


def test_execute_job_synthetic_receipts_and_saved_predictions(tmp_path, monkeypatch):
    m = impl()
    d = importlib.import_module('research.adapter_improvements_20261018_v1.decoder')
    from types import SimpleNamespace
    from research.adapter_improvements_20261018_v1.contracts import read, resolve_binding
    rng = np.random.default_rng(99)
    values = {}
    inventory = {}
    for scope, split, n in (('A_fit', 'fit', 9), ('A_fit', 'A_stop', 4), ('A60', 'fit', 13), ('A60', 'V_select', 4)):
        arrays = dict(phi=rng.normal(size=(n, 48, 42)).astype(np.float32),
                      target=rng.normal(size=(n, 48, 21)).astype(np.float32),
                      anchor=rng.normal(size=(n, 1, 21)).astype(np.float32),
                      mask=np.ones((n, 48, 21), bool))
        path = tmp_path/(scope+'_'+split+'.npz')
        np.savez(path, **arrays)
        values[str(path)] = arrays
        inventory[scope, split] = dict(path=path, manifest=dict(seconds=0.),
                                        origins=np.arange(n, dtype=np.int64),
                                        key=dict(scaler_sha='synthetic-scaler', PE_checkpoint_sha256='synthetic-PE'))
    old_rows = []
    for stage, split, lambdas in (('A_fit', 'A_stop', (.01, .1, 1.)), ('A60', 'V_select', (.01,))):
        fitting, evaluation = values[str(inventory[stage, 'fit']['path'])], values[str(inventory[stage, split]['path'])]
        stats = d.prepare_statistics(fitting['phi'], fitting['target']-fitting['anchor'], fitting['mask'])
        for arm in ('D_SHARED', 'D_HORIZON'):
            for lam in lambdas:
                state = d.fit_statistics(stats, arm, lam)
                _, score = m._evaluate(evaluation, state)
                old_rows.append(dict(method='B2_DECODER_'+arm.removeprefix('D_'), stage=stage,
                                      lambda_value=lam, MSE=score['MSE'], MAE=score['MAE']))
    old = dict(rows=old_rows, original_selected_epoch=1.)
    modules = dict(legacy=SimpleNamespace(F_SHA='synthetic-F-state'), blas_controller=__import__('contextlib').nullcontext())
    monkeypatch.setattr(m, '_input_contract', lambda job, ctx: (old, {}, inventory, modules, dict(columns=[str(i) for i in range(21)])))
    monkeypatch.setattr(m, '_load_bound', lambda entry, modules, task, check: values[str(entry['path'])])
    ctx = dict(repo=tmp_path, cache=tmp_path/'new_cache', report=tmp_path/'report', deadline=time.time()+60)
    result = m.execute_job(dict(job_id='synthetic_decoder', seed=92601, scientific_identity='synthetic'), ctx)
    assert result['status'] == 'COMPLETE'
    public = read(result['artifact_paths'][0])
    assert len(public['rows']) == 4 and len(public['selection_trials']) == 40
    assert len(public['original_B03_reconstruction']) == 8
    for row in public['rows']:
        path = resolve_binding(row['prediction_binding'], ctx)
        with np.load(path) as archive:
            score = m.score_arrays(archive['pred_normalized'], archive['target_normalized'], archive['observed'])
        assert score['MSE'] == row['MSE']
        assert row['saved_array_independent_FP64'] == 'PASS'
    assert public['new_F_calls'] == 0
