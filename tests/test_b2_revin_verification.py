import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


def impl():
    try:
        return importlib.import_module('research.b2_revin_20261018_v1.verification')
    except ModuleNotFoundError:
        pytest.fail('Independent saved-prediction verifier is missing')


def test_independent_metric_is_equal_channel_macro_and_ignores_masked_canary():
    prediction = np.array([[[2., 1.], [np.nan, 1.]]], dtype=np.float32)
    target = np.array([[[0., 0.], [999., 0.]]], dtype=np.float32)
    observed = np.array([[[True, True], [False, True]]])
    values = dict(pred_normalized=prediction, target_normalized=target, observed=observed)
    scored = impl()._independent_metrics(values)
    assert scored['MSE'] == 2.5
    assert scored['MAE'] == 1.5
    assert scored['n_targets'] == 3
    assert scored['channel_counts'] == [1, 2]
    assert scored['horizon_counts'] == [[1, 1], [0, 1]]
    assert scored['MSE'] != 2.


def test_n0_channel_is_undefined_and_nonfinite_observed_target_is_rejected():
    values = dict(pred_normalized=np.ones((2, 2, 2), np.float32),
                  target_normalized=np.zeros((2, 2, 2), np.float32),
                  observed=np.ones((2, 2, 2), bool))
    values['observed'][:, :, 1] = False
    with pytest.raises(ValueError, match='observed target'):
        impl()._independent_metrics(values)
    values['observed'][:] = True
    values['target_normalized'][0, 0, 1] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        impl()._independent_metrics(values)


def test_receipt_verification_preserves_unrounded_bit_values():
    m = impl()
    metrics = dict(MSE=0.123456789012345, MAE=0.234567890123456, n_targets=10,
                   observed=10, channel_counts=[5, 5])
    assert m._check_receipt(metrics, dict(metrics)) is None
    with pytest.raises(ValueError, match='MSE'):
        m._check_receipt(metrics, dict(metrics, MSE=.123457))
    with pytest.raises(ValueError, match='observed'):
        m._check_receipt(metrics, dict(metrics, observed=9))


def pair(revin, cap=3, seed=92601, dataset='jena'):
    values = dict(pred_normalized=np.zeros((2, 2, 2), np.float32),
                  target_normalized=np.ones((2, 2, 2), np.float32),
                  observed=np.ones((2, 2, 2), bool), origins=np.array([10, 12]),
                  channel_ids=np.array(['a', 'b']), mean=np.array([0., 0.], np.float32),
                  std=np.array([1., 1.], np.float32))
    row = dict(run_id='B2_' + ('ON' if revin else 'OFF') + f'_cap{cap}',
               job_id='B2_' + ('ON' if revin else 'OFF'), dataset=dataset, seed=seed,
               family='B2', revin=revin, cap=cap, C=2, K=1, L=4, H=2,
               core_dtype='float32', point_statistic='quantile_0.5',
               units='global_A60_normalized', data_sha='d', model_sha='f',
               scaler_sha256='s', training_recipe={'epochs':[.25, 1., 3.]},
               initial_core_sha256='identical_initial_P_E_D')
    row.update(impl()._independent_metrics(values))
    return row, values


@pytest.mark.parametrize('changed', [dict(seed=92602), dict(dataset='pems32'), dict(cap=6),
                                  dict(initial_core_sha256='other_version'), dict(model_sha='other_F')])
def test_matched_comparison_rejects_cohort_seed_cap_or_backbone_version(changed):
    candidate, left = pair(True)
    control, right = pair(False)
    control.update(changed)
    with pytest.raises(ValueError, match='Matched comparison'):
        impl()._compare(candidate, control, left, right, 'REVIN')


def test_pair_origin_statistics_reconstruct_macro_mse_and_mae_delta():
    candidate, left = pair(True)
    control, right = pair(False)
    left['pred_normalized'][:] = 0.5
    candidate.update(impl()._independent_metrics(left))
    result, origins = impl()._compare(candidate, control, left, right, 'REVIN')
    assert result['MSE_delta'] == -.75
    assert result['MAE_delta'] == -.5
    assert result['MSE_percent_delta'] == -75.
    assert result['MAE_percent_delta'] == -50.
    assert result['match_status'] == 'MATCHED'
    assert result['candidate_run_id'] == candidate['run_id']
    assert len(origins) == 4
    assert sum(row['SSE_diff'] for row in origins) == -6.
    assert sum(row['SAE_diff'] for row in origins) == -4.


def test_comparison_rejects_changed_saved_target_or_mask():
    candidate, left = pair(True)
    control, right = pair(False)
    right['target_normalized'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='target_normalized'):
        impl()._compare(candidate, control, left, right, 'REVIN')


def selections():
    searches = {}
    for enabled in (False, True):
        for seed in (92601, 92602):
            name = f'B2_jena_REVIN_{"ON" if enabled else "OFF"}_s{seed}'
            values = [2., 1.5, 1., .9] if enabled else [2., 1.4, .8, 1.]
            searches[(name, 3)] = dict(status='COMPLETE', cap=3,
                spec=dict(dataset='jena', family='B2', seed=seed, revin=enabled),
                curves=[dict(epoch=epoch, astop_mse=mse) for epoch, mse in zip([0., .25, 1., 3.], values)],
                selected_epoch=3. if enabled else 1.)
    decision = dict(status='SEALED', cohort='jena', cap=3, selection_split='A_stop', V_used=False,
        matched_arms=[name for name, _ in searches], extend_all_to6=True, still_improving_at_cap=True,
        common_revin=False, astop_two_seed_means={'OFF':.8, 'ON':.9},
        selections={name:dict(epoch=result['selected_epoch'],
        astop_mse=min(row['astop_mse'] for row in result['curves'])) for (name, cap), result in searches.items()})
    return decision, searches


def test_decision_is_recomputed_only_from_all_four_astop_curves():
    decision, searches = selections()
    assert impl()._check_decision(decision, searches)['common_revin'] is False
    decision['common_revin'] = True
    with pytest.raises(ValueError, match='normalization'):
        impl()._check_decision(decision, searches)


@pytest.mark.parametrize('changed', [dict(V_used=True), dict(extend_all_to6=False),
                                  dict(matched_arms=['one_only'])])
def test_decision_rejects_v_use_and_asymmetric_extension(changed):
    decision, searches = selections()
    decision.update(changed)
    with pytest.raises(ValueError, match='decision'):
        impl()._check_decision(decision, searches)


@pytest.fixture
def canonical(tmp_path):
    root = Path(__file__).resolve().parents[1]
    base = root / 'research/informs_broad_validation_20261015_v1'
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    modules = {key: importlib.import_module(name) for key, name in
        [('fit', 'fit_base'), ('data', 'data_contract'), ('training', 'training')]}
    values = np.random.default_rng(12).normal(size=(14, 2)).astype(np.float32)
    observed = np.ones_like(values, bool)
    observed[12, 1] = False
    task = dict(physical=values, observed=observed, columns=['a', 'b'], L=4, H=2,
        data_sha256=modules['data'].array_sha256(values),
        split_ranges=dict(A_fit=(0, 8), A_stop=(8, 10), M_cal=(10, 12), V_select=(12, 14), T_final=(14, 16)))
    mean, std, _ = modules['data'].fit_statistics(task, 10)
    _, target, mask = modules['training'].FastWindows(task, mean, std, 'cpu').get([12])
    path = tmp_path / 'cache' / 'prediction.npz'
    path.parent.mkdir(parents=True)
    archive = dict(pred_normalized=target.numpy()+.25, target_normalized=target.numpy(),
        observed=mask.numpy(), origins=np.array([12], np.int64), channel_ids=np.array(['a', 'b']))
    np.savez(path, **archive)
    ctx = dict(repo=root, report=tmp_path/'report', cache=tmp_path/'cache', asset=tmp_path, modules=modules)
    row = dict(prediction_binding=impl().contracts.binding(path, ctx))
    return ctx, task, row, archive, path


def test_missing_saved_scaler_is_explicitly_reconstructed_from_fixed_a60(canonical):
    ctx, task, row, _, path = canonical
    before = path.read_bytes()
    values, proof = impl()._canonical_values(row, task, ctx)
    assert proof['scaler_provenance'] == 'FIXED_A60_RECONSTRUCTED_NOT_SAVED'
    assert proof['reconstructed_scaler_arrays'] == ['mean', 'std']
    assert 'mean' not in proof['saved_array_names']
    assert values['mean'].dtype == np.float32
    assert path.read_bytes() == before


@pytest.mark.parametrize('field', ['target_normalized', 'observed', 'origins', 'channel_ids'])
def test_actual_fixed_data_rejects_saved_target_mask_origin_or_channel_canary(canonical, field):
    ctx, task, row, archive, path = canonical
    if field == 'target_normalized':
        archive[field][0, 0, 0] += 1
    elif field == 'observed':
        archive[field][0, 0, 0] = False
    elif field == 'origins':
        archive[field][0] = 11
    else:
        archive[field] = archive[field][::-1]
    np.savez(path, **archive)
    row['prediction_binding'] = impl().contracts.binding(path, ctx)
    with pytest.raises(ValueError):
        impl()._canonical_values(row, task, ctx)


def test_prediction_source_hash_binds_deployment_sources_and_training_parent(canonical):
    ctx, _, _, _, _ = canonical
    m = impl()
    files = {'fit_base.py':m.contracts.sha(ctx['modules']['fit'].__file__)}
    digest = m.contracts.fingerprint(dict(deployment_sources=files, training_source_sha='NO_TRAINING_FIXED_BASELINE'))
    m._check_source(files, digest, ctx, training_source_sha='NO_TRAINING_FIXED_BASELINE')
    with pytest.raises(ValueError, match='aggregate'):
        m._check_source(files, digest, ctx, training_source_sha='different_training_parent')


def test_ridge_keeps_fp64_compute_distinct_from_saved_fp32_predictions(canonical):
    ctx, task, base, _, path = canonical
    ctx['repo'] = path.parent.parent
    m = impl()
    artifact = ctx['cache'] / 'ridge_parameters.npz'
    np.savez(artifact, coefficients=np.ones((2, 7, 2), np.float64))
    name = 'BASELINE_jena_Ridge'
    row = dict(base, run_id=name, dataset='jena', family='RAW_RIDGE', seed=92601,
               revin=False, cap=0, C=2, K=0, L=4, H=2, source_kind='HISTORICAL_VERIFIED',
               status='HISTORICAL_VERIFIED', training_status='HISTORICAL_SCOPE',
               bundle_binding=m.contracts.binding(artifact, ctx))
    rows, _, _ = m._load_rows(ctx, {}, {(name, 0):row}, {}, {'jena':3}, {'jena':task}, {})
    assert rows[0]['core_dtype'] == 'float64'
    assert rows[0]['prediction_dtype'] == 'float32'
    assert rows[0]['training_status'] == 'HISTORICAL_SCOPE'


@pytest.mark.parametrize('cap', [3, 6])
def test_last_strictly_improving_checkpoint_is_explicit_training_cap(canonical, cap):
    ctx, task, base, _, path = canonical
    ctx['repo'] = path.parent.parent
    m = impl()
    name = 'B0_jena_REVIN_OFF_s92601'
    epochs = [.25, 1., 3.] if cap == 3 else [.25, 1., 3., 4., 5., 6.]
    curves = [dict(epoch=epoch, astop_mse=2.-index*.1) for index, epoch in enumerate([0.]+epochs)]
    summary = dict(seconds=1., updates=4, exposures=cap*3, trainable_parameters=16)
    spec = dict(m.contracts.RECIPE, family='B0', run_id=name, dataset='jena', seed=92601,
                revin=False, latent_k=1, epochs=epochs, device='cpu', core_dtype='float32')
    mean, std, _ = ctx['modules']['data'].fit_statistics(task, 10)
    state = dict(E=torch.zeros(2, 1), D=torch.zeros(7, 1, 2), center=torch.zeros(2))
    contract = dict(source_sha='source', config_sha='configuration', data_sha=task['data_sha256'],
                    model_sha='frozen_F', scaler_sha=ctx['modules']['fit'].scaler_fingerprint(mean, std))
    bundle = dict(spec=spec, model=state, normalization=dict(mean=mean, std=std, fit_end=10),
                  contract=contract, selected_search_epoch=float(cap), search_summary=summary,
                  refit_summary=summary, model_state_sha=m.contracts.fingerprint({key:
                  ctx['modules']['data'].array_sha256(value.numpy()) for key, value in state.items()}))
    folder = ctx['cache'] / 'runs' / name / f'cap{cap}'
    (folder/'search').mkdir(parents=True)
    torch.save(dict(model=state), folder/'search'/'checkpoint_000000.pt')
    torch.save(bundle, folder/'selected_bundle.pt')
    search = dict(summary=summary, selected_epoch=float(cap), curves=curves)
    row = dict(base, run_id=name, dataset='jena', family='B0', seed=92601, revin=False,
               cap=cap, C=2, K=1, L=4, H=2, source_kind='NEW_TRAINED', status='COMPLETE',
               bundle_binding=m.contracts.binding(folder/'selected_bundle.pt', ctx))
    rows, _, _ = m._load_rows(ctx, {}, {(name, cap):row}, {(name, cap):search},
                            {'jena':cap}, {'jena':task}, {})
    assert rows[0]['training_status'] == 'TRAINING_CAP_STILL_IMPROVING'
    assert rows[0]['cap'] == cap
    assert rows[0]['selected_epoch'] == cap
