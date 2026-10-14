"""Synthetic CPU checks only; no existing scientific NPZ is opened."""
import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('saved_diagnostics', HERE / 'diagnose_saved_outputs.py')
diag = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diag
SPEC.loader.exec_module(diag)


def bundle(prediction=None, scale='normalized'):
    shape = (4, 5, 3)
    target = np.zeros(shape, np.float64)
    target[:, :, 1] = 1.
    observed = np.ones(shape, bool)
    observed[0, :3, 1] = False
    observed[:, :, 2] = False
    target[~observed] = np.nan
    if prediction is None:
        prediction = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 20
    mean, std = np.array([10., -2., 100.]), np.array([2., 4., 1.])
    splits = {split: {'prediction': prediction.copy(), 'target': target.copy(),
                      'observed': observed.copy(), 'origins': np.array([0, 8, 16, 24], np.int64) + offset}
              for split, offset in [('fit', 0), ('tune', 40), ('val', 80)]}
    if scale == 'physical':
        for values in splits.values():
            for field in ('prediction', 'target'):
                values[field] = values[field] * std + mean
    return diag.Bundle(mean, std, splits, scale, {'path': 'synthetic', 'sha256': 'synthetic'})


def save_bundle(path, source):
    values = {split + '_' + key: value for split, arrays in source.splits.items() for key, value in arrays.items()}
    with path.open('xb') as handle:
        np.savez_compressed(handle, mean=source.mean, std=source.std, **values)


def test_partial_observation_uses_channel_macro_not_global_micro():
    error = np.array([[[1., 10., np.nan], [1., 10., np.nan], [1., 10., np.nan]]])
    mask = np.array([[[True, True, False], [True, False, False], [True, False, False]]])
    result = diag.score(error, mask)
    assert result['normalized_mse'] == 50.5
    assert result['normalized_mae'] == 5.5
    assert result['active_channels'] == 2
    assert result['observed_count'] == 4
    assert result['normalized_mse'] != (3 + 100) / 4


def test_exact_condition_rejects_target_stats_mask_and_origin_changes():
    raw = bundle()
    for field in ('mean', 'std', 'target', 'observed', 'origins'):
        changed = deepcopy(raw)
        if field in ('mean', 'std'):
            getattr(changed, field)[0] += 1e-12
        elif field == 'observed':
            changed.splits['tune'][field][0, 0, 0] = False
        else:
            changed.splits['tune'][field].flat[0] += 1 if field == 'origins' else 1e-12
        with pytest.raises(ValueError, match='EXACT_CONDITION_MISMATCH'):
            diag.validate_pair(raw, changed)
    changed = deepcopy(raw)
    changed.std = changed.std.astype(np.float32)
    with pytest.raises(ValueError, match='EXACT_CONDITION_MISMATCH'):
        diag.validate_pair(raw, changed)


def test_d0_tail_and_partitions_reconstruct_macro_gap_with_missing_channel():
    raw = bundle()
    candidate = deepcopy(raw)
    for values in candidate.splits.values():
        values['prediction'][:, :, 0] += np.array([[-1., 2., -3., 4., -5.]])
        values['prediction'][:, :, 1] *= .25
    result = diag.diagnose_pair(raw, candidate, 'candidate', 1, tail_quantiles=(.5, .9), time_bins=3)
    assert max(row['absolute_difference'] for row in result['reconstruction']) < 1e-12
    assert all(row['normalized_mse_contribution'] == 0. for row in result['tails'] if row['channel_index'] == 2)
    row = next(row for row in result['channels'] if row['split'] == 'fit' and row['channel_index'] == 0)
    assert row['physical_mse'] == row['normalized_mse'] * 4
    assert row['physical_mae'] == row['normalized_mae'] * 2
    assert row['normalized_signed_bias'] != row['normalized_mae']


def test_tail_raw_cutoff_is_paired_and_own_tail_is_not():
    raw_error = np.array([[[0.], [1.], [2.], [3.]]])
    error = np.array([[[30.], [20.], [1.], [0.]]])
    mask = np.ones_like(error, bool)
    rows = diag.tail_rows(raw_error, error, mask, raw_error, mask, 'x', 1, 'val', ['c'], (.5,))
    raw_tail = next(row for row in rows if row['cutoff_reference'] == 'same_split_raw' and row['region'] == 'tail')
    own_tail = next(row for row in rows if row['cutoff_reference'] == 'same_split_own' and row['region'] == 'tail')
    assert raw_tail['normalized_mse_contribution'] == .25
    assert raw_tail['paired_same_events'] is True
    assert own_tail['normalized_mse_contribution'] == 325.
    assert own_tail['paired_same_events'] is False
    assert own_tail['cutoff_normalized_abs_error'] != own_tail['raw_cutoff_normalized_abs_error']


def test_tail_ties_are_all_included_and_fit_unobserved_channel_stays_body():
    error = np.ones((1, 4, 2))
    mask = np.ones_like(error, bool)
    fit_mask = mask.copy()
    fit_mask[:, :, 1] = False
    rows = diag.tail_rows(error, error, mask, error, fit_mask, 'x', 1, 'val', ['a', 'b'], (.9,))
    ties = next(row for row in rows if row['cutoff_reference'] == 'same_split_raw' and row['region'] == 'tail' and row['channel_index'] == 0)
    absent = next(row for row in rows if row['cutoff_reference'] == 'fit_raw' and row['region'] == 'body' and row['channel_index'] == 1)
    assert ties['selected_observed_count'] == 4
    assert absent['cutoff_normalized_abs_error'] is None
    assert absent['selected_observed_count'] == 4


def test_duplicate_or_nonmonotonic_origins_rejected():
    for origins in (np.array([0, 0, 16, 24]), np.array([0, 16, 8, 24])):
        payload = deepcopy(bundle().splits['fit'])
        payload['origins'] = origins
        with pytest.raises(ValueError, match='unique and strictly increasing'):
            diag.validate_split(payload, 3)


def test_zero_variance_channel_uses_saved_positive_effective_scale_without_refit():
    raw = bundle(np.zeros((4, 5, 3)))
    raw.std[:] = 1.
    diag.validate_statistics(raw.mean, raw.std)
    rows = diag.diagnose_pair(raw, raw, 'RAW', 'shared_reference')['channels']
    constant = next(row for row in rows if row['split'] == 'fit' and row['channel_index'] == 0)
    assert constant['target_normalized_std'] == 0.
    assert constant['normalized_mse'] == 0.
    assert constant['stored_fit_std'] == 1.
    invalid = raw.std.copy()
    invalid[0] = 0.
    with pytest.raises(ValueError, match='positive'):
        diag.validate_statistics(raw.mean, invalid)


def test_split_overlap_is_reported_without_independent_generalization_claim():
    raw = bundle()
    raw.splits['tune']['origins'] = np.array([1, 9, 17, 25])
    raw.splits['val']['origins'] = np.array([80, 81, 82, 83])
    roles = {'fit': 'training', 'tune': 'model_selection', 'val': 'reused_development'}
    info = diag.split_overlap_info(raw, roles)
    assert info['target_interval_overlap'][0]['overlapping_target_window_pairs'] == 4
    assert next(row for row in info['within_split_target_interval_overlap'] if row['split'] == 'val')['overlapping_target_window_pairs'] == 6
    assert info['independent_generalization_established'] is False
    with pytest.raises(ValueError, match='Explicit split roles'):
        diag.split_overlap_info(raw, {**roles, 'val': 'independent_evaluation'})


def test_physical_and_normalized_storage_give_same_scores():
    normalized, physical = bundle(), bundle(scale='physical')
    expected = diag.diagnose_pair(normalized, normalized, 'RAW', 'shared_reference')
    actual = diag.diagnose_pair(physical, physical, 'RAW', 'shared_reference')
    for left, right in zip(expected['summary'], actual['summary']):
        assert right['normalized_mse'] == pytest.approx(left['normalized_mse'], abs=1e-13)
        assert right['normalized_mae'] == pytest.approx(left['normalized_mae'], abs=1e-13)


def test_seed_metric_average_is_not_prediction_ensemble():
    rows = [{'split': 'val', 'method': 'x', 'seed': 1, 'normalized_mse': 1.},
            {'split': 'val', 'method': 'x', 'seed': 2, 'normalized_mse': 1.}]
    means = diag.seed_means(rows, ['split', 'method'])
    assert means[0]['normalized_mse'] == 1.
    assert means[0]['seed_count'] == 2
    assert not any('ci' in key for key in means[0])
    assert np.mean([1., -1.]) ** 2 == 0.
    with pytest.raises(ValueError, match='Duplicate seed'):
        diag.seed_means([rows[0], rows[0]], ['split', 'method'])


def test_rank_projection_modes_have_distinct_mean_handling():
    values = np.array([[[-2., 10.], [-1., 10.], [1., 10.], [2., 10.]]])
    origin = diag.fit_projection(values, 1, 'through_origin')
    centered = diag.fit_projection(values, 1, 'mean_centered')
    np.testing.assert_allclose(origin.center, [0., 0.])
    np.testing.assert_allclose(centered.center, [0., 10.])
    assert np.mean((origin.apply(values) - values) ** 2) > 0
    np.testing.assert_allclose(centered.apply(values), values, atol=1e-14)
    assert centered.retained_fit_energy_fraction == 1.
    assert origin.fit_energy != centered.fit_energy


def test_rank_basis_is_fit_only_and_actual_target_score_is_separate():
    raw = bundle()
    last = {split: np.zeros((4, 3)) for split in raw.splits}
    rows, arrays = diag.rank_projection_diagnostics(raw, last, 1)
    changed = deepcopy(raw)
    changed.splits['tune']['target'][changed.splits['tune']['observed']] += 1000
    changed.splits['tune']['prediction'] *= -3
    changed_rows, changed_arrays = diag.rank_projection_diagnostics(changed, last, 1)
    for mode in ('through_origin', 'mean_centered'):
        np.testing.assert_array_equal(arrays[mode + '_basis'], changed_arrays[mode + '_basis'])
        np.testing.assert_array_equal(arrays[mode + '_center'], changed_arrays[mode + '_center'])
        row = next(item for item in rows if item['split'] == 'tune' and item['projection_mode'] == mode)
        changed_row = next(item for item in changed_rows if item['split'] == 'tune' and item['projection_mode'] == mode)
        assert changed_row['normalized_mse'] > row['normalized_mse']
        assert row['raw_forecast_approximation_normalized_mse'] != row['normalized_mse']
        assert row['basis_fitted_split'] == 'fit'
        assert row['nonlinear_decoder_lower_bound'] is False
        assert row['requires_full_c_raw_forecast'] is True


def test_rank_zero_energy_and_full_rank_identity():
    zero = np.zeros((2, 4, 3))
    for mode in ('through_origin', 'mean_centered'):
        fitted = diag.fit_projection(zero, 2, mode)
        assert fitted.effective_fit_rank == 0
        assert fitted.retained_fit_energy_fraction is None
        np.testing.assert_array_equal(fitted.apply(zero), zero)
    values = np.arange(24.).reshape(2, 4, 3)
    full = diag.fit_projection(values, 3, 'mean_centered')
    np.testing.assert_allclose(full.apply(values), values, atol=2e-14)
    with pytest.raises(ValueError, match='Invalid rank'):
        diag.fit_projection(values, 0, 'through_origin')


def test_source_sha_and_size_guard_and_protected_test(tmp_path):
    path = tmp_path / 'raw.npz'
    save_bundle(path, bundle())
    loaded = diag.load_bundle(path, ['fit'], 'normalized', diag.sha256(path))
    assert loaded.source['sha256'] == diag.sha256(path)
    with pytest.raises(ValueError, match='SOURCE_SHA_MISMATCH'):
        diag.load_bundle(path, ['fit'], 'normalized', '0' * 64)
    with pytest.raises(ValueError, match='uncompressed size'):
        diag.load_bundle(path, ['fit'], 'normalized', max_bytes=1)
    with pytest.raises(PermissionError, match='Protected TEST'):
        diag.checked_path(tmp_path / 'test.npz')
    with pytest.raises(PermissionError, match='EVAL read'):
        diag.load_bundle(tmp_path / 'missing.npz', ['eval'], 'normalized')


def test_eval_roles_checked_before_opening_any_target(tmp_path):
    config = {'raw': {'path': str(tmp_path / 'missing.npz')}, 'splits': ['fit', 'eval'],
              'split_roles': {'fit': 'training', 'eval': 'independent_evaluation'}, 'storage_scale': 'normalized'}
    with pytest.raises(ValueError, match='supported split roles'):
        diag.execute(config, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_unsupported_test_split_is_rejected_before_source_io(tmp_path, monkeypatch):
    def must_not_read(*args, **kwargs):
        raise AssertionError('Source was read before split gate')
    monkeypatch.setattr(diag, 'sha256', must_not_read)
    monkeypatch.setattr(diag.zipfile, 'ZipFile', must_not_read)
    monkeypatch.setattr(diag.np, 'load', must_not_read)
    with pytest.raises(ValueError, match='Unsupported or protected split request'):
        diag.load_bundle(tmp_path / 'missing.npz', ['test'], 'normalized')


def test_synthetic_end_to_end_exclusive_outputs_last_alignment_and_seed_means(tmp_path):
    raw = bundle()
    raw_path = tmp_path / 'raw.npz'
    save_bundle(raw_path, raw)
    candidate_paths = []
    for seed in (1, 2):
        candidate = deepcopy(raw)
        for arrays in candidate.splits.values():
            arrays['prediction'] += seed * .1
        path = tmp_path / ('candidate' + str(seed) + '.npz')
        save_bundle(path, candidate)
        candidate_paths.append(path)
    last_path = tmp_path / 'last.npz'
    payload = {'mean': raw.mean, 'std': raw.std}
    for split in raw.splits:
        payload[split + '_last'] = np.zeros((4, 3))
        payload[split + '_origins'] = raw.splits[split]['origins']
    with last_path.open('xb') as handle:
        np.savez_compressed(handle, **payload)
    config = {'dataset': 'SYNTHETIC_ONLY', 'raw': {'path': str(raw_path), 'sha256': diag.sha256(raw_path)},
              'storage_scale': 'normalized', 'splits': ['fit', 'tune', 'val'],
              'split_roles': {'fit': 'training', 'tune': 'model_selection', 'val': 'reused_development'},
              'candidates': [{'name': 'x', 'seed': seed, 'path': str(path)} for seed, path in zip((1, 2), candidate_paths)],
              'k': 1, 'last': {'path': str(last_path)}, 'time_bins': 3, 'tail_quantiles': [.9]}
    output = tmp_path / 'out'
    result = diag.execute(config, output)
    assert result['status'] == 'COMPLETE_SAVED_OUTPUT_DIAGNOSTICS'
    assert result['experiment_complete'] is False
    assert result['rank_projection_rows'] == 6
    assert len(result['sources']) == 3
    assert result['new_inference'] is False
    assert (output / 'summary_seed_metric_means.csv').exists()
    assert (output / 'rank_projection.npz').exists()
    before = diag.sha256(output / 'RESULT.json')
    with pytest.raises(FileExistsError, match='fresh directory'):
        diag.execute(config, output)
    assert diag.sha256(output / 'RESULT.json') == before
    loaded = diag.load_bundle(raw_path, config['splits'], 'normalized')
    bad = dict(payload)
    bad['tune_origins'] = bad['tune_origins'] + 1
    bad_path = tmp_path / 'last_bad.npz'
    with bad_path.open('xb') as handle:
        np.savez_compressed(handle, **bad)
    with pytest.raises(ValueError, match='EXACT_CONDITION_MISMATCH'):
        diag.load_last(bad_path, loaded)


def test_manifest_requires_sha_and_explicit_fit_normalization(tmp_path):
    import json
    config = {'schema': 'smallk_saved_diagnostics_v1', 'dataset': 'synthetic',
              'raw': {'path': 'raw.npz'}, 'candidates': [{'name': 'x', 'seed': 1, 'path': 'x.npz'}],
              'storage_scale': 'normalized', 'splits': ['fit'], 'split_roles': {'fit': 'training'},
              'normalization_origin_split': 'fit'}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    args = argparse.Namespace(source_manifest=str(path), raw=None, candidate=None)
    with pytest.raises(ValueError, match='require SHA256'):
        diag.resolve_config(args)
    config['raw']['sha256'] = '0' * 64
    config['candidates'][0]['sha256'] = '1' * 64
    config['normalization_origin_split'] = 'all'
    path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError, match='FIT statistics'):
        diag.resolve_config(args)
