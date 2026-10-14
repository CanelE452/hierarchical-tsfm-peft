"""CPU synthetic checks; no previous scientific arrays or model are opened."""
import inspect
import json

import numpy as np
import pytest

from research.smallk_failure_followup_20261014 import projected_past_probe as pp


def fixture(channels=5):
    rng = np.random.default_rng(92601)
    splits = {}
    for index, name in enumerate(('fit', 'tune', 'val')):
        history = rng.normal(size=(4, 9, channels)).astype(np.float32)
        raw = linear_forecast(history)
        observed = np.ones_like(raw, dtype=bool)
        observed[0, 0, 0] = False
        target = raw + np.float32(.2)
        target[~observed] = np.nan
        splits[name] = {'history': history, 'raw_forecast': raw, 'target': target,
                        'observed': observed, 'history_observed': np.ones_like(history, bool),
                        'origins': np.arange(4, dtype=np.int64) * 20 + 100 * index}
    return splits


def linear_forecast(history):
    value = .7 * history[:, -1:] + .3 * history[:, -2:-1]
    return np.repeat(value, 3, axis=1).astype(history.dtype)


def test_basis_uses_fit_raw_delta_and_has_all_predeclared_candidates():
    values = fixture(11)['fit']
    family = pp.fit_bases(values['raw_forecast'], values['history'][:, -1])
    assert [(basis.name, basis.rank) for basis in family] == [
        ('identity_C', 11), ('through_origin_quarter', 2), ('through_origin_half', 5),
        ('through_origin_full_C_rotated', 11), ('centered_quarter', 2),
        ('centered_half', 5), ('centered_full_C_rotated', 11)]
    assert tuple(inspect.signature(pp.fit_bases).parameters) == ('fit_raw_forecast', 'fit_last')
    delta = (values['raw_forecast'].astype(np.float64) - values['history'][:, -1:, :]).reshape(-1, 11)
    through = next(b for b in family if b.name == 'through_origin_full_C_rotated')
    centered = next(b for b in family if b.name == 'centered_full_C_rotated')
    assert np.allclose(through.eigenvalues.sum(), np.mean(delta ** 2, axis=0).sum())
    assert np.allclose(centered.mean, delta.mean(0))
    assert np.allclose(centered.eigenvalues.sum(), np.mean((delta - delta.mean(0)) ** 2, axis=0).sum())
    for basis in family:
        assert np.allclose(basis.matrix.T @ basis.matrix, np.eye(basis.rank), atol=1e-12)
        for column in basis.matrix.T:
            assert column[np.argmax(np.abs(column))] >= 0
    again = pp.fit_bases(values['raw_forecast'], values['history'][:, -1])
    assert all(np.array_equal(left.matrix, right.matrix) for left, right in zip(family, again))


def test_through_origin_and_centered_fit_are_distinct():
    delta = np.array([[[10., 1.]], [[10., -1.]], [[10., 2.]], [[10., -2.]]])
    family = pp.fit_bases(delta, np.zeros((4, 2)))
    through = next(b for b in family if b.name == 'through_origin_quarter')
    centered = next(b for b in family if b.name == 'centered_quarter')
    assert np.allclose(through.matrix[:, 0], [1., 0.])
    assert np.allclose(centered.matrix[:, 0], [0., 1.])


@pytest.mark.parametrize('mode', ['through_origin', 'centered'])
def test_linear_shared_temporal_forecast_commutes_with_fixed_projection(mode):
    values = fixture()['fit']
    basis = next(b for b in pp.fit_bases(values['raw_forecast'], values['history'][:, -1])
                 if b.name == mode + '_quarter')
    result = pp.probe_split(values['history'], values['raw_forecast'], basis, linear_forecast, batch_size=2)
    assert np.allclose(result['projected_past_forecast'], result['projected_raw_forecast'], atol=3e-7)
    assert np.array_equal(result['latent_last'], result['latent_past'][:, -1:])
    assert result['latent_future'].shape == (4, 3, basis.rank)


def test_centered_decode_restores_the_fit_mean_perpendicular_component():
    basis = pp.Basis('centered_manual', 'centered', 'manual', np.array([[1.], [0.]]),
                     np.array([3., 4.]), np.array([1., 0.]))
    raw_last = np.array([[5., 6.]])
    raw_forecast = np.array([[[7., 16.]]])
    assert np.allclose(pp.project_raw_forecast(raw_forecast, raw_last, basis), [[[7., 10.]]])
    latent_future = np.array([[[7.]]])
    assert np.allclose(pp.decode_forecast(latent_future, raw_last, basis), [[[7., 10.]]])
    assert np.allclose(pp.mean_perpendicular(basis), [0., 4.])


def test_full_rank_rotation_reconstructs_information_but_can_change_a_nonlinear_forecast():
    angle = .6
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    basis = pp.Basis('through_origin_full_C_rotated', 'through_origin', 'full_C_rotated',
                     rotation, np.zeros(2), np.ones(2))
    history = np.array([[[.2, .7], [.9, -.3], [1.4, .8]]], np.float32)
    nonlinear = lambda x: np.repeat(x[:, -1:] + x[:, -1:] ** 2, 3, axis=1)
    result = pp.probe_split(history, nonlinear(history), basis, nonlinear)
    assert np.max(np.abs(result['reconstructed_history'] - history)) < 3e-7
    assert np.max(np.abs(result['projected_raw_forecast'] - nonlinear(history))) < 5e-7
    assert np.max(np.abs(result['projected_past_forecast'] - result['projected_raw_forecast'])) > .1


def test_identity_is_a_separate_forecast_sanity_check():
    values = fixture()['fit']
    basis = pp.fit_bases(values['raw_forecast'], values['history'][:, -1])[0]
    result = pp.probe_split(values['history'], values['raw_forecast'], basis, linear_forecast)
    assert np.array_equal(result['latent_past'], values['history'])
    assert np.array_equal(result['latent_future'], values['raw_forecast'])
    assert np.allclose(result['projected_past_forecast'], values['raw_forecast'], atol=1e-7)


def test_scores_use_observed_channel_macro_and_keep_physical_scale_separate():
    prediction = np.array([[[1., 10.], [1., 999.], [1., 999.]]])
    target = np.zeros_like(prediction)
    observed = np.array([[[True, True], [True, False], [True, False]]])
    target[~observed] = np.nan
    normalized = pp.observed_score(prediction, target, observed)
    physical = pp.observed_score(prediction, target, observed, np.array([2., 4.]))
    assert normalized['mse'] == 50.5
    assert normalized['mae'] == 5.5
    assert normalized['observed_counts'] == [3, 1]
    assert physical['mse'] == 802.
    assert normalized['mse'] != (3. + 100.) / 4.
    empty = pp.observed_score(prediction, target, np.zeros_like(observed))
    assert empty['mse'] is None
    assert empty['active_channels'] == 0


def test_target_and_mask_changes_never_change_the_fixed_basis(tmp_path):
    splits = fixture()
    first = pp.run_probe(splits, linear_forecast, np.zeros(5, np.float32), np.ones(5, np.float32),
                         tmp_path / 'first', provenance={'fixture': True})
    for values in splits.values():
        values['target'] = values['target'] * 1000
        values['observed'][:] = False
    second = pp.run_probe(splits, linear_forecast, np.zeros(5, np.float32), np.ones(5, np.float32),
                          tmp_path / 'second', provenance={'fixture': True})
    assert first['status'] == second['status'] == 'COMPLETE'
    assert len(first['candidates']) == len(second['candidates']) == 7
    for left, right in zip(first['candidates'], second['candidates']):
        with np.load(tmp_path / 'first' / left['basis_artifact']['path']) as a:
            with np.load(tmp_path / 'second' / right['basis_artifact']['path']) as b:
                assert np.array_equal(a['basis'], b['basis'])
                assert np.array_equal(a['delta_mean'], b['delta_mean'])


def test_saved_schema_contains_all_candidates_and_exact_input_provenance(tmp_path):
    splits = fixture()
    output = tmp_path / 'run'
    result = pp.run_probe(splits, linear_forecast, np.zeros(5, np.float32), np.ones(5, np.float32),
                          output, provenance={'source_sha256': 'synthetic_source'})
    assert result['status'] == 'COMPLETE'
    assert result['scientific_execution'] is False
    assert len(result['candidates']) == 7
    assert set(result['split_roles']) == {'fit', 'tune', 'val'}
    assert len(result['artifacts']) == 3 + 7 + 7 * 3
    for candidate in result['candidates']:
        assert set(candidate['splits']) == {'fit', 'tune', 'val'}
        for split, row in candidate['splits'].items():
            with np.load(output / row['artifact']['path']) as arrays:
                assert {'projected_raw_forecast', 'projected_past_forecast', 'latent_past',
                        'latent_last', 'latent_future', 'raw_last', 'reconstructed_history'} <= set(arrays.files)
                assert np.array_equal(arrays['raw_last'], splits[split]['history'][:, -1])
            assert set(row['scores']) == {'raw', 'projected_raw', 'projected_past'}
            assert row['scores']['raw']['active_channels'] == 5
    with np.load(output / 'INPUT_fit.npz') as arrays:
        assert np.array_equal(arrays['origins'], splits['fit']['origins'])
        assert np.array_equal(arrays['history_normalized'], splits['fit']['history'])
        assert arrays['observed'].dtype == np.bool_
    assert json.loads((output / 'RESULT.json').read_text())['provenance']['source_sha256'] == 'synthetic_source'
    with pytest.raises(FileExistsError):
        pp.run_probe(splits, linear_forecast, np.zeros(5, np.float32), np.ones(5, np.float32),
                     output, provenance={'fixture': True})


def test_partial_outputs_and_path_specific_failure_are_preserved(tmp_path):
    count = 0

    def failing_predictor(history):
        nonlocal count
        count += 1
        if count == 4:
            raise RuntimeError('SYNTHETIC_PREDICTOR_FAILURE')
        return linear_forecast(history)

    output = tmp_path / 'partial'
    with pytest.raises(RuntimeError, match='SYNTHETIC_PREDICTOR_FAILURE'):
        pp.run_probe(fixture(), failing_predictor, np.zeros(5, np.float32), np.ones(5, np.float32),
                     output, provenance={'fixture': True})
    result = json.loads((output / 'RESULT.json').read_text())
    assert result['status'] == 'FAILED_PRESERVED'
    assert result['failure']['candidate'] == 'through_origin_quarter'
    assert result['failure']['split'] == 'fit'
    assert result['failure']['path'].endswith('through_origin_quarter__fit.npz')
    assert (output / 'identity_C__val.npz').is_file()


def test_identity_mismatch_is_retained_and_blocks_unfair_comparison(tmp_path):
    splits = fixture()
    splits['fit']['raw_forecast'] = splits['fit']['raw_forecast'] + 10
    output = tmp_path / 'mismatch'
    with pytest.raises(ValueError, match='IDENTITY_FORECAST_VS_STORED_RAW'):
        pp.run_probe(splits, linear_forecast, np.zeros(5, np.float32), np.ones(5, np.float32),
                     output, provenance={'fixture': True})
    result = json.loads((output / 'RESULT.json').read_text())
    assert result['status'] == 'FAILED_PRESERVED'
    assert result['failure']['candidate'] == 'identity_C'
    assert (output / 'identity_C__fit.npz').is_file()


def test_protected_test_and_unmetered_actual_execution_are_rejected(tmp_path):
    splits = fixture()
    splits['test'] = splits['val']
    with pytest.raises(ValueError, match='SPLIT_SCOPE'):
        pp.run_probe(splits, linear_forecast, np.zeros(5, np.float32), np.ones(5, np.float32),
                     tmp_path / 'test', provenance={})
    with pytest.raises(ValueError, match='SHARED_LIMIT_REQUIRED'):
        pp.run_probe(fixture(), linear_forecast, np.zeros(5, np.float32), np.ones(5, np.float32),
                     tmp_path / 'actual', provenance={}, actual=True)


def test_basis_fit_failure_names_fit_input_and_preserves_all_inputs(tmp_path):
    output = tmp_path / 'basis_failure'
    with pytest.raises(ValueError, match='D1_REQUIRES_AT_LEAST_TWO_CHANNELS'):
        pp.run_probe(fixture(1), linear_forecast, np.zeros(1, np.float32), np.ones(1, np.float32),
                     output, provenance={'fixture': True})
    result = json.loads((output / 'RESULT.json').read_text())
    assert result['failure']['phase'] == 'basis_fit'
    assert result['failure']['split'] == 'fit'
    assert result['failure']['path'].endswith('INPUT_fit.npz')
    assert len(result['artifacts']) == 3
