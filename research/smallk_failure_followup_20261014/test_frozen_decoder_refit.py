"""Synthetic CPU contracts; no old NPZ, checkpoint, model, GPU or TEST access."""
import copy
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from research.smallk_failure_followup_20261014 import frozen_decoder_refit as dr


def state_fixture(k=2, c=3):
    rng = np.random.default_rng(17)
    return {
        'codec.decode.weight': rng.normal(size=(c, k)).astype(np.float32),
        'codec.decode.bias': rng.normal(size=c).astype(np.float32),
        'residual.0.weight': rng.normal(size=(32, k)).astype(np.float32),
        'residual.0.bias': rng.normal(size=32).astype(np.float32),
        'residual.2.weight': rng.normal(size=(c, 32)).astype(np.float32),
        'residual.2.bias': rng.normal(size=c).astype(np.float32),
    }


def independent_original(future, last, raw_last, state):
    def decode(z):
        t = torch.from_numpy(z.copy())
        linear = F.linear(t, torch.from_numpy(state['codec.decode.weight']),
                          torch.from_numpy(state['codec.decode.bias']))
        hidden = F.gelu(F.linear(t, torch.from_numpy(state['residual.0.weight']),
                                torch.from_numpy(state['residual.0.bias'])), approximate='none')
        return linear + F.linear(hidden, torch.from_numpy(state['residual.2.weight']),
                                 torch.from_numpy(state['residual.2.bias']))
    return (torch.from_numpy(raw_last.copy())[:, None] + decode(future) - decode(last)).numpy()


def bundle_fixture():
    state = state_fixture()
    rng = np.random.default_rng(19)
    splits = {}
    for index, split in enumerate(('fit', 'tune', 'val')):
        future = rng.normal(size=(9, 4, 2)).astype(np.float32)
        last = rng.normal(size=(9, 1, 2)).astype(np.float32)
        raw_last = rng.normal(size=(9, 3)).astype(np.float32)
        target = np.repeat(raw_last[:, None], 4, axis=1) + future @ np.array(
            [[1., 2., -1.], [.5, -.2, 1.]], np.float32)
        observed = rng.uniform(size=target.shape) > .15
        target[~observed] = np.nan
        splits[split] = {'z_future': future, 'z_last': last, 'raw_last': raw_last,
                         'original_prediction': independent_original(future, last, raw_last, state),
                         'raw_forecast': np.repeat(raw_last[:, None], 4, axis=1) + np.float32(.4),
                         'target': target, 'observed': observed,
                         'origins': np.arange(9, dtype=np.int64) * 8 + 100 * index}
    return state, splits


def run_fixture(tmp_path, *, splits=None, shared_limit=None):
    state, original = bundle_fixture()
    return dr.run_decoder_refit(
        tmp_path / 'fresh', splits=original if splits is None else splits, codec_state=state,
        mean=np.zeros(3, np.float32), std=np.ones(3, np.float32),
        source_record={'fixture_only': True}, synthetic=True, shared_limit=shared_limit)


def test_original_factorization_preserves_actual_gelu_class_and_float32_order():
    assert hasattr(dr, 'DecoderState'), 'frozen decoder extraction is not implemented'
    raw_state, splits = bundle_fixture()
    state = dr.DecoderState.from_codec_state(raw_state)
    values = splits['fit']
    original = dr.original_anchored_prediction(values['z_future'], values['z_last'], values['raw_last'], state)
    assert np.array_equal(original, values['original_prediction'])
    factored = dr.factored_original_prediction(values['z_future'], values['z_last'], values['raw_last'], state)
    assert np.allclose(factored, original, atol=1e-5, rtol=2e-6)
    assert factored.dtype == np.float32


def test_nonlinear_hidden_difference_cannot_be_reconstructed_from_pre_q_delta():
    raw = state_fixture(k=1, c=1)
    raw['residual.0.weight'][:] = 1.
    raw['residual.0.bias'][:] = 0.
    state = dr.DecoderState.from_codec_state(raw)
    future = np.array([[[1.]], [[3.]]], np.float32)
    last = np.array([[[0.]], [[2.]]], np.float32)
    features = dr.decoder_features(future, last, state)
    assert np.array_equal(features[:, :, :1], np.ones((2, 1, 1), np.float32))
    assert not np.allclose(features[0, :, 1:], features[1, :, 1:])


def test_linear_only_source_head_reduces_to_original_anchored_linear_class():
    raw, splits = bundle_fixture()
    raw['residual.2.weight'][:] = 0.
    state = dr.DecoderState.from_codec_state(raw)
    values = splits['fit']
    expected = values['raw_last'][:, None] + (values['z_future'] - values['z_last']) @ raw['codec.decode.weight'].T
    factored = dr.factored_original_prediction(values['z_future'], values['z_last'], values['raw_last'], state)
    assert np.allclose(factored, expected, atol=1e-6)


def test_no_centering_or_intercept_and_zero_delta_preserves_current_anchor():
    features = np.array([[[2.]], [[4.]], [[6.]]], np.float32)
    last = np.array([[10.], [20.], [30.]], np.float32)
    target = last[:, None] + 3 * features
    head = dr.fit_output_head(features, target, last, np.ones_like(target, bool), 0.)
    assert head.weight.dtype == np.float32
    assert np.allclose(head.weight, [[3.]])
    assert np.allclose(head.feature_rms, [np.sqrt(56. / 3)])
    assert np.array_equal(head.predict(np.zeros_like(features), last), last[:, None])


def test_masked_per_channel_ridge_ignores_unobserved_nan_and_keeps_missing_channel_original():
    features = np.array([[[1., 0.]], [[0., 1.]], [[1., 1.]]], np.float32)
    target = np.array([[[2., np.nan, np.nan]], [[3., 4., np.nan]], [[np.nan, 10., np.nan]]], np.float32)
    mask = np.isfinite(target)
    fallback = np.array([[7., 8., 9.], [11., 12., 13.]], np.float32)
    head = dr.fit_output_head(features, target, np.zeros((3, 3), np.float32), mask, 0., fallback_weight=fallback)
    assert np.allclose(head.weight[:, :2], [[2., 6.], [3., 4.]], atol=1e-6)
    assert np.array_equal(head.weight[:, 2], fallback[:, 2])
    assert head.observed_counts.tolist() == [2, 2, 0]
    assert head.missing_fit_channels == [2]


def test_ridge_alpha_is_per_observed_channel_mean_loss_and_fit_rms_only():
    features = np.array([[[1.]], [[3.]]], np.float32)
    target = 2 * features
    head = dr.fit_output_head(features, target, np.zeros((2, 1), np.float32), np.ones_like(target, bool), 1.)
    assert np.allclose(head.feature_rms, [np.sqrt(5.)])
    assert np.allclose(head.weight, [[1.]])
    before = head.feature_rms.copy()
    head.predict(np.array([[[10000.]]], np.float32), np.zeros((1, 1), np.float32))
    assert np.array_equal(head.feature_rms, before)


def test_refit_can_be_exported_back_into_same_original_decoder_architecture():
    raw, splits = bundle_fixture()
    values = splits['fit']
    state = dr.DecoderState.from_codec_state(raw)
    features = dr.decoder_features(values['z_future'], values['z_last'], state)
    head = dr.fit_output_head(features, values['target'], values['raw_last'], values['observed'], .01,
                              fallback_weight=state.outgoing_weight)
    exported = head.decoder_state(state)
    assert np.array_equal(exported['residual.0.weight'], raw['residual.0.weight'])
    assert np.array_equal(exported['residual.0.bias'], raw['residual.0.bias'])
    expected = independent_original(values['z_future'], values['z_last'], values['raw_last'], exported)
    assert np.allclose(head.predict(features, values['raw_last']), expected, atol=2e-5, rtol=2e-6)


def test_target_labels_are_separate_and_teacher_uses_identical_actual_fit_mask():
    _, splits = bundle_fixture()
    values = splits['fit']
    objectives = dr.objective_targets(values['target'], values['raw_forecast'])
    assert np.array_equal(objectives['Y'], values['target'], equal_nan=True)
    assert np.array_equal(objectives['RAW'], values['raw_forecast'])
    mask = values['observed']
    assert np.allclose(objectives['MIXED'][mask], .5 * (values['target'][mask] + values['raw_forecast'][mask]))
    assert np.isnan(objectives['MIXED'][~mask]).all()


def test_tune_selects_earliest_exact_tie_and_validation_never_changes_fit_or_selection(tmp_path):
    raw, splits = bundle_fixture()
    first = run_fixture(tmp_path / 'first', splits=splits)
    changed = copy.deepcopy(splits)
    changed['val']['target'][changed['val']['observed']] += np.float32(10000.)
    changed['val']['z_future'] *= np.float32(10.)
    changed['val']['original_prediction'] = independent_original(
        changed['val']['z_future'], changed['val']['z_last'], changed['val']['raw_last'], raw)
    second = run_fixture(tmp_path / 'second', splits=changed)
    for objective in ('Y', 'RAW', 'MIXED'):
        left, right = first['refits'][objective], second['refits'][objective]
        assert left['selected_alpha'] == right['selected_alpha']
        assert left['feature_rms_sha256'] == right['feature_rms_sha256']
        assert left['selected_weight_sha256'] == right['selected_weight_sha256']
    assert dr.select_alpha([{'alpha': 0., 'tune_mse': 1.}, {'alpha': .1, 'tune_mse': 1.}]) == 0


def test_actual_y_tune_error_selects_ridge_alpha_rather_than_fit_or_teacher_loss(tmp_path):
    raw = state_fixture(k=1, c=1)
    raw['codec.decode.weight'][:] = 0.
    raw['residual.0.weight'][:] = 0.
    raw['residual.2.weight'][:] = 0.
    splits = {}
    for split in ('fit', 'tune', 'val'):
        future = np.arange(1, 4, dtype=np.float32).reshape(3, 1, 1)
        last = np.zeros_like(future)
        raw_last = np.zeros((3, 1), np.float32)
        target = future * (np.float32(3.) if split == 'fit' else np.float32(0.))
        splits[split] = {'z_future': future, 'z_last': last, 'raw_last': raw_last,
                         'original_prediction': independent_original(future, last, raw_last, raw),
                         'raw_forecast': np.zeros_like(target), 'target': target,
                         'observed': np.ones_like(target, bool), 'origins': np.arange(3, dtype=np.int64)}
    result = dr.run_decoder_refit(tmp_path / 'fresh', splits=splits, codec_state=raw,
                                  mean=np.zeros(1, np.float32), std=np.ones(1, np.float32),
                                  source_record={'fixture_only': True}, synthetic=True)
    assert result['refits']['Y']['selected_alpha'] == 10.
    assert result['refits']['RAW']['selected_alpha'] == 0.
    assert result['refits']['MIXED']['selected_alpha'] == 10.


def test_run_records_all_alphas_scores_origins_stats_and_source_and_preserves_readonly_inputs(tmp_path):
    raw, splits = bundle_fixture()
    before = copy.deepcopy(splits)
    for values in splits.values():
        for value in values.values():
            value.setflags(write=False)
    result = run_fixture(tmp_path, splits=splits)
    assert result['status'] == 'COMPLETE'
    assert result['scientific_execution'] is False
    assert result['split_roles']['val'] == 'reused_development'
    for objective in ('Y', 'RAW', 'MIXED'):
        assert [row['alpha'] for row in result['refits'][objective]['alpha_grid']] == list(dr.ALPHAS)
    assert (tmp_path / 'fresh' / 'METRICS.csv').is_file()
    assert (tmp_path / 'fresh' / 'REPORT_KO.md').is_file()
    with np.load(tmp_path / 'fresh' / 'INPUT_fit.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['origins'], splits['fit']['origins'])
        assert saved['mean'].dtype == np.float32
        assert saved['z_future'].dtype == np.float32
    with np.load(tmp_path / 'fresh' / 'SOURCE_DECODER.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['residual.0.weight'], raw['residual.0.weight'])
    for split, values in splits.items():
        for name, value in values.items():
            assert np.array_equal(value, before[split][name], equal_nan=True)
    with pytest.raises(FileExistsError):
        run_fixture(tmp_path, splits=splits)


@pytest.mark.parametrize('bad', ['test', 'float64', 'delta_only', 'no_tune_observed', 'hidden_wrong'])
def test_source_schema_guards_fail_before_any_output(tmp_path, bad):
    raw, splits = bundle_fixture()
    if bad == 'test':
        splits['test'] = splits['val']
    elif bad == 'float64':
        splits['fit']['z_future'] = splits['fit']['z_future'].astype(np.float64)
    elif bad == 'delta_only':
        del splits['fit']['z_last']
    elif bad == 'no_tune_observed':
        splits['tune']['observed'][:] = False
    else:
        raw['residual.0.weight'] = np.zeros((31, 2), np.float32)
    with pytest.raises(ValueError):
        dr.run_decoder_refit(tmp_path / 'bad', splits=splits, codec_state=raw,
                             mean=np.zeros(3, np.float32), std=np.ones(3, np.float32),
                             source_record={'fixture_only': True}, synthetic=True)
    assert not (tmp_path / 'bad').exists()


def test_actual_execution_requires_root_gate_source_verification_and_existing_shared_limit(tmp_path):
    raw, splits = bundle_fixture()
    with pytest.raises(ValueError, match='EXECUTION_GATE'):
        dr.run_decoder_refit(tmp_path / 'actual', splits=splits, codec_state=raw,
                             mean=np.zeros(3, np.float32), std=np.ones(3, np.float32),
                             source_record={})
    with pytest.raises(ValueError, match='SHARED_LIMIT'):
        dr.run_decoder_refit(tmp_path / 'actual', splits=splits, codec_state=raw,
                             mean=np.zeros(3, np.float32), std=np.ones(3, np.float32),
                             source_record={}, execution_authorized=True)
    assert not (tmp_path / 'actual').exists()


def test_caller_source_attestation_cannot_substitute_a_different_decoder_state():
    raw, splits = bundle_fixture()
    state = dr.DecoderState.from_codec_state(raw)
    record = {'bi_complete': True, 'architecture': 'ANCHOR_NONLINEAR_D', 'input_length': 512,
              'verification': {name: True for name in (
                  'source_checkpoint', 'source_snapshots', 'same_original_main_E_F',
                  'absolute_latents', 'original_stats_origins_masks', 'split_roles')},
              'decoder_state_sha256': '0' * 64}
    with pytest.raises(ValueError, match='DECODER_STATE_SHA256'):
        dr._validate(splits, np.zeros(3, np.float32), np.ones(3, np.float32), state, record, False)


@pytest.mark.parametrize('bad', ['verified_sha', 'missing_alias', 'wrong_dataset_split'])
def test_actual_source_lineage_aliases_hashes_and_dataset_split_roles_are_enforced(bad):
    raw = state_fixture(k=4, c=17)
    state = dr.DecoderState.from_codec_state(raw)
    splits = {}
    for split, count in (('fit', 256), ('tune', 64), ('val', 30)):
        zeros = np.zeros((count, 48, 17), np.float32)
        splits[split] = {'z_future': np.zeros((count, 48, 4), np.float32),
                         'z_last': np.zeros((count, 1, 4), np.float32),
                         'raw_last': np.zeros((count, 17), np.float32),
                         'original_prediction': zeros, 'raw_forecast': zeros, 'target': zeros,
                         'observed': np.ones_like(zeros, bool), 'origins': np.arange(count, dtype=np.int64)}
    record = {'bi_complete': True, 'architecture': 'ANCHOR_NONLINEAR_D', 'input_length': 512,
              'dataset': 'robin', 'seed': 92601, 'decoder_state_sha256': dr.decoder_state_sha256(state),
              'verification': {name: True for name in (
                  'source_checkpoint', 'source_snapshots', 'same_original_main_E_F',
                  'absolute_latents', 'original_stats_origins_masks', 'split_roles')},
              'source_artifacts': [{'alias': alias, 'path': 'caller-fixture-only/' + alias,
                                    'sha256': '1' * 64, 'verified_sha256': '1' * 64}
                                   for alias in ('MAIN_CP', 'MAIN_PREDICTION', 'RAW_PREDICTION', 'ORIGINAL_SOURCE')]}
    dr._validate(splits, np.zeros(17, np.float32), np.ones(17, np.float32), state, record, False)
    if bad == 'verified_sha':
        record['source_artifacts'][0]['verified_sha256'] = '2' * 64
        message = 'VERIFIED_SHA256_MISMATCH'
    elif bad == 'missing_alias':
        record['source_artifacts'].pop()
        message = 'ALIASES_REQUIRED'
    else:
        splits['eval'] = splits.pop('val')
        message = 'SPLITS_REQUIRED'
    with pytest.raises(ValueError, match=message):
        dr._validate(splits, np.zeros(17, np.float32), np.ones(17, np.float32), state, record, False)


def test_observed_macro_scores_keep_channel_weight_equal_and_physical_scale_separate():
    prediction = np.array([[[2., 3.]], [[2., 99.]]], np.float32)
    target = np.array([[[0., 0.]], [[0., np.nan]]], np.float32)
    mask = np.isfinite(target)
    normalized = dr.observed_score(prediction, target, mask)
    physical = dr.observed_score(prediction, target, mask, np.array([10., 1.], np.float32))
    assert normalized['mse'] == 6.5
    assert normalized['mae'] == 2.5
    assert normalized['observed_counts'] == [2, 1]
    assert physical['mse'] == 204.5
    assert physical['mae'] == 11.5


def test_original_source_parity_failure_keeps_reference_arrays_and_does_not_refit(tmp_path):
    _, splits = bundle_fixture()
    splits['fit']['original_prediction'] += np.float32(1.)
    with pytest.raises(ValueError, match='PARITY'):
        run_fixture(tmp_path, splits=splits)
    result = json.loads((tmp_path / 'fresh' / 'RESULT.json').read_text(encoding='utf-8'))
    assert result['status'] == 'FAILED_PRESERVED'
    assert result['refits'] == {}
    assert (tmp_path / 'fresh' / 'REFERENCE_fit.npz').is_file()


def test_local_failure_preserves_fresh_arrays_and_location_receipt(tmp_path):
    class Limit:
        def __init__(self):
            self.calls = 0
        def check(self):
            self.calls += 1
            if self.calls == 4:
                raise RuntimeError('fixture cap')
    with pytest.raises(RuntimeError, match='fixture cap'):
        run_fixture(tmp_path, shared_limit=Limit())
    result = json.loads((tmp_path / 'fresh' / 'RESULT.json').read_text(encoding='utf-8'))
    assert result['status'] == 'FAILED_PRESERVED'
    assert result['failure']['phase']
    assert (tmp_path / 'fresh' / 'INPUT_fit.npz').is_file()
