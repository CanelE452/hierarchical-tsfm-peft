"""SOURCE_READY_NOT_EXECUTED: synthetic integration fixtures, no real assets.

Any future fixture execution must be inside the fresh approved runtime. The
coefficient fixtures below are manufactured arrays, not trained model results.
Temporary NPZ descriptors intentionally use original probes' raw-byte schema.
"""
import ast
import copy
import hashlib
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_matched_readout_v1 as runner
import run_condition_probes as probes
import stage_separation_math_v1 as old
import matched_readout_contract_v1 as contract


def manufactured_old_n2(tmp_path):
    past = np.linspace(-1., 1., 3*512, dtype=np.float32).reshape(3, 512, 1)
    raw_last = np.array([[.2, -.3], [.4, -.5], [.6, -.7]], np.float32)
    raw_features = old.fixed_past_features(past)
    state = old.FeatureState(old._readonly(np.zeros(33)), old._readonly(np.ones(33)), 1, 3, old.array_sha(raw_features))
    coefficients = np.zeros((33, 48, 2), np.float64)
    coefficients[0, :, 0] = .125
    coefficients[-1, :, 1] = -.5
    intercept = np.repeat(np.array([[.1, -.2]], np.float64), 48, axis=0)
    counts = np.full((48, 2), 3, np.int64)
    solution = old.RidgeSolution('N2_DIRECT_C', .01, old._readonly(coefficients), old._readonly(intercept),
        old._readonly(counts, np.int64), old._readonly([144, 144], np.int64), old._readonly(np.full((48, 2), 144.)),
        old._readonly(np.full((48, 2), 2), np.int64), old._readonly(np.full((48, 2), 5.)), state.sha256, True)
    model = old.NativeModel(state, solution)
    feature_record = probes.save_npz(tmp_path/'FEATURES.npz', {'center': state.center, 'scale': state.scale})
    coefficients_record = probes.save_npz(tmp_path/'COEFFICIENTS.npz', runner.solution_arrays(solution))
    solution_metadata = {'kind': solution.kind, 'alpha': solution.alpha, 'feature_state_sha256': state.sha256,
                         'solution_sha256': solution.sha256, 'with_intercept': True}
    row = {'status': 'COMPLETE', 'diagnostic_only': False, 'selected_index': 3, 'selected_alpha': .01,
           'selected_model_sha256': model.sha256, 'selected_coefficient_artifact': coefficients_record,
           'selected_model_metadata': {'solution': solution_metadata}, 'selected_artifacts': {},
           'alpha_grid': [{'alpha': a, 'tune_actual_Y_mse': 1. if i == 3 else 2.,
                          'model_sha256': model.sha256, 'coefficient_artifact': coefficients_record}
                         for i, a in enumerate(contract.ALPHAS)]}
    state_record = {'artifact': feature_record, 'sha256': state.sha256, 'latent_channels': 1,
                    'fit_rows': 3, 'fit_raw_features_sha256': state.fit_raw_features_sha256}
    stage = {'status': 'COMPLETE', 'dataset': 'robin', 'seed': 92601, 'old_E_D_F_updated': False,
             'F_loaded': False, 'E_forward_replayed': False, 'codec_before': {'old_algorithm': 'fixture'},
             'codec_after': {'old_algorithm': 'fixture'}, 'codec_terminal_hash': {'old_algorithm': 'fixture'},
             'feature_state': state_record, 'families': {'N2': row}}
    return stage, model, past, raw_last


def test_gate_structurally_precedes_output_numpy_contract_or_asset_loader():
    tree = ast.parse(inspect.getsource(runner.execute))
    body = tree.body[0].body
    assert isinstance(body[0], ast.Import) and body[0].names[0].name == 'matched_readout_runtime_v1'
    assert isinstance(body[1], ast.Assign) and body[1].value.func.attr == 'ContinuationLimit'
    assert body[1].value.keywords[0].arg == 'gpu' and body[1].value.keywords[0].value.value is False
    source = inspect.getsource(runner.execute)
    gate = source.index('runtime.ContinuationLimit(')
    for token in ('output.mkdir(', 'import numpy', 'import matched_readout_math', 'probes.load_bridge(', 'probes.load_checked_npz('):
        assert gate < source.index(token)
    assert 'torch.load' not in source and 'import_old_engine' not in source and '.cuda(' not in source


def test_constructor_rejection_creates_no_output_or_asset_load(tmp_path, monkeypatch):
    class SyntheticGateDenied(RuntimeError):
        pass
    def denied(tag, gpu):
        assert gpu is False
        raise SyntheticGateDenied('synthetic approval absent')
    monkeypatch.setitem(sys.modules, 'matched_readout_runtime_v1', SimpleNamespace(ContinuationLimit=denied))
    monkeypatch.setattr(runner, 'CACHE', tmp_path/'uncreated')
    args = SimpleNamespace(tag='synthetic_gate', dataset='robin', seed=92601)
    with pytest.raises(SyntheticGateDenied):
        runner.execute(args)
    assert not (tmp_path/'uncreated').exists()


def test_NPZ_descriptor_matches_original_raw_byte_manifest(tmp_path):
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    record = probes.save_npz(tmp_path/'RAW_BYTES.npz', {'values': values})
    declared = record['arrays']['values']
    assert declared == {'shape': [3, 4], 'dtype': 'float64',
                        'sha256': hashlib.sha256(values.tobytes()).hexdigest()}
    assert declared['sha256'] != old.array_sha(values)
    loaded, _ = probes.load_checked_npz(record)
    np.testing.assert_array_equal(loaded['values'], values)
    wrong = copy.deepcopy(record)
    wrong['arrays']['values']['sha256'] = old.array_sha(values)
    with pytest.raises(ValueError, match='BRIDGE_ARRAY_CONTENT_MISMATCH'):
        probes.load_checked_npz(wrong)


def test_old_N2_state_coefficients_hash_and_prediction_exact_restore(tmp_path):
    stage, model, past, last = manufactured_old_n2(tmp_path)
    contract.validate_stage_reuse(stage, 'robin', 92601)
    restored, records = runner.restore_reused_n2(stage, probes)
    assert len(records) == 2
    assert restored.feature_state.sha256 == model.feature_state.sha256
    assert restored.solution.sha256 == model.solution.sha256
    assert restored.sha256 == model.sha256
    expected_innovation = (model.feature_state.transform(past) @ model.solution.coefficients.reshape(33, -1)
                           + model.solution.intercept.reshape(-1)).reshape(3, 48, 2).astype(np.float32)
    expected = last[:, None] + expected_innovation
    np.testing.assert_array_equal(restored.direct_forecast(past, last), expected)


def test_changed_selected_coefficient_valid_file_still_rejected_by_state_SHA(tmp_path):
    stage, model, _, _ = manufactured_old_n2(tmp_path)
    changed = {key: value.copy() for key, value in runner.solution_arrays(model.solution).items()}
    changed['coefficients'][0, 0, 0] += 1.
    stage['families']['N2']['selected_coefficient_artifact'] = probes.save_npz(tmp_path/'CHANGED.npz', changed)
    with pytest.raises(ValueError, match='REUSED_N2_MODEL_SHA'):
        runner.restore_reused_n2(stage, probes)


def test_changed_feature_state_or_selected_model_hash_rejected(tmp_path):
    stage, _, _, _ = manufactured_old_n2(tmp_path)
    changed = copy.deepcopy(stage)
    changed['feature_state']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='REUSED_N2_FEATURE_STATE_SHA'):
        runner.restore_reused_n2(changed, probes)
    changed = copy.deepcopy(stage)
    changed['families']['N2']['selected_model_sha256'] = '0'*64
    with pytest.raises(ValueError, match='REUSED_N2_MODEL_SHA'):
        runner.restore_reused_n2(changed, probes)


def test_wrong_condition_selection_or_candidate_source_rejected(tmp_path):
    stage, _, _, _ = manufactured_old_n2(tmp_path)
    for dataset, seed in (('beijing_aotizhongxin', 92601), ('robin', 92602)):
        with pytest.raises(ValueError, match='LINEAGE'):
            contract.validate_stage_reuse(stage, dataset, seed)
    changed = copy.deepcopy(stage)
    changed['families']['N2']['selected_coefficient_artifact'] = {
        **changed['families']['N2']['selected_coefficient_artifact'], 'sha256': '0'*64}
    with pytest.raises(ValueError, match='EARLIEST_N2_SELECTION'):
        contract.validate_stage_reuse(changed, 'robin', 92601)
    changed = copy.deepcopy(stage)
    changed['families']['N2']['selected_index'] = 0
    with pytest.raises(ValueError, match='EARLIEST_N2_SELECTION'):
        contract.validate_stage_reuse(changed, 'robin', 92601)


def test_old_N2_selected_input_contract_keeps_unobserved_NaN_and_rejects_changes():
    target = np.array([[[1., np.nan], [2., 3.]]], np.float32)
    observed = np.isfinite(target)
    references = {'target': target, 'observed': observed, 'origins': np.array([512], np.int64),
                  'raw_last': np.array([[.2, -.3]], np.float32), 'mean': np.zeros(2, np.float32),
                  'std': np.array([1., 2.], np.float32)}
    for name, value in references.items():
        contract.exact(value.copy(), value, 'same:' + name)
    for name in ('target', 'origins', 'std'):
        changed = references[name].copy()
        changed.flat[0] += 1
        with pytest.raises(ValueError, match='EXACT_OLD_INPUT'):
            contract.exact(changed, references[name], 'changed:' + name)
    with pytest.raises(ValueError, match='EXACT_OLD_INPUT'):
        contract.exact(target.astype(np.float64), target, 'cast')
    changed = target.copy(); changed[0, 0, 1] = 0.
    with pytest.raises(ValueError, match='EXACT_OLD_INPUT'):
        contract.exact(changed, target, 'future_missing_fill')


def test_missing_E_cache_is_local_and_RAW_route_runs_before_it():
    source = inspect.getsource(runner.execute)
    raw_position = source.index("fit_route('RAW33C'")
    cache_position = source.index('stage_path = probes.resolve(args.stage_native_result)')
    assert raw_position < cache_position
    assert "result['E_cache_or_N2_reuse_failure'] = failure" in source
    assert "setdefault('E512K', {'status': 'NOT_RUN'" in source
    assert "if isinstance(error, runtime.BudgetHeld): raise" in source
    assert 'forecast(past_by_split[split], raw_splits[split][\'raw_last\'])' in source
    assert source.index("fit_route('E512K'") < source.index('model, records = restore_reused_n2(')
    assert 'declared = by_stem(stage, \'INPUT_\' + split)' in source
    assert 'arrays, record = probes.load_checked_npz(declared)' in source
    assert 'cached_records[split] = declared' in source


def test_forecast_inputs_do_not_accept_future_targets_or_masks():
    import matched_readout_math_v1 as math
    assert tuple(inspect.signature(math.ReadoutModel.forecast).parameters) == ('self', 'past', 'raw_last')
    assert tuple(inspect.signature(old.NativeModel.direct_forecast).parameters) == ('self', 'latent_past', 'raw_last')


def test_artifact_lookup_rejects_missing_or_duplicate_SOURCE():
    value = {'artifacts': [{'path': '/synthetic/INPUT_fit.npz'}]}
    assert runner.by_stem(value, 'INPUT_fit')['path'].endswith('INPUT_fit.npz')
    with pytest.raises(ValueError, match='EXACT_SINGLE_STAGE_ARTIFACT'):
        runner.by_stem(value, 'INPUT_tune')
    with pytest.raises(ValueError, match='EXACT_SINGLE_STAGE_ARTIFACT'):
        runner.by_stem({'artifacts': value['artifacts']*2}, 'INPUT_fit')


def test_terminal_resource_failure_cannot_be_claimed_complete():
    source = inspect.getsource(runner.execute)
    assert "if receipt.get('terminal_status') != 'COMPLETE':" in source
    assert "result.update(status=receipt['terminal_status']" in source
    assert "if initialized: runtime.write_new(output / 'RESULT.json', result)" in source
    assert 'resource_receipt' in source


def manufactured_cached_D2_lineage(tmp_path):
    latent = np.arange(2*512*2, dtype=np.float32).reshape(2, 512, 2) / np.float32(100.)
    target = np.ones((2, 48, 3), np.float32)
    observed = np.ones(target.shape, bool)
    target[0, 0, 1] = np.nan; observed[0, 0, 1] = False
    arrays = {'latent_past': latent, 'z_last': latent[:, -1:].copy(), 'target': target,
        'observed': observed, 'origins': np.array([512, 576], np.int64),
        'raw_last': np.zeros((2, 3), np.float32), 'history_observed': np.ones((2, 512, 3), bool),
        'mean': np.zeros(3, np.float32), 'std': np.ones(3, np.float32),
        'raw_forecast': np.zeros((2, 48, 3), np.float32), 'original_prediction': np.ones((2, 48, 3), np.float32)}
    absolute = probes.save_npz(tmp_path/'ABSOLUTE_LATENTS_fit.npz', arrays)
    stage_input = probes.save_npz(tmp_path/'INPUT_fit.npz', arrays)
    provenance = {'input_bridge': {'sha256': 'b'*64},
                  'input_artifacts': [{'sha256': '1'*64}, {'sha256': '2'*64}, {'sha256': '3'*64}]}
    cp = {'sha256': 'c'*64}
    d2 = {'status': 'COMPLETE', 'stage': 'D2', 'dataset': 'robin', 'seed': 92601,
          'MAIN_codec_frozen_pass': True, 'F_frozen_pass': True,
          'F_before': {'old_algorithm': probes.OLD_F_HASH, 'bi_algorithm': probes.BI_F_HASH},
          'F_after': {'old_algorithm': probes.OLD_F_HASH, 'bi_algorithm': probes.BI_F_HASH},
          'MAIN_checkpoint': {'checkpoint': cp}, 'bridge_provenance': provenance, 'artifacts': [absolute]}
    d2_path = tmp_path/'D2_SYNTHETIC.json'
    probes.write(d2_path, d2)
    stage = {'D2_source': {'path': str(d2_path), 'sha256': probes.sha(d2_path)}}
    return stage, provenance, cp, {'fit': stage_input}, {'fit': arrays}


def test_cached_lineage_uses_declared_array_manifest_not_stripped_verified_record(tmp_path):
    stage, provenance, cp, declared, cached = manufactured_cached_D2_lineage(tmp_path)
    loaded, stripped = probes.load_checked_npz(declared['fit'])
    assert 'arrays' in declared['fit'] and 'arrays' not in stripped
    np.testing.assert_array_equal(loaded['latent_past'], cached['fit']['latent_past'])
    checked = runner.verify_cached_lineage(probes, stage, 'robin', 92601, provenance, cp, declared, cached)
    assert checked['sha256'] == stage['D2_source']['sha256']
    # Reproduces the earlier adapter bug instead of blessing a file-only record.
    with pytest.raises(KeyError, match='arrays'):
        runner.verify_cached_lineage(probes, stage, 'robin', 92601, provenance, cp, {'fit': stripped}, cached)


def test_cached_lineage_rejects_wrong_latent_bytes_even_with_matching_inputs(tmp_path):
    stage, provenance, cp, declared, cached = manufactured_cached_D2_lineage(tmp_path)
    changed = {name: value.copy() for name, value in cached['fit'].items()}
    changed['latent_past'][0, 0, 0] += np.float32(1.)
    with pytest.raises(ValueError, match='D2_TO_STAGE_CACHED_ARRAY_LINEAGE:fit:latent_past'):
        runner.verify_cached_lineage(probes, stage, 'robin', 92601, provenance, cp, declared, {'fit': changed})
    manifest_changed = copy.deepcopy(declared)
    manifest_changed['fit']['arrays']['latent_past']['sha256'] = old.array_sha(cached['fit']['latent_past'])
    with pytest.raises(ValueError, match='D2_TO_STAGE_CACHED_ARRAY_LINEAGE:fit:latent_past'):
        runner.verify_cached_lineage(probes, stage, 'robin', 92601, provenance, cp, manifest_changed, cached)


def test_cached_lineage_requires_exact_D2_CP_bridge_seed_and_input_files(tmp_path):
    stage, provenance, cp, declared, cached = manufactured_cached_D2_lineage(tmp_path)
    with pytest.raises(ValueError, match='ORIGINAL_D2_CP_BRIDGE_CACHED_E_LINEAGE'):
        runner.verify_cached_lineage(probes, stage, 'robin', 92602, provenance, cp, declared, cached)
    with pytest.raises(ValueError, match='ORIGINAL_D2_CP_BRIDGE_CACHED_E_LINEAGE'):
        runner.verify_cached_lineage(probes, stage, 'robin', 92601, provenance, {'sha256': '0'*64}, declared, cached)
    changed = copy.deepcopy(provenance)
    changed['input_bridge']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='ORIGINAL_D2_CP_BRIDGE_CACHED_E_LINEAGE'):
        runner.verify_cached_lineage(probes, stage, 'robin', 92601, changed, cp, declared, cached)
    changed = copy.deepcopy(provenance)
    changed['input_artifacts'][0]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='ORIGINAL_D2_CP_BRIDGE_CACHED_E_LINEAGE'):
        runner.verify_cached_lineage(probes, stage, 'robin', 92601, changed, cp, declared, cached)


def test_prior_canonical_audit_binds_exact_producer_SHA(tmp_path, monkeypatch):
    producer_record = {'path': '/synthetic/native_result.json', 'sha256': 'a'*64}
    rows = [{'dataset': d, 'seed': s, 'producer_sha256': 'a'*64 if (d, s) == ('robin', 92601) else str(i+1)*64}
            for i, (d, s) in enumerate((('robin', 92601), ('robin', 92602),
                                       ('beijing_aotizhongxin', 92601), ('beijing_aotizhongxin', 92602)))]
    audit = {'status': 'PASS', 'failed_count': 0, 'conditions': rows}
    audit_path = tmp_path/'SYNTHETIC_AUDIT.json'; probes.write(audit_path, audit)
    audit_record = {'role': 'native_audit', 'path': str(audit_path), 'sha256': probes.sha(audit_path)}
    manifest = {'generator_result_status': 'COMPLETE', 'native_audit_gate': {'passed': True},
                'input_records': [{'role': 'native_result', **producer_record}, audit_record]}
    manifest_path = tmp_path/'SYNTHETIC_MANIFEST.json'; probes.write(manifest_path, manifest)
    monkeypatch.setattr(runner, 'CANONICAL_STAGE_REPORT_SHA', probes.sha(manifest_path))
    records = runner.verify_prior_audit(probes, manifest_path, producer_record, 'robin', 92601)
    assert len(records) == 2
    with pytest.raises(ValueError, match='STAGE_PRODUCER_NOT_BOUND_TO_CANONICAL_REPORT'):
        runner.verify_prior_audit(probes, manifest_path, {**producer_record, 'sha256': 'f'*64}, 'robin', 92601)
    audit_bad = copy.deepcopy(audit); audit_bad['conditions'][0]['producer_sha256'] = 'f'*64
    bad_audit_path = tmp_path/'BAD_BINDING_AUDIT.json'; probes.write(bad_audit_path, audit_bad)
    manifest_bad = copy.deepcopy(manifest)
    manifest_bad['input_records'][1] = {'role': 'native_audit', 'path': str(bad_audit_path), 'sha256': probes.sha(bad_audit_path)}
    bad_manifest_path = tmp_path/'BAD_BINDING_MANIFEST.json'; probes.write(bad_manifest_path, manifest_bad)
    monkeypatch.setattr(runner, 'CANONICAL_STAGE_REPORT_SHA', probes.sha(bad_manifest_path))
    with pytest.raises(ValueError, match='ACTUAL_NATIVE_AUDIT_PRODUCER_SHA_BINDING_REQUIRED'):
        runner.verify_prior_audit(probes, bad_manifest_path, producer_record, 'robin', 92601)
