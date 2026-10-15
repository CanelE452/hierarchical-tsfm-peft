"""FIT-only purged conditional STATIC readout stability; no live F/model/GPU."""
from pathlib import Path
import argparse
import gc
import json
import math
import shutil
import sys
import time

import run_horizon_decoder_diagnostic_v1 as horizon

old, HERE, ROOT, BASE, OLD, require = horizon.old, horizon.HERE, horizon.ROOT, horizon.BASE, horizon.OLD, horizon.require
SPACES = ('PCA_K8', 'RAW_C')
ALPHAS = horizon.ALPHAS
PLAN_SHA = '86e407534891b7129d5d7a74772283c3d9113b668b63725d005ebaf114593cc1'
HORIZON_RESULT_SHA = '5f158382640dac74ec63cc3d1005b385c898821d3c032d51b543f2c9b27d7dac'
HORIZON_AUDIT_SHA = '0f65053536403a3b7b8f53fca3b9f8274fb1a406eaf6dfdcb73bb0467fffe698'
HORIZON_CONTROLLER_SHA = '151da5b0b4cc4cacdb9d7a547d12e7837dcdcb5ee4c5047a1d6e3993afe525c6'
HORIZON_AUDITOR_SHA = 'accb6fb775d9b12ca6e39c0be7e062545b98d429089ee32e0ac46ed65edae613'
SOURCE_HASHES = {
    'horizon_decoder_delta_v1.py': '406252e940ae676e7a78f69e067f5b3fa70456bdb1ea27d57169b8964ceabe0e',
    'horizon_correction_error_terms_v1.py': '806a66415834394d1b5efb07bc95fdbd7d103e6167c15abaf7be03f64026a187',
    'causal_lag_decoder_v1.py': 'f9846499e2cb9c8a322b25cb1e2c328a5eb3ff059d674a20ba172258c207ee6b',
    'decoder_metric_probe_v1.py': '75dd4159dfb8aed3d070bb9e123588ce81ff83fa9847687d4a4fae14166d50cc',
    'fixed_decoder_span_v1.py': '687d856bb48aaaf82c8c1c2a9e8842dbd43ece98148bf44e99ed22943dc5d506',
    'continuation_runtime_v1.py': '83db8dc961cdf66dc2b0b499c717f89651faebddbbc4f56f939d5d890793b2d5',
}


def load_gates(args):
    require(all(isinstance(v, str) and len(v) == 64 for v in (PLAN_SHA, HORIZON_RESULT_SHA, HORIZON_AUDIT_SHA, HORIZON_CONTROLLER_SHA, HORIZON_AUDITOR_SHA, *SOURCE_HASHES.values())), 'PURGED_SOURCE_GATES_NOT_REGISTERED')
    old.bound({'path': str(HERE / 'PLAN_PURGED_READOUT_STABILITY_KO_v1.md'), 'sha256': PLAN_SHA})
    old.bound({'path': str(HERE / 'run_horizon_decoder_diagnostic_v1.py'), 'sha256': HORIZON_CONTROLLER_SHA})
    for name, digest in SOURCE_HASHES.items():
        old.bound({'path': str(HERE / name), 'sha256': digest})
    refs = {'horizon_diagnostic': old.record(args.horizon_result), 'horizon_audit': old.record(args.horizon_audit)}
    require(refs['horizon_diagnostic']['sha256'] == HORIZON_RESULT_SHA and refs['horizon_audit']['sha256'] == HORIZON_AUDIT_SHA, 'EXACT_ACCEPTED_HORIZON_RESULT_AUDIT')
    diagnostic, audit = old.read(args.horizon_result), old.read(args.horizon_audit)
    require(diagnostic['status'] == 'COMPLETE' and audit['status'] == 'PASS' and not diagnostic['failed_paths'] and not audit['partial_paths'], 'COMPLETE_HORIZON_AND_FULL_PASS_REQUIRED')
    require(diagnostic['all_24_locks_before_DEV'] is True and diagnostic['protected_TEST_accessed'] is audit['protected_TEST_accessed'] is False, 'ACCEPTED_HORIZON_LOCKS_AND_TEST_SCOPE')
    require(all(diagnostic.get(k) == v for k, v in {'fits_completed': 24, 'alpha_candidates_completed': 240, 'ZERO_candidates_completed': 24, 'candidate_FIT_TUNE_evaluations_completed': 528, 'selected_DEV_evaluations_completed': 24, 'resolved_selection_locks': 24}.items()), 'FULL_HORIZON_COUNTS')
    require(audit['totals']['decoder_fit_paths_checked'] == 24 and audit['totals']['selected_FP32_arithmetic_predictions_checked'] == 72, 'HORIZON_AUDIT_COVERAGE')
    old.bound(audit['diagnostic'], refs['horizon_diagnostic'])
    old.bound(audit['audit_source'], {'path': str(HERE / 'audit_horizon_decoder_diagnostic_v1.py'), 'sha256': HORIZON_AUDITOR_SHA})
    old.bound(diagnostic['controller_source'], {'path': str(HERE / 'run_horizon_decoder_diagnostic_v1.py'), 'sha256': HORIZON_CONTROLLER_SHA})
    cross, science, parentrefs = horizon.load_gates(args)
    for key in ('cross_moment_diagnostic', 'cross_moment_audit'):
        old.bound(diagnostic[key], parentrefs[key])
    refs.update(parentrefs)
    return diagnostic, audit, cross, science, refs


def partition_origins(origins):
    import numpy as np
    require(isinstance(origins, np.ndarray) and origins.dtype == np.int64 and origins.shape == (256,), 'EXACT_FIT256_INT64_ORIGINS')
    require(np.unique(origins).size == 256, 'UNIQUE_FIT_ORIGINS_REQUIRED')
    order = np.argsort(origins, kind='stable')
    a, b, c = order[:128], order[128:192], order[192:]
    first_b, first_c = int(origins[b[0]]), int(origins[c[0]])
    keep_a, keep_b = origins[a] <= first_b - 48, origins[b] <= first_c - 48
    parts = {'A': a[keep_a], 'B': b[keep_b], 'C': c, 'A_candidate': a, 'B_candidate': b,
             'A_purged': a[~keep_a], 'B_purged': b[~keep_b], 'ordered_source': order}
    require(all(parts[key].size for key in ('A', 'B', 'C')), 'NONEMPTY_PURGED_A_B_C_REQUIRED')
    return parts, {'purge': {'A_candidate_count': 128, 'B_candidate_count': 64, 'C_count': 64,
        'A_removed_count': int((~keep_a).sum()), 'B_removed_count': int((~keep_b).sum()),
        'first_B_origin': first_b, 'first_C_origin': first_c,
        'A_predicate': 'origin+47 < first_B_origin', 'B_predicate': 'origin+47 < first_C_origin',
        'origin_means_first_future_timestamp': True, 'history_overlap_permitted': True,
        'complete_L_plus_H_separation_claimed': False, 'original_input_order_sorted': bool(np.array_equal(order, np.arange(256)))}}


def timestamp_facts(origins, observed):
    import numpy as np
    require(origins.dtype == np.int64 and origins.ndim == 1 and origins.size and observed.dtype == np.bool_ and observed.shape[:2] == (len(origins), 48), 'MATCHED_COHORT_ORIGINS_AND_H48_MASK')
    timestamps = origins[:, None] + np.arange(48, dtype=np.int64)
    unique, counts = np.unique(timestamps, return_counts=True)
    multiplicity, frequencies = np.unique(counts, return_counts=True)
    return {'origin_count': len(origins), 'origins_min': int(origins.min()), 'origins_max': int(origins.max()),
        'first_target_timestamp': int(unique[0]), 'last_target_timestamp': int(unique[-1]),
        'horizon_slots': int(timestamps.size), 'unique_target_timestamps': len(unique), 'max_target_multiplicity': int(counts.max()),
        'target_multiplicity_histogram': [{'multiplicity': int(m), 'unique_timestamp_count': int(n)} for m, n in zip(multiplicity, frequencies)],
        'observed_counts_per_channel': observed.sum(axis=(0, 1)).tolist(),
        'unique_observed_target_timestamps_per_channel': [int(np.unique(timestamps[observed[:, :, c]]).size) for c in range(observed.shape[2])],
        'independent_repeat_count': None}


def subset(raw, indices):
    return {**raw, **{key: raw[key][indices] for key in ('target', 'observed', 'origins')}}


def score_prediction(prediction, raw):
    import numpy as np
    from decoder_metric_probe_v1 import masked_scores
    complete = np.broadcast_to(raw['observed'].all(2, keepdims=True), raw['observed'].shape)
    return {'status': 'COMPLETE', 'scores_all_observed': masked_scores(prediction, raw['target'], raw['observed'], raw['std']),
        'scores_complete_mask': masked_scores(prediction, raw['target'], complete, raw['std']) if complete.any() else {'status': 'NO_COMPLETE_C'}}


def fit_A(features, baseline, target, observed, indices):
    import horizon_decoder_delta_v1 as module
    x = features[indices]
    scale, constant = module.feature_scale(x)
    response = horizon.fit_response(target[indices], observed[indices], baseline[indices])
    return module.fit_path(module.gated_features(x, 'STATIC_DELTA'), response, observed[indices], scale, constant)


def select_B(candidates):
    order = ['ALPHA_' + format(a, 'g') for a in ALPHAS] + ['ZERO_DELTA']
    valid = [row for row in candidates if row['status'] == 'COMPLETE' and math.isfinite(row.get('B_MSE', float('nan')))]
    require(bool(valid), 'FINITE_B_SELECTION_REQUIRED')
    return min(valid, key=lambda row: (row['B_MSE'], order.index(row['id'])))


def resolve_paths(result, fit, seal, registry):
    resolved, locks = {}, []
    for space in SPACES:
        row = result['routes'].setdefault(space, {})
        try:
            row = fit(space)
            require(row['status'] == 'COMPLETE', 'ALL_ELEVEN_A_B_CANDIDATES_REQUIRED')
        except Exception as error:
            row = result['routes'].get(space, row)
            row.update(status='FAILED_PATH_PRESERVED', failure={'type': type(error).__name__, 'message': str(error)})
            result['failed_paths'].append({'space': space, 'stage': 'A_B_FIT_SELECTION', 'failure': row['failure']})
        row['selection_lock'] = seal(space, row)
        result['routes'][space] = row
        locks.append({'space': space, 'status': row['status'], 'selection_lock': row['selection_lock']})
        if row['status'] == 'COMPLETE':
            resolved[space] = row
        gc.collect()
    result['selection_registry'] = registry(locks)
    result['totals']['resolved_selection_locks'] = len(locks)
    result['both_locks_before_C'] = True
    return resolved


def infer_C(result, resolved, infer):
    require(result.get('both_locks_before_C') is True and result['totals']['resolved_selection_locks'] == 2, 'BOTH_B_LOCKS_AND_REGISTRY_BEFORE_C')
    for space, row in resolved.items():
        try:
            row['C'] = infer(space, row)
            require(row['C']['status'] == 'COMPLETE', 'FINITE_SELECTED_C_REQUIRED')
            result['totals']['selected_C_evaluations_completed'] += 1
        except Exception as error:
            result['failed_paths'].append({'space': space, 'stage': 'SELECTED_C', 'failure': {'type': type(error).__name__, 'message': str(error)}})


def execute(args):
    diagnostic, audit, cross, science, refs = load_gates(args)
    output = old.fresh_output(args.tag)
    sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    import numpy as np
    import horizon_decoder_delta_v1 as module
    import horizon_correction_error_terms_v1 as error_terms
    import fixed_decoder_span_v1 as io
    import continuation_runtime_v1 as rt
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    from threadpoolctl import threadpool_limits
    require(tuple(module.ALPHAS) == ALPHAS, 'REGISTERED_ALPHA_MENU_UNCHANGED')
    output.mkdir(parents=True, exist_ok=False)
    meter = rt.Meter('purged_readout_' + args.tag, gpu=False)
    result = {'status': 'RUNNING', **refs, 'routes': {}, 'failed_paths': [],
        'totals': {key: 0 for key in ('fits_attempted', 'fits_completed', 'alpha_candidates_completed', 'ZERO_candidates_completed', 'candidate_A_B_evaluations_completed', 'selected_C_evaluations_completed', 'resolved_selection_locks')},
        'planned_counts': {'fits_completed': 2, 'alpha_candidates_completed': 20, 'ZERO_candidates_completed': 2, 'candidate_A_B_evaluations_completed': 44, 'selected_C_evaluations_completed': 2, 'resolved_selection_locks': 2},
        'conditional_readout_diagnostic': True, 'fixed_full_FIT_upstream': True, 'upstream_used_A_outside_data': True,
        'independent_holdout': False, 'end_to_end_temporal_validation': False, 'strict_K_only': False,
        'new_F_forward_calls': 0, 'new_model_objects_loaded': 0, 'new_optimizer_updates': 0, 'new_E_D_training_updates': 0,
        'new_backbone_training': False, 'whole_deployment_cost_measured': False, 'new_coefficient_cost_status': 'NOT_MEASURED',
        'protected_TEST_accessed': False, 'TUNE_DEV_arrays_deserialized': False, 'TUNE_DEV_scored': False,
        'lineage_protection_may_hash_TUNE_DEV_file_bytes': True,
        'C_targets_loaded_with_full_FIT_archive': True, 'C_outputs_or_scores_before_locks': False,
        'NEW_SELECTED_C_predictions_or_C_scores_before_locks': False,
        'C_baseline_and_features_available_from_fixed_full_FIT_context_before_locks': True,
        'C_mask_and_timestamp_facts_computed_before_locks': True,
        'C_outputs_or_scores_scope': 'New selected corrected C predictions, C score/error evaluation; excludes fixed full-FIT baselines/features and mask/timestamp facts',
        'independent_OS_C_access_events_verified': False,
        'both_locks_before_C': False, 'new_DEV_winners_selected': 0, 'same_capacity_PCA_and_RAW_claimed': False,
        'inherited_horizon_data_protocol': 'origin is first future timestamp; response origin..origin+47',
        'source_records': [], 'input_records': [], 'protected_prior_records': []}
    protected, checked, contexts = [], {}, {}
    failure = None
    try:
        install_protected_test_guard()
        meter.protected_check()
        protected = prior_file_records(diagnostic, audit, cross, science, refs)
        result['protected_prior_records'] = protected
        folder = output / 'sources'
        folder.mkdir()
        source_paths = [Path(__file__), HERE / 'test_run_purged_readout_stability_v1.py', HERE / 'PLAN_PURGED_READOUT_STABILITY_KO_v1.md',
            HERE / 'run_horizon_decoder_diagnostic_v1.py', HERE / 'run_new_direction_direct_ridge_v1.py',
            HERE / 'run_linear_e_adaptation_v1.py', OLD / 'prepare_inputs.py', *[HERE / name for name in SOURCE_HASHES]]
        for path in source_paths:
            destination = folder / path.name
            shutil.copyfile(path, destination)
            result['source_records'].append({**old.record(path), 'snapshot_path': str(destination)})
        result.update(controller_source=old.record(__file__), module_source=old.record(module.__file__),
            error_terms_module_source=old.record(error_terms.__file__), plan=old.record(HERE / 'PLAN_PURGED_READOUT_STABILITY_KO_v1.md'))
        def load(record):
            return io.checked_payload({**record, 'path': str(old.bound(record))}, meter, checked)
        source = science['datasets']['robin']['spaces']['PCA']['conditions']['L7_RECON']['splits']['fit']['input_source']
        for space in SPACES:
            old.bound(diagnostic['datasets']['robin']['spaces'][space]['contexts']['fit']['input_source'], source)
        raw = load(source)
        require(raw['history'].shape == (256, 512, 17) and raw['target'].shape == raw['observed'].shape == (256, 48, 17), 'ROBIN_FIT256_C17_L512_H48')
        require(raw['history'].dtype == raw['target'].dtype == np.float32 and raw['observed'].dtype == np.bool_ and np.isfinite(raw['target'][raw['observed']]).all(), 'TRUE_OBSERVED_FP32_FIT_TARGETS')
        parts, partition = partition_origins(raw['origins'])
        partition['cohorts'] = {label: timestamp_facts(raw['origins'][parts[label]], raw['observed'][parts[label]]) for label in ('A', 'B', 'C')}
        result['partition'] = partition
        result['partition_artifact'] = io.save_npz(output / 'PARTITION.npz', {'source_origins': raw['origins'],
            **{key + '_indices': value for key, value in parts.items()}})
        def context(space):
            if space in contexts:
                return contexts[space]
            cache = state = cached_source = state_source = None
            if space == 'PCA_K8':
                owner = cross['datasets']['robin']['conditions'][space]
                entry = owner['splits']['fit']
                old.bound(entry['input_source'], source)
                cached_source, state_source = entry['artifact'], owner['state_artifact']
                parent_context = diagnostic['datasets']['robin']['spaces'][space]['contexts']['fit']
                old.bound(cached_source, parent_context['cached_latent_source'])
                old.bound(state_source, parent_context['frozen_state_source'])
                cache, state = load(cached_source), load(state_source)
                require(old.byte_equal(cache['origins'], raw['origins']) and state['E'].shape == (17, 8), 'EXACT_FIXED_K8_FIT_CACHE_ORIGINS')
            stream = horizon.stream_payload(raw['history'], raw['raw_last'], raw['raw_forecast'], space, 'F', cache, state)
            metadata = {'input_source': source, 'cached_latent_source': cached_source, 'frozen_state_source': state_source,
                'stream_checks': stream['checks'], 'array_bindings': {key: {'shape': list(stream[key].shape), 'dtype': str(stream[key].dtype), 'sha256': io.array_sha(stream[key])} for key in ('features', 'baseline', 'latent_history', 'latent_future')}}
            contexts[space] = stream, metadata
            return stream, metadata
        def detailed(prediction, baseline, cohort, kind):
            return {**score_prediction(prediction, cohort), 'temporal_diagnostics': horizon.temporal_scores(prediction, cohort),
                'error_terms': error_terms.diagnose(baseline, prediction, cohort['target'], cohort['observed'], require_zero=kind == 'ZERO_DELTA')}
        def run_route(space):
            stream, metadata = context(space)
            row = {'status': 'RUNNING', 'space': space, 'mode': 'F', 'arm': 'STATIC_DELTA', 'source_context': metadata,
                'K': stream['latent_history'].shape[2], 'additional_slope_parameters': stream['features'].shape[2] * 17, 'candidates': [], 'baseline_scores': {}}
            result['routes'][space] = row
            result['totals']['fits_attempted'] += 1
            started = time.perf_counter()
            states, fitmeta = fit_A(stream['features'], stream['baseline'], raw['target'], raw['observed'], parts['A'])
            summary, fitarrays = horizon.fit_info(fitmeta)
            summary.update(only_A_rows_used_for_readout_fit=True, only_A_rows_used_for_RMS=True,
                B_C_labels_used_for_fit=False, fixed_upstream_uses_full_FIT=True)
            row.update(fit_metadata=summary, fit_wall_seconds=time.perf_counter() - started,
                RMS_source='A-only actual-F ungated features; upstream full-FIT state remains fixed')
            row['fit_artifact'] = io.save_npz(output / (space + '_A_FIT_INFO.npz'), fitarrays)
            result['totals']['fits_completed'] += 1
            zero = {'alpha': None, 'coefficient_FP64': np.zeros_like(states[0]['coefficient_FP64']), 'coefficient_FP32': np.zeros_like(states[0]['coefficient_FP32'])}
            for state in [*states, zero]:
                kind = 'ZERO_DELTA' if state['alpha'] is None else 'ALPHA'
                identifier = kind if kind == 'ZERO_DELTA' else 'ALPHA_' + format(state['alpha'], 'g')
                candidate = {'id': identifier, 'kind': kind, 'alpha': state['alpha'], 'status': 'FAILED_CANDIDATE_PRESERVED'}
                row['candidates'].append(candidate)
                try:
                    arrays = {k: v for k, v in state.items() if k != 'alpha'}
                    candidate['coefficient_artifact'] = io.save_npz(output / (space + '_' + identifier + '_COEFFICIENTS.npz'), arrays)
                    candidate['coefficient_array_records'], candidate['coefficient_value_sha256'] = horizon.parent.coefficient_records(state)
                    predictions = {}
                    for label in ('A', 'B'):
                        indices = parts[label]
                        cohort = subset(raw, indices)
                        prediction, check = horizon.predict_candidate(stream['baseline'][indices], stream['features'][indices], state, 'STATIC_DELTA', kind)
                        candidate[label] = score_prediction(prediction, cohort)
                        candidate.setdefault('parity_checks', {})[label] = check
                        result['totals']['candidate_A_B_evaluations_completed'] += 1
                        predictions[label + '_prediction_FP32'] = prediction
                        predictions[label + '_origins'] = cohort['origins']
                    candidate['prediction_artifact'] = io.save_npz(output / (space + '_' + identifier + '_A_B.npz'), predictions)
                    candidate['B_MSE'] = candidate['B']['scores_all_observed']['normalized']['mse']
                    candidate['status'] = 'COMPLETE'
                    result['totals']['ZERO_candidates_completed' if kind == 'ZERO_DELTA' else 'alpha_candidates_completed'] += 1
                    meter.check({'stage': 'A_FIT_B_SELECTION_ONLY', 'space': space, 'candidate': identifier})
                except Exception as error:
                    candidate['failure'] = {'type': type(error).__name__, 'message': str(error)}
            require(len(row['candidates']) == 11 and all(c['status'] == 'COMPLETE' for c in row['candidates']), 'COMPLETE_ELEVEN_CANDIDATE_MENU')
            winner = select_B(row['candidates'])
            selected_predictions = load(winner['prediction_artifact'])
            for label in ('A', 'B'):
                indices = parts[label]
                cohort = subset(raw, indices)
                row[label] = detailed(selected_predictions[label + '_prediction_FP32'], stream['baseline'][indices], cohort, winner['kind'])
                row['baseline_scores'][label] = score_prediction(stream['baseline'][indices], cohort)
            row.update(status='COMPLETE', selected_candidate=winner['id'], selected_kind=winner['kind'], selected_alpha=winner['alpha'],
                selected_B_MSE=winner['B_MSE'], selected_coefficients=winner['coefficient_artifact'], selected_coefficient_value_sha256=winner['coefficient_value_sha256'],
                selected_A_B_prediction_artifact=winner['prediction_artifact'], selected_A_B_reused_without_new_inference=True,
                selection_rule='Own actual FP32 B observed channel-macro MSE; registered alpha order then ZERO on exact tie')
            return row
        def seal(space, row):
            path = output / (space + '_B_SELECTION_LOCK.json')
            rt.write_new(path, {'space': space, 'status': row['status'], **{key: row.get(key) for key in ('selected_candidate', 'selected_kind', 'selected_alpha', 'selected_B_MSE', 'selected_coefficients', 'selected_coefficient_value_sha256')},
                'C_outputs_or_scores_created_before_lock': False,
                'NEW_SELECTED_C_predictions_or_C_scores_before_lock': False,
                'C_outputs_or_scores_scope': result['C_outputs_or_scores_scope'],
                'C_targets_in_same_FIT_archive_already_loaded': True,
                'C_baseline_and_features_available_from_fixed_full_FIT_context_before_lock': True,
                'C_mask_and_timestamp_facts_computed_before_lock': True, 'sealed_unix': time.time()})
            return old.record(path)
        def registry(locks):
            path = output / 'B_SELECTION_REGISTRY.json'
            rt.write_new(path, {'status': 'BOTH_ROUTES_RESOLVED_BEFORE_C', 'conditions': locks,
                'C_outputs_or_scores_created_before_seal': False,
                'NEW_SELECTED_C_predictions_or_C_scores_before_seal': False,
                'C_outputs_or_scores_scope': result['C_outputs_or_scores_scope'],
                'C_baseline_and_features_available_from_fixed_full_FIT_context_before_seal': True,
                'C_mask_and_timestamp_facts_computed_before_seal': True, 'sealed_unix': time.time()})
            return old.record(path)
        def infer(space, row):
            old.bound(result['selection_registry'])
            old.bound(row['selection_lock'])
            state = load(row['selected_coefficients'])
            require(horizon.parent.coefficient_records(state)[1] == row['selected_coefficient_value_sha256'], 'LOCKED_COEFFICIENT_VALUES_UNCHANGED')
            stream, _ = context(space)
            indices = parts['C']
            cohort = subset(raw, indices)
            prediction, check = horizon.predict_candidate(stream['baseline'][indices], stream['features'][indices], state, 'STATIC_DELTA', row['selected_kind'])
            entry = detailed(prediction, stream['baseline'][indices], cohort, row['selected_kind'])
            entry.update(artifact=io.save_npz(output / (space + '_SELECTED_C.npz'), {'prediction_FP32': prediction, 'origins': cohort['origins']}), parity_check=check)
            row['baseline_scores']['C'] = score_prediction(stream['baseline'][indices], cohort)
            return entry
        with threadpool_limits(limits=2):
            resolved = resolve_paths(result, run_route, seal, registry)
            infer_C(result, resolved, infer)
        result['comparisons'] = {}
        for label in ('B', 'C'):
            if all(result['routes'][space].get(label, {}).get('status') == 'COMPLETE' for space in SPACES):
                pca, rawroute = (result['routes'][space] for space in SPACES)
                p = pca[label]['scores_all_observed']['normalized']['mse']
                r = rawroute[label]['scores_all_observed']['normalized']['mse']
                result['comparisons'][label] = {'PCA_selected_minus_ZERO': p - pca['baseline_scores'][label]['scores_all_observed']['normalized']['mse'],
                    'Raw_selected_minus_ZERO': r - rawroute['baseline_scores'][label]['scores_all_observed']['normalized']['mse'], 'PCA_selected_minus_Raw_selected': p - r}
        result['input_records'] = list(checked.values())
        result['status'] = 'COMPLETE' if not result['failed_paths'] and all(result['totals'][key] == value for key, value in result['planned_counts'].items()) else 'PARTIAL_COMPLETED'
        json.dumps(result, allow_nan=False)
    except Exception as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
        result.update(status='FAILED_PRESERVED', failure=failure)
    finally:
        result['input_records'] = list(checked.values())
        try:
            for item in [*protected, *checked.values(), *result['source_records']]:
                old.bound(item)
            for item in result['source_records']:
                require(old.record(item['snapshot_path'])['sha256'] == item['sha256'], 'EXECUTED_SOURCE_SNAPSHOT_UNCHANGED')
            meter.protected_check()
            result['protected_and_loaded_inputs_unchanged_after_run'] = True
        except Exception as error:
            integrity_failure = {'type': type(error).__name__, 'message': str(error)}
            result.update(status='FAILED_PRESERVED', integrity_failure=integrity_failure,
                protected_and_loaded_inputs_unchanged_after_run=False)
            failure = failure or integrity_failure
        result['resource_terminal'] = meter.close(result['status'], failure)
        rt.write_new(output / 'RESULT.json', result)
    print(json.dumps({'status': result['status'], 'result': str(output / 'RESULT.json'), 'totals': result['totals'], 'failure': failure}), flush=True)
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('tag', 'cross-result', 'cross-audit', 'horizon-result', 'horizon-audit'):
        parser.add_argument('--' + key, required=True)
    raise SystemExit(execute(parser.parse_args()))
