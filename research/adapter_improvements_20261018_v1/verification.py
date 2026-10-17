"""Independent saved-evidence audit. Import performs no I/O or model loading.

``verify(ctx, write=False)`` is read-only. The default additionally writes only
VALIDATION_REPORT.json. Missing work and unperformed publication gates remain
explicitly separate from the numerical validity of completed scored results.
"""
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from .contracts import (NAME, STATES, SUCCESS, atomic_json, digest, read,
                        resolve_binding, sha, utc, validate_plan, verify_references)


REQUIRED_PUBLIC = (
    'README_KO.md', 'RESULT_ANALYSIS_KO.md', 'METHOD_CONTRACT.md', 'METHOD_REFERENCES.md',
    'REPRODUCE_KO.md', 'EXPERIMENT_PLAN.json', 'RESOURCE_LEDGER.json', 'RESUME_QUEUE.json',
    'RUNS.csv', 'COMPARISONS.csv', 'COST.csv', 'CHANNEL_HORIZON_STATS.csv', 'SAME_UPDATE.csv',
    'MULTIK_AUDIT.csv', 'PAIRED_ORIGIN_EFFECTS.csv', 'ARTIFACT_MANIFEST.json', 'PPT_TABLES_KO.md', 'PUBLIC_PRIVATE_ARTIFACTS.md')
SCORED_KINDS = {'training', 'decoder', 'decoder_transfer', 'lift_jobs', 'reuse_pca'}
CORE_ARRAYS = {'pred_normalized', 'target_normalized', 'observed', 'origins', 'channel_ids'}
OPTIONAL_ARRAYS = {'mean', 'std', 'base_pred_normalized', 'anchor'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def array_signature(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256_raw_C_bytes=hashlib.sha256(value.tobytes()).hexdigest())


def _encoded_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def independent_statistics(prediction, target, mask):
    """FP64 SSE/SAE/counts, without importing any producer metric function."""
    p, y, m = np.asarray(prediction), np.asarray(target), np.asarray(mask)
    _require(p.ndim == 3 and min(p.shape) > 0 and p.shape == y.shape == m.shape,
             'prediction_target_mask_axes')
    _require(m.dtype == np.bool_, 'original_boolean_mask_required')
    _require(np.isfinite(p[m]).all() and np.isfinite(y[m]).all(), 'nonfinite_observed_prediction_target')
    error = np.where(m, p.astype(np.float64)-y.astype(np.float64), 0.)
    squared, absolute = np.square(error), np.abs(error)
    hc = np.sum(m, axis=0, dtype=np.int64)
    hs, ha = np.sum(squared, axis=0, dtype=np.float64), np.sum(absolute, axis=0, dtype=np.float64)
    cc = np.sum(hc, axis=0, dtype=np.int64)
    _require(np.all(cc > 0), 'registered_channel_N_c_zero')
    cs = np.sum(squared, axis=(0, 1), dtype=np.float64)
    ca = np.sum(absolute, axis=(0, 1), dtype=np.float64)

    def metric(sse, sae, count):
        defined = bool(np.all(count > 0))
        return dict(MSE=math.fsum((sse/count).tolist())/len(count) if defined else None,
                    MAE=math.fsum((sae/count).tolist())/len(count) if defined else None,
                    status='DEFINED' if defined else 'UNDEFINED',
                    coverage_channels=int(np.count_nonzero(count)), registered_channels=len(count))

    value = metric(cs, ca, cc)
    bands = []
    for indices in np.array_split(np.arange(p.shape[1]), 3):
        s, a, count = hs[indices].sum(0), ha[indices].sum(0), hc[indices].sum(0)
        bands.append(dict(horizon_start=int(indices[0])+1 if len(indices) else None,
                          horizon_end=int(indices[-1])+1 if len(indices) else None,
                          channel_sse=s.tolist(), channel_sae=a.tolist(), channel_counts=count.tolist(),
                          **metric(s, a, count)))
    os = np.sum(squared, axis=1, dtype=np.float64)
    oa = np.sum(absolute, axis=1, dtype=np.float64)
    oc = np.sum(m, axis=1, dtype=np.int64)
    origins = [dict(origin_index=i, channel_sse=s.tolist(), channel_sae=a.tolist(),
                    channel_counts=count.tolist(), **metric(s, a, count))
               for i, (s, a, count) in enumerate(zip(os, oa, oc))]
    _require(np.allclose(np.sum([b['channel_sse'] for b in bands], axis=0), cs, atol=1e-12, rtol=1e-12)
             and np.allclose(np.sum([b['channel_sae'] for b in bands], axis=0), ca, atol=1e-12, rtol=1e-12)
             and np.array_equal(np.sum([b['channel_counts'] for b in bands], axis=0), cc),
             'band_sufficient_statistics_do_not_reconstruct_whole')
    value.update(channel_sse=cs.tolist(), channel_sae=ca.tolist(), channel_counts=cc.tolist(),
                 horizon_sse=hs.tolist(), horizon_sae=ha.tolist(), horizon_counts=hc.tolist(),
                 origin_sse=os.tolist(), origin_sae=oa.tolist(), origin_counts=oc.tolist(),
                 origin_statistics=origins, bands=bands, observed=int(cc.sum()), n_origins=len(p),
                 metric_accumulation_dtype='float64')
    return value


def _compare_statistics(row, stats):
    for name in ('MSE', 'MAE'):
        _require(name in row and row[name] is not None and np.isfinite(float(row[name]))
                 and abs(float(row[name])-stats[name]) <= 1e-12, 'saved_FP64_'+name+'_mismatch')
    for name in ('channel_sse', 'channel_sae', 'channel_counts', 'horizon_sse', 'horizon_sae',
                 'horizon_counts', 'origin_sse', 'origin_sae', 'origin_counts'):
        if name in row:
            actual, expected = np.asarray(row[name]), np.asarray(stats[name])
            _require(actual.shape == expected.shape and
                     (np.array_equal(actual, expected) if 'count' in name else
                      np.allclose(actual, expected, atol=1e-12, rtol=1e-12)), name+'_mismatch')
    for name in ('observed', 'n_origins'):
        if name in row:
            _require(int(row[name]) == stats[name], name+'_mismatch')
    if 'bands' in row:
        _require(len(row['bands']) == 3, 'band_count_mismatch')
        for reported, actual in zip(row['bands'], stats['bands']):
            for key in ('horizon_start', 'horizon_end', 'MSE', 'MAE'):
                if key in reported:
                    a, b = reported[key], actual[key]
                    _require(a == b if a is None or b is None else abs(float(a)-float(b)) <= 1e-12,
                             'band_'+key+'_mismatch')
            for key in ('status', 'coverage_channels'):
                if key in reported:
                    _require(reported[key] == actual[key], 'band_'+key+'_mismatch')
            for key in ('channel_sse', 'channel_sae', 'channel_counts'):
                _require(key in reported and np.allclose(reported[key], actual[key], atol=1e-12, rtol=1e-12),
                         'band_'+key+'_mismatch')
    if 'origin_statistics' in row:
        _require(len(row['origin_statistics']) == len(stats['origin_statistics']), 'origin_statistics_count')
        for a, b in zip(row['origin_statistics'], stats['origin_statistics']):
            for key in ('origin_index', 'MSE', 'MAE'):
                if key in a:
                    _require(a[key] == b[key] if a[key] is None or b[key] is None else
                             abs(float(a[key])-float(b[key])) <= 1e-12, 'origin_'+key+'_mismatch')


def _fingerprints(arrays, row, kind):
    fingerprints = row.get('array_fingerprints', {})
    convention = 'dtype+json_shape+C_bytes' if kind in {'decoder', 'decoder_transfer', 'lift_jobs'} else 'raw_C_bytes'
    for name, expected in fingerprints.items():
        _require(name in arrays, 'fingerprint_array_missing')
        value = arrays[name]
        expected_sha = expected.get('sha256') if isinstance(expected, dict) else expected
        actual_sha = _encoded_sha(value) if convention == 'dtype+json_shape+C_bytes' else array_signature(value)['sha256_raw_C_bytes']
        _require(expected_sha == actual_sha, 'array_fingerprint_'+name+'_mismatch')
        if isinstance(expected, dict):
            _require(expected.get('shape') == list(value.shape) and expected.get('dtype') == str(value.dtype),
                     'array_fingerprint_'+name+'_axes_dtype_mismatch')
    return convention


def _scaler_sha(arrays):
    if not {'mean', 'std'}.issubset(arrays):
        return None
    values = {}
    for name in ('mean', 'std'):
        value = array_signature(arrays[name])
        values[name] = dict(shape=value['shape'], dtype=value['dtype'], sha256=value['sha256_raw_C_bytes'])
    return digest(values)


def _proof(row, result, job, ctx):
    artifact = row.get('prediction_binding')
    _require(isinstance(artifact, dict), 'scored_result_missing_prediction_binding')
    path = resolve_binding(artifact, ctx)
    if row.get('prediction_sha256'):
        _require(row['prediction_sha256'] == sha(path), 'prediction_NPZ_sha_mismatch')
    with np.load(path, allow_pickle=False) as archive:
        _require(CORE_ARRAYS.issubset(archive.files) and set(archive.files).issubset(CORE_ARRAYS|OPTIONAL_ARRAYS),
                 'prediction_archive_schema_mismatch')
        arrays = {key: archive[key] for key in archive.files}
    p, origins, channels = arrays['pred_normalized'], arrays['origins'], arrays['channel_ids']
    stats = independent_statistics(p, arrays['target_normalized'], arrays['observed'])
    _require(origins.ndim == 1 and np.issubdtype(origins.dtype, np.integer) and len(origins) == len(p)
             and (len(origins) < 2 or np.all(np.diff(origins) > 0)), 'origin_order_or_axes_mismatch')
    _require(channels.ndim == 1 and len(channels) == p.shape[2]
             and len(np.unique(channels)) == len(channels), 'channel_order_or_axes_mismatch')
    for name, expected in [('C', p.shape[2]), ('H', p.shape[1]), ('n_origins', len(p))]:
        if row.get(name) is not None:
            _require(int(row[name]) == expected, 'registered_'+name+'_mismatch')
    evaluation = row.get('evaluation', 'V_select')
    if evaluation == 'V_select' and len(origins) > 1:
        _require(np.all(np.diff(origins) == p.shape[1]), 'V_origin_stride_mismatch')
    cohort = row.get('dataset', job.get('cohort'))
    if cohort in ('jena', 'pems32'):
        c, h, k = (21, 48, 6) if cohort == 'jena' else (32, 12, 8)
        _require(p.shape[1:] == (h, c) and int(row.get('L', 512)) == 512
                 and (row.get('method') == 'RAW' or int(row.get('K', job.get('K', k))) == k),
                 'registered_dataset_C_K_L_H_mismatch')
    _compare_statistics(row, stats)
    convention = _fingerprints(arrays, row, job['kind'])
    scaler = _scaler_sha(arrays)
    claimed_scaler = row.get('scaler_sha256')
    if scaler is not None and claimed_scaler:
        _require(scaler == claimed_scaler, 'saved_scaler_sha_mismatch')
    if row.get('logical_F_rows') is not None:
        _require(int(row['logical_F_rows']) == len(p)*int(row.get('K', job.get('K', 0))), 'logical_F_rows_mismatch')
    _require(row.get('selection_split', 'A_stop') == 'A_stop', 'selection_split_is_not_A_stop')
    exposure = str(row.get('V_exposure', '')).lower()
    _require(not any(text in exposure for text in ('independent_holdout', 'unseen_holdout', 'unused_holdout')),
             'V_independent_holdout_claim')
    canonical = {label: array_signature(arrays[key]) for label, key in
                 [('target', 'target_normalized'), ('mask', 'observed'), ('origins', 'origins'), ('channel_ids', 'channel_ids')]}
    return dict(source_job_id=job['job_id'], method=row.get('method'), stage=row.get('stage'), evaluation=evaluation,
                source_run_id=row.get('run_id', job['job_id']+':'+str(row.get('method'))),
                epoch=row.get('epoch'), selected_epoch=row.get('selected_epoch'),
                prediction_reference=artifact['reference'], prediction_sha256=artifact['sha256'],
                canonical_cohort_signature=canonical,
                scaler_sha256_ifproven=scaler or claimed_scaler,
                scaler_evidence='saved mean/std arrays' if scaler else 'bound producer receipt' if claimed_scaler else None,
                metric_scale_ifproven='normalized' if row.get('units') or scaler else None,
                F_snapshot_ifproven=row.get('model_sha', result.get('model_sha', result.get('frozen_F_weight_hash', result.get('F_weight_hash')))),
                device=row.get('device', job.get('device')), recipe_binding=dict(recipe=result.get('recipe'),
                     scientific_identity=job['scientific_identity'], source_sha=row.get('source_sha', result.get('source_sha'))),
                checkpoint_binding=row.get('checkpoint_binding', result.get('bundle_binding')),
                parent_PE_checkpoint_sha256=row.get('PE_checkpoint_sha256'),
                base_checkpoint=row.get('base_checkpoint', row.get('parent_bundle_binding')),
                calibrator_binding=row.get('refiner_bundle'), state_binding=row.get('state_binding'),
                units=row.get('units'), source_fingerprint_convention=convention,
                C=p.shape[2], H=p.shape[1], K=row.get('K', job.get('K')), L=row.get('L'),
                head_dtype=row.get('head_dtype', row.get('core_dtype', str(p.dtype))),
                prediction_dtype=str(p.dtype),
                dataset=row.get('dataset', job.get('cohort')), seed=row.get('seed', job.get('seed')),
                updates=row.get('updates'), exposures=row.get('exposures'), sampler_sha256=row.get('sampler_sha256'),
                statistics=stats, validation_status='PASS')


def _rows(result):
    for collection in ('rows', 'common_endpoint_rows'):
        for index, row in enumerate(result.get(collection, [])):
            yield collection, index, row


def _producer_sources(result, plan, ctx, job):
    pinned = plan.get('source_hashes', {})
    for field in ('scientific_source_hashes', 'source_hashes', 'source_files'):
        for name, expected in result.get(field, {}).items():
            matches = [value for relative, value in pinned.items()
                       if relative == name or ('/' not in name and Path(relative).name == name)]
            if job['kind'] == 'multik_audit' and field == 'source_hashes' and not matches:
                relative = Path(name)
                _require(relative.is_relative_to(Path('research/informs_broad_validation_20261015_v1'))
                         and name.endswith(('.json', '.json.gz', '.csv', '.md')),
                         'multik_artifact_source_scope:'+name)
                resolve_binding(dict(reference='repo://'+name, sha256=expected), ctx)
                continue
            _require(len(matches) == 1 and matches[0] == expected, 'producer_source_seal_mismatch:'+name)


def _initialization(result, job, ctx):
    if job['kind'] != 'initialization':
        return
    _require(result.get('status') == 'COMPLETE' and result.get('converged') is True, 'DCA_COMPLETE_not_converged')
    scope = result.get('fit_scope', {})
    _require(scope.get('name') == job.get('scope') and scope.get('start') == 0
             and scope.get('fit_end', 0) > 10 and scope.get('covariance_rows') == scope['fit_end']
             and scope.get('lag_pair_max_row') == scope['fit_end']-1, 'DCA_fitting_boundary_contract')
    artifact = dict(reference='cache://'+result['cache_binding'], sha256=result['cache_sha256'])
    path = resolve_binding(artifact, ctx)
    with np.load(path, allow_pickle=False) as arrays:
        e64, e32 = arrays['E64'], arrays['E32']
        _require(e64.dtype == np.float64 and e32.dtype == np.float32 and np.array_equal(e64.astype(np.float32), e32),
                 'DCA_E64_E32_dtype_cast_parity')
        _require(e32.ndim == 2 and np.isfinite(e32).all() and np.linalg.matrix_rank(e32.astype(np.float64)) == e32.shape[1],
                 'DCA_initialization_rank_or_finite_values')
        _require(_encoded_sha(e32) == result['diagnostics']['E32_sha256'], 'DCA_saved_E32_fingerprint')
    receipt = read(path.with_name('receipt.json'))
    identity = receipt.get('identity', {})
    _require(receipt.get('public') == {key: value for key, value in result.items() if key != 'private_artifacts'}
             and identity.get('key') == result.get('identity_sha256')
             and digest(identity.get('binding')) == identity.get('key'), 'DCA_scope_cache_receipt_identity')
    _require(identity['binding']['fit_scope'] == scope, 'DCA_scope_cache_fitting_range')


def _selection(result, job, ctx):
    if job['kind'] not in SCORED_KINDS or job['kind'] == 'reuse_pca':
        return
    rows = result.get('rows', [])
    if not rows:
        return
    if job['kind'] == 'training':
        _require(result.get('selection_sealed_before_V') is True, 'training_selection_not_sealed_before_V')
        fit = result.get('fit_receipt', {})
        _require(int(fit.get('search_fit_end', 0)) < int(fit.get('refit_fit_end', 0)), 'training_A_fit_A60_scopes_not_separate')
        for row in rows:
            _require(row.get('fit') == 'fresh_A60' and row.get('selection_split') == 'A_stop', 'training_selection_scope_mismatch')
        if job.get('init_kind') == 'DCA':
            init = result.get('initialization_binding', {})
            _require(init.get('status') == 'COMPLETE' and init.get('converged') is True
                     and int(init.get('fit_scope', {}).get('fit_end', -1)) == int(fit['refit_fit_end']),
                     'DCA_selected_refit_initialization_scope_or_convergence')
            _require(bool(init.get('cache_sha256')), 'DCA_initialization_cache_binding_missing')
        return
    seals = [row['selection_seal'] for row in rows if row.get('selection_seal')]
    _require(len(seals) == len(rows) and all(value == seals[0] for value in seals), 'selection_seal_missing_or_inconsistent')
    seal = read(resolve_binding(seals[0], ctx))
    body = {key: value for key, value in seal.items() if key != 'selection_sha256'}
    _require(digest(body) == seal.get('selection_sha256'), 'selection_seal_digest_mismatch')
    _require(seal.get('V_features_loaded', seal.get('V_loaded', seal.get('selection', {}).get('V_loaded', False))) is False,
             'V_was_loaded_before_selection_seal')
    if job['kind'] in {'decoder', 'decoder_transfer'}:
        trials = result.get('selection_trials', [])
        _require(len(trials) == 40 and len(result.get('rows', [])) == 4, 'decoder_registered_40_grid_or_four_arms_missing')
        expected_arms = {'D_SHARED', 'D_HORIZON', 'D_LINEAR', 'D_GAIN'}
        _require({row.get('arm') for row in rows} == expected_arms, 'decoder_four_arm_identity')
        grid = {(arm, lam, gamma) for arm in expected_arms for lam in (.0001, .001, .01, .1, 1.)
                for gamma in ((1., 10., 100.) if arm in {'D_LINEAR', 'D_GAIN'} else (1.,))}
        _require({(t['arm'], t['lambda_value'], t.get('gamma', 1.)) for t in trials} == grid,
                 'decoder_registered_grid_values_changed')
        _require(set(result.get('selections', {})) == expected_arms, 'decoder_selection_arm_missing')
        _require(seal.get('scientific_identity') == job['scientific_identity'], 'selection_scientific_identity_mismatch')
        for arm, selected in result.get('selections', {}).items():
            candidates = [trial for trial in trials if trial.get('arm') == arm]
            chosen = min(candidates, key=lambda trial: (trial['MSE'], trial['lambda_value'], trial.get('gamma', 1.)))
            _require(all(chosen[key] == selected[key] for key in ('MSE', 'lambda_value', 'gamma')), 'decoder_exact_selection_mismatch')
        for row in rows:
            selected = result['selections'][row['arm']]
            _require(row['lambda_value'] == selected['lambda_value'] and row['gamma'] == selected['gamma'],
                     'decoder_refit_configuration_not_selected')
    if job['kind'] == 'lift_jobs':
        _require(result.get('selection_sealed_before_V') is True and result.get('new_F_calls') == 0,
                 'LIFT_selection_or_extra_foundation_calls')
        _require({row.get('method') for row in rows} == {'B2_LIFT', 'B2_OUTPUT_GAIN'}, 'LIFT_same_base_gain_pair_missing')


def _cost(row, proofs, ctx):
    if not row.get('cost_valid'):
        _require(row.get('status') != 'COMPLETE', 'complete_cost_marked_invalid')
        return dict(status='UNVERIFIED', reason='route_not_valid_for_ranking')
    _require(row.get('status') == 'COMPLETE', 'cost_valid_false_status')
    artifact = row.get('accuracy_prediction_binding')
    _require(isinstance(artifact, dict), 'cost_missing_exact_accuracy_archive')
    resolve_binding(artifact, ctx)
    matches = [p for p in proofs if p['prediction_reference'] == artifact['reference']
               and p['prediction_sha256'] == artifact['sha256']
               and p['source_job_id'] == row.get('accuracy_job_id')
               and p['method'] == row.get('method') and p['evaluation'] == 'V_select']
    _require(len(matches) == 1, 'cost_accuracy_job_method_archive_join_mismatch')
    proof = matches[0]
    _require(row.get('accuracy_run_id') == proof['source_run_id'], 'cost_accuracy_run_ID_mismatch')
    for name in ('saved_V_parity', 'serialized_reload_saved_V_parity', 'serialized_reload_parity'):
        parity = row.get(name, {})
        _require(parity.get('status') == 'PASS' and parity.get('n_origins') == proof['statistics']['n_origins'],
                 name+'_not_full_saved_V_PASS')
        _require(float(parity.get('atol', 1e-5)) == 1e-5 and float(parity.get('rtol', 1e-4)) == 1e-4,
                 'unregistered_FP32_route_tolerance')
    _require(row.get('warmups') == 10 and row.get('timed_trials') == 20, 'cost_warmup_timed_protocol_mismatch')
    _require(row.get('batch_size') in (1, 8), 'cost_unregistered_batch')
    _require(row.get('backbone_dtype') == 'float32', 'cost_foundation_dtype_changed')
    _require(row.get('head_dtype') == proof['head_dtype'] and row.get('prediction_dtype') == proof['prediction_dtype'],
             'cost_scored_head_prediction_dtype_mismatch')
    k = row.get('C') if row.get('method') in {'RAW', 'LORA'} else row.get('K')
    _require(k is not None and row.get('F_rows_per_timed_batch') == [int(k)*int(row['batch_size'])], 'cost_timed_F_rows_mismatch')
    _require(len(row.get('cold_observations', [])) == 2 and bool(row.get('whole_path')),
             'cost_cold_warm_scope_missing')
    if '_source_trials' in row:
        trials = [t for t in row['_source_trials'] if t.get('accuracy_run_id', t.get('run_id')) == row['accuracy_run_id']
                  and int(t['batch_size']) == row['batch_size']]
        warm = [t for t in trials if t['phase'] == 'warmup']
        timed = [t for t in trials if t['phase'] == 'timed']
        _require(len(warm) == 10 and len(timed) == 20, 'raw_cost_trial_warmup_timed_counts')
        _require(all(int(t['F_logical_streams']) == int(k)*row['batch_size'] for t in timed), 'raw_cost_trial_F_rows')
        milliseconds = np.asarray([float(t['whole_seconds'])*1000 for t in timed], np.float64)
        _require(np.isfinite(milliseconds).all() and np.all(milliseconds > 0), 'raw_cost_trial_invalid_duration')
        q25, median, q75, p95 = np.quantile(milliseconds, [.25, .5, .75, .95])
        summaries = dict(median_batch_ms=median, IQR_batch_ms=q75-q25, p95_batch_ms=p95,
                         median_ms_per_origin=median/row['batch_size'], IQR_ms_per_origin=(q75-q25)/row['batch_size'],
                         p95_ms_per_origin=p95/row['batch_size'])
        for field, expected in summaries.items():
            _require(field in row and abs(float(row[field])-float(expected)) <= 1e-12, 'raw_cost_'+field+'_mismatch')
    for cost_key, proof_key in [('accuracy_bundle_binding', 'checkpoint_binding'),
                                ('accuracy_base_checkpoint_binding', 'base_checkpoint'),
                                ('accuracy_refiner_binding', 'calibrator_binding')]:
        expected, actual = proof.get(proof_key), row.get(cost_key)
        if expected is not None:
            _require(actual == expected, 'cost_'+cost_key+'_join_mismatch')
    if proof.get('parent_PE_checkpoint_sha256'):
        parent = row.get('accuracy_bundle_binding') or row.get('accuracy_base_checkpoint_binding')
        _require(parent and parent['sha256'] == proof['parent_PE_checkpoint_sha256'], 'cost_decoder_parent_PE_checkpoint_mismatch')
    if proof.get('state_binding'):
        _require(proof['state_binding'] in row.get('_source_private_artifacts', []), 'cost_exact_decoder_head_state_missing')
    return dict(status='PASS', accuracy_job_id=proof['source_job_id'], prediction_sha256=artifact['sha256'])


def _historical_cost_proof(row, ctx):
    """Independently read an old Raw manifest, including its published gzip."""
    import gzip
    artifact = row['accuracy_prediction_binding']
    manifests = []
    for item in row.get('accuracy_source_evidence', []):
        path = resolve_binding(item, ctx)
        if path.suffix == '.gz':
            raw = gzip.decompress(path.read_bytes())
            value = json.loads(raw)
            if 'archive_sha256' in value:
                manifests.append(value)
        elif path.suffix == '.json':
            value = read(path)
            if 'archive_sha256' in value:
                manifests.append(value)
    matching = [value for value in manifests if value['archive_sha256'] == artifact['sha256']]
    _require(len(matching) == 1, 'historical_cost_accuracy_manifest_not_bound')
    manifest = matching[0]
    _require(manifest.get('split') == 'V_select', 'historical_cost_is_not_saved_V')
    path = resolve_binding(artifact, ctx)
    with np.load(path, allow_pickle=False) as archive:
        stats = independent_statistics(archive['pred_normalized'], archive['target_normalized'], archive['observed'])
    synthetic = dict(stats, method=row['method'], stage='selected_refit', evaluation='V_select',
                     run_id=row.get('accuracy_run_id'), C=row.get('C'), H=row.get('H'), K=row.get('K'), L=row.get('L'),
                     prediction_binding=artifact, array_fingerprints=manifest.get('arrays', {}),
                     scaler_sha256=manifest.get('scaler_sha'), model_sha=row.get('accuracy_foundation_sha256'),
                     dataset=row.get('dataset'), seed=row.get('seed'), units=row.get('accuracy_units', 'normalized_A60'))
    for key in ('MSE', 'MAE'):
        if key in manifest:
            _require(abs(float(manifest[key])-stats[key]) <= 1e-12, 'historical_manifest_'+key+'_mismatch')
    return _proof(synthetic, {}, dict(job_id=row['accuracy_job_id'], kind='historical',
                  scientific_identity='manifest:'+artifact['sha256']), ctx)


def _safe_reason(error, ctx):
    text = str(error)
    for name in ('repo', 'cache', 'previous_cache', 'asset', 'report'):
        if name in ctx:
            text = text.replace(str(Path(ctx[name]).resolve()), name+'://')
    text = re.sub(r'[A-Za-z]:[\\/][^\s\"\']+', '[private-path]', text)
    text = re.sub(r'/(?:home|Users|tmp|mnt)/[^\s\"\']+', '[private-path]', text)
    return type(error).__name__+': '+text[:260]


class _Checks:
    def __init__(self, ctx):
        self.ctx, self.values = ctx, []

    def add(self, identifier, status, **details):
        self.values.append(dict(id=identifier, status=status, **details))

    def run(self, identifier, function):
        try:
            result = function()
            self.add(identifier, 'PASS')
            return result
        except Exception as error:
            self.add(identifier, 'FAIL', reason=_safe_reason(error, self.ctx))
            return None


def _protected(ctx):
    path = Path(ctx['cache'])/'PROTECTED_PARENT_FILES.json'
    inventory, start = read(path), read(Path(ctx['report'])/'START_STATE.json')
    _require(start.get('protected_parent_inventory_sha256') == sha(path), 'protected_inventory_anchor_mismatch')
    _require(start.get('protected_historical_file_count') == len(inventory), 'protected_inventory_count_mismatch')
    root = Path(ctx['repo']).resolve()
    for relative, expected in inventory.items():
        target = (root/relative).resolve()
        _require(target.is_relative_to(root) and target.is_file() and sha(target) == expected, 'protected_parent_file_changed:'+relative)
    return dict(file_count=len(inventory), inventory_sha256=sha(path))


def _resources(ctx, plan):
    cache = Path(ctx['cache'])
    ledger = read(cache/'CAMPAIGN.json')
    _require(ledger.get('campaign') == NAME and ledger['started_unix'] == plan['campaign_started_unix'], 'persistent_campaign_start_mismatch')
    _require(ledger['limits'] == plan['limits'], 'persistent_resource_limits_changed')
    charges, seen = [], set()
    journal = cache/'GPU_USAGE.jsonl'
    for line in journal.read_text(encoding='utf-8').splitlines() if journal.exists() else []:
        row = json.loads(line)
        identifier, seconds = row.get('lease_id'), float(row['charged_process_seconds'])
        _require(identifier and identifier not in seen and np.isfinite(seconds) and seconds >= 0, 'duplicate_or_invalid_GPU_charge')
        seen.add(identifier)
        charges.append(seconds)
    gpu = math.fsum(charges)
    _require(abs(gpu-float(ledger['gpu_seconds'])) <= 1e-6, 'GPU_journal_ledger_mismatch')
    _require(gpu <= float(ledger['limits']['gpu_seconds']), 'GPU_budget_exceeded')
    size = 0
    for path in cache.rglob('*'):
        try:
            if path.is_file():
                size += path.stat().st_size
        except FileNotFoundError:
            # A live worker may atomically replace its pending checkpoint.
            continue
    _require(size <= int(ledger['limits']['cache_bytes']), 'new_cache_budget_exceeded')
    lineage = ledger.get('lineage', [])
    verify_references(lineage, ctx)
    download = int(ledger.get('download_reserved_bytes', 0))
    manifest = Path(ctx['report'])/'OFFICIAL_SOURCE_MANIFEST.json'
    if manifest.exists():
        download += sum(int(s.get('download_bytes', 0)) for s in read(manifest).get('sources', []))
    usage = cache/'DOWNLOAD_USAGE.jsonl'
    if usage.exists():
        download += sum(int(json.loads(line)['reserved_bytes']) for line in usage.read_text().splitlines())
    _require(download <= int(ledger['limits']['download_bytes']), 'download_budget_exceeded')
    return dict(gpu_seconds=gpu, gpu_journal_entries=len(charges), new_cache_bytes=size,
                conservatively_accounted_download_bytes=download, lineage_entries=len(lineage),
                elapsed_wall_seconds=max(0., time.time()-ledger['started_unix']),
                wall_limit_seconds=ledger['limits']['wall_seconds'], active_GPU_lease=(cache/'ACTIVE_GPU_LEASE.json').exists())


def _csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _cell(value):
    if value in ('', None):
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def _csv_integer(value):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError('public_integer_metadata_missing') from error
    _require(number.is_finite() and number == number.to_integral_value(), 'public_integer_metadata_invalid')
    return int(number)


def _paired_origins(report, by_id, verified):
    comparison_path = report/'COMPARISONS.csv'
    _require(comparison_path.is_file(), 'paired_comparison_inventory_missing')
    all_comparisons = _csv_rows(comparison_path)
    comparison_ids = [row.get('comparison_id') for row in all_comparisons]
    _require(len(set(comparison_ids)) == len(comparison_ids), 'paired_comparison_ID_duplicated')
    expected = {row['comparison_id']: row for row in all_comparisons
                if row.get('match_status') == 'MATCHED' and row.get('stage') == 'selected_refit'
                and row.get('evaluation') == 'V_select'}
    seen = {key: set() for key in expected}
    lengths = {}

    def integer(value):
        try:
            return _csv_integer(value)
        except ValueError as error:
            raise ValueError('paired_integer_metadata_invalid') from error

    def metric(value, expected_value):
        if expected_value is None:
            _require(value in ('', None, 'UNKNOWN', 'UNDEFINED'), 'paired_undefined_metric_was_zero_filled')
        else:
            _require(value not in ('', None, 'UNKNOWN', 'UNDEFINED') and
                     np.isfinite(float(value)) and abs(float(value)-expected_value) <= 1e-12, 'paired_metric_or_delta_mismatch')

    for row in _csv_rows(report/'PAIRED_ORIGIN_EFFECTS.csv'):
        comparison_id = row.get('comparison_id')
        _require(comparison_id in expected, 'paired_unregistered_or_unmatched_comparison_ID')
        comparison = expected[comparison_id]
        for field in ('candidate_row_id', 'control_row_id', 'candidate', 'control', 'cohort', 'seed'):
            _require(str(row.get(field)) == str(comparison.get(field)), 'paired_comparison_'+field+'_join_mismatch')
        origin = integer(row.get('origin_index'))
        _require(origin not in seen[comparison_id], 'paired_duplicate_origin_index')
        seen[comparison_id].add(origin)
        scored = []
        for label in ('candidate', 'control'):
            run_row = by_id.get(row[label+'_row_id'])
            _require(run_row is not None and run_row.get('stage') == 'selected_refit'
                     and run_row.get('evaluation') == 'V_select', 'paired_RUNS_scope_mismatch')
            bound = _cell(run_row.get('prediction_binding'))
            proof = verified.get((run_row.get('job_id'), bound.get('reference'), bound.get('sha256'))) if isinstance(bound, dict) else None
            _require(proof is not None, 'paired_saved_prediction_proof_missing')
            n_origins = proof['statistics']['n_origins']
            lengths[comparison_id] = n_origins
            _require(0 <= origin < n_origins, 'paired_origin_index_outside_saved_cohort')
            _require(integer(row.get('n_origins')) == n_origins == integer(run_row.get('n_origins')),
                     'paired_n_origins_mismatch')
            channels = len(proof['statistics']['channel_counts'])
            _require(integer(row.get('C')) == channels == integer(run_row.get('C')), 'paired_registered_C_mismatch')
            _require(integer(row.get('evaluation_stride')) == proof['H'] == integer(run_row.get('evaluation_stride')),
                     'paired_evaluation_stride_mismatch')
            _require(row.get('horizon_overlap') in (False, 'False', 'false', '0', 0), 'paired_horizon_overlap_mismatch')
            if comparison.get('horizon_overlap') not in (None, ''):
                _require(comparison['horizon_overlap'] in (False, 'False', 'false', '0', 0), 'paired_comparison_horizon_overlap_mismatch')
            for field, expected_value in [('n_origins', n_origins), ('evaluation_stride', proof['H'])]:
                if comparison.get(field) not in (None, '', 'UNKNOWN'):
                    _require(integer(comparison[field]) == expected_value, 'paired_comparison_'+field+'_mismatch')
            _require(str(row.get(label)) == str(run_row.get('method')) and
                     str(row.get('cohort')) == str(run_row.get('cohort')) and
                     str(row.get('seed')) == str(run_row.get('seed')), 'paired_RUNS_method_cohort_seed_mismatch')
            for field in ('source_file', 'source_sha256', 'source_collection', 'source_row_index'):
                _require(str(row.get(label+'_'+field)) == str(run_row.get(field)), 'paired_'+label+'_'+field+'_mismatch')
            for field, value in [('prediction_reference', proof['prediction_reference']),
                                 ('prediction_sha256', proof['prediction_sha256'])]:
                _require(row.get(label+'_'+field) == value, 'paired_'+label+'_'+field+'_mismatch')
            origin_stats = proof['statistics']['origin_statistics'][origin]
            _require(len(origin_stats['channel_counts']) == channels, 'paired_origin_channel_count_axes')
            _require(row.get(label+'_status') == origin_stats['status'] and
                     integer(row.get(label+'_coverage_channels')) == origin_stats['coverage_channels'],
                     'paired_origin_status_or_coverage_mismatch')
            for name in ('MSE', 'MAE'):
                metric(row.get(label+'_'+name), origin_stats[name])
            scored.append(origin_stats)
        left, right = scored
        _require(left['channel_counts'] == right['channel_counts'], 'paired_origin_observation_counts_mismatch')
        defined = left['status'] == right['status'] == 'DEFINED'
        _require(row.get('status') == ('DEFINED' if defined else 'UNDEFINED'), 'paired_fixed_C_status_mismatch')
        for name in ('MSE', 'MAE'):
            metric(row.get('delta_'+name), left[name]-right[name] if defined else None)
    for comparison_id, origins in seen.items():
        _require(comparison_id in lengths and origins == set(range(lengths[comparison_id])), 'paired_origin_coverage_incomplete')
    return dict(comparisons=len(expected), origins=sum(len(indices) for indices in seen.values()),
                global_metric_rule='Global channel SSE/count remains independent of mean origin macro.')


def _public_tables(ctx, proofs, costs):
    report = Path(ctx['report'])
    path = report/'RUNS.csv'
    if not path.exists():
        return dict(status='UNVERIFIED', reason='RUNS.csv_missing')
    verified = {(p['source_job_id'], p['prediction_reference'], p['prediction_sha256']): p for p in proofs}
    rows = _csv_rows(path)
    by_id = {}
    for row in rows:
        if row.get('row_id'):
            _require(row['row_id'] not in by_id, 'duplicate_RUNS_row_id')
            by_id[row['row_id']] = row
        artifact = _cell(row.get('prediction_binding'))
        if isinstance(artifact, dict):
            proof = verified.get((row.get('job_id'), artifact.get('reference'), artifact.get('sha256')))
            _require(proof is not None, 'RUNS_scored_archive_not_independently_verified')
            for key in ('MSE', 'MAE'):
                _require(row.get(key) not in ('', None) and abs(float(row[key])-proof['statistics'][key]) <= 1e-12,
                         'RUNS_'+key+'_mismatch')
            for field, label in [('target_sha256', 'target'), ('mask_sha256', 'mask'),
                                 ('origins_sha256', 'origins'), ('channel_sha256', 'channel_ids')]:
                if row.get('signature_proof') == 'PASS':
                    _require(row.get(field) == digest(proof['canonical_cohort_signature'][label]),
                             'RUNS_canonical_'+field+'_mismatch')
        if row.get('source_file') and row.get('source_sha256'):
            source_ref = row['source_file']
            if '://' in source_ref:
                source = resolve_binding(dict(reference=source_ref, sha256=row['source_sha256']), ctx)
            else:
                source = (Path(ctx['repo'])/source_ref).resolve()
            if not source.is_file():
                source = (report/row['source_file']).resolve()
            _require(source.is_relative_to(Path(ctx['repo']).resolve()) or source.is_relative_to(report.resolve()), 'RUNS_source_escapes_root')
            _require(source.is_file() and sha(source) == row['source_sha256'], 'RUNS_source_JSON_hash_mismatch')
            if source.suffix == '.json' and row.get('source_collection') in ('rows', 'common_endpoint_rows'):
                raw = read(source)[row['source_collection']][_csv_integer(row['source_row_index'])]
                for key in ('MSE', 'MAE'):
                    if raw.get(key) is not None:
                        _require(row.get(key) not in ('', None, 'UNKNOWN') and abs(float(row[key])-float(raw[key])) <= 1e-12,
                                 'RUNS_source_row_'+key+'_mismatch')
    comparisons = report/'COMPARISONS.csv'
    if comparisons.exists():
        for row in _csv_rows(comparisons):
            if row.get('match_status') != 'MATCHED':
                continue
            a, b = by_id.get(row.get('candidate_row_id')), by_id.get(row.get('control_row_id'))
            _require(a is not None and b is not None, 'comparison_row_join_missing')
            def proof_for(run_row):
                bound = _cell(run_row.get('prediction_binding'))
                return verified.get((run_row.get('job_id'), bound.get('reference'), bound.get('sha256'))) if isinstance(bound, dict) else None
            ap, bp = proof_for(a), proof_for(b)
            _require(ap is not None and bp is not None, 'MATCHED_comparison_requires_two_verified_saved_predictions')
            _require(ap['canonical_cohort_signature'] == bp['canonical_cohort_signature'], 'MATCHED_cohort_mismatch')
            for field in ('scaler_sha256_ifproven', 'F_snapshot_ifproven'):
                _require(ap.get(field) is not None and ap[field] == bp.get(field), 'MATCHED_missing_or_different_'+field)
            for field in ('cohort', 'seed', 'C', 'K', 'L', 'H', 'stage', 'evaluation', 'device',
                          'core_dtype', 'point_statistic', 'recipe_signature', 'backend_signature', 'source_signature'):
                _require(a.get(field) not in ('', None, 'UNKNOWN') and a[field] == b.get(field), 'MATCHED_'+field+'_mismatch')
            if a.get('stage') == 'search':
                for field in ('epoch', 'updates', 'exposures', 'sampler_sha256'):
                    _require(ap.get(field) is not None and ap[field] == bp.get(field), 'same_update_missing_or_different_'+field)
            for key in ('MSE', 'MAE'):
                if a.get(key) not in ('', None) and b.get(key) not in ('', None):
                    delta = float(a[key])-float(b[key])
                    _require(abs(float(row['delta_'+key])-delta) <= 1e-12, 'comparison_delta_'+key+'_mismatch')
                    for percent_name in ('delta_'+key+'_pct', 'pct_'+key, 'delta_pct_'+key):
                        if row.get(percent_name) not in ('', None) and float(b[key]) != 0:
                            _require(abs(float(row[percent_name])-100*delta/float(b[key])) <= 1e-10, 'comparison_percent_sign_or_value')
    statistics_path = report/'CHANNEL_HORIZON_STATS.csv'
    if statistics_path.exists():
        seen = set()
        for row in _csv_rows(statistics_path):
            run_row = by_id.get(row.get('row_id'))
            _require(run_row is not None, 'statistics_RUNS_row_join_missing')
            if row.get('record_origin') not in (None, ''):
                _require(row['record_origin'] == run_row.get('record_origin'), 'statistics_record_origin_mismatch')
            bound = _cell(run_row.get('prediction_binding'))
            if not isinstance(bound, dict):
                _require(run_row.get('record_origin') == 'CONTEXTUAL', 'statistics_without_saved_prediction_proof')
                continue
            proof = verified.get((run_row.get('job_id'), bound.get('reference'), bound.get('sha256')))
            _require(proof is not None, 'statistics_archive_not_verified')
            h, c = _csv_integer(row['horizon'])-1, _csv_integer(row['channel_id'])
            key = row['row_id'], h, c
            _require(key not in seen, 'duplicate_channel_horizon_statistics')
            seen.add(key)
            stat = proof['statistics']
            _require(0 <= h < len(stat['horizon_counts']) and 0 <= c < len(stat['channel_counts']), 'statistics_axis_outside_registered_H_C')
            for label, field in [('SSE', 'horizon_sse'), ('SAE', 'horizon_sae'), ('count', 'horizon_counts')]:
                _require(np.isclose(float(row[label]), stat[field][h][c], atol=1e-12, rtol=1e-12), 'public_statistics_'+label+'_mismatch')
        for row_id, run_row in by_id.items():
            if any(key[0] == row_id for key in seen):
                _require(sum(key[0] == row_id for key in seen) == _csv_integer(run_row['H'])*_csv_integer(run_row['C']),
                         'incomplete_channel_horizon_statistics')
    paired_path = report/'PAIRED_ORIGIN_EFFECTS.csv'
    paired_audit = None
    if paired_path.exists():
        paired_audit = _paired_origins(report, by_id, verified)
    cost_csv = report/'COST.csv'
    if cost_csv.exists():
        valid = {(row.get('_source_job_id'), row.get('accuracy_job_id'), row.get('method'),
                  row.get('device'), str(row.get('batch_size'))): row
                 for row in costs if row.get('cost_valid') and row.get('_verification_status') == 'PASS'}
        for row in _csv_rows(cost_csv):
            if _cell(row.get('cost_valid')) in (True, 'True', 'true', 1):
                producer_method = 'RAW' if row.get('method') == 'Raw' else row.get('method')
                key = (row.get('job_id'), row.get('accuracy_job_id'), producer_method,
                       row.get('device'), row.get('batch_size'))
                _require(key in valid,
                         'ranked_COST_row_has_no_valid_exact_route')
                original = valid[key]
                for key in ('median_ms_per_origin', 'p95_ms_per_origin', 'median_batch_ms'):
                    if key in original:
                        _require(row.get(key) not in ('', None, 'UNKNOWN') and abs(float(row[key])-float(original[key])) <= 1e-12,
                                 'COST_'+key+'_mismatch')
    return dict(rows=len(rows), comparisons_checked=comparisons.exists(), cost_table_checked=cost_csv.exists(),
                paired_origin_table_checked=paired_path.exists(), paired_origin_statistics=paired_audit)


def _figures(ctx):
    report = Path(ctx['report'])
    index = read(report/'FIGURE_INDEX.json')
    if index.get('reporting_source_sha256'):
        reporting = Path(ctx['repo'])/'research'/NAME/'reporting.py'
        _require(reporting.is_file() and sha(reporting) == index['reporting_source_sha256'],
                 'separate_reporting_source_seal_mismatch')
    verified, missing = [], []
    figures = index.get('figures', index.get('files', []))
    _require(bool(figures), 'figure_index_has_no_entries')
    for figure in figures:
        if figure.get('status') not in ('COMPLETE', 'PASS', 'GENERATED', 'AVAILABLE'):
            _require(bool(figure.get('reason')), 'missing_figure_reason_required')
            missing.append(figure.get('figure_id'))
            continue
        paths = {}
        for key in ('png', 'svg', 'source_csv'):
            path = (report/figure[key]).resolve()
            _require(path.is_relative_to(report.resolve()) and path.is_file() and path.stat().st_size > 0,
                     'figure_'+key+'_missing_or_escaping')
            paths[key] = path
        _require(sha(paths['source_csv']) == figure['source_sha256'], 'figure_source_CSV_hash_mismatch')
        rows = _csv_rows(paths['source_csv'])
        _require(len(rows) == int(figure['n_rows']), 'figure_source_row_count_mismatch')
        _require(paths['png'].read_bytes().startswith(b'\x89PNG\r\n\x1a\n') and '<svg' in paths['svg'].read_text(encoding='utf-8'),
                 'figure_not_valid_PNG_SVG_container')
        verified.append(figure.get('figure_id'))
    return dict(verified_figure_ids=verified, missing_figure_ids=missing,
                visual_labels_and_public_regeneration='UNVERIFIED_requires_separate_render_and_replay_evidence')


def _manifest(ctx):
    root = Path(ctx['repo']).resolve()
    report = Path(ctx['report']).resolve()
    manifest_path = report/'ARTIFACT_MANIFEST.json'
    manifest, seen = read(manifest_path), set()
    for item in manifest.get('files', []):
        relative = item['path']
        path = (root/relative).resolve()
        _require(relative not in seen and not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                 'manifest_duplicate_or_escaping_path')
        seen.add(relative)
        _require(path.is_relative_to(root) and path.is_file() and not (root/relative).is_symlink(), 'manifest_not_ordinary_workspace_file')
        _require(path not in (manifest_path, report/'VALIDATION_REPORT.json'), 'manifest_self_validation_hash_cycle')
        _require('.cache' not in Path(relative).parts and path.suffix.lower() not in {'.pt', '.pth', '.npz', '.safetensors', '.bin'},
                 'manifest_contains_private_model_or_array_asset')
        _require(sha(path) == item['sha256'] and path.stat().st_size == item['bytes'], 'manifest_file_hash_or_size_mismatch')
    _require(bool(seen), 'manifest_has_no_public_files')
    return dict(public_files=len(seen), manifest_sha256=sha(manifest_path), self_and_validation_excluded=True)


def _compact_proof(proof):
    """Summarize only after every full-statistics audit has finished."""
    full = proof['statistics']
    scalar_fields = ('MSE', 'MAE', 'status', 'coverage_channels', 'registered_channels',
                     'observed', 'n_origins', 'metric_accumulation_dtype')
    summary = {key: full[key] for key in scalar_fields if key in full}
    if 'bands' in full:
        band_fields = ('horizon_start', 'horizon_end', 'MSE', 'MAE', 'status',
                       'coverage_channels', 'registered_channels')
        summary['bands'] = [{key: band[key] for key in band_fields if key in band} for band in full['bands']]

    def dimensions(value):
        shape = []
        while isinstance(value, (list, tuple)):
            shape.append(len(value))
            if not value:
                break
            value = value[0]
        return shape

    blocks = {key: dict(sha256=digest(value), shape=dimensions(value))
              for key, value in full.items() if key not in scalar_fields}
    return dict(proof, statistics=summary, full_statistics_sha256=digest(full),
                statistics_blocks=blocks,
                statistics_serialization='summary_after_full_audit; canonical_JSON_SHA256_v1')


def verify(ctx, *, write=True):
    """Audit immutable evidence; optionally publish this audit, never run science."""
    ctx = dict(ctx)
    report = Path(ctx['report'])
    checks, proofs, results, cost_rows = _Checks(ctx), [], {}, []
    plan_path = Path(ctx.get('plan_path', report/'EXPERIMENT_PLAN.json'))
    plan = checks.run('plan:read', lambda: read(plan_path))
    plans = [] if plan is None else [(plan_path, plan)]
    for supplement in sorted((report/'SUPPLEMENTAL_PLANS').glob('*.json')):
        child = checks.run('supplemental_plan:read:'+supplement.stem, lambda p=supplement: read(p))
        if child is not None:
            plans.append((supplement, child))
    seen_jobs, job_plans = set(), []
    for source_path, own_plan in plans:
        def validate(p=own_plan):
            from .run import validate_context_plan
            actual = dict(ctx, ledger=read(Path(ctx['cache'])/'CAMPAIGN.json'))
            validate_plan(p, ctx['repo'])
            validate_context_plan(p, actual)
            return dict(plan_sha256=p['plan_sha256'], scientific_sources=len(p['source_hashes']))
        identifier = 'plan:scientific_controller_and_resume_guards' if source_path == plan_path else 'supplemental_plan:guards:'+source_path.stem
        checks.run(identifier, validate)
        for job in own_plan.get('jobs', []):
            if job['job_id'] in seen_jobs:
                checks.add('plan:duplicate_job:'+job['job_id'], 'FAIL', reason='job_ID_reused_across_immutable_plans')
                continue
            seen_jobs.add(job['job_id'])
            job_plans.append((job, own_plan))
    if (Path(ctx['cache'])/'PROTECTED_PARENT_FILES.json').exists():
        checks.run('parent:protected_bytes', lambda: _protected(ctx))
    else:
        checks.add('parent:protected_bytes', 'UNVERIFIED', reason='protected_inventory_missing')
    resource_audit = None
    if plan is not None:
        resource_audit = checks.run('resources:persistent_ledger_journal_and_limits', lambda: _resources(ctx, plan))
    statuses, pending, unscored = Counter(), [], []
    for job, own_plan in job_plans:
        identifier = job['job_id']
        path = report/'receipts'/(identifier+'.json')
        if not path.exists():
            statuses['PENDING'] += 1
            pending.append(identifier)
            checks.add('receipt:'+identifier, 'UNVERIFIED', reason='registered_job_has_no_receipt')
            continue
        receipt = checks.run('receipt:read:'+identifier, lambda p=path: read(p))
        if receipt is None:
            continue
        status = receipt.get('status')
        statuses[str(status)] += 1
        if status in ('RUNNING', 'PLANNED'):
            pending.append(identifier)
        def receipt_check():
            _require(status in STATES, 'unregistered_receipt_status')
            _require(receipt.get('job_id') == identifier and receipt.get('scientific_identity') == job['scientific_identity'],
                     'receipt_job_scientific_identity_mismatch')
            if receipt.get('plan_sha256') is not None:
                _require(receipt['plan_sha256'] == own_plan['plan_sha256'], 'receipt_plan_identity_mismatch')
            for relative, expected in receipt.get('artifact_hashes', {}).items():
                resolve_binding(dict(reference='report://'+relative, sha256=expected), ctx)
            value = receipt.get('result', {})
            if value.get('status') is not None:
                _require(value['status'] == status, 'receipt_and_result_status_mismatch')
            if status in SUCCESS:
                for dependency in job.get('dependency_ids', []):
                    dependency_path = report/'receipts'/(dependency+'.json')
                    _require(dependency_path.is_file() and read(dependency_path).get('status') in SUCCESS,
                             'completed_job_has_incomplete_dependency')
                _require(bool(receipt.get('artifact_hashes')), 'successful_receipt_missing_artifacts')
                _require(value.get('status') == status, 'receipt_and_result_status_mismatch')
                public_job = report/'jobs'/(identifier+'.json')
                if public_job.exists() and 'jobs/'+identifier+'.json' in receipt['artifact_hashes']:
                    _require(read(public_job) == value, 'public_job_and_receipt_result_differ')
                verify_references(value, ctx)
                return value
            return {}
        result = checks.run('receipt:identity_and_artifacts:'+identifier, receipt_check)
        if not result or status not in SUCCESS:
            continue
        results[identifier] = result
        checks.run('producer_sources:'+identifier, lambda r=result, p=own_plan, j=job: _producer_sources(r, p, ctx, j))
        checks.run('initialization:'+identifier, lambda r=result, j=job: _initialization(r, j, ctx))
        checks.run('selection:'+identifier, lambda r=result, j=job: _selection(r, j, ctx))
        scientific_gates_passed = all(check['status'] != 'FAIL' for check in checks.values
                                     if check['id'] in ('producer_sources:'+identifier, 'initialization:'+identifier,
                                                        'selection:'+identifier))
        scored = list(_rows(result))
        trial_rows = []
        if job['kind'] == 'costs' or result.get('family') == 'WHOLE_PATH_COST':
            def load_trials():
                trial_files = [relative for relative in receipt.get('artifact_hashes', {}) if relative.endswith('.trials.csv')]
                if any(row.get('cost_valid') for row in result.get('rows', [])):
                    _require(len(trial_files) == 1, 'valid_cost_requires_bound_raw_trial_CSV')
                return _csv_rows(report/trial_files[0]) if len(trial_files) == 1 else []
            trial_rows = checks.run('cost_trials:'+identifier, load_trials) or []
        if job['kind'] in SCORED_KINDS and not result.get('rows'):
            checks.add('scores:'+identifier, 'FAIL', reason='successful_scored_job_has_no_results')
        for collection, index, row in scored:
            if job['kind'] == 'costs' or result.get('family') == 'WHOLE_PATH_COST':
                cost_rows.append(dict(row, _source_job_id=identifier,
                                      _source_private_artifacts=result.get('private_artifacts', []), _source_trials=trial_rows))
                continue
            if row.get('MSE') is None and row.get('MAE') is None:
                continue
            if not row.get('prediction_binding') and collection == 'common_endpoint_rows' and job['kind'] == 'reuse_pca':
                checks.add('prediction:'+identifier+':'+collection+':'+str(index), 'UNVERIFIED',
                           reason='historical_endpoint_metric_only_not_new_saved_prediction_verification')
                unscored.append(dict(job_id=identifier, collection=collection, index=index))
                continue
            proof = checks.run('prediction:'+identifier+':'+collection+':'+str(index),
                               lambda r=row, v=result, j=job: _proof(r, v, j, ctx))
            if proof is not None and scientific_gates_passed:
                proof.update(source_collection=collection, source_row_index=index)
                proofs.append(proof)
    for row in cost_rows:
        if row.get('cost_valid') and row.get('accuracy_source_kind') == 'REUSED_HISTORICAL_SCORED_RAW':
            key = (row.get('accuracy_job_id'), row.get('accuracy_prediction_binding', {}).get('sha256'))
            if not any((p['source_job_id'], p['prediction_sha256']) == key for p in proofs):
                proof = checks.run('prediction:historical_cost:'+str(row.get('accuracy_job_id')),
                                   lambda r=row: _historical_cost_proof(r, ctx))
                if proof is not None:
                    proofs.append(proof)
    cohorts = {}
    for proof in proofs:
        if proof['evaluation'] != 'V_select':
            continue
        key = (proof['dataset'], proof['scaler_sha256_ifproven'])
        previous = cohorts.setdefault(key, proof['canonical_cohort_signature'])
        checks.add('cohort:'+proof['source_job_id']+':'+str(proof['method']),
                   'PASS' if previous == proof['canonical_cohort_signature'] else 'FAIL',
                   reason='canonical_saved_V_target_mask_origins_channel_order')
    for index, row in enumerate(cost_rows):
        verified = checks.run('cost:'+row['_source_job_id']+':'+str(index), lambda r=row: _cost(r, proofs, ctx))
        row['_verification_status'] = verified.get('status') if verified else 'FAIL'
        if verified and verified.get('status') == 'UNVERIFIED':
            checks.values[-1].update(status='UNVERIFIED', reason=verified['reason'])
    public_table_audit = None
    if (report/'RUNS.csv').exists():
        checked = checks.run('public:tables_and_exact_joins', lambda: _public_tables(ctx, proofs, cost_rows))
        public_table_audit = checked
        if checked and checked.get('status') == 'UNVERIFIED':
            checks.add('public:table_reconstruction', 'UNVERIFIED', reason=checked['reason'])
    else:
        checks.add('public:tables_and_exact_joins', 'UNVERIFIED', reason='public_tables_not_generated')
    if (report/'FIGURE_INDEX.json').exists():
        checks.run('public:figure_sources_and_files', lambda: _figures(ctx))
    else:
        checks.add('public:figure_sources_and_files', 'UNVERIFIED', reason='figure_index_missing')
    if (report/'ARTIFACT_MANIFEST.json').exists():
        checks.run('public:artifact_manifest_bytes', lambda: _manifest(ctx))
    else:
        checks.add('public:artifact_manifest_bytes', 'UNVERIFIED', reason='public_manifest_missing')
    missing = [name for name in REQUIRED_PUBLIC if not (report/name).is_file()]
    checks.add('public:required_artifact_coverage', 'UNVERIFIED' if missing else 'PASS', missing=missing)
    for gate in ('public_visual_inspection', 'public_figure_regeneration', 'publication_remote_commit_and_paths'):
        checks.add(gate, 'UNVERIFIED', reason='requires_separate_actual_evidence; never inferred from local audit')
    failures = [item for item in checks.values if item['status'] == 'FAIL']
    complete = bool(plan) and all(status in SUCCESS for status in statuses) and not pending and not failures and not missing
    value = dict(schema='adapter-improvements-independent-verification-v1', checked_utc=utc(),
                 status='FAIL' if failures else 'PASS', scientific_status='COMPLETE' if complete else 'PARTIAL',
                 publication_status='UNVERIFIED', plan_sha256=(plan or {}).get('plan_sha256'),
                 supplemental_plan_sha256=[p['plan_sha256'] for path, p in plans if path != plan_path],
                 resource_accounting=resource_audit,
                 public_table_audit=public_table_audit,
                 verifier_source_sha256=sha(__file__), checks=checks.values, failures=failures,
                 unverified_gates=[item for item in checks.values if item['status'] == 'UNVERIFIED'],
                 coverage=dict(registered_jobs=len(job_plans), statuses=dict(statuses),
                               pending_job_ids=pending, verified_prediction_archives=len(proofs),
                               verified_cost_routes=sum(row['_verification_status'] == 'PASS' for row in cost_rows),
                               metric_only_historical_endpoints=unscored, missing_public_artifacts=missing),
                 verified_prediction_archives=[_compact_proof(proof) for proof in proofs],
                 interpretation='PASS validates completed bound evidence only; PARTIAL and UNVERIFIED gates remain explicit. V is reused development data.')
    if write:
        atomic_json(report/'VALIDATION_REPORT.json', value)
    return value
