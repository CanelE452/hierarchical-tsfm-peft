"""Fresh CPU-only matched readouts; import/describe does not read real assets.
Actual execution requires new human authorization and a fixed fresh manifest.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name / 'matched_readout_v1'
DATASETS = {'robin': (17, 4, 'val', 30), 'beijing_aotizhongxin': (11, 2, 'eval', 146)}
SEEDS = (92601, 92602)
CANONICAL_STAGE_REPORT_SHA = 'd32b50a75fdaeaf8c771ff25077b52e7e38ab98dc40a5be12cc6282d4badb84d'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


def safe_tag(tag):
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in tag):
        raise ValueError('FRESH_SAFE_TAG_REQUIRED')
    return tag


def by_stem(stage, stem):
    matches = [r for r in stage['artifacts'] if Path(r['path']).stem == stem]
    if len(matches) != 1:
        raise ValueError('EXACT_SINGLE_STAGE_ARTIFACT_REQUIRED:' + stem)
    return matches[0]


def restore_reused_n2(stage, probes):
    import stage_separation_math_v1 as old
    record = stage['feature_state']
    arrays, checked_feature = probes.load_checked_npz(record['artifact'])
    state = old.FeatureState(arrays['center'], arrays['scale'], record['latent_channels'],
                             record['fit_rows'], record['fit_raw_features_sha256'])
    if state.sha256 != record['sha256']:
        raise ValueError('REUSED_N2_FEATURE_STATE_SHA')
    selected = stage['families']['N2']
    arrays, checked_coefficients = probes.load_checked_npz(selected['selected_coefficient_artifact'])
    meta = selected['selected_model_metadata']['solution']
    solution = old.RidgeSolution(meta['kind'], meta['alpha'],
        *(arrays[k] for k in ('coefficients', 'intercept', 'observed_counts', 'channel_counts',
          'normal_equation_denominators', 'design_ranks', 'regularized_gram_conditions')),
        meta['feature_state_sha256'], meta['with_intercept'])
    model = old.NativeModel(state, solution)
    if solution.sha256 != meta['solution_sha256'] or model.sha256 != selected['selected_model_sha256']:
        raise ValueError('REUSED_N2_MODEL_SHA')
    return model, [checked_feature, checked_coefficients]


def solution_arrays(solution):
    return {name: getattr(solution, name) for name in ('coefficients', 'intercept', 'observed_counts',
        'channel_counts', 'normal_equation_denominators', 'design_ranks', 'regularized_gram_conditions')}


def verify_prior_audit(probes, manifest_path, stage_record, dataset, seed):
    manifest_record = probes.verified_file({'path': str(probes.resolve(manifest_path)), 'sha256': CANONICAL_STAGE_REPORT_SHA})
    manifest = probes.read(manifest_record['path'])
    if manifest.get('generator_result_status') != 'COMPLETE' or manifest.get('native_audit_gate', {}).get('passed') is not True:
        raise ValueError('COMPLETE_CANONICAL_STAGE_REPORT_REQUIRED')
    records = manifest['input_records']
    if not any(r['role'] == 'native_result' and r['sha256'].lower() == stage_record['sha256'] for r in records):
        raise ValueError('STAGE_PRODUCER_NOT_BOUND_TO_CANONICAL_REPORT')
    audit_sources = [r for r in records if r['role'] == 'native_audit']
    if len(audit_sources) != 1: raise ValueError('EXACT_ONE_CANONICAL_NATIVE_AUDIT_REQUIRED')
    audit_record = probes.verified_file({**audit_sources[0], 'sha256': audit_sources[0]['sha256'].lower()})
    audit = probes.read(audit_record['path'])
    if audit.get('status') != 'PASS' or audit.get('failed_count') != 0 or len(audit.get('conditions', [])) != 4:
        raise ValueError('ACTUAL_FOUR_CONDITION_NATIVE_AUDIT_PASS_REQUIRED')
    matches = [r for r in audit['conditions'] if r['dataset'] == dataset and r['seed'] == seed]
    if len(matches) != 1 or matches[0]['producer_sha256'] != stage_record['sha256']:
        raise ValueError('ACTUAL_NATIVE_AUDIT_PRODUCER_SHA_BINDING_REQUIRED')
    return [manifest_record, audit_record]


def verify_cached_lineage(probes, stage, dataset, seed, provenance, cp, cached_records, cached):
    import numpy as np
    record = probes.verified_file(stage['D2_source'])
    d2 = probes.read(record['path'])
    if (d2.get('status') != 'COMPLETE' or d2.get('stage') != 'D2' or d2.get('dataset') != dataset
            or d2.get('seed') != seed or d2.get('MAIN_codec_frozen_pass') is not True
            or d2.get('F_frozen_pass') is not True or d2.get('F_before') != d2.get('F_after')
            or d2.get('F_before') != {'old_algorithm': probes.OLD_F_HASH, 'bi_algorithm': probes.BI_F_HASH}
            or d2['MAIN_checkpoint']['checkpoint']['sha256'] != cp['sha256']
            or d2['bridge_provenance']['input_bridge']['sha256'] != provenance['input_bridge']['sha256']
            or {r['sha256'] for r in d2['bridge_provenance']['input_artifacts']} != {r['sha256'] for r in provenance['input_artifacts']}):
        raise ValueError('ORIGINAL_D2_CP_BRIDGE_CACHED_E_LINEAGE_REQUIRED')
    for split, arrays in cached.items():
        candidates = [r for r in d2['artifacts'] if Path(r['path']).stem == 'ABSOLUTE_LATENTS_' + split]
        if len(candidates) != 1: raise ValueError('EXACT_D2_ABSOLUTE_LATENTS_RECORD_REQUIRED')
        for name in ('latent_past', 'z_last', 'target', 'observed', 'origins', 'raw_last', 'history_observed',
                     'mean', 'std', 'raw_forecast', 'original_prediction'):
            declared = candidates[0]['arrays'][name]
            value = arrays[name]
            if (declared['shape'] != list(value.shape) or declared['dtype'] != str(value.dtype)
                    or declared['sha256'] != probes.array_sha(np.ascontiguousarray(value))
                    or declared != cached_records[split]['arrays'][name]):
                raise ValueError('D2_TO_STAGE_CACHED_ARRAY_LINEAGE:' + split + ':' + name)
    return record


def execute(args):
    import matched_readout_runtime_v1 as runtime
    limit = runtime.ContinuationLimit('matched_readout_' + safe_tag(args.tag), gpu=False)
    output = CACHE / args.tag
    result = {'status': 'RUNNING', 'dataset': args.dataset, 'seed': args.seed,
        'scientific_execution': True, 'protected_TEST_accessed': False, 'GPU_execution': False,
        'E_D_F_loaded': False, 'E_D_F_forward_replayed': False, 'E_D_F_updated': False,
        'future_oracle_read': False, 'old_campaign_restart': False, 'independent_holdout': False,
        'whole_goal_complete': False, 'families': {}, 'references': {}, 'artifacts': [],
        'source_records': [], 'source_snapshots': [], 'data_contract': {},
        'maximum_new_candidates': 14, 'old_N2_retrained': False,
        'selection': 'actual-Y normalized observed-channel macro TUNE MSE; earliest exact tie',
        'old_E33K_and_NEW_routes_have_different_input_capacity': True,
        'permission_gate': limit.authorization_record, 'run_manifest': limit.run_manifest_record}
    initialized = False
    started = time.perf_counter()
    try:
        output.mkdir(parents=True, exist_ok=False)
        initialized = True
        import numpy as np
        import prepare_inputs as inputs
        import run_condition_probes as probes
        import frozen_decoder_refit as scores_module
        import matched_readout_contract_v1 as contract
        import matched_readout_math_v1 as math
        from threadpoolctl import threadpool_limits
        inputs.install_protected_test_guard()
        full, fast = limit.checkfull, limit.shared.checkfast
        result['numpy_version'] = np.__version__
        def save(name, arrays):
            full()
            record = probes.save_npz(output / (name + '.npz'), arrays)
            result['artifacts'].append(record)
            full()
            return record
        def score(prediction, values):
            if prediction.dtype != np.float32 or prediction.shape != values['target'].shape or not np.isfinite(prediction).all():
                raise ValueError('FINITE_FP32_ACTUAL_N48C_PREDICTION_REQUIRED')
            return {'normalized': scores_module.observed_score(prediction, values['target'], values['observed']),
                    'physical_per_channel_diagnostic': scores_module.observed_score(prediction, values['target'], values['observed'], std)}
        full()
        bridge, raw_splits, mean, std, provenance = probes.load_bridge(args.input_bridge, args.dataset, inputs.verify_phase_barrier)
        result['bridge_provenance'] = provenance
        result['source_records'].extend(probes.verify_source_records(bridge))
        result['source_records'].extend(provenance['input_artifacts'] + [provenance['input_bridge']])
        result['source_records'].append(probes.check_original_raw(bridge, args.dataset, raw_splits, mean, std))
        for split, values in raw_splits.items():
            full()
            result['data_contract'][split] = contract.validate_raw_split(args.dataset, split, values, mean, std)
            save('RAW_INPUT_' + split, {**values, 'mean': mean, 'std': std})
            result['references'][split] = {'RAW_BOLT': score(values['raw_forecast'], values)}
        sources = (Path(__file__), HERE/'matched_readout_math_v1.py', HERE/'matched_readout_contract_v1.py',
                   HERE/'matched_readout_runtime_v1.py', HERE/'stage_separation_math_v1.py',
                   HERE/'run_condition_probes.py', HERE/'prepare_inputs.py', HERE/'frozen_decoder_refit.py',
                   HERE/'fast_storage_scan_v1.py', HERE/'PLAN_MATCHED_READOUT_INFORMATION_KO_v1.md',
                   HERE/'METHOD_LOCK_MATCHED_READOUT_v1.md')
        folder = output / 'sources'; folder.mkdir()
        for path in sources:
            full()
            digest = probes.sha(path)
            destination = folder / path.name
            with path.open('rb') as source, destination.open('xb') as snapshot:
                shutil.copyfileobj(source, snapshot)
            if probes.sha(destination) != digest:
                raise ValueError('SOURCE_SNAPSHOT_SHA')
            result['source_snapshots'].append({'path': str(path), 'sha256': digest, 'snapshot_path': str(destination)})

        def fit_route(route, past_by_split):
            row = {'status': 'RUNNING', 'new_training': True, 'diagnostic_only': False,
                   'alpha_grid': [], 'splits': {}, 'selected_artifacts': {}}
            result['families'][route] = row
            start = time.perf_counter()
            try:
                full()
                factor_start = time.perf_counter()
                with threadpool_limits(limits=2):
                    prepared = math.prepare_route(route, past_by_split['fit'], raw_splits['fit']['target'],
                        raw_splits['fit']['raw_last'], raw_splits['fit']['observed'], check=fast)
                row['new_FIT_factorization_wall_seconds'] = time.perf_counter() - factor_start
                row['preparation'] = prepared.metadata()
                state = prepared.feature_state
                row['feature_state'] = {**state.metadata(), 'artifact': save(route + '_FIT_FEATURE_STATE',
                    {'center': state.center, 'scale': state.scale})}
                candidates = []
                for index, alpha in enumerate(math.ALPHAS):
                    full()
                    candidate_start = time.perf_counter()
                    with threadpool_limits(limits=2):
                        model = prepared.fit(alpha)
                    entry = {'alpha': alpha, 'model_metadata': model.metadata(), 'model_sha256': model.sha256,
                        'coefficient_artifact': save(route + '_alpha_' + str(index) + '_COEFFICIENTS', solution_arrays(model.solution))}
                    for split in ('fit', 'tune'):
                        fast()
                        with threadpool_limits(limits=2):
                            prediction = model.forecast(past_by_split[split], raw_splits[split]['raw_last'])
                        entry[split + '_scores'] = score(prediction, raw_splits[split])
                        entry[split + '_artifact'] = save(route + '_alpha_' + str(index) + '_' + split,
                            {'prediction': prediction, 'origins': raw_splits[split]['origins']})
                    entry['tune_actual_Y_mse'] = entry['tune_scores']['normalized']['mse']
                    entry['FIT_apply_inference_save_wall_seconds'] = time.perf_counter() - candidate_start
                    row['alpha_grid'].append(entry)
                    candidates.append(model)
                index = math.select_alpha(row['alpha_grid'])
                chosen, model = row['alpha_grid'][index], candidates[index]
                row.update(selected_index=index, selected_alpha=chosen['alpha'], selected_model_metadata=model.metadata(),
                           selected_model_sha256=model.sha256, selected_coefficient_artifact=chosen['coefficient_artifact'])
                for split, values in raw_splits.items():
                    full()
                    with threadpool_limits(limits=2):
                        prediction = model.forecast(past_by_split[split], values['raw_last'])
                    row['splits'][split] = score(prediction, values)
                    row['selected_artifacts'][split] = save(route + '_SELECTED_' + split,
                        {'prediction': prediction, 'target': values['target'], 'observed': values['observed'],
                         'origins': values['origins'], 'raw_last': values['raw_last'], 'mean': mean, 'std': std})
                row.update(status='COMPLETE', total_family_wall_seconds=time.perf_counter() - start)
                full()
            except Exception as error:
                row.update(status='BUDGET_HELD_PRESERVED' if isinstance(error, runtime.BudgetHeld) else 'FAILED_PRESERVED',
                           failure={'type': type(error).__name__, 'message': str(error)},
                           total_family_wall_seconds=time.perf_counter() - start)
                if isinstance(error, runtime.BudgetHeld): raise

        # A missing cached E/N2 artifact cannot cancel this independently valid RAW route.
        fit_route('RAW33C', {s: v['history'] for s, v in raw_splits.items()})
        try:
            full()
            stage_path = probes.resolve(args.stage_native_result)
            stage_record = probes.verified_file({'path': str(stage_path), 'sha256': probes.sha(stage_path)})
            stage = probes.read(stage_path)
            row = contract.validate_stage_reuse(stage, args.dataset, args.seed)
            result['source_records'].extend(verify_prior_audit(probes, args.stage_report_manifest, stage_record, args.dataset, args.seed))
            if stage['bridge_provenance']['input_bridge']['sha256'] != provenance['input_bridge']['sha256']:
                raise ValueError('OLD_N2_EXACT_INPUT_BRIDGE_REQUIRED')
            cp_key = ('robin' if args.dataset == 'robin' else 'beijing') + '_' + str(args.seed)
            cp = probes.verified_file(bridge['declared_asset_hashes']['checkpoints_file_hash_only'][cp_key])
            if cp['sha256'] != stage['MAIN_checkpoint']['checkpoint']['sha256']:
                raise ValueError('OLD_N2_EXACT_MAIN_CHECKPOINT_REQUIRED')
            old_update, old_hash = probes.EXPECTED_CODECS[args.dataset, args.seed]
            if stage['MAIN_checkpoint']['selected_update'] != old_update or stage['codec_before']['old_algorithm'] != old_hash:
                raise ValueError('LOCKED_ORIGINAL_MAIN_METADATA_REQUIRED')
            result['stage_N2_source'] = stage_record
            result['original_CP_file_hash_only'] = cp
            result['source_records'].extend((stage_record, cp))
            cached, cached_records = {}, {}
            for split, values in raw_splits.items():
                full()
                declared = by_stem(stage, 'INPUT_' + split)
                arrays, record = probes.load_checked_npz(declared)
                result['data_contract'][split].update(contract.validate_split(args.dataset, split, values, arrays, mean, std))
                cached[split] = arrays
                cached_records[split] = declared
                result['source_records'].append(record)
                save('CACHED_E_INPUT_' + split, arrays)
            d2_record = verify_cached_lineage(probes, stage, args.dataset, args.seed, provenance, cp, cached_records, cached)
            result['source_records'].append(d2_record)
            result['cached_E_lineage'] = {'original_D2_source': d2_record,
                'original_bridge_and_input_artifact_SHA_bound': True, 'D2_to_stage_cached_array_metadata_and_bytes_bound': True,
                'independent_E_forward_replayed': False}
            # Valid E data remains usable even if old N2 coefficient reuse later fails.
            fit_route('E512K', {s: v['latent_past'] for s, v in cached.items()})
            model, records = restore_reused_n2(stage, probes)
            model.feature_state.transform(cached['fit']['latent_past'], require_fit_artifact=True)
            result['source_records'].extend(records)
            reused = {'status': 'RUNNING', 'new_training': False, 'selected_alpha': row['selected_alpha'],
                      'selected_model_sha256': model.sha256, 'splits': {}, 'selected_artifacts': {},
                      'historical_training_wall_seconds': None, 'coefficient_source': row['selected_coefficient_artifact'],
                      'feature_source': stage['feature_state']['artifact'], 'cached_prediction_checks': {}}
            result['families']['E33K_REUSED'] = reused
            for split, values in raw_splits.items():
                full()
                stored, record = probes.load_checked_npz(row['selected_artifacts'][split])
                for field in ('target', 'observed', 'origins', 'raw_last', 'mean', 'std'):
                    expected = mean if field == 'mean' else std if field == 'std' else values[field]
                    contract.exact(stored[field], expected, 'reused_N2:' + split + ':' + field)
                with threadpool_limits(limits=2):
                    prediction = model.direct_forecast(cached[split]['latent_past'], values['raw_last'])
                parity = probes.compare_prediction(prediction, stored['prediction'])
                if not parity['pass']: raise ValueError('REUSED_N2_COEFFICIENT_FORECAST_PARITY:' + split)
                reused['cached_prediction_checks'][split] = parity
                reused['splits'][split] = score(prediction, values)
                reused['selected_artifacts'][split] = save('E33K_REUSED_SELECTED_' + split,
                    {'prediction': prediction, 'target': values['target'], 'observed': values['observed'],
                     'origins': values['origins'], 'raw_last': values['raw_last'], 'mean': mean, 'std': std})
                result['source_records'].append(record)
                result['references'][split]['ORIGINAL_MAIN'] = score(cached[split]['original_prediction'], values)
            reused['status'] = 'COMPLETE'
            full()
        except Exception as error:
            if isinstance(error, runtime.BudgetHeld): raise
            failure = {'type': type(error).__name__, 'message': str(error)}
            result['E_cache_or_N2_reuse_failure'] = failure
            result['families'].setdefault('E33K_REUSED', {}).update(status='FAILED_PRESERVED', failure=failure)
            result['families'].setdefault('E512K', {'status': 'NOT_RUN', 'reason': failure})
        for record in result['source_records']:
            full(); probes.verified_file(record)
        for record in result['source_snapshots']:
            if probes.sha(record['path']) != record['sha256'] or probes.sha(record['snapshot_path']) != record['sha256']:
                raise ValueError('SOURCE_CHANGED_DURING_EXECUTION')
        result['status'] = 'COMPLETE' if all(result['families'].get(r, {}).get('status') == 'COMPLETE'
            for r in ('RAW33C', 'E512K', 'E33K_REUSED')) else 'FAILED_PRESERVED'
        full()
    except Exception as error:
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error, runtime.BudgetHeld) else 'FAILED_PRESERVED',
                      failure={'type': type(error).__name__, 'message': str(error)})
    finally:
        result['body_wall_seconds'] = time.perf_counter() - started
        receipt = limit.close(result['status'], result.get('failure'))
        result['resource_receipt'] = receipt
        if receipt.get('terminal_status') != 'COMPLETE':
            result.update(status=receipt['terminal_status'], resource_terminal_error=receipt.get('terminal_error'))
        if initialized: runtime.write_new(output / 'RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe', action='store_true'); mode.add_argument('--execute', action='store_true')
    parser.add_argument('--dataset', choices=tuple(DATASETS)); parser.add_argument('--seed', type=int, choices=SEEDS)
    for name in ('input-bridge', 'stage-native-result', 'tag'): parser.add_argument('--' + name)
    parser.add_argument('--stage-report-manifest', default=str(HERE/'report_stage_separation_v2/MANIFEST.json'))
    args = parser.parse_args()
    if args.describe:
        print(json.dumps({'status': 'SOURCE_ONLY_NOT_EXECUTED', 'new_routes': ['RAW33C', 'E512K'],
                          'reused': 'old N2 E33K', 'maximum_new_candidates_per_condition': 14,
                          'actual_assets_opened': False, 'GPU_or_F_used': False})); return 0
    if not all((args.dataset, args.seed, args.input_bridge, args.stage_native_result, args.tag)):
        parser.error('execute needs exact old dataset/seed/input bridge/stage N2 result and fresh tag')
    result = execute(args)
    print(json.dumps({'status': result['status'], 'dataset': args.dataset, 'seed': args.seed}))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
