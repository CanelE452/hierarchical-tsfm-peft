"""Fresh gated CPU stage separation; default mode describes source only.

O0/O1 use actual future observations for diagnostics only. N1/N2/N3 expose
past-only prediction APIs. No original engine, F, CUDA or optimizer is loaded.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name / 'stage_separation_v1' / 'native'
DATASETS = {'robin': (17, 4, 'val', 30), 'beijing_aotizhongxin': (11, 2, 'eval', 146)}
SEEDS = (92601, 92602)
ATOL, RTOL = 1e-5, 2e-6
sys.path.insert(0, str(HERE))


def safe_tag(tag):
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in tag):
        raise ValueError('FRESH_SAFE_TAG_REQUIRED')
    return tag


def exact_array(left, right, label):
    import numpy as np
    if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right, equal_nan=True):
        raise ValueError('EXACT_ORIGINAL_ARRAY_REQUIRED:' + label)


def validate_d2_identity(d2, dataset, seed, probes):
    expected_splits = {'fit', 'tune', 'val' if dataset == 'robin' else 'eval'}
    if (d2.get('status') != 'COMPLETE' or d2.get('stage') != 'D2'
            or d2.get('dataset') != dataset or d2.get('seed') != seed
            or d2.get('F_frozen_pass') is not True or d2.get('MAIN_codec_frozen_pass') is not True
            or d2.get('F_before') != {'old_algorithm': probes.OLD_F_HASH, 'bi_algorithm': probes.BI_F_HASH}
            or d2.get('F_before') != d2.get('F_after')
            or set(d2.get('source_GPU_D_vs_stored_MAIN', {})) != expected_splits
            or not all(v.get('pass') is True for v in d2.get('source_GPU_D_vs_stored_MAIN', {}).values())):
        raise ValueError('COMPLETE_ORIGINAL_D2_IDENTITY_AND_FROZEN_LINEAGE_REQUIRED')


def validate_oracle_identity(oracle, dataset, seed, cp_sha, d2_sha, splits):
    if (oracle.get('status') != 'COMPLETE' or oracle.get('dataset') != dataset
            or oracle.get('seed') != seed or oracle.get('diagnostic_only') is not True
            or oracle.get('future_observations_used') is not True
            or oracle.get('original_codec_frozen_pass') is not True
            or oracle.get('codec_before') != oracle.get('codec_after')
            or oracle['MAIN_checkpoint']['checkpoint']['sha256'] != cp_sha
            or oracle['D2_source']['sha256'] != d2_sha
            or set(oracle.get('artifacts', {})) != set(splits)):
        raise ValueError('SAME_CP_D2_COMPLETE_DIAGNOSTIC_ORACLE_REQUIRED')
    for split in splits:
        if not oracle.get('checks', {}).get(split) or not all(
                row.get('pass') is True for row in oracle['checks'][split].values()):
            raise ValueError('ORACLE_PREFIX_AND_CACHED_PARITY_REQUIRED:' + split)


def solution_arrays(solution):
    return {name: getattr(solution, name) for name in ('coefficients', 'intercept', 'observed_counts',
        'channel_counts', 'normal_equation_denominators', 'design_ranks', 'regularized_gram_conditions')}


def model_arrays(model):
    import stage_separation_math_v1 as math
    if isinstance(model, math.NativeModel):
        return solution_arrays(model.solution)
    return {**solution_arrays(model.scaled_solution), 'effective_weight': model.weight,
            'feature_rms': model.feature_rms, 'hidden_weight': model.hidden_weight,
            'hidden_bias': model.hidden_bias}


def solution_metadata(solution):
    return {'kind': solution.kind, 'alpha': solution.alpha, 'with_intercept': solution.with_intercept,
            'feature_state_sha256': solution.feature_state_sha256, 'solution_sha256': solution.sha256,
            'diagnostic_nonfinite_rank_condition_values_are_in_NPZ_only': True}


def restore_solution(meta, arrays):
    import stage_separation_math_v1 as math
    solution = math.RidgeSolution(meta['kind'], meta['alpha'],
        *(arrays[name] for name in ('coefficients', 'intercept', 'observed_counts', 'channel_counts',
          'normal_equation_denominators', 'design_ranks', 'regularized_gram_conditions')),
        meta['feature_state_sha256'], meta['with_intercept'])
    if solution.sha256 != meta['solution_sha256']:
        raise ValueError('RESTORED_SOLUTION_STATE_SHA_MISMATCH')
    return solution


def load_selected_models(producer, probes):
    """Caller must construct its approved runtime before this asset-loading API.

    Return N1/N2/N3 model objects (None for unavailable families), original CPU
    DecoderState and shared FeatureState. No engine/model/CUDA is imported.
    """
    import stage_separation_math_v1 as math
    import frozen_decoder_refit as fdr
    if producer.get('status') != 'COMPLETE':
        raise ValueError('COMPLETE_NATIVE_PRODUCER_REQUIRED_FOR_DEPLOYMENT_COST')
    record = producer['feature_state']
    a, _ = probes.load_checked_npz(record['artifact'])
    state = math.FeatureState(a['center'], a['scale'], record['latent_channels'], record['fit_rows'],
                              record['fit_raw_features_sha256'])
    if state.sha256 != record['sha256']:
        raise ValueError('RESTORED_SHARED_FEATURE_STATE_SHA_MISMATCH')
    decoder_arrays, _ = probes.load_checked_npz(producer['source_decoder_artifact'])
    decoder = fdr.DecoderState.from_codec_state(decoder_arrays)
    if fdr.decoder_state_sha256(decoder) != producer['source_decoder_state_sha256']:
        raise ValueError('RESTORED_ORIGINAL_DECODER_STATE_SHA_MISMATCH')
    restored = {}
    for family in ('N1', 'N2', 'N3'):
        row = producer['families'].get(family, {})
        if row.get('status') != 'COMPLETE':
            restored[family] = None
            continue
        a, _ = probes.load_checked_npz(row['selected_coefficient_artifact'])
        meta = row['selected_model_metadata']
        solution = restore_solution(meta['solution'], a)
        if family != 'N3':
            model = math.NativeModel(state, solution)
        else:
            model = math.DecoderCalibration(meta['kind'], meta['alpha'], a['effective_weight'], a['feature_rms'],
                a['observed_counts'], solution, a['hidden_weight'], a['hidden_bias'], meta['fixed_n1_model_sha256'])
        if model.sha256 != row['selected_model_sha256']:
            raise ValueError('RESTORED_SELECTED_MODEL_SHA_MISMATCH:' + family)
        restored[family] = model
    if restored['N3'] is not None and (restored['N1'] is None or
            restored['N3'].fixed_n1_model_sha256 != restored['N1'].sha256):
        raise ValueError('RESTORED_N3_MUST_BIND_SELECTED_N1')
    return {**restored, 'decoder': decoder, 'feature_state': state}


def predict_selected_cpu(family, model, latent_past, raw_last, decoder_state, n1_model=None):
    """Same normalized CPU prediction recipe as the scientific producer."""
    import frozen_decoder_refit as fdr
    if family == 'N1':
        latent = model.latent_forecast(latent_past)
        return fdr.original_anchored_prediction(latent, latent_past[:, -1:], raw_last, decoder_state)
    if family == 'N2':
        return model.direct_forecast(latent_past, raw_last)
    if family == 'N3':
        return model.forecast_from_selected_n1(n1_model, latent_past, raw_last)
    raise ValueError('PAST_ONLY_N1_N2_N3_FAMILY_REQUIRED')


def execute(args):
    import stage_separation_runtime_v1 as runtime
    limit = runtime.ContinuationLimit('stage_native_' + safe_tag(args.tag), gpu=False)
    output = CACHE / args.tag
    result = {'status': 'RUNNING', 'dataset': args.dataset, 'seed': args.seed,
        'scientific_execution': True, 'protected_TEST_accessed': False, 'F_loaded': False,
        'F_forward_replayed': False, 'E_forward_replayed': False, 'old_campaign_restart': False,
        'independent_holdout': False, 'whole_goal_complete': False, 'families': {}, 'references': {},
        'artifacts': [], 'source_records': [], 'timings': {}, 'permission_gate': limit.authorization_record,
        'run_manifest': limit.run_manifest_record, 'parity_tolerance': {'atol': ATOL, 'rtol': RTOL},
        'selection': 'actual-Y TUNE normalized observed-channel macro MSE; earliest exact tie',
        'maximum_new_candidates': 28, 'candidate_evaluation_scope': 'all FIT/TUNE; selected reused VAL/EVAL only',
        'normalization': 'original FIT statistics unchanged',
        'native_loss': 'channel-macro sum_NH(mask*error^2)/Nchannel + alpha*sum_h||W_scaled||^2',
        'teacher_loss_is_actual_Y_prediction_loss': False,
        'physical_macro_is_primary': False, 'oracle_is_deployable': False}
    initialized = False
    source_state = None
    started = time.perf_counter()
    try:
        output.mkdir(parents=True, exist_ok=False)
        initialized = True
        import numpy as np
        import torch
        import prepare_inputs as inputs
        import run_condition_probes as probes
        import frozen_decoder_refit as fdr
        import stage_separation_math_v1 as math
        import stage_separation_oracle_v1 as contracts
        from threadpoolctl import threadpool_limits
        inputs.install_protected_test_guard()
        full, fast = limit.checkfull, limit.shared.checkfast
        def saved(name, arrays):
            full()
            record = probes.save_npz(output / (name + '.npz'), arrays)
            result['artifacts'].append(record)
            full()
            return record
        def scores(pred, values):
            math.masked_macro_mse(pred, values['target'], values['observed'])
            return {'normalized': fdr.observed_score(pred, values['target'], values['observed']),
                    'physical_per_channel_diagnostic': fdr.observed_score(pred, values['target'], values['observed'], std)}
        def predict_original(z, values):
            fast()
            prediction = fdr.original_anchored_prediction(z, values['z_last'], values['raw_last'], decoder)
            fast()
            return prediction
        load_start = time.perf_counter()
        bridge, split_inputs, mean, std, provenance = probes.load_bridge(args.input_bridge, args.dataset, inputs.verify_phase_barrier)
        result['bridge_provenance'] = provenance
        result['source_records'].extend(probes.verify_source_records(bridge))
        result['source_records'].extend(provenance['input_artifacts'])
        result['source_records'].append(provenance['input_bridge'])
        result['source_records'].append(probes.check_original_raw(bridge, args.dataset, split_inputs, mean, std))
        d2_path = probes.resolve(args.d2_result)
        contracts.assert_no_protected_test_path(d2_path, role='D2_cached_latents')
        d2_record = {'path': str(d2_path), 'sha256': probes.sha(d2_path)}
        d2 = probes.read(d2_path)
        validate_d2_identity(d2, args.dataset, args.seed, probes)
        result['D2_source'] = d2_record
        result['source_records'].append(d2_record)
        for item in d2['source_snapshot']:
            result['source_records'].append(probes.verified_file(item))
            if probes.sha(item['snapshot_path']) != item['sha256']:
                raise ValueError('OLD_D2_SOURCE_SNAPSHOT_CHANGED')
        checkpoint_key = ('robin' if args.dataset == 'robin' else 'beijing') + '_' + str(args.seed)
        cp = probes.verified_file(bridge['declared_asset_hashes']['checkpoints_file_hash_only'][checkpoint_key])
        cp_meta = d2['MAIN_checkpoint']
        if Path(cp['path']) != probes.resolve(cp_meta['checkpoint']['path']) or cp['sha256'] != cp_meta['checkpoint']['sha256']:
            raise ValueError('EXACT_OLD_MAIN_CP_REQUIRED')
        case_record = probes.verified_file(cp_meta['case'])
        case = probes.read(case_record['path'])
        old_update, old_hash = probes.EXPECTED_CODECS[args.dataset, args.seed]
        if (case['dataset'] != args.dataset or case['seed'] != args.seed
                or case['architecture'] != 'ANCHOR_NONLINEAR_D'
                or case['training']['selected_update'] != old_update
                or case['training']['selected_codec_hash'] != old_hash
                or probes.resolve(case['training']['selected_checkpoint']['path']) != Path(cp['path'])
                or case['training']['selected_checkpoint']['sha256'] != cp['sha256']):
            raise ValueError('PREDECLARED_OLD_MAIN_CASE_CP_IDENTITY')
        full()
        checkpoint = torch.load(cp['path'], map_location='cpu', weights_only=True)
        if (checkpoint['seed'] != args.seed or checkpoint['architecture'] != 'ANCHOR_NONLINEAR_D'
                or checkpoint['method'] != 'MAIN' or checkpoint['weight'] != 0. or checkpoint['update'] != old_update):
            raise ValueError('OLD_MAIN_CP_METADATA')
        source_state = checkpoint['codec']
        c, k, evaluation, eval_n = DATASETS[args.dataset]
        if tuple(source_state['fixed_basis'].shape) != (c, k) or probes.old_state_hash(source_state) != old_hash:
            raise ValueError('EXACT_OLD_MAIN_CODEC_STATE_REQUIRED')
        result['MAIN_checkpoint'] = cp_meta
        result['codec_before'] = {'old_algorithm': probes.old_state_hash(source_state), 'bi_algorithm': probes.bi_state_hash(source_state)}
        if (cp_meta['selected_codec_hash_old_algorithm'] != result['codec_before']['old_algorithm']
                or cp_meta['selected_codec_hash_bi_algorithm'] != result['codec_before']['bi_algorithm']
                or cp_meta['original_prediction_batch_size'] != 8):
            raise ValueError('D2_EXACT_CODEC_DUAL_HASH_BATCH8_REQUIRED')
        result['source_records'].extend((cp, case_record))
        decoder = fdr.DecoderState.from_codec_state(source_state)
        result['source_decoder_state_sha256'] = fdr.decoder_state_sha256(decoder)
        result['source_decoder_artifact'] = saved('SOURCE_DECODER', decoder.arrays)
        splits = {}
        original_npz, main_source = probes.load_checked_npz(bridge['declared_asset_hashes']['predictions'][args.dataset]['main_' + str(args.seed)])
        original = probes.compare_reference(original_npz, split_inputs, mean, std)
        result['source_records'].append(main_source)
        for split, values in split_inputs.items():
            full()
            record = next(r for r in d2['artifacts'] if Path(r['path']).name == 'ABSOLUTE_LATENTS_' + split + '.npz')
            cached, checked = probes.load_checked_npz(record)
            contracts.validate_d2_cached_latent_payload(cached)
            for name in ('target', 'observed', 'origins', 'raw_last', 'history_observed', 'raw_forecast'):
                exact_array(cached[name], values[name], split + ':' + name)
            for name, value in (('mean', mean), ('std', std), ('original_prediction', original[split]),
                                ('z_last', cached['latent_past'][:, -1:])):
                exact_array(cached[name], value, split + ':' + name)
            if cached['latent_past'].shape != (len(values['history']), 512, k) or len(values['history']) != {'fit': 256, 'tune': 64, evaluation: eval_n}[split]:
                raise ValueError('LOCKED_SPLIT_N_L_K')
            splits[split] = cached
            result['source_records'].append(checked)
            saved('INPUT_' + split, cached)
            replay = predict_original(cached['z_future'], cached)
            parity = probes.compare_prediction(replay, original[split])
            if not parity['pass']:
                raise ValueError('CPU_ORIGINAL_D_STORED_MAIN_PARITY:' + split)
            result['references'][split] = {'ORIGINAL_MAIN': scores(original[split], cached),
                'RAW': scores(cached['raw_forecast'], cached), 'CPU_SOURCE_D_PARITY': parity}
            saved('REFERENCE_' + split, {'CPU_SOURCE_D': replay, 'ORIGINAL_MAIN': original[split], 'RAW': cached['raw_forecast']})
        try:
            full()
            refit_record = probes.verified_file(d2['probe_result'])
            old_refit = probes.read(refit_record['path'])
            if old_refit.get('status') != 'COMPLETE' or old_refit['source_record']['dataset'] != args.dataset or old_refit['source_record']['seed'] != args.seed:
                raise ValueError('D2_REFIT_COMPLETE_IDENTITY')
            if old_refit['source_decoder_state_sha256'] != result['source_decoder_state_sha256']:
                raise ValueError('D2_REFIT_SAME_ORIGINAL_DECODER_REQUIRED')
            result['D2_existing_F_decoder_refits'] = {'status': 'COMPLETE', 'source': refit_record, 'refits': old_refit['refits']}
            result['source_records'].append(refit_record)
        except Exception as error:
            if isinstance(error, runtime.BudgetHeld):
                raise
            result['D2_existing_F_decoder_refits'] = {'status': 'UNAVAILABLE', 'failure': {'type': type(error).__name__, 'message': str(error)}}
        try:
            full()
            from run_saved_diagnostics import verify_native_lineage
            native_lineage = verify_native_lineage(probes, bridge, args.dataset, args.seed)
            native_npz, native_source = probes.load_checked_npz(native_lineage['native_prediction'])
            native_preds = probes.compare_reference(native_npz, split_inputs, mean, std)
            result['SAME_MAIN_NATIVE_existing'] = {'status': 'COMPLETE', 'lineage': native_lineage,
                'splits': {s: scores(native_preds[s], splits[s]) for s in splits}, 'new_training': False}
            result['source_records'].extend(native_lineage[name] for name in ('control_json', 'MAIN_checkpoint', 'native_prediction', 'native_models'))
        except Exception as error:
            if isinstance(error, runtime.BudgetHeld):
                raise
            result['SAME_MAIN_NATIVE_existing'] = {'status': 'UNAVAILABLE', 'failure': {'type': type(error).__name__, 'message': str(error)}, 'new_training': False}
        result['timings']['input_CP_CPU_decoder_parity_and_reference_reuse_wall_seconds'] = time.perf_counter() - load_start
        snapshot_folder = output / 'sources'
        snapshot_folder.mkdir()
        result['source_snapshots'] = []
        for path in (Path(__file__).resolve(), HERE / 'stage_separation_math_v1.py', HERE / 'stage_separation_runtime_v1.py',
                     HERE / 'frozen_decoder_refit.py', HERE / 'run_condition_probes.py', HERE / 'run_saved_diagnostics.py',
                     HERE / 'stage_separation_oracle_v1.py', HERE / 'fast_storage_scan_v1.py',
                     HERE / 'PLAN_STAGE_SEPARATION_KO_v1.md', HERE / 'METHOD_LOCK_STAGE_SEPARATION_v1.md'):
            full()
            source_sha = probes.sha(path)
            destination = snapshot_folder / path.name
            with path.open('rb') as src, destination.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            if probes.sha(destination) != source_sha:
                raise ValueError('SOURCE_SNAPSHOT_HASH_MISMATCH')
            result['source_snapshots'].append({'path': str(path), 'sha256': source_sha, 'snapshot_path': str(destination)})
            full()
        fit = splits['fit']
        full()
        training_start = time.perf_counter()
        state = math.fit_feature_state(fit['latent_past'])
        result['feature_state'] = {'sha256': state.sha256, 'latent_channels': k, 'fit_rows': state.fit_rows,
            'fit_raw_features_sha256': state.fit_raw_features_sha256,
            'new_FIT_training_wall_seconds': time.perf_counter() - training_start,
            'artifact': saved('SHARED_FIT_FEATURE_STATE', {'center': state.center, 'scale': state.scale})}
        result['controls'] = {}
        full()
        baseline_start = time.perf_counter()
        intercept = math.fit_intercept_only(fit['target'], fit['raw_last'], fit['observed'])
        result['controls']['FIT_INTERCEPT_ONLY'] = {'model_sha256': intercept.sha256,
            'new_FIT_training_wall_seconds': time.perf_counter() - baseline_start,
            'coefficient_artifact': saved('FIT_INTERCEPT_ONLY_COEFFICIENTS', {'innovation': intercept.innovation, 'observed_counts': intercept.observed_counts}), 'splits': {}}
        result['controls']['CURRENT_C_PERSISTENCE'] = {'new_FIT_training_wall_seconds': 0., 'splits': {}}
        for split, values in splits.items():
            for name, prediction in (('CURRENT_C_PERSISTENCE', math.persistence_forecast(values['raw_last'])),
                                     ('FIT_INTERCEPT_ONLY', intercept.forecast(values['raw_last']))):
                result['controls'][name]['splits'][split] = scores(prediction, values)
                saved(name + '_' + split, {'prediction': prediction, 'origins': values['origins']})

        def family(name, prepare, forecast):
            row = {'status': 'RUNNING', 'diagnostic_only': name == 'O1', 'alpha_grid': [], 'splits': {}, 'selected_artifacts': {}}
            result['families'][name] = row
            family_start = time.perf_counter()
            try:
                full()
                t = time.perf_counter()
                with threadpool_limits(limits=2):
                    prepared = prepare()
                row['new_FIT_factorization_wall_seconds'] = time.perf_counter() - t
                row['factorization_groups'] = len(prepared.ridge.groups)
                row['zero_observed_output_counts'] = int((prepared.ridge.counts == 0).sum())
                full()
                candidates = []
                for index, alpha in enumerate(math.ALPHAS):
                    full()
                    t = time.perf_counter()
                    model = prepared.fit(alpha)
                    solution = model.solution if isinstance(model, math.NativeModel) else model.scaled_solution
                    entry = {'alpha': alpha, 'model_sha256': model.sha256,
                        'solution': solution_metadata(solution),
                        'coefficient_artifact': saved(name + '_alpha_' + str(index) + '_COEFFICIENTS', model_arrays(model))}
                    for split in ('fit', 'tune'):
                        fast()
                        prediction, latent = forecast(model, splits[split])
                        entry[split + '_scores'] = scores(prediction, splits[split])
                        payload = {'prediction': prediction, 'origins': splits[split]['origins']}
                        if latent is not None:
                            payload['predicted_latent_future'] = latent
                        entry[split + '_artifact'] = saved(name + '_alpha_' + str(index) + '_' + split, payload)
                    entry['tune_actual_Y_mse'] = entry['tune_scores']['normalized']['mse']
                    entry['fit_apply_inference_save_wall_seconds'] = time.perf_counter() - t
                    row['alpha_grid'].append(entry)
                    candidates.append(model)
                    full()
                selected_index = math.select_alpha(row['alpha_grid'])
                model = candidates[selected_index]
                chosen = row['alpha_grid'][selected_index]
                row.update(selected_index=selected_index, selected_alpha=model.solution.alpha if isinstance(model, math.NativeModel) else model.alpha,
                    selected_model_sha256=model.sha256, selected_coefficient_artifact=chosen['coefficient_artifact'],
                    selected_model_metadata={'kind': chosen['solution']['kind'], 'alpha': chosen['alpha'],
                        'solution': chosen['solution'], 'fixed_n1_model_sha256': getattr(model, 'fixed_n1_model_sha256', None)})
                for split, values in splits.items():
                    full()
                    prediction, latent = forecast(model, values)
                    row['splits'][split] = scores(prediction, values)
                    payload = {'prediction': prediction, 'target': values['target'], 'observed': values['observed'],
                        'origins': values['origins'], 'mean': mean, 'std': std, 'raw_last': values['raw_last']}
                    if latent is not None:
                        payload['predicted_latent_future'] = latent
                    row['selected_artifacts'][split] = saved(name + '_SELECTED_' + split, payload)
                row.update(status='COMPLETE', total_family_wall_seconds=time.perf_counter() - family_start)
                full()
                return model
            except Exception as error:
                row.update(status='BUDGET_HELD_PRESERVED' if isinstance(error, runtime.BudgetHeld) else 'FAILED_PRESERVED',
                           failure={'type': type(error).__name__, 'message': str(error)},
                           total_family_wall_seconds=time.perf_counter() - family_start)
                if isinstance(error, runtime.BudgetHeld):
                    raise
                return None

        # This independent path always precedes the optional future-oracle branch.
        n2 = family('N2', lambda: math.prepare_n2(state, fit['latent_past'], fit['target'], fit['raw_last'], fit['observed']),
                    lambda model, v: (model.direct_forecast(v['latent_past'], v['raw_last']), None))
        oracle_splits = None
        try:
            if not args.oracle_result:
                raise FileNotFoundError('OPTIONAL_ORACLE_RESULT_NOT_SUPPLIED')
            full()
            oracle_path = probes.resolve(args.oracle_result)
            contracts.assert_no_protected_test_path(oracle_path, role='future_oracle_diagnostic')
            oracle_record = {'path': str(oracle_path), 'sha256': probes.sha(oracle_path)}
            oracle = probes.read(oracle_path)
            validate_oracle_identity(oracle, args.dataset, args.seed, cp['sha256'], d2_record['sha256'], splits)
            result['oracle_source'] = oracle_record
            result['source_records'].append(oracle_record)
            oracle_splits = {}
            for split, values in splits.items():
                full()
                a, checked = probes.load_checked_npz(oracle['artifacts'][split])
                for name in ('target', 'observed', 'origins', 'raw_last', 'mean', 'std'):
                    exact_array(a[name], values[name], 'oracle:' + split + ':' + name)
                exact_array(a['current_past_latent'], values['latent_past'], 'oracle_current_past:' + split)
                exact_array(a['current_z_last'], values['z_last'], 'oracle_current_last:' + split)
                expected_mask = math.complete_prefix_latent_mask(values['observed'], k)
                exact_array(a['complete_future_prefix_mask'], expected_mask[:, :, 0], 'oracle_complete_prefix:' + split)
                if a['oracle_z'].dtype != np.float32 or a['oracle_z'].shape != (len(values['target']), 48, k) or not np.isfinite(a['oracle_z']).all():
                    raise ValueError('FINITE_ORACLE_ABSOLUTE_Z_FP32_N48K_REQUIRED')
                result['source_records'].append(checked)
                oracle_splits[split] = a
                saved('ORACLE_INPUT_DIAGNOSTIC_' + split, a)
            teacher_mask = math.complete_prefix_latent_mask(fit['observed'], k)
            result['teacher_contract'] = {'mask_rule': 'all original C observed through entire future prefix; repeated K',
                'fit_counts_by_horizon_K': teacher_mask.sum(axis=0).tolist(),
                'actual_Y_fit_channel_counts': fit['observed'].sum((0, 1)).tolist(),
                'artifact': saved('FIT_N1_TEACHER', {'oracle_z': oracle_splits['fit']['oracle_z'],
                    'latent_innovation': oracle_splits['fit']['oracle_z'] - fit['z_last'], 'teacher_observed': teacher_mask,
                    'actual_Y_observed': fit['observed'], 'origins': fit['origins']})}
        except Exception as error:
            oracle_splits = None
            result['oracle_branch_failure'] = {'type': type(error).__name__, 'message': str(error)}
            if isinstance(error, runtime.BudgetHeld):
                raise
        if oracle_splits is not None:
            hidden_weight, hidden_bias = decoder.arrays['residual.0.weight'], decoder.arrays['residual.0.bias']
            result['O0_ORACLE_OLD_D'] = {'diagnostic_only': True, 'splits': {}, 'artifacts': {}}
            for split, values in splits.items():
                full()
                pred = predict_original(oracle_splits[split]['oracle_z'], values)
                result['O0_ORACLE_OLD_D']['splits'][split] = scores(pred, values)
                result['O0_ORACLE_OLD_D']['artifacts'][split] = saved('O0_ORACLE_OLD_D_' + split, {'prediction': pred, 'origins': values['origins']})
                # Predeclared math/Torch hidden feature parity, no tolerance widening.
                np_features = math.decoder_features(oracle_splits[split]['oracle_z'], values['z_last'], hidden_weight, hidden_bias)
                cpu_features = fdr.decoder_features(oracle_splits[split]['oracle_z'], values['z_last'], decoder)
                parity = probes.compare_prediction(np_features, cpu_features)
                if not parity['pass']:
                    result['oracle_decoder_feature_parity_failure'] = {'split': split, **parity}
            if 'oracle_decoder_feature_parity_failure' not in result:
                family('O1', lambda: math.prepare_o1(oracle_splits['fit']['oracle_z'], fit['z_last'], fit['target'],
                    fit['raw_last'], fit['observed'], hidden_weight, hidden_bias, decoder.outgoing_weight),
                    lambda model, v: (model.forecast(oracle_splits[next(s for s in splits if splits[s] is v)]['oracle_z'], v['z_last'], v['raw_last']), None))
            else:
                result['families']['O1'] = {'status': 'FAILED_PRESERVED', 'failure': result['oracle_decoder_feature_parity_failure']}
            def native_forecast(model, v):
                latent = model.latent_forecast(v['latent_past'])
                return predict_original(latent, v), latent
            n1 = family('N1', lambda: math.prepare_n1(state, fit['latent_past'], oracle_splits['fit']['oracle_z'], fit['observed']), native_forecast)
            if n1 is not None:
                result['N3_numpy_CPU_hidden_parity'] = {}
                for split, values in splits.items():
                    full()
                    latent = n1.latent_forecast(values['latent_past'])
                    numpy_features = math.decoder_features(latent, values['z_last'], hidden_weight, hidden_bias)
                    cpu_features = fdr.decoder_features(latent, values['z_last'], decoder)
                    result['N3_numpy_CPU_hidden_parity'][split] = probes.compare_prediction(numpy_features, cpu_features)
                def n3_forecast(model, v):
                    # No caller-supplied future tensor enters the deployment API.
                    prediction = model.forecast_from_selected_n1(n1, v['latent_past'], v['raw_last'])
                    latent = n1.latent_forecast(v['latent_past'])
                    return prediction, latent
                if all(p['pass'] for p in result['N3_numpy_CPU_hidden_parity'].values()):
                    n3 = family('N3', lambda: math.prepare_n3(n1, fit['latent_past'], fit['target'], fit['raw_last'], fit['observed'],
                        hidden_weight, hidden_bias, decoder.outgoing_weight), n3_forecast)
                else:
                    n3 = None
                    result['families']['N3'] = {'status': 'FAILED_PRESERVED', 'failure': {'type': 'FixedParityFailure', 'message': 'Numpy/Torch hidden feature parity failed at fixed 1e-5/2e-6'}}
                if n3 is not None:
                    result['N3_fixed_N1_contract'] = {'fixed_n1_model_sha256': n1.sha256,
                        'selected_n1_alpha': n1.solution.alpha, 'N1_coefficients_reselected_for_N3': False, 'splits': {}}
                    for split in splits:
                        a, _ = probes.load_checked_npz(result['families']['N1']['selected_artifacts'][split])
                        b, _ = probes.load_checked_npz(result['families']['N3']['selected_artifacts'][split])
                        exact_array(a['predicted_latent_future'], b['predicted_latent_future'], 'N1_N3_LATENT:' + split)
                        result['N3_fixed_N1_contract']['splits'][split] = {'array_equal': True,
                            'latent_sha256': math.array_sha(a['predicted_latent_future'])}
            else:
                result['families']['N3'] = {'status': 'NOT_RUN', 'reason': 'selected N1 unavailable'}
        else:
            for name in ('O1', 'N1', 'N3'):
                result['families'][name] = {'status': 'NOT_RUN', 'reason': 'verified oracle branch unavailable'}
        result['codec_after'] = {'old_algorithm': probes.old_state_hash(source_state), 'bi_algorithm': probes.bi_state_hash(source_state)}
        if result['codec_before'] != result['codec_after'] or any(v.device.type != 'cpu' for v in source_state.values()):
            raise ValueError('OLD_CODEC_MUTATED_OR_NOT_CPU')
        result['old_E_D_F_updated'] = False
        result['F_reference_scope'] = 'producer D2 dual-hash frozen proof only; no independent F replay in this CPU process'
        for record in result['source_records']:
            full()
            probes.verified_file(record)
        for record in result['source_snapshots']:
            if probes.sha(record['path']) != record['sha256'] or probes.sha(record['snapshot_path']) != record['sha256']:
                raise ValueError('SOURCE_CHANGED_DURING_SCIENTIFIC_EXECUTION')
        result['status'] = 'COMPLETE' if all(result['families'].get(n, {}).get('status') == 'COMPLETE' for n in ('O1', 'N1', 'N2', 'N3')) else 'FAILED_PRESERVED'
        if result['status'] != 'COMPLETE':
            result['failure'] = {'type': 'IncompleteIndependentFamilies', 'message': 'Successful independent family artifacts retained; inspect every family status.'}
        full()
    except Exception as error:
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error, runtime.BudgetHeld) else 'FAILED_PRESERVED',
                      failure={'type': type(error).__name__, 'message': str(error)})
    finally:
        if source_state is not None:
            import run_condition_probes as probes
            terminal_hash = {'old_algorithm': probes.old_state_hash(source_state), 'bi_algorithm': probes.bi_state_hash(source_state)}
            result['codec_terminal_hash'] = terminal_hash
            if result.get('codec_before') != terminal_hash:
                result.update(status='FAILED_PRESERVED', failure={'type': 'FrozenStateChanged', 'message': 'OLD_CODEC_CHANGED'})
        result['total_runner_wall_seconds'] = time.perf_counter() - started
        receipt = limit.close(result['status'], result.get('failure'))
        result['resource_receipt'] = receipt
        if receipt.get('terminal_status', result['status']) != 'COMPLETE':
            result['status'] = receipt['terminal_status']
            if receipt.get('terminal_error') is not None:
                result['resource_terminal_error'] = receipt['terminal_error']
        if initialized:
            runtime.write_new(output / 'RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--dataset', choices=tuple(DATASETS))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--input-bridge')
    parser.add_argument('--d2-result')
    parser.add_argument('--oracle-result')
    parser.add_argument('--tag')
    args = parser.parse_args()
    if args.describe:
        print(json.dumps({'status': 'SOURCE_ONLY_NOT_EXECUTED', 'actual_assets_opened': False,
            'scope': 'CPU native ridge/O0/O1/N1/N2/N3; optional oracle failure preserves independent N2',
            'runtime': 'fresh stage_separation_runtime_v1 approval/fixed manifest required',
            'maximum_candidates': 28, 'protected_TEST': False}))
        return 0
    if not all((args.dataset, args.seed, args.input_bridge, args.d2_result, args.tag)):
        parser.error('execute needs exact old dataset/seed/input bridge/D2 producer/fresh tag')
    result = execute(args)
    print(json.dumps({'status': result['status'], 'dataset': args.dataset, 'seed': args.seed, 'tag': args.tag}))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
