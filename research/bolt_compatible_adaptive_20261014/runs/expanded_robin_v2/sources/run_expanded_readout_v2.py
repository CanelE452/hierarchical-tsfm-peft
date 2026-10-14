"""Canonical-array repair of a failed FIT512 audit; original attempt preserved."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'research/smallk_failure_followup_20261014'
sys.path[:0] = [str(HERE), str(OLD), str(ROOT)]
CONTRACTS = {
    'robin': ROOT/'research/smallk_predictive_alignment_20261013/run_dev_v1/robin_contract.json',
    'beijing_aotizhongxin': ROOT/'.cache/smallk_predictive_alignment_20261013/beijing_repair/raw_dataset/beijing_aotizhongxin/b04da438b2f331ac/contract.json',
}


def legal_fit_origins(bounds, maximum=512):
    import numpy as np
    lo, hi = bounds
    legal = np.arange(lo+512, hi-48+1, dtype=np.int64)
    if not len(legal):
        raise ValueError('NO_LEGAL_WITHIN_FIT_WINDOWS')
    return legal[np.linspace(0, len(legal)-1, maximum).astype(int)] if len(legal) > maximum else legal


def canonicalize_payload(values):
    import numpy as np
    canonical = {key: np.ascontiguousarray(value) for key, value in values.items()}
    for key, value in canonical.items():
        original = values[key]
        if (value.dtype != original.dtype or value.shape != original.shape
                or not np.array_equal(value, original, equal_nan=True) or not value.flags.c_contiguous):
            raise ValueError('CANONICAL_LAYOUT_MUST_PRESERVE_VALUES_SHAPE_DTYPE:' + key)
    return canonical


def verify_repair_gate(args, probes):
    failed_record = probes.verified_file({'path': args.failed_result, 'sha256': args.failed_result_sha256})
    audit_record = probes.verified_file({'path': args.failed_audit, 'sha256': args.failed_audit_sha256})
    diagnosis_record = probes.verified_file({'path': args.layout_diagnosis, 'sha256': args.layout_diagnosis_sha256})
    failed, audit, diagnosis = (probes.read(r['path']) for r in (failed_record, audit_record, diagnosis_record))
    if (failed.get('phase') != 'MATCHED_FIT512_ONE_DIAGNOSTIC' or failed.get('dataset') != args.dataset
            or audit.get('status') != 'FAIL' or audit.get('scientific_execution') is not True
            or audit.get('failure', {}).get('message') != 'EXACT_ARRAY_MISMATCH:dense_FIT_center:SELF33'
            or diagnosis.get('status') != 'DIAGNOSTIC_COMPLETE' or diagnosis.get('raw_history_values_exact') is not True
            or diagnosis.get('producer_record', {}).get('sha256') != failed_record['sha256']):
        raise ValueError('FAILED_AUDIT_AND_SAME_PRODUCER_LAYOUT_DIAGNOSIS_REQUIRED')
    layouts = diagnosis['raw_history_layouts']
    summaries = diagnosis['routes']['SELF33']['variant_statistics']
    if (layouts['generated_original_layout']['C_contiguous'] is not False
            or layouts['saved_NPZ']['C_contiguous'] is not True
            or summaries['generated_original_layout']['center_vs_saved']['exact_equal'] is not True
            or summaries['generated_C_contiguous']['center_vs_saved']['exact_equal'] is not False):
        raise ValueError('PROVEN_LAYOUT_DEPENDENT_FIT_STATISTICS_REQUIRED')
    return failed, {'failed_producer': failed_record, 'failed_audit': audit_record, 'layout_diagnosis': diagnosis_record,
                    'same_diagnostic_repair_not_new_independent_experiment': True,
                    'preserved_original_attempt_excluded_from_valid_scientific_counts': True,
                    'canonicalization': 'Every payload array C-contiguous before statistics, solver, prediction and saving; no value/shape/dtype change'}


def execute(args, limit):
    import numpy as np
    import readout_math_v1 as math
    import run_condition_probes as probes
    import prepare_inputs as inputs
    import matched_readout_contract_v1 as contract
    from threadpoolctl import threadpool_limits
    inputs.install_protected_test_guard()
    full, fast = limit.checkfull, limit.shared.checkfast
    output = HERE/'runs'/args.tag
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'RUNNING', 'phase': 'MATCHED_FIT512_ONE_DIAGNOSTIC', 'dataset': args.dataset,
              'scientific_execution': True, 'protected_TEST_accessed': False, 'GPU_execution': False,
              'model_weights_loaded': False, 'independent_holdout': False, 'old_campaign_restart': False,
              'deterministic_ridge_no_seed_replication': True, 'families': {}, 'feature_states': {},
              'inputs': {}, 'artifacts': [], 'source_records': [], 'source_snapshots': [], 'references': {},
              'data_contract': {}, 'new_model_bundles': 0, 'reused_model_bundles': 0, 'pool_comparisons': 0,
              'permission_gate': limit.authorization_record, 'run_manifest': limit.run_manifest_record,
              'overlapping_training_windows_not_independent_examples': True,
              'selection': 'one global normalized channel-macro TUNE MSE; earliest exact tie'}
    started = time.perf_counter()
    try:
        full()
        failed, correction = verify_repair_gate(args, probes)
        result['correction'] = correction
        result['repair_attempt'] = True
        result['canonical_payload_order'] = 'C'
        result['source_records'].extend(correction[name] for name in ('failed_producer', 'failed_audit', 'layout_diagnosis'))
        first_path = Path(args.first_result).resolve()
        if not first_path.is_relative_to(HERE/'runs'):
            raise ValueError('NEW_CAMPAIGN_FIRST_RESULT_REQUIRED')
        first_record = probes.verified_file({'path': str(first_path), 'sha256': probes.sha(first_path)})
        first = probes.read(first_path)
        if (first.get('status') != 'COMPLETE' or first.get('dataset') != args.dataset
                or first['branch_lock']['next_branch'] != 'MATCHED_FIT512_ONE_DIAGNOSTIC'):
            raise ValueError('FIRST_TUNE_OVERFIT_GATE_REQUIRED')
        if failed['first256_result']['sha256'] != first_record['sha256']:
            raise ValueError('SAME_FIRST256_PRODUCER_REQUIRED_FOR_REPAIR')
        for candidate in (HERE/'runs').glob('*/RESULT.json'):
            prior = probes.read(candidate)
            if prior.get('phase') == result['phase'] and prior.get('dataset') == args.dataset and prior.get('status') == 'COMPLETE':
                if probes.sha(candidate) != correction['failed_producer']['sha256']:
                    raise ValueError('ONLY_ONE_FAILED_AUDIT_REPAIR_ATTEMPT_ALLOWED')
        result['first256_result'] = first_record
        result['source_records'].append(first_record)
        result['split_roles'] = first['split_roles']
        result['bridge_provenance'] = first['bridge_provenance']
        raw = {}
        for split in ('fit', 'tune'):
            values, checked = probes.load_checked_npz(first['inputs'][split])
            raw[split] = canonicalize_payload(values)
            result['source_records'].append(checked)
        original_fit = raw['fit']
        mean, std = original_fit['mean'], original_fit['std']
        contract_path = CONTRACTS[args.dataset]
        source_contract = probes.read(contract_path)
        result['source_records'].append(probes.verified_file({'path': str(contract_path), 'sha256': probes.sha(contract_path)}))
        bounds = source_contract['bounds']['fit']
        if args.dataset == 'robin':
            source_record = {'path': str(probes.resolve(source_contract['source'])), 'sha256': source_contract['source_sha256']}
            checked = probes.verified_file(source_record)
            with np.load(checked['path'], allow_pickle=False) as source:
                physical = (source['x'][:bounds[1]] * source['std'] + source['mean']).astype(np.float32)
                observed = source['finite'][:bounds[1]].copy()
            physical[~observed] = np.nan
            result['raw_source_note'] = 'Original normalized trainval x member decoded; only FIT prefix used; FP32 inverse as original bridge. Member contains previously exposed VAL rows.'
        else:
            checked = probes.verified_file(source_contract['payloads']['fit_tune'])
            with np.load(checked['path'], allow_pickle=False) as source:
                physical = source['physical'][:bounds[1]].copy()
                observed = source['mask'][:bounds[1]].copy()
                contract.exact(source['mean'], mean, 'original_FIT_mean')
                contract.exact(source['std'], std, 'original_FIT_std')
            result['raw_source_note'] = 'Existing FIT/TUNE physical payload; only FIT prefix used. EVAL target payload absent.'
        result['source_records'].append(checked)

        def windows(origins):
            past = np.stack([physical[o-512:o] for o in origins]).astype(np.float32)
            past_mask = np.stack([observed[o-512:o] for o in origins]).astype(bool)
            future_mask = np.stack([observed[o:o+48] for o in origins]).astype(bool)
            future = np.stack([physical[o:o+48] for o in origins]).astype(np.float32)
            history = inputs.old_window_preprocess(past, past_mask, mean, std)
            target = inputs.target_from_physical_observed(future, future_mask, mean, std)
            return canonicalize_payload({'history': history, 'history_observed': past_mask, 'target': target,
                    'observed': future_mask, 'origins': origins, 'raw_last': history[:, -1].copy(), 'mean': mean, 'std': std})

        reconstructed = windows(original_fit['origins'])
        for key in reconstructed:
            contract.exact(reconstructed[key], original_fit[key], 'original256_reconstruction:'+key)
        origins = legal_fit_origins(bounds)
        raw['fit'] = windows(origins)
        result['original256_reconstruction_exact_pass'] = True
        result['original_FIT_statistics_unchanged'] = True
        result['legal_FIT_bounds'] = bounds
        result['expanded_FIT_origin_count'] = len(origins)
        result['expanded_FIT_origin_first_last'] = [int(origins[0]), int(origins[-1])]
        result['expanded_and_original_origin_overlap_count'] = int(np.intersect1d(origins, original_fit['origins']).size)

        def save(name, arrays):
            full()
            record = probes.save_npz(output/(name+'.npz'), arrays)
            result['artifacts'].append(record)
            return record

        def scores(prediction, values):
            return {'normalized': math.observed_score(prediction, values['target'], values['observed']),
                    'physical_per_channel_diagnostic': math.observed_score(prediction, values['target'], values['observed'], std)}

        for split, values in raw.items():
            result['inputs'][split] = save('RAW_INPUT_'+split, values)
            result['data_contract'][split] = {'windows': len(values['origins']), 'history_length': 512,
                'forecast_length': 48, 'raw_channels': len(mean), 'FIT_statistics_unchanged': True,
                'target_mask_anchor_stats_exactly_matched': True}
        shared_mean = math.fit_shared_mean(raw['fit']['target'], raw['fit']['raw_last'], raw['fit']['observed'])
        result['shared_mean_artifact'] = save('SHARED_MEAN', shared_mean)
        folder = output/'sources'
        folder.mkdir()
        for path in (Path(__file__), HERE/'readout_math_v1.py', HERE/'adaptive_runtime_v1.py',
                     OLD/'matched_readout_math_v1.py', OLD/'stage_separation_math_v1.py', OLD/'prepare_inputs.py'):
            full()
            target = folder/path.name
            with path.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            if probes.sha(path) != probes.sha(target):
                raise ValueError('SOURCE_SNAPSHOT_HASH_MISMATCH')
            result['source_snapshots'].append({'path': str(path), 'sha256': probes.sha(path), 'snapshot_path': str(target)})
        models_by_route, selected = {}, {}
        for route in ('SELF33', 'JOINT33'):
            full()
            tick = time.perf_counter()
            with threadpool_limits(limits=2):
                prepared = (math.prepare_self('SELF33', raw['fit']['history'], raw['fit']['target'], raw['fit']['raw_last'], raw['fit']['observed'], check=fast)
                    if route == 'SELF33' else math.joint.prepare_route('RAW33C', raw['fit']['history'], raw['fit']['target'], raw['fit']['raw_last'], raw['fit']['observed'], check=fast))
            state = prepared.feature_state
            result['feature_states'][route] = {**state.metadata(), 'artifact': save(route+'_FIT_FEATURE_STATE', {'center': state.center, 'scale': state.scale})}
            row = {'status': 'RUNNING', 'new_training': True, 'preparation': prepared.metadata(), 'alpha_grid': [],
                   'pool': [], 'splits': {}, 'selected_artifacts': {}, 'new_FIT_factorization_wall_seconds': time.perf_counter()-tick}
            result['families'][route] = row
            for kind in ('PERSIST', 'MEAN'):
                row['pool'].append({'kind': kind, **{s+'_scores': scores(math.control_forecast(kind, v['raw_last'], shared_mean), v) for s, v in raw.items()}})
            models = []
            for i, alpha in enumerate(math.ALPHAS):
                full()
                with threadpool_limits(limits=2):
                    model = prepared.fit(alpha)
                arrays = model.arrays if route == 'SELF33' else {k: getattr(model.solution, k) for k in math.SOLUTION_KEYS}
                entry = {'kind': 'RIDGE', 'alpha': alpha, 'model_metadata': model.metadata(), 'model_sha256': model.sha256,
                         'coefficient_artifact': save(route+'_alpha_'+str(i)+'_COEFFICIENTS', arrays)}
                for split, values in raw.items():
                    with threadpool_limits(limits=2):
                        prediction = model.forecast(values['history'], values['raw_last'])
                    entry[split+'_scores'] = scores(prediction, values)
                    entry[split+'_artifact'] = save(route+'_alpha_'+str(i)+'_'+split, {'prediction': prediction, 'origins': values['origins']})
                entry['tune_actual_Y_mse'] = entry['tune_scores']['normalized']['mse']
                row['alpha_grid'].append(entry)
                models.append(model)
                result['new_model_bundles'] += 1
            row['pool'].extend(row['alpha_grid'])
            index = math.select_pool(row['pool'])
            entry = row['pool'][index]
            row.update(selected_index=index, selected_kind=entry['kind'], selected_alpha=entry.get('alpha'))
            if entry['kind'] == 'RIDGE':
                row.update(selected_model_metadata=entry['model_metadata'], selected_model_sha256=entry['model_sha256'],
                           selected_coefficient_artifact=entry['coefficient_artifact'])
            ordered = sorted(p['tune_scores']['normalized']['mse'] for p in row['pool'])
            row['tune_selection_margin_mse'] = ordered[1]-ordered[0]
            for split, values in raw.items():
                with threadpool_limits(limits=2):
                    prediction = (models[index-2].forecast(values['history'], values['raw_last']) if entry['kind'] == 'RIDGE'
                        else math.control_forecast(entry['kind'], values['raw_last'], shared_mean))
                selected[route, split] = prediction
                row['splits'][split] = scores(prediction, values)
                row['selected_artifacts'][split] = save(route+'_SELECTED_'+split,
                    {'prediction': prediction, **{k: values[k] for k in ('target', 'observed', 'origins', 'raw_last', 'mean', 'std')}})
            row['status'] = 'COMPLETE'
            models_by_route[route] = models
            result['pool_comparisons'] += 9
        gate = math.common_information_gate(selected['SELF33', 'tune'], selected['JOINT33', 'tune'], raw['tune']['target'], raw['tune']['observed'])
        lock = {'status': 'LOCKED_BEFORE_NEW_DEV_SCORING', 'dataset': args.dataset, 'locked_unix': time.time(),
            'scientific_inputs_only': ['FIT', 'TUNE'], 'new_DEV_scored_before_lock': False,
            'raw_trainval_member_contains_previously_exposed_DEV_rows': args.dataset == 'robin',
            'new_features_and_targets_constructed_from_FIT_only': True, 'common_information_gate': gate,
            'next_branch': 'SMALLK_PRESERVATION_COMPARISON' if gate['passed'] else 'STOP_COMMON_REPRESENTATION_NO_GAIN_AFTER_MATCHED_FIT512',
            'selected_routes': {r: {'selected_index': result['families'][r]['selected_index'], 'selected_kind': result['families'][r]['selected_kind'],
                'selected_alpha': result['families'][r]['selected_alpha']} for r in ('SELF33', 'JOINT33')}}
        probes.write(output/'BRANCH_LOCK.json', lock)
        result['branch_lock'] = {'path': str(output/'BRANCH_LOCK.json'), 'sha256': probes.sha(output/'BRANCH_LOCK.json'), **lock}
        full()
        final_split = 'val' if args.dataset == 'robin' else 'eval'
        values, checked = probes.load_checked_npz(first['inputs'][final_split])
        values = canonicalize_payload(values)
        result['source_records'].append(checked)
        result['inputs'][final_split] = save('RAW_INPUT_'+final_split, values)
        result['references']['tune'] = first['references']['tune']
        result['references'][final_split] = first['references'][final_split]
        for route in ('SELF33', 'JOINT33'):
            full()
            row = result['families'][route]
            with threadpool_limits(limits=2):
                prediction = (models_by_route[route][row['selected_index']-2].forecast(values['history'], values['raw_last']) if row['selected_kind'] == 'RIDGE'
                    else math.control_forecast(row['selected_kind'], values['raw_last'], shared_mean))
            row['splits'][final_split] = scores(prediction, values)
            row['selected_artifacts'][final_split] = save(route+'_SELECTED_'+final_split,
                {'prediction': prediction, **{k: values[k] for k in ('target', 'observed', 'origins', 'raw_last', 'mean', 'std')}})
        result['status'] = 'COMPLETE'
    except Exception as error:
        import adaptive_runtime_v1 as runtime
        result['status'] = 'BUDGET_HELD_PRESERVED' if isinstance(error, runtime.BudgetHeld) else 'FAILED_PRESERVED'
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        result['wall_seconds'] = time.perf_counter()-started
        terminal = limit.close(result['status'], result.get('failure'))
        result['resource_terminal'] = terminal
        result['status'] = terminal['terminal_status']
        probes.write(output/'RESULT.json', result)
        print(json.dumps({'result': str(output/'RESULT.json'), 'status': result['status'], 'failure': result.get('failure'),
                          'next_branch': result.get('branch_lock', {}).get('next_branch'), 'wall_seconds': result['wall_seconds']}), flush=True)
    return 0 if result['status'] == 'COMPLETE' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dataset', choices=tuple(CONTRACTS), required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--first-result', required=True)
    parser.add_argument('--failed-result', required=True)
    parser.add_argument('--failed-result-sha256', required=True)
    parser.add_argument('--failed-audit', required=True)
    parser.add_argument('--failed-audit-sha256', required=True)
    parser.add_argument('--layout-diagnosis', required=True)
    parser.add_argument('--layout-diagnosis-sha256', required=True)
    args = parser.parse_args()
    if not args.execute:
        return 0
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(name, '2')
    import adaptive_runtime_v1 as runtime
    limit = runtime.ContinuationLimit(args.tag, gpu=False)
    try:
        return execute(args, limit)
    except BaseException as error:
        if not limit.closed:
            limit.close('FAILED_PRESERVED', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
