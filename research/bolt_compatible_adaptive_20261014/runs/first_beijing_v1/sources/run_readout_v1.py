"""Fresh guarded SELF33/JOINT33/SELF512 diagnostic; no scientific reads on import."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'research/smallk_failure_followup_20261014'
sys.path[:0] = [str(HERE), str(OLD), str(ROOT)]
PRODUCERS = {
    'robin': ROOT / '.cache/smallk_failure_followup_20261014/matched_readout_v1/matched_robin_s92601_v1/RESULT.json',
    'beijing_aotizhongxin': ROOT / '.cache/smallk_failure_followup_20261014/matched_readout_v1/matched_beijing_s92601_v1/RESULT.json',
}
EXPECTED_MATH_SHA = '304b5ae9bf5fa1cc3957cf14c17bea4d2f24fd9c87b00e271e1fc77d01c0088f'
EXPECTED_STAGE_SHA = '8ce0ceb4d7247e05d517f2658aad92e3835e46134cfd48d7c2ad482fe69a2eab'


def raw_artifact(producer, split):
    rows = [r for r in producer['artifacts'] if Path(r['path']).stem == 'RAW_INPUT_' + split]
    if len(rows) != 1:
        raise ValueError('EXACT_ONE_ORIGINAL_RAW_INPUT_REQUIRED:' + split)
    return rows[0]


def execute(args, limit):
    import numpy as np
    import readout_math_v1 as math
    import run_condition_probes as probes
    import prepare_inputs as inputs
    import matched_readout_contract_v1 as contract
    from threadpoolctl import threadpool_limits
    inputs.install_protected_test_guard()
    full, fast = limit.checkfull, limit.shared.checkfast
    output = HERE / 'runs' / args.tag
    output.parent.mkdir(exist_ok=True)
    output.mkdir(exist_ok=False)
    result = {'status': 'RUNNING', 'dataset': args.dataset, 'scientific_execution': True,
        'protected_TEST_accessed': False, 'GPU_execution': False, 'model_weights_loaded': False,
        'independent_holdout': False, 'old_campaign_restart': False, 'deterministic_ridge_no_seed_replication': True,
        'families': {}, 'feature_states': {}, 'references': {}, 'inputs': {}, 'artifacts': [],
        'source_records': [], 'source_snapshots': [], 'data_contract': {},
        'permission_gate': limit.authorization_record, 'run_manifest': limit.run_manifest_record,
        'new_model_bundles': 0, 'reused_model_bundles': 0, 'pool_comparisons': 0,
        'selection': 'one global normalized observed-channel macro TUNE MSE; earliest exact tie',
        'numpy_version': np.__version__}
    started = time.perf_counter()
    terminal = None
    try:
        full()
        path = PRODUCERS[args.dataset]
        producer_record = probes.verified_file({'path': str(path), 'sha256': probes.sha(path)}, role='fixed_before_outcome_joint_producer')
        producer = probes.read(path)
        if producer.get('status') != 'COMPLETE' or producer.get('dataset') != args.dataset or producer.get('seed') != 92601:
            raise ValueError('FIXED_COMPLETED_MATCHED_PRODUCER_REQUIRED')
        result['old_producer'] = producer_record
        result['source_records'].append(producer_record)
        bridge_record = probes.verified_file(producer['bridge_provenance']['input_bridge'])
        bridge = probes.read(bridge_record['path'])
        if bridge.get('status') != 'COMPLETE' or args.dataset not in bridge.get('datasets', {}):
            raise ValueError('ORIGINAL_COMPLETED_INPUT_BRIDGE_REQUIRED')
        phase = bridge['phase_barrier']
        for name in ('audit', 'report', 'manifest'):
            probes.verified_file(phase[name])
        result['phase_barrier'] = inputs.verify_phase_barrier(*(probes.resolve(phase[name]['path']) for name in ('audit', 'report', 'manifest')))
        result['bridge_provenance'] = producer['bridge_provenance']
        result['source_records'].append(bridge_record)
        result['split_roles'] = bridge['datasets'][args.dataset]['split_roles']

        def save(name, arrays):
            full()
            record = probes.save_npz(output / (name + '.npz'), arrays)
            result['artifacts'].append(record)
            full()
            return record

        def scores(prediction, values):
            if prediction.dtype != np.float32 or prediction.shape != values['target'].shape or not np.isfinite(prediction).all():
                raise ValueError('FINITE_FP32_N48C_REQUIRED')
            return {'normalized': math.observed_score(prediction, values['target'], values['observed']),
                    'physical_per_channel_diagnostic': math.observed_score(prediction, values['target'], values['observed'], std)}

        raw = {}
        mean = std = None
        final_split = 'val' if args.dataset == 'robin' else 'eval'

        def load_raw(split):
            nonlocal mean, std
            full()
            arrays, record = probes.load_checked_npz(raw_artifact(producer, split))
            bridge_rows = [r for r in bridge['artifacts'] if r.get('dataset') == args.dataset and r.get('split') == split]
            if len(bridge_rows) != 1:
                raise ValueError('EXACT_ORIGINAL_BRIDGE_SPLIT_REQUIRED')
            original, original_record = probes.load_checked_npz(bridge_rows[0])
            for key in arrays:
                if key not in original:
                    raise ValueError('ORIGINAL_BRIDGE_ARRAY_REQUIRED:' + key)
                contract.exact(arrays[key], original[key], split + ':' + key)
            if mean is None:
                mean, std = arrays['mean'], arrays['std']
            contract.exact(arrays['mean'], mean, split + ':FIT_mean')
            contract.exact(arrays['std'], std, split + ':FIT_std')
            result['data_contract'][split] = contract.validate_raw_split(args.dataset, split, arrays, mean, std)
            if np.any(np.diff(arrays['origins']) <= 0):
                raise ValueError('STRICTLY_INCREASING_TIME_ORIGINS_REQUIRED')
            raw[split] = arrays
            result['inputs'][split] = save('RAW_INPUT_' + split, arrays)
            result['source_records'].extend((record, original_record))
            result['references'][split] = {'RAW_BOLT': scores(arrays['raw_forecast'], arrays)}
            return arrays

        # DEV payload is deliberately absent until all selection and branch fields are sealed.
        load_raw('fit')
        load_raw('tune')
        shared_mean = math.fit_shared_mean(raw['fit']['target'], raw['fit']['raw_last'], raw['fit']['observed'])
        result['shared_mean_artifact'] = save('SHARED_MEAN', shared_mean)
        folder = output / 'sources'
        folder.mkdir()
        for source in (Path(__file__), HERE/'readout_math_v1.py', HERE/'adaptive_runtime_v1.py',
                       OLD/'matched_readout_math_v1.py', OLD/'stage_separation_math_v1.py',
                       OLD/'run_condition_probes.py', OLD/'prepare_inputs.py', OLD/'matched_readout_contract_v1.py'):
            full()
            target = folder / source.name
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            if probes.sha(source) != probes.sha(target):
                raise ValueError('SOURCE_SNAPSHOT_PARITY_REQUIRED')
            result['source_snapshots'].append({'path': str(source), 'sha256': probes.sha(source), 'snapshot_path': str(target)})

        models_by_route = {}

        def fit_new(route):
            full()
            t0 = time.perf_counter()
            with threadpool_limits(limits=2):
                if route == 'JOINT33':
                    prepared = math.joint.prepare_route('RAW33C', raw['fit']['history'], raw['fit']['target'],
                        raw['fit']['raw_last'], raw['fit']['observed'], check=fast)
                else:
                    prepared = math.prepare_self(route, raw['fit']['history'], raw['fit']['target'],
                        raw['fit']['raw_last'], raw['fit']['observed'], check=fast)
            row = {'status': 'RUNNING', 'new_training': True, 'alpha_grid': [], 'preparation': prepared.metadata(),
                   'new_FIT_factorization_wall_seconds': time.perf_counter()-t0, 'splits': {}, 'selected_artifacts': {}}
            result['families'][route] = row
            state = prepared.feature_state
            result['feature_states'][route] = {**state.metadata(), 'artifact': save(route+'_FIT_FEATURE_STATE', {'center': state.center, 'scale': state.scale})}
            models = []
            for index, alpha in enumerate(math.ALPHAS):
                full()
                with threadpool_limits(limits=2):
                    model = prepared.fit(alpha)
                arrays = model.arrays if route != 'JOINT33' else {k: getattr(model.solution, k) for k in math.SOLUTION_KEYS}
                entry = {'alpha': alpha, 'kind': 'RIDGE', 'model_metadata': model.metadata(),
                         'model_sha256': model.sha256, 'coefficient_artifact': save(route+'_alpha_'+str(index)+'_COEFFICIENTS', arrays)}
                for split in ('fit', 'tune'):
                    full()
                    with threadpool_limits(limits=2):
                        prediction = model.forecast(raw[split]['history'], raw[split]['raw_last'])
                    entry[split+'_scores'] = scores(prediction, raw[split])
                    entry[split+'_artifact'] = save(route+'_alpha_'+str(index)+'_'+split,
                        {'prediction': prediction, 'origins': raw[split]['origins']})
                entry['tune_actual_Y_mse'] = entry['tune_scores']['normalized']['mse']
                row['alpha_grid'].append(entry)
                models.append(model)
                result['new_model_bundles'] += 1
            models_by_route[route] = models
            row['status'] = 'COMPLETE'

        def reuse_joint():
            full()
            for name, expected in (('matched_readout_math_v1.py', EXPECTED_MATH_SHA), ('stage_separation_math_v1.py', EXPECTED_STAGE_SHA)):
                records = [r for r in producer['source_snapshots'] if Path(r['path']).name == name]
                if len(records) != 1 or records[0]['sha256'] != expected or probes.sha(OLD/name) != expected:
                    raise ValueError('REUSED_RIDGE_EXACT_SOURCE_REQUIRED:' + name)
                if probes.sha(records[0]['snapshot_path']) != expected:
                    raise ValueError('REUSED_RIDGE_ORIGINAL_SNAPSHOT_REQUIRED:' + name)
            old = producer['families']['RAW33C']
            if old.get('status') != 'COMPLETE' or tuple(r['alpha'] for r in old['alpha_grid']) != math.ALPHAS:
                raise ValueError('COMPLETE_ORIGINAL_SEVEN_ALPHA_REQUIRED')
            feature_metadata = old['feature_state']
            feature_arrays, checked_state = probes.load_checked_npz(feature_metadata['artifact'])
            computed = math.joint.fit_feature_state('RAW33C', raw['fit']['history'])
            if computed.sha256 != feature_metadata['sha256']:
                raise ValueError('REUSED_FIT_FEATURE_CONTENT_SHA_MISMATCH')
            contract.exact(feature_arrays['center'], computed.center, 'FIT_feature_center')
            contract.exact(feature_arrays['scale'], computed.scale, 'FIT_feature_scale')
            row = {'status': 'RUNNING', 'new_training': False, 'alpha_grid': [], 'preparation': old['preparation'],
                   'reuse_gate': [], 'splits': {}, 'selected_artifacts': {}, 'old_selected_index': old['selected_index'],
                   'old_selected_alpha': old['selected_alpha'], 'old_seven_pool_preserved': True}
            models = []
            for index, entry in enumerate(old['alpha_grid']):
                full()
                coefficients, checked = probes.load_checked_npz(entry['coefficient_artifact'])
                model = math.joint.restore_model(feature_metadata, feature_arrays, entry['model_metadata'], coefficients)
                np.testing.assert_array_equal(coefficients['observed_counts'], raw['fit']['observed'].sum(axis=0))
                channel_counts = raw['fit']['observed'].sum(axis=(0, 1))
                np.testing.assert_array_equal(coefficients['channel_counts'], channel_counts)
                np.testing.assert_array_equal(coefficients['normal_equation_denominators'], np.tile(channel_counts, (48, 1)))
                fresh = {'alpha': entry['alpha'], 'kind': 'RIDGE', 'model_metadata': entry['model_metadata'],
                         'model_sha256': model.sha256, 'coefficient_artifact': entry['coefficient_artifact']}
                checks = {'alpha': entry['alpha'], 'model_sha256': model.sha256, 'splits': {}}
                for split in ('fit', 'tune'):
                    full()
                    stored, checked_prediction = probes.load_checked_npz(entry[split+'_artifact'])
                    contract.exact(stored['origins'], raw[split]['origins'], 'reused_candidate_origins:'+split)
                    with threadpool_limits(limits=2):
                        prediction = model.forecast(raw[split]['history'], raw[split]['raw_last'])
                    parity = probes.compare_prediction(prediction, stored['prediction'])
                    if not parity['pass']:
                        raise ValueError('REUSED_COEFFICIENT_PREDICTION_PARITY_FAILED')
                    fresh[split+'_scores'] = scores(prediction, raw[split])
                    for metric in ('mse', 'mae'):
                        if not np.isclose(fresh[split+'_scores']['normalized'][metric], entry[split+'_scores']['normalized'][metric], atol=1e-7, rtol=1e-7):
                            raise ValueError('REUSED_NORMALIZED_SCORE_PARITY_FAILED')
                    fresh[split+'_artifact'] = save('JOINT33_REUSED_alpha_'+str(index)+'_'+split,
                        {'prediction': prediction, 'origins': raw[split]['origins']})
                    result['source_records'].append(checked_prediction)
                    checks['splits'][split] = parity
                fresh['tune_actual_Y_mse'] = fresh['tune_scores']['normalized']['mse']
                row['alpha_grid'].append(fresh)
                row['reuse_gate'].append(checks)
                result['source_records'].append(checked)
                models.append(model)
            result['source_records'].append(checked_state)
            result['feature_states']['JOINT33'] = {**feature_metadata, 'artifact': feature_metadata['artifact']}
            result['families']['JOINT33'] = row
            models_by_route['JOINT33'] = models
            result['reused_model_bundles'] += 7
            row['status'] = 'COMPLETE'

        fit_new('SELF33')
        try:
            reuse_joint()
        except Exception as error:
            import adaptive_runtime_v1 as runtime
            if isinstance(error, runtime.BudgetHeld):
                raise
            result['joint_reuse_failure_preserved'] = {'type': type(error).__name__, 'message': str(error)}
            fit_new('JOINT33')
        fit_new('SELF512')
        selected_predictions = {}
        for route in math.ROUTES:
            full()
            row = result['families'][route]
            row['pool'] = []
            for kind in ('PERSIST', 'MEAN'):
                control = {'kind': kind}
                for split in ('fit', 'tune'):
                    prediction = math.control_forecast(kind, raw[split]['raw_last'], shared_mean)
                    control[split+'_scores'] = scores(prediction, raw[split])
                row['pool'].append(control)
            row['pool'].extend(row['alpha_grid'])
            chosen_index = math.select_pool(row['pool'])
            selected = row['pool'][chosen_index]
            row.update(selected_index=chosen_index, selected_kind=selected['kind'], selected_alpha=selected.get('alpha'))
            if selected['kind'] == 'RIDGE':
                row.update(selected_model_metadata=selected['model_metadata'], selected_model_sha256=selected['model_sha256'],
                           selected_coefficient_artifact=selected['coefficient_artifact'])
            ordered = sorted(v['tune_scores']['normalized']['mse'] for v in row['pool'])
            row['tune_selection_margin_mse'] = ordered[1]-ordered[0]
            row['tune_selection_margin_rel_pct'] = 100*(ordered[1]/ordered[0]-1) if ordered[0] > 1e-8 else None
            for split in ('fit', 'tune'):
                values = raw[split]
                if selected['kind'] == 'RIDGE':
                    with threadpool_limits(limits=2):
                        prediction = models_by_route[route][chosen_index-2].forecast(values['history'], values['raw_last'])
                else:
                    prediction = math.control_forecast(selected['kind'], values['raw_last'], shared_mean)
                row['splits'][split] = scores(prediction, values)
                row['selected_artifacts'][split] = save(route+'_SELECTED_'+split,
                    {'prediction': prediction, **{k: values[k] for k in ('target', 'observed', 'origins', 'raw_last')}, 'mean': mean, 'std': std})
                selected_predictions[route, split] = prediction
            result['pool_comparisons'] += 9

        gate = math.common_information_gate(selected_predictions['SELF33', 'tune'], selected_predictions['JOINT33', 'tune'],
            raw['tune']['target'], raw['tune']['observed'])
        overfit = (result['families']['JOINT33']['splits']['fit']['normalized']['mse'] < result['families']['SELF33']['splits']['fit']['normalized']['mse']
            and gate['joint_tune']['mse'] >= gate['self_tune']['mse'])
        branch = 'SMALLK_PRESERVATION_COMPARISON' if gate['passed'] else 'MATCHED_FIT512_ONE_DIAGNOSTIC' if overfit else 'STOP_COMMON_REPRESENTATION_WEAK_EVIDENCE'
        lock = {'status': 'LOCKED_BEFORE_DEV_PAYLOAD_READ', 'dataset': args.dataset, 'locked_unix': time.time(),
                'scientific_inputs_only': ['FIT', 'TUNE'], 'DEV_payload_opened_before_lock': False,
                'common_information_gate': gate, 'joint_fit_good_tune_bad': bool(overfit), 'next_branch': branch,
                'selected_routes': {r: {'selected_index': result['families'][r]['selected_index'],
                    'selected_kind': result['families'][r]['selected_kind'], 'selected_alpha': result['families'][r]['selected_alpha']} for r in math.ROUTES},
                'Raw_and_SELF512_method_adoption_is_a_separate_gate': True,
                'old_producer_json_contains_previously_exposed_DEV_scores': True,
                'old_DEV_fields_not_used_for_new_selection_or_branch': True}
        probes.write(output/'BRANCH_LOCK.json', lock)
        result['branch_lock'] = {'path': str(output/'BRANCH_LOCK.json'), 'sha256': probes.sha(output/'BRANCH_LOCK.json'), **lock}
        full()
        load_raw(final_split)
        values = raw[final_split]
        for route in math.ROUTES:
            full()
            row = result['families'][route]
            if row['selected_kind'] == 'RIDGE':
                with threadpool_limits(limits=2):
                    prediction = models_by_route[route][row['selected_index']-2].forecast(values['history'], values['raw_last'])
            else:
                prediction = math.control_forecast(row['selected_kind'], values['raw_last'], shared_mean)
            row['splits'][final_split] = scores(prediction, values)
            row['selected_artifacts'][final_split] = save(route+'_SELECTED_'+final_split,
                {'prediction': prediction, **{k: values[k] for k in ('target', 'observed', 'origins', 'raw_last')}, 'mean': mean, 'std': std})
        result['reference_shared_controls'] = {}
        for kind in ('PERSIST', 'MEAN'):
            result['reference_shared_controls'][kind] = {s: scores(math.control_forecast(kind, v['raw_last'], shared_mean), v) for s, v in raw.items()}
        full()
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
        print(json.dumps({'result': str(output/'RESULT.json'), 'status': result['status'],
                          'new_model_bundles': result['new_model_bundles'], 'reused_model_bundles': result['reused_model_bundles'],
                          'next_branch': result.get('branch_lock', {}).get('next_branch'), 'wall_seconds': result['wall_seconds']}), flush=True)
    return 0 if result['status'] == 'COMPLETE' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dataset', choices=tuple(PRODUCERS), required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'scientific_execution': False, 'first_stage': 'SELF33 JOINT33 SELF512 same nine-candidate pool'}))
        return 0
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(name, '2')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
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
