"""CPU-only whole cost for the one gated FIT512 SELF33/JOINT33 pair."""
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
ROUTES = ('SELF33', 'JOINT33')
REPEATS, WARMUP = 6, 2


def schedule():
    return [list(ROUTES if i % 2 == 0 else ROUTES[::-1]) for i in range(REPEATS)]


def measure_pair(pipelines, physical, observed, references, origins, output, check, cost):
    import numpy as np
    import psutil
    process = psutil.Process()
    rows = []
    output = Path(output)
    output.mkdir(exist_ok=False)
    for batch in (1, 8):
        folder = output / ('B' + str(batch))
        folder.mkdir(exist_ok=False)
        values, masks = physical[:batch], observed[:batch]
        for order in schedule()[:WARMUP]:
            for route in order:
                check()
                pipelines[route].whole(values, masks)
        samples = {route: [] for route in ROUTES}
        for block, order in enumerate(schedule()):
            for position, route in enumerate(order):
                check()
                started = time.perf_counter()
                normalized, physical_prediction = pipelines[route].whole(values, masks)
                seconds = time.perf_counter()-started
                parity = cost.parity(normalized, references[route][:batch])
                payload = cost.save_npz(folder / (route + '_r' + str(block) + '.npz'),
                    {'normalized': normalized, 'physical': physical_prediction, 'origins': origins[:batch]})
                item = {'route': route, 'batch': batch, 'block': block, 'position': position,
                    'seconds': seconds, 'normalized_parity': parity, 'sample_artifact': payload,
                    'resident_rss_bytes': int(process.memory_info().rss)}
                with (folder / (route + '_r' + str(block) + '.json')).open('x', encoding='utf-8') as handle:
                    json.dump(item, handle, ensure_ascii=False, indent=2, allow_nan=False)
                samples[route].append({**item, 'normalized': normalized.copy(), 'physical': physical_prediction.copy()})
                check()
                if not parity['pass']:
                    raise ValueError('EXPANDED_SELECTED_COST_SCIENTIFIC_PARITY_FAILED:' + route)
        for route in ROUTES:
            records = samples[route]
            seconds = [r['seconds'] for r in records]
            payload = cost.save_npz(folder / (route + '_TIMED.npz'), {
                'normalized': np.stack([r['normalized'] for r in records]),
                'physical': np.stack([r['physical'] for r in records]),
                'block': np.array([r['block'] for r in records], np.int64),
                'position': np.array([r['position'] for r in records], np.int64),
                'origins': origins[:batch]})
            rows.append({'dataset': 'robin', 'route': route, 'status': 'COMPLETE', 'batch': batch,
                'repeats': REPEATS, 'warmup_each_route': WARMUP, 'seconds': seconds,
                'positions': [r['position'] for r in records], 'blocks': [r['block'] for r in records],
                'median_seconds': float(np.median(seconds)),
                'median_seconds_per_window': float(np.median(seconds))/batch,
                'sampled_peak_rss_bytes': max(r['resident_rss_bytes'] for r in records),
                'normalized_parity': [r['normalized_parity'] for r in records],
                'timed_outputs': payload, 'sample_artifacts': [r['sample_artifact'] for r in records],
                'GPU_executed': False, 'GPU_resident_model': False,
                'paired_RAW_ratio': None,
                'whole_includes': 'actual causal CPU fill, unchanged FIT normalization, selected CPU readout, CPU inverse, materialized physical CPU return',
                'whole_excludes': 'load, references, guards, hashing, disk save',
                'hardware_trial_scope': 'separate CPU-only process; two routes each position3times; no fresh Raw comparator or cross-trial speed claim'})
    return rows


def execute(args, limit):
    import numpy as np
    import prepare_inputs as inputs
    import run_condition_probes as probes
    import readout_math_v1 as math
    import cost_readout_v1 as cost
    from threadpoolctl import threadpool_limits
    inputs.install_protected_test_guard()
    output = HERE / 'cost_runs' / args.tag
    output.parent.mkdir(exist_ok=True)
    output.mkdir(exist_ok=False)
    result = {'status': 'RUNNING', 'phase': 'MATCHED_FIT512_ONE_DIAGNOSTIC_CPU_COST', 'rows': [],
        'planned_rows': 4, 'new_training_updates': 0, 'GPU_execution': False,
        'protected_TEST_accessed': False, 'independent_holdout': False,
        'source_records': [], 'source_snapshots': [],
        'permission_gate': limit.authorization_record, 'run_manifest': limit.run_manifest_record}
    failure = None
    try:
        if not args.no_contention_attested:
            raise ValueError('ROOT_NO_OTHER_NEW_CAMPAIGN_JOB_ATTESTATION_REQUIRED')
        full, fast = limit.checkfull, limit.shared.checkfast
        full()
        producer_record = probes.verified_file({'path': str(probes.resolve(args.readout_result)), 'sha256': probes.sha(probes.resolve(args.readout_result))})
        producer = probes.read(producer_record['path'])
        if (producer.get('status') != 'COMPLETE' or producer.get('dataset') != 'robin'
                or producer.get('phase') != 'MATCHED_FIT512_ONE_DIAGNOSTIC'
                or producer.get('pool_comparisons') != 18 or producer.get('GPU_execution') is not False
                or producer.get('permission_gate', {}).get('sha256') != limit.authorization_record['sha256']
                or producer.get('run_manifest', {}).get('sha256') != limit.run_manifest_record['sha256']):
            raise ValueError('APPROVED_COMPLETE_ROBIN_FIT512_DIAGNOSTIC_REQUIRED')
        audit_record = probes.verified_file({'path': str(probes.resolve(args.audit)), 'sha256': probes.sha(probes.resolve(args.audit))})
        audit = probes.read(audit_record['path'])
        bound = audit.get('producer_record', {})
        if (audit.get('status') != 'PASS' or bound.get('sha256') != producer_record['sha256']
                or audit.get('pool_candidates_checked') != 18):
            raise ValueError('EXACT_EXPANDED_PRODUCER_INDEPENDENT_AUDIT_PASS_REQUIRED')
        result.update(producer_record=producer_record, independent_scientific_audit=audit_record)
        result['source_records'].extend((producer_record, audit_record))
        snapshots = output / 'sources'
        snapshots.mkdir(exist_ok=False)
        for source in (Path(__file__), HERE/'cost_readout_v1.py', HERE/'readout_math_v1.py', HERE/'adaptive_runtime_v1.py'):
            target = snapshots / source.name
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            if probes.sha(source) != probes.sha(target):
                raise ValueError('EXPANDED_COST_SOURCE_SNAPSHOT_REQUIRED')
            result['source_snapshots'].append({'path': str(source), 'sha256': probes.sha(source), 'snapshot_path': str(target)})
        bridge_record = probes.verified_file(producer['bridge_provenance']['input_bridge'])
        bridge = probes.read(bridge_record['path'])
        phase = bridge['phase_barrier']
        for name in ('audit', 'report', 'manifest'):
            result['source_records'].append(probes.verified_file(phase[name]))
        inputs.verify_phase_barrier(*(probes.resolve(phase[name]['path']) for name in ('audit', 'report', 'manifest')))
        record_rows = [r for r in bridge['artifacts'] if r.get('dataset') == 'robin' and r.get('split') == 'val']
        if len(record_rows) != 1:
            raise ValueError('EXACT_ORIGINAL_ROBIN_VAL_PHYSICAL_INPUT_REQUIRED')
        raw, physical_record = probes.load_checked_npz(record_rows[0])
        scientific_raw, scientific_record = probes.load_checked_npz(producer['inputs']['val'])
        for key, value in scientific_raw.items():
            if key not in raw or not np.array_equal(value, raw[key], equal_nan=True):
                raise ValueError('EXPANDED_COST_DEV_CONDITION_MISMATCH:' + key)
        cost.validate_physical_inputs(raw['history_physical'], raw['history_observed'], raw['history'], raw['mean'], raw['std'], raw['origins'])
        shared_mean, mean_record = probes.load_checked_npz(producer['shared_mean_artifact'])
        result['source_records'].extend((bridge_record, physical_record, scientific_record, mean_record))
        pipelines, references, metadata, objects = {}, {}, {}, {}
        for route in ROUTES:
            family = producer['families'][route]
            kind = family['selected_kind']
            metadata[route] = {'selected_kind': kind, 'selected_alpha': family['selected_alpha'], 'FIT_rows': 512}
            if kind == 'RIDGE':
                state = producer['feature_states'][route]
                feature_arrays, state_record = probes.load_checked_npz(state['artifact'])
                coefficients, coefficient_record = probes.load_checked_npz(family['selected_coefficient_artifact'])
                model = (math.joint.restore_model(state, feature_arrays, family['selected_model_metadata'], coefficients)
                    if route == 'JOINT33' else math.restore_self_model(state, feature_arrays, family['selected_model_metadata'], coefficients))
                if model.sha256 != family['selected_model_sha256']:
                    raise ValueError('EXPANDED_SELECTED_MODEL_SHA_MISMATCH')
                objects[route] = model
                predictor = lambda history, model=model: model.forecast(history, history[:, -1])
                result['source_records'].extend((state_record, coefficient_record))
                metadata[route]['selected_model_sha256'] = model.sha256
            elif kind in ('PERSIST', 'MEAN'):
                predictor = lambda history, kind=kind: math.control_forecast(kind, history[:, -1], shared_mean)
            else:
                raise ValueError('DECLARED_EXPANDED_SELECTED_KIND_REQUIRED')
            pipelines[route] = cost.CPUOutputPipeline(predictor, raw['mean'], raw['std'])
            prediction_arrays, prediction_record = probes.load_checked_npz(family['selected_artifacts']['val'])
            for key in ('origins', 'target', 'observed', 'mean', 'std', 'raw_last'):
                if not np.array_equal(prediction_arrays[key], raw[key], equal_nan=True):
                    raise ValueError('EXPANDED_SELECTED_COST_PREDICTION_CONDITION_MISMATCH')
            references[route] = prediction_arrays['prediction']
            result['source_records'].append(prediction_record)
        result['metadata'] = metadata
        with threadpool_limits(limits=2):
            result['rows'] = measure_pair(pipelines, raw['history_physical'], raw['history_observed'], references,
                raw['origins'], output/'measured', fast, cost)
        for record in result['source_records'] + result['source_snapshots']:
            probes.verified_file(record)
            if 'snapshot_path' in record and probes.sha(record['snapshot_path']) != record['sha256']:
                raise ValueError('EXPANDED_COST_SOURCE_CHANGED_AFTER')
        result.update(status='COMPLETE', source_before_after_SHA_pass=True,
            no_fresh_RAW_baseline=True, cross_trial_speed_superiority_claim=False,
            physical_input_source=physical_record, no_inverse_normalized_input_proxy=True)
        full()
    except Exception as error:
        import adaptive_runtime_v1 as runtime
        failure = {'type': type(error).__name__, 'message': str(error)}
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error, runtime.BudgetHeld) else 'FAILED_PRESERVED', failure=failure)
    finally:
        terminal = limit.close(result['status'], failure)
        result['resource_terminal'] = terminal
        result['status'] = terminal['terminal_status']
        with (output/'RESULT.json').open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({'expanded_cost_result': str(output/'RESULT.json'), 'status': result['status'], 'rows': len(result['rows'])}), flush=True)
    return 0 if result['status'] == 'COMPLETE' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--readout-result', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--no-contention-attested', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'scientific_execution': False, 'planned_rows': 4}))
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
