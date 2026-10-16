"""Independent stdlib audit of explicitly sealed report core-cost statistics."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import sys


PLAN_REL = 'diagnostics/multik_postfinal_resource_1792096163404501000/PLAN.json'
PLAN_SHA = '45908dfa6bf5354c655bc879143d81a440484927885acda8ea4989cbf9808601'
REPORT_SOURCE_SHA = 'e9a4a05abae5d696bd54d9e82b0c14d7b7d3add794fb5ac196e7f950da9f4bf6'
ABS_TOL = 1e-9
REL_TOL = 1e-7
DIRECT = {'DIRECT_G', 'DIRECT_B2', 'DIRECT_B4', 'CURRENT', 'SEASONAL', 'RAW_RIDGE'}


def inside(root, name):
    if (not isinstance(name, str) or not name or '\\' in name or PurePosixPath(name).is_absolute()
            or any(x in ('', '.', '..') or ':' in x for x in name.split('/'))):
        raise ValueError('UNSAFE_RELATIVE_PATH')
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('RELATIVE_PATH_ESCAPED_ROOT')
    return path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def finite(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError('NONFINITE_OR_NEGATIVE_TIMER')
    return number


def equal(actual, expected):
    return math.isfinite(float(expected)) and math.isclose(actual, float(expected), abs_tol=ABS_TOL, rel_tol=REL_TOL)


def linear_p90(values):
    values = sorted(values)
    position = (len(values) - 1) * .9
    lower = math.floor(position); upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def expected_F(spec, batch, channels):
    family = spec['family']
    if family in DIRECT:
        return 0, 0
    if family in {'COMBINATION', 'ALPHA', 'RAW_CALIBRATED'}:
        children = [spec[key] for key in ('g_route', 'b_route') if key in spec
                    and not (key == 'b_route' and spec.get('variant') == 'past_control')]
        counts = [expected_F(child, batch, channels) for child in children]
        return sum(x[0] for x in counts), sum(x[1] for x in counts)
    if family in {'RAW', 'LORA'}:
        streams = batch * channels
    elif family in {'B0', 'B2', 'B4', 'LAG_CAUSAL', 'LAG_BIDIR', 'ADAPTS', 'PCA_FIXED_BOLT', 'NESTED_PCA'}:
        streams = batch * int(spec['latent_k'])
    else:
        raise ValueError('UNSUPPORTED_F_COUNTER_FAMILY')
    micro = spec.get('row_microbatch', spec.get('F_row_microbatch', 'full' if family == 'RAW' else 128))
    calls = 1 if micro in (None, 'full', 0) else math.ceil(streams / int(micro))
    return streams, calls


def audit_receipt(receipt, trials, report_rows, inputs):
    condition_ids = receipt.get('run_ids', [receipt.get('run_id')])
    if not condition_ids or None in condition_ids or len(set(condition_ids)) != len(condition_ids):
        raise ValueError('INVALID_CONDITION_ID_SET')
    balanced = 'run_ids' in receipt
    requests = [(phase, int(receipt[key + ('_per_route' if balanced else '')]), instrumented)
                for phase, key, instrumented in [('warmup', 'warmups', False), ('timed', 'timed_trials', False), ('stage_probe', 'stage_trials', True)]]
    if any(count < 0 for _, count, _ in requests) or requests[1][1] < 1:
        raise ValueError('INVALID_DECLARED_TRIAL_COUNTS')
    minimum = all(count >= threshold for (_, count, _), threshold in zip(requests, (10, 20, 20)))
    protocol_complete = receipt['status'] == 'COMPLETE_PROTOCOL' and minimum
    expected_order = [(phase, repeat, position, condition_ids[(repeat + position) % len(condition_ids)], instrumented)
                      for phase, count, instrumented in requests for repeat in range(count) for position in range(len(condition_ids))]
    actual_order = [(row['phase'], int(row['trial']), int(row['order_position']), row['run_id'], row['stage_instrumentation'] == 'True') for row in trials]
    if actual_order != expected_order:
        raise ValueError('PHASE_TRIAL_BALANCED_ROTATION_OR_STAGE_FLAG_MISMATCH')
    batch = int(receipt['batch_size']); channels = int(receipt['C'])
    bindings = receipt['accuracy_bindings']
    if set(bindings) != set(condition_ids):
        raise ValueError('BINDING_CONDITION_ID_SET_MISMATCH')
    report_by_id = {row['cost_run_id']: row for row in report_rows}
    if len(report_by_id) != len(report_rows) or set(report_by_id) != set(condition_ids):
        raise ValueError('SEALED_REPORT_CONDITION_SET_MISMATCH')
    for condition in condition_ids:
        binding = bindings[condition]; spec = binding['cost_route_spec']; accuracy = binding['accuracy_route_spec']
        if (binding.get('status') != 'BOUND_EXACT_ROUTE' or binding['condition_id'] != condition
                or spec['run_id'] != condition or spec.get('accuracy_run_id') != binding['accuracy_run_id']
                or accuracy['run_id'] != binding['accuracy_run_id']
                or fingerprint(spec) != binding['cost_route_spec_sha256']
                or fingerprint(accuracy) != binding['accuracy_route_spec_sha256']):
            raise ValueError('EXACT_ACCURACY_SPEC_BINDING_MISMATCH')
        ignored = {'run_id', 'accuracy_run_id', 'row_microbatch', 'cost_origins'}
        if {k:v for k,v in spec.items() if k not in ignored} != {k:v for k,v in accuracy.items() if k not in ignored}:
            raise ValueError('COST_ONLY_FIELDS_CHANGE_ACCURACY_ROUTE')
        entry = inputs.get(('public', binding['accuracy_manifest_public_relative_path']))
        if not entry or entry['sha256'] != binding['accuracy_manifest_sha256']:
            raise ValueError('ACCURACY_MANIFEST_NOT_BOUND_IN_REPORT_INPUTS')
        if binding['data_sha256'] != receipt['data_sha256']:
            raise ValueError('RECEIPT_DATA_BINDING_DIFFERS')
        if not balanced and binding['scaler_sha'] != receipt['scaler_sha']:
            raise ValueError('RECEIPT_SCALER_BINDING_DIFFERS')
        for artifact in binding['artifact_bindings'].values():
            entry = inputs.get(('private', artifact['cache_relative_path']))
            if not entry or entry['sha256'] != artifact['sha256']:
                raise ValueError('ARTIFACT_NOT_BOUND_IN_REPORT_METADATA')
        expected_streams, expected_calls = expected_F(spec, batch, channels)
        recorded_micro = spec.get('row_microbatch', 'full' if spec['family'] == 'RAW' else 128)
        for row in [r for r in trials if r['run_id'] == condition]:
            if (int(row['batch_size']) != batch or int(row['F_logical_streams']) != expected_streams
                    or int(row['F_physical_streams']) != expected_streams or int(row['F_calls']) != expected_calls):
                raise ValueError('PHYSICAL_LOGICAL_F_STREAM_OR_CALL_COUNT_MISMATCH')
            if row['row_microbatch'] != str(recorded_micro):
                raise ValueError('RECORDED_TOP_LEVEL_ROW_MICROBATCH_MISMATCH')
            if row['stage_instrumentation'] not in ('True', 'False'):
                raise ValueError('NONBOOLEAN_STAGE_FLAG')
            finite(row['whole_seconds'])
            stage_values = [value for key, value in row.items() if key.startswith('stage_') and key.endswith('_seconds') and value != '']
            for value in stage_values:
                finite(value)
            if (row['phase'] != 'stage_probe' and stage_values) or (row['phase'] == 'stage_probe' and not stage_values):
                raise ValueError('STAGE_COLUMNS_DO_NOT_MATCH_INSTRUMENTED_PHASE')
    saved = {(s['run_id'], s['phase']): s for s in receipt['summaries']}
    expected_keys = {(condition, phase) for condition in condition_ids for phase, count, _ in requests if phase != 'warmup' and count}
    if len(saved) != len(receipt['summaries']) or set(saved) != expected_keys:
        raise ValueError('SAVED_SUMMARY_KEY_SET_MISMATCH')
    result = []
    for condition in condition_ids:
        for phase, count, instrumented in requests:
            if phase == 'warmup' or not count:
                continue
            values = [finite(row['whole_seconds']) for row in trials if (row['run_id'], row['phase']) == (condition, phase)]
            computed = dict(median_seconds=statistics.median(values), mean_seconds=math.fsum(values) / len(values),
                            min_seconds=min(values), max_seconds=max(values), p90_seconds=linear_p90(values))
            original = saved[(condition, phase)]
            if (original['trials'] != len(values) or original['batch_size'] != batch
                    or original['stage_instrumentation'] is not instrumented
                    or original.get('stage_sum_equals_whole_claimed') is not False
                    or original['condition_id'] != condition
                    or original['accuracy_run_id'] != bindings[condition]['accuracy_run_id']
                    or any(not equal(value, original[key]) for key, value in computed.items())):
                raise ValueError('SAVED_MEDIAN_MEAN_MIN_MAX_P90_OR_FLAGS_MISMATCH')
            report = report_by_id[condition]
            if (report['run_id'] != bindings[condition]['accuracy_run_id'] or int(report['batch_size']) != batch
                    or report['scope'] != receipt['scope'] or report['dataset'] != receipt['dataset']
                    or report['accuracy_binding_status'] != 'BOUND_EXACT_ROUTE'
                    or report['status'] != ('COMPLETE_PROTOCOL' if protocol_complete else 'PARTIAL_COST_PROTOCOL')
                    or report['trials_sha256'] != receipt['trials_sha256']):
                raise ValueError('COSTS_REPORT_CONDITION_IDENTITY_MISMATCH')
            if phase == 'timed' and (int(report['timed_trials']) != len(values)
                    or not equal(computed['median_seconds'], report['median_seconds'])
                    or not equal(computed['mean_seconds'], report['mean_seconds'])):
                raise ValueError('SEALED_COSTS_REPORT_NUMBERS_DIFFER')
            result.append(dict(cost_run_id=condition, accuracy_run_id=bindings[condition]['accuracy_run_id'],
                phase=phase, trials=len(values), batch_size=batch, scope=receipt['scope'],
                stage_instrumentation=instrumented, **computed))
    if receipt['status'] == 'COMPLETE_PROTOCOL' and not minimum:
        raise ValueError('FALSE_COMPLETE_PROTOCOL_MINIMUM')
    return result, dict(actual_trials=len(trials), conditions=len(condition_ids), complete_protocol=protocol_complete)


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def execute(args, public, plan):
    if not args.root_triggered:
        raise ValueError('EXPLICIT_ROOT_REPORT_TERMINAL_TRIGGER_REQUIRED')
    cache = Path(os.environ['INFORMS_CACHE_ROOT']).resolve()
    if (cache / 'ACTIVE_GPU_LEASE.json').exists():
        raise ValueError('ACTIVE_GPU_COST_OWNER_PRESENT')
    report = inside(public, args.report_relative)
    paths = {name: report / name for name in ('COSTS.csv', 'INPUT_MANIFEST.json', 'REPORT_MANIFEST.json')}
    expected = {'COSTS.csv': args.costs_sha256, 'INPUT_MANIFEST.json': args.inputs_sha256, 'REPORT_MANIFEST.json': args.report_manifest_sha256}
    if any(file_sha(path) != expected[name] for name, path in paths.items()):
        raise ValueError('EXPLICIT_REPORT_SHA_MISMATCH')
    report_manifest = json.loads(paths['REPORT_MANIFEST.json'].read_bytes())
    if report_manifest.get('source_sha256') != REPORT_SOURCE_SHA:
        raise ValueError('REPORT_GENERATOR_SOURCE_CHANGED')
    outputs = {row['file']: row for row in report_manifest['outputs']}
    if any(outputs[name]['sha256'] != expected[name] or outputs[name]['bytes'] != paths[name].stat().st_size for name in ('COSTS.csv', 'INPUT_MANIFEST.json')):
        raise ValueError('REPORT_OUTPUT_MANIFEST_BINDING_MISMATCH')
    rows = list(csv.DictReader(paths['COSTS.csv'].open(encoding='utf-8-sig', newline='')))
    original_inputs = json.loads(paths['INPUT_MANIFEST.json'].read_bytes())['inputs']
    inputs = {(entry['scope'], entry['file']): entry for entry in original_inputs}
    if len(inputs) != len(original_inputs):
        raise ValueError('DUPLICATE_REPORT_INPUT_BINDING')
    frozen = {name: (public / name).read_bytes() for name in plan['source39']}
    if {name: hashlib.sha256(raw).hexdigest() for name, raw in frozen.items()} != plan['source39']:
        raise ValueError('SOURCE39_CHANGED')
    registry = (public / 'EXPERIMENT_REGISTRY.csv').read_bytes(); lock = (public / 'SELECTION_LOCK.json').read_bytes()
    records = []; details = []; bindings = {}
    for relative in sorted({row['receipt_relative_path'] for row in rows}):
        record = dict(receipt_relative_path=relative)
        try:
            path = inside(public, relative); entry = inputs.get(('public', relative))
            if not entry or file_sha(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
                raise ValueError('RECEIPT_SHA_OR_BYTES_NOT_BOUND_BY_REPORT')
            receipt = json.loads(path.read_bytes()); bindings[relative] = entry
            if receipt.get('source_sha256') != {name: plan['source39'][name] for name in ('deployment.py', 'measure_costs.py')}:
                raise ValueError('MEASUREMENT_SOURCE_SHA_DIFFERS')
            trial_path = inside(path.parent, receipt['trials_file']); trial_relative = trial_path.relative_to(public).as_posix()
            trial_entry = inputs.get(('public', trial_relative))
            if (not trial_entry or trial_path.stat().st_size != trial_entry['bytes']
                    or trial_entry['bytes'] > 20*1024**2 or file_sha(trial_path) != trial_entry['sha256']
                    or trial_entry['sha256'] != receipt['trials_sha256']):
                raise ValueError('TRIAL_GZIP_SHA_OR_BYTES_NOT_BOUND_BY_REPORT')
            bindings[trial_relative] = trial_entry
            with gzip.open(trial_path, 'rt', encoding='utf-8', newline='') as stream:
                trials = list(csv.DictReader(stream))
            computed, counts = audit_receipt(receipt, trials, [row for row in rows if row['receipt_relative_path'] == relative], inputs)
            for row in computed:
                row.update(receipt_relative_path=relative, receipt_sha256=entry['sha256'], trials_sha256=trial_entry['sha256'])
            details.extend(computed); record.update(status='PASS_RECORDED_NUMBERS', **counts)
        except Exception as error:
            record.update(status='FAILED_PRESERVED', error_type=type(error).__name__, reason_code=str(error) if isinstance(error, ValueError) else 'MISSING_OR_UNSUPPORTED_RECORDED_STRUCTURE')
        records.append(record)
    if (any((public / name).read_bytes() != raw for name, raw in frozen.items())
            or registry != (public / 'EXPERIMENT_REGISTRY.csv').read_bytes() or lock != (public / 'SELECTION_LOCK.json').read_bytes()
            or any(file_sha(path) != expected[name] for name, path in paths.items())
            or any(file_sha(inside(public, name)) != item['sha256'] for name, item in bindings.items())):
        raise ValueError('SOURCE_REGISTRY_LOCK_REPORT_OR_STATS_CHANGED_DURING_AUDIT')
    out = inside(public, args.output); out.mkdir(parents=True, exist_ok=False)
    failures = [row['receipt_relative_path'] for row in records if row['status'] == 'FAILED_PRESERVED']
    result = dict(status='PASS_ALL_REFERENCED_COST_NUMBERS' if not failures else 'FAILED_RECEIPTS_PRESERVED',
        report_relative_path=args.report_relative, report_input_SHA=expected, receipts=records, failed_receipts=failures,
        report_condition_count=len(rows), summary_rows=len(details), source39_registry_lock_preserved=True,
        absolute_tolerance=ABS_TOL, relative_tolerance=REL_TOL, tolerance_changed_during_execution=False,
        p90_algorithm='sorted linear interpolation at (n-1)*0.9; stdlib only',
        mean_algorithm='math.fsum(values)/n; compared within fixed requested tolerance',
        original_failed_cost_scopes_not_fabricated=True, model_NPZ_weights_or_GPU_read=False,
        historical_memory_remeasured=False, stored_prediction_SHA_recomputed=False, new_measurement=False)
    new_json(out / 'AUDIT_RESULT.json', result); new_json(out / 'FILE_BINDINGS.json', bindings)
    if details:
        with (out / 'SUMMARY_RECALCULATIONS.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(details[0])); writer.writeheader(); writer.writerows(details)
    print(json.dumps(dict(status=result['status'], receipts=len(records), conditions=len(rows), failed=len(failures), output_relative_path=args.output)))
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--root-triggered', action='store_true')
    for name in ('report-relative', 'costs-sha256', 'inputs-sha256', 'report-manifest-sha256', 'output'):
        parser.add_argument('--' + name)
    args = parser.parse_args(); public = Path(os.environ['INFORMS_PUBLIC_ROOT']).resolve()
    if file_sha(public / PLAN_REL) != PLAN_SHA or file_sha(public / 'report_campaign.py') != REPORT_SOURCE_SHA:
        raise ValueError('PINNED_STATIC_SOURCE_OR_PLAN_CHANGED')
    plan = json.loads((public / PLAN_REL).read_bytes())
    if args.execute:
        if not all((args.report_relative, args.costs_sha256, args.inputs_sha256, args.report_manifest_sha256, args.output)):
            parser.error('Explicit sealed report metadata+SHA and new output required')
        raise SystemExit(execute(args, public, plan))
    print(json.dumps(dict(status='PREPARATION_ONLY_ACTUAL_REPORT_AND_TRIALS_NOT_READ', absolute_tolerance=ABS_TOL, relative_tolerance=REL_TOL,
        arrays_models_or_GPU=False, framework_imports=[n for n in ('numpy', 'torch', 'pandas') if n in sys.modules])))


if __name__ == '__main__':
    main()
