"""Independent stdlib audit of recorded K-transition costs; no new measurement."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import sys


PINS_REL = 'diagnostics/multik_execution_ready_1792086454973961300/INPUT_PINS.json'
PINS_SHA = 'b520924528606a1f2eb83f6302a247cfd9b1f4e5a179adb2f1ff83b5fc3c2ad0'
PLAN_REL = 'diagnostics/multik_postfinal_resource_1792096163404501000/PLAN.json'
PLAN_SHA = '45908dfa6bf5354c655bc879143d81a440484927885acda8ea4989cbf9808601'
LAUNCHER_REL = 'diagnostics/multik_postfinal_resource_1792096163404501000/original_multik_launcher.py'
LAUNCHER_SHA = 'fe629d2586165984f3fe980c19340a1e15aba84aa6fc87a81482c09e849bb7b3'
TIMER_SCOPE = 'Instrumented guarded K-transition plus full CPU physical pipeline; explicit boundary-check overhead is reported separately'
MEASURED_SOURCES = ('measure_multik_transition.py', 'deployment.py', 'route_catalog.py',
                    'measure_costs.py', 'ordered_pilot.py', 'networks.py', 'resources.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inside(root, name):
    if (not isinstance(name, str) or not name or '\\' in name or
            PurePosixPath(name).is_absolute() or any(x in ('', '.', '..') or ':' in x for x in name.split('/'))):
        raise ValueError('UNSAFE_RELATIVE_PATH')
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('ESCAPED_RELATIVE_PATH')
    return path


def pinned(path, expected):
    raw = Path(path).read_bytes()
    if sha(raw) != expected:
        raise ValueError('PINNED_SHA_MISMATCH')
    return json.loads(raw)


def trial_bytes(row):
    return (json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + '\n').replace('\n', '\r\n').encode('utf-8')


def edges(ks, repeats):
    ks = sorted(ks)
    if len(ks) != 3 or len(set(ks)) != 3 or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError('INVALID_K_GRID')
    pairs = [(a, b) for a in ks for b in ks if a != b]
    return [(repeat, position, a, b) for repeat in range(repeats)
            for position, (a, b) in enumerate(pairs[repeat % 6:] + pairs[:repeat % 6])]


def ledger(ks, rows):
    result = []
    for phase, requested in [('warmup', 10), ('timed', 20)]:
        for a in ks:
            for b in ks:
                if a == b:
                    continue
                group = [r for r in rows if (r['phase'], r['from_K'], r['to_K']) == (phase, a, b)]
                within = sum(r['resource_within_limit'] for r in group)
                result.append(dict(phase=phase, from_K=a, to_K=b, requested=requested,
                    actual_complete=len(group), remaining=requested-len(group),
                    completed_within_resource=within, completed_after_resource_guard=len(group)-within,
                    remaining_within_resource=requested-within))
    return result


def number(value, signed=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (not signed and value < 0):
        raise ValueError('NONFINITE_OR_NEGATIVE_DURATION_OR_BYTES')


def audit_scope(scope, originals):
    strategy = scope['strategy']; name = strategy + '/RESULT.json'
    result = json.loads(originals[name]); rows = result.get('rows', [])
    if any(result.get(key) != scope[key] for key in ('strategy', 'dataset', 'batch')):
        raise ValueError('SCOPE_IDENTITY_MISMATCH')
    if result.get('timer_scope') != TIMER_SCOPE:
        raise ValueError('TIMER_SCOPE_MISMATCH')
    expected_order = [(phase, *entry) for phase, count in [('warmup', 10), ('timed', 20)]
                      for entry in edges(scope['K_grid'], count)]
    actual_order = [(r['phase'], r['repeat'], r['order_position'], r['from_K'], r['to_K']) for r in rows]
    if len(rows) > 180 or actual_order != expected_order[:len(rows)]:
        raise ValueError('DIRECTED_TRANSITION_ORDER_OR_COUNT_MISMATCH')
    trials = sorted(n for n in originals if '/actual_trials/' in n)
    expected_trials = [strategy + f'/actual_trials/trial_{i:06}.json' for i in range(len(rows))]
    if trials != expected_trials:
        raise ValueError('DURABLE_TRIAL_INDEX_SET_MISMATCH')
    for path, row in zip(trials, rows):
        if json.loads(originals[path]) != row or trial_bytes(row) != originals[path]:
            raise ValueError('TRIAL_ROW_OR_EXACT_BYTES_MISMATCH')
        if (row.get('strategy') != strategy or row.get('batch_size') != scope['batch']
                or row.get('accuracy_run_id') != scope['route_ids'][str(row['to_K'])]
                or row.get('F_streams') != scope['batch'] * row['to_K']
                or row.get('F_physical_streams') != scope['batch'] * row['to_K']
                or row.get('numeric_prediction_valid') is not True
                or type(row.get('resource_within_limit')) is not bool):
            raise ValueError('TRIAL_ROUTE_STREAM_OR_VALIDITY_MISMATCH')
        for key, value in row.items():
            if '_seconds' in key:
                number(value, signed=key == 'other_clock_overhead_seconds')
        between = row['controller_guard_seconds_between_segments']
        switching = row['controller_guard_seconds_inside_switch']
        if (row['controller_guard_seconds_inside_whole'] != between + switching
                or row['derived_seconds_minus_observed_controller_guard'] != max(
                    0., row['transition_plus_physical_seconds'] - between - switching)):
            raise ValueError('GUARD_OR_DERIVED_TIMER_ARITHMETIC_MISMATCH')
        if (row.get('resource_event_after_prediction') is not None) == row['resource_within_limit']:
            raise ValueError('RESOURCE_EVENT_FLAG_MISMATCH')
    if rows:
        actual_csv = list(csv.DictReader(io.StringIO(originals[strategy + '/TRIALS.csv'].decode('utf-8'), newline='')))
        expected_csv = [{k: '' if v is None else str(v) for k, v in row.items()} for row in rows]
        if actual_csv != expected_csv:
            raise ValueError('CSV_ROW_VALUE_OR_ORDER_MISMATCH')
    numeric = json.loads(originals[strategy + '/NUMERIC_EXECUTION.json'])
    if numeric.get('rows') != rows or numeric.get('transition_ledger') != result['transition_ledger']:
        raise ValueError('NUMERIC_DURABLE_ROWS_OR_LEDGER_MISMATCH')
    computed = ledger(scope['K_grid'], rows)
    if len(result['transition_ledger']) != 12 or any(
            any(record.get(k) != v for k, v in expected.items())
            for record, expected in zip(result['transition_ledger'], computed)):
        raise ValueError('INDEPENDENT_LEDGER_MISMATCH')
    saved = result.get('summaries')
    expected_summaries = []
    for entry in computed:
        if entry['phase'] != 'timed':
            continue
        group = [r for r in rows if (r['phase'], r['from_K'], r['to_K']) == ('timed', entry['from_K'], entry['to_K'])]
        expected_summaries.append(dict(entry,
            median_transition_plus_physical_seconds=statistics.median([r['transition_plus_physical_seconds'] for r in group]) if group else None,
            median_physical_pipeline_seconds=statistics.median([r['physical_pipeline_seconds'] for r in group]) if group else None))
    if saved is not None and (len(saved) != 6 or any(
            any(record.get(k) != v for k, v in expected.items()) for record, expected in zip(saved, expected_summaries))):
        raise ValueError('SAVED_MEDIAN_OR_SUMMARY_LEDGER_MISMATCH')
    memory = result.get('memory', {}); storage = result.get('storage', {})
    for key in ('RSS_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes', 'resident_route_count', 'foundation_instances'):
        if key in memory:
            number(memory[key])
            if type(memory[key]) is not int:
                raise ValueError('NONINTEGER_MEMORY_COUNT')
    for value in memory.get('deduplicated_resident_weight_bytes_by_device', {}).values():
        number(value)
        if type(value) is not int:
            raise ValueError('NONINTEGER_WEIGHT_BYTES')
    artifacts = storage.get('artifacts', {})
    if artifacts:
        for key, item in artifacts.items():
            inside(Path('.'), key); number(item['bytes'])
        if (storage.get('actual_artifact_bytes') != sum(x['bytes'] for x in artifacts.values())
                or storage.get('unique_artifact_paths') != len(artifacts)
                or storage.get('unique_content_sha256_count') != len({x['sha256'] for x in artifacts.values()})):
            raise ValueError('STORED_ARTIFACT_STORAGE_TOTAL_MISMATCH')
    complete = bool(result.get('exact_minimum_protocol_complete'))
    if complete and (len(rows) != 180 or any(x['remaining_within_resource'] for x in computed)
            or result.get('status') != 'COMPLETE_PROTOCOL'
            or result.get('numeric_execution_status') != 'COMPLETE_REQUESTED_TRIALS'
            or result.get('resource_status') != 'WITHIN_OBSERVED_BOUNDARIES'
            or result.get('post_measure_validation_status') != 'PASS' or saved is None):
        raise ValueError('FALSE_COMPLETE_PROTOCOL_FLAG')
    return result, computed, expected_summaries


def summaries(scope, result, computed, result_relative, result_sha):
    output = []; memory = result.get('memory', {}); storage = result.get('storage', {})
    for entry in computed:
        for validity in ('all_recorded', 'resource_valid_only'):
            group = [r for r in result['rows'] if
                     (r['phase'], r['from_K'], r['to_K']) == (entry['phase'], entry['from_K'], entry['to_K'])
                     and (validity == 'all_recorded' or r['resource_within_limit'])]
            output.append(dict(scope_id=scope['scope_id'], dataset=scope['dataset'], batch=scope['batch'],
                strategy=scope['strategy'], phase=entry['phase'], from_K=entry['from_K'], to_K=entry['to_K'],
                validity_scope=validity, actual_count=len(group), requested_count=entry['requested'],
                resource_valid_count=entry['completed_within_resource'],
                median_transition_plus_physical_seconds=statistics.median([r['transition_plus_physical_seconds'] for r in group]) if group else None,
                median_physical_pipeline_seconds=statistics.median([r['physical_pipeline_seconds'] for r in group]) if group else None,
                saved_summary_present=result.get('summaries') is not None,
                protocol_complete=bool(result.get('exact_minimum_protocol_complete')),
                original_result_status=result.get('status'), original_resource_status=result.get('resource_status'),
                RSS_bytes=memory.get('RSS_bytes'), CUDA_peak_allocated_bytes=memory.get('peak_allocated_bytes'),
                CUDA_peak_reserved_bytes=memory.get('peak_reserved_bytes'),
                resident_CPU_weight_bytes=memory.get('deduplicated_resident_weight_bytes_by_device', {}).get('cpu'),
                resident_CUDA_weight_bytes=memory.get('deduplicated_resident_weight_bytes_by_device', {}).get('cuda'),
                unique_artifact_paths=storage.get('unique_artifact_paths'), artifact_storage_bytes=storage.get('actual_artifact_bytes'),
                timer_scope=TIMER_SCOPE, guard_overhead_included=True, RSS_is_peak=False,
                result_relative_path=result_relative, result_sha256=result_sha))
    return output


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def execute(args, public, pins, plan):
    cache = Path(os.environ['INFORMS_CACHE_ROOT']).resolve()
    if not args.root_triggered or (cache / 'ACTIVE_GPU_LEASE.json').exists():
        raise ValueError('ROOT_TERMINAL_TRIGGER_AND_NO_ACTIVE_COST_OWNER_REQUIRED')
    batch_path = inside(public, args.batch_relative)
    batch = pinned(batch_path, args.batch_sha256)
    terminal = pinned(inside(public, args.terminal_result), args.terminal_sha256)
    closure = terminal.get('central_closure') or {}
    if (terminal.get('multik_batch_result') != batch or closure.get('status') != 'CLOSED'
            or not terminal.get('source39_preserved') or not terminal.get('immutable_selection_and_registry_preserved')):
        raise ValueError('TERMINAL_EXACT_BATCH_OR_CLOSURE_NOT_BOUND')
    actual_closure = pinned(inside(cache, closure['cache_relative_path']), closure['sha256'])
    if any(actual_closure.get(k) != closure.get(k) for k in ('lease_id', 'status', 'charged_process_seconds', 'deadline_overrun_seconds')):
        raise ValueError('CENTRAL_CLOSURE_FIELDS_DIFFER')
    if batch.get('pins_sha256') != args.pins_sha256 or not batch.get('finished_unix'):
        raise ValueError('BATCH_NOT_TERMINAL_OR_PINS_DIFFER')
    number(batch['owner_deadline_unix'])
    if batch['owner_deadline_unix'] > pins['hard_wall_deadline_unix']:
        raise ValueError('OWNER_DEADLINE_EXCEEDS_ORIGINAL_WALL')
    recorded = batch['scope_ledger']; queue = pins['queue']
    if len(recorded) != 20 or [x['scope_id'] for x in recorded] != [x['scope_id'] for x in queue]:
        raise ValueError('FULL20_QUEUE_MISMATCH')
    if any(any(row.get(k) != v for k, v in registered.items() if k != 'status') for row, registered in zip(recorded, queue)):
        raise ValueError('FULL20_SCIENTIFIC_SCOPE_CHANGED')
    if (batch.get('requested_scopes') != 20
            or batch.get('actual_numeric_trials') != sum(r.get('actual_numeric_trials', 0) for r in recorded)
            or batch.get('actual_resource_valid_trials') != sum(r.get('actual_resource_valid_trials', 0) for r in recorded)
            or batch.get('complete_scopes') != sum(bool(r.get('exact_minimum_protocol_complete')) for r in recorded)):
        raise ValueError('GLOBAL_BATCH_COUNTS_DIFFER_FROM_EXACT20_ROWS')
    frozen = {name: (public / name).read_bytes() for name in plan['source39']}
    if {name: sha(raw) for name, raw in frozen.items()} != plan['source39']:
        raise ValueError('SOURCE39_CHANGED')
    registry_raw = (public / 'EXPERIMENT_REGISTRY.csv').read_bytes()
    lock_raw = (public / 'SELECTION_LOCK.json').read_bytes()
    registry = list(csv.DictReader(io.StringIO(registry_raw.decode('utf-8-sig'), newline='')))
    byid = {r['experiment_id']: r for r in registry}
    if len(byid) != len(registry):
        raise ValueError('DUPLICATE_REGISTERED_IDS')
    for identity, route in pins['routes'].items():
        if any(byid.get(identity, {}).get(k) != v for k, v in route['registry_contract'].items()):
            raise ValueError('REGISTERED_ROUTE_OR_SEED_CHANGED')
        pinned(inside(public, route['manifest_public_relative_path']), route['manifest_sha256'])
    scope_records = []; detail = []; all_bindings = {}
    for entry in pins['plans'].values():
        pinned(inside(public, entry['public_relative_path']), entry['sha256'])
        all_bindings[entry['public_relative_path']] = entry['sha256']
    for scope, row in zip(queue, recorded):
        record = dict(scope_id=scope['scope_id'], original_status=row['status'],
                      original_exact_minimum_protocol_complete=bool(row.get('exact_minimum_protocol_complete')))
        output = row.get('output_public_relative_path')
        if not output:
            if row.get('actual_numeric_trials', 0) or row.get('actual_resource_valid_trials', 0) or row.get('exact_minimum_protocol_complete'):
                record.update(status='FAILED_PRESERVED', reason_code='COUNT_OR_COMPLETE_WITHOUT_OUTPUT')
            else:
                record.update(status='NOT_EXECUTED_NO_RECORDED_OUTPUT', actual_numeric_trials=0)
            scope_records.append(record); continue
        try:
            if not output.startswith('costs/multik_transition_'):
                raise ValueError('UNEXPECTED_COST_SOURCE_ROOT')
            folder = inside(public, output)
            paths = [p for p in folder.rglob('*') if p.is_file()]
            if any(p.is_symlink() or p.suffix not in ('.json', '.csv') for p in paths):
                raise ValueError('UNEXPECTED_SCOPE_FILE_OR_SYMLINK')
            originals = {p.relative_to(folder).as_posix(): p.read_bytes() for p in paths}
            file_map = {name: sha(raw) for name, raw in originals.items()}
            all_bindings.update({output + '/' + name: value for name, value in file_map.items()})
            if file_map != row.get('result_files_sha256'):
                raise ValueError('RECEIPT_FILESET_OR_SHA_MISMATCH')
            source_queue_path = folder.parent / 'FULL_QUEUE_PLAN.json'
            source_queue_raw = source_queue_path.read_bytes(); source_queue = json.loads(source_queue_raw)
            if (source_queue.get('pins_sha256') != PINS_SHA or source_queue.get('queue') != queue
                    or source_queue.get('owner_deadline_unix', math.inf) > pins['hard_wall_deadline_unix']):
                raise ValueError('SCOPE_ORIGINAL_QUEUE_OR_DEADLINE_CHANGED')
            all_bindings[source_queue_path.relative_to(public).as_posix()] = sha(source_queue_raw)
            result, computed, _ = audit_scope(scope, originals)
            if (len(result['rows']) != row.get('actual_numeric_trials')
                    or sum(r['resource_within_limit'] for r in result['rows']) != row.get('actual_resource_valid_trials')):
                raise ValueError('BATCH_ACTUAL_RESOURCE_COUNTS_MISMATCH')
            if result.get('source_sha256') != {name: plan['source39'][name] for name in MEASURED_SOURCES}:
                raise ValueError('MEASURED_SOURCE_CONTRACT_DIFFERS')
            if set(result['accuracy_bindings']) != set(scope['route_ids']):
                raise ValueError('CANONICAL_ACCURACY_K_SET_DIFFERS')
            for k, identity in scope['route_ids'].items():
                binding = result['accuracy_bindings'][k]; route = pins['routes'][identity]
                expected = dict(accuracy_run_id=identity, accuracy_manifest_sha256=route['manifest_sha256'],
                    accuracy_prediction_model_sha=route['selected_model_sha'], data_sha256=route['data_sha'],
                    scaler_sha=route['scaler_sha'], foundation_state_sha256=pins['foundation_state_sha256'])
                if any(binding.get(key) != value for key, value in expected.items()):
                    raise ValueError('CANONICAL_ACCURACY_BINDING_DIFFERS')
            expected_artifacts = {pins['routes'][identity]['artifact']['relative_path']:
                                  {key: pins['routes'][identity]['artifact'][key] for key in ('bytes', 'sha256')}
                                  for identity in scope['route_ids'].values()}
            if result.get('storage', {}).get('artifacts') != expected_artifacts:
                raise ValueError('STORED_ARTIFACT_METADATA_DIFFERS_FROM_PINNED_ROUTES')
            child = json.loads(originals['BATCH_RECEIPT.json'])
            if (child.get('plan_sha256') != pins['plans'][scope['dataset']]['sha256']
                    or child.get('source_sha256') != plan['source39']['measure_multik_transition.py']):
                raise ValueError('CHILD_PLAN_OR_WRITER_PIN_DIFFERS')
            if bool(result.get('exact_minimum_protocol_complete')) != bool(row.get('exact_minimum_protocol_complete')):
                raise ValueError('BATCH_FINAL_PROTOCOL_FLAG_DIFFERS')
            relative = output + '/' + scope['strategy'] + '/RESULT.json'
            detail.extend(summaries(scope, result, computed, relative, file_map[scope['strategy'] + '/RESULT.json']))
            record.update(status='PASS_RECORDED_NUMBERS_COMPLETE_PROTOCOL' if result.get('exact_minimum_protocol_complete') else 'PASS_RECORDED_NUMBERS_PARTIAL_PROTOCOL',
                actual_numeric_trials=len(result['rows']), resource_valid_trials=sum(r['resource_within_limit'] for r in result['rows']),
                saved_summaries_present=result.get('summaries') is not None, result_relative_path=relative,
                result_sha256=file_map[scope['strategy'] + '/RESULT.json'])
        except Exception as error:
            record.update(status='FAILED_PRESERVED', error_type=type(error).__name__,
                          reason_code=str(error) if isinstance(error, ValueError) else 'MISSING_OR_UNSUPPORTED_RECORDED_STRUCTURE')
        scope_records.append(record)
    if any((public / name).read_bytes() != raw for name, raw in frozen.items()) or (public / 'EXPERIMENT_REGISTRY.csv').read_bytes() != registry_raw or (public / 'SELECTION_LOCK.json').read_bytes() != lock_raw:
        raise ValueError('SOURCE_REGISTRY_OR_LOCK_CHANGED_DURING_AUDIT')
    if sha(batch_path.read_bytes()) != args.batch_sha256:
        raise ValueError('FINAL_BATCH_CHANGED_DURING_AUDIT')
    if any(sha(inside(public, name).read_bytes()) != value for name, value in all_bindings.items()):
        raise ValueError('RECORDED_STATISTICS_CHANGED_DURING_AUDIT')
    out = inside(public, args.output); out.mkdir(parents=True, exist_ok=False)
    failures = [x['scope_id'] for x in scope_records if x['status'] == 'FAILED_PRESERVED']
    total = sum(x.get('actual_numeric_trials', 0) for x in scope_records if x['status'].startswith('PASS_'))
    result = dict(status='PASS_ALL_RECORDED_NUMBERS' if not failures else 'FAILED_SCOPES_PRESERVED',
        full20_original_status=batch['status'], original_batch_claims_full20_complete=bool(batch.get('actual_cost_raw_statistics_complete')),
        global_owner_resource_valid=terminal.get('resource_valid'), central_closure=closure,
        audited_scope_count=sum(x['status'].startswith('PASS_') for x in scope_records), actual_numeric_trials_audited=total,
        failed_scope_ids=failures, scope_records=scope_records, source39_preserved=True,
        registry_sha256=sha(registry_raw), selection_lock_sha256=sha(lock_raw),
        batch_relative_path=args.batch_relative, batch_sha256=args.batch_sha256, pins_sha256=args.pins_sha256,
        terminal_result_relative_path=args.terminal_result, terminal_result_sha256=args.terminal_sha256,
        auditor_sha256=sha(Path(__file__).read_bytes()), no_old_partial166_double_counted=True,
        timer_scope=TIMER_SCOPE, median_algorithm='statistics.median; saved summaries timed/all-recorded; resource-valid-only reported separately',
        p90_not_computed=True, memory_remeasured=False, RSS_is_peak=False,
        guard_and_derived_formula='Exact original operation order; no tolerance change',
        substage_sum_equals_whole_claimed=False,
        historical_prediction_SHA_recomputed_from_arrays=False, model_artifact_bytes_rehashed=False,
        artifact_identity_scope='Recorded metadata matched original pinned routes; no checkpoint or foundation bytes read',
        arrays_models_GPU_training_or_new_measurement=False)
    new_json(out / 'AUDIT_RESULT.json', result)
    new_json(out / 'FILE_BINDINGS.json', all_bindings)
    if detail:
        for row in detail:
            row['global_owner_resource_valid'] = terminal.get('resource_valid')
            row['global_owner_deadline_overrun_seconds'] = closure.get('deadline_overrun_seconds')
        with (out / 'TRANSITION_SUMMARIES.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(detail[0])); writer.writeheader(); writer.writerows(detail)
    print(json.dumps(dict(status=result['status'], scope_count=result['audited_scope_count'], actual_trials=total,
                         failures=len(failures), output_relative_path=args.output)))
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--root-triggered', action='store_true')
    for name in ('batch-relative', 'batch-sha256', 'pins-sha256', 'terminal-result', 'terminal-sha256', 'output'):
        parser.add_argument('--' + name)
    args = parser.parse_args(); public = Path(os.environ['INFORMS_PUBLIC_ROOT']).resolve()
    pins = pinned(public / PINS_REL, PINS_SHA); plan = pinned(public / PLAN_REL, PLAN_SHA)
    if sha((public / LAUNCHER_REL).read_bytes()) != LAUNCHER_SHA:
        raise ValueError('ORIGINAL_CLASSIFIER_RESUME_SOURCE_CHANGED')
    if args.execute:
        if not all((args.batch_relative, args.batch_sha256, args.pins_sha256, args.terminal_result, args.terminal_sha256, args.output)):
            parser.error('Explicit final BATCH/pins/terminal SHA and new output required')
        if args.pins_sha256 != PINS_SHA:
            parser.error('Original pins SHA is required')
        raise SystemExit(execute(args, public, pins, plan))
    print(json.dumps(dict(status='PREPARATION_ONLY_CURRENT_COST_NOT_READ', original_queue_scopes=len(pins['queue']),
        original_launcher_sha256=LAUNCHER_SHA, arrays_models_GPU_or_new_measurement=False,
        current_cost_glob_or_numeric_audit=False, framework_imports=[n for n in ('numpy', 'torch', 'pandas') if n in sys.modules])))


if __name__ == '__main__':
    main()
