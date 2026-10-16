"""Actual runtime-K transitions; CUDA execution borrows an existing owner lease."""
from __future__ import annotations

import argparse
import collections
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

from resources import BudgetExceeded, ResourceBusy

STRATEGIES = ('shared', 'posthoc', 'nested', 'independent_hot', 'independent_cold')


def balanced_edges(ks, repeats):
    ks = tuple(sorted(ks))
    if len(ks) != 3 or len(set(ks)) != 3 or any(not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError('Exactly three distinct positive integer K values required')
    if not isinstance(repeats, int) or repeats < 0:
        raise ValueError('Nonnegative integer repeat count required')
    edges = [(a, b) for a in ks for b in ks if a != b]
    result = []
    for repeat in range(repeats):
        order = edges[repeat % len(edges):] + edges[:repeat % len(edges)]
        result.extend((repeat, position, a, b) for position, (a, b) in enumerate(order))
    return result


def storage_inventory(specs, cache):
    from data_contract import sha256_file
    cache = Path(cache).resolve()
    files = {}
    for spec in specs.values():
        relative = spec['artifact_cache_relative_path']
        path = (cache / relative).resolve()
        if not path.is_relative_to(cache) or not path.is_file() or sha256_file(path) != spec['artifact_sha256']:
            raise ValueError('Artifact storage identity is absent, outside cache, or changed')
        key = path.relative_to(cache).as_posix()
        files[key] = {'bytes': path.stat().st_size, 'sha256': spec['artifact_sha256']}
    return {'unique_artifact_paths': len(files), 'actual_artifact_bytes': sum(x['bytes'] for x in files.values()),
            'unique_content_sha256_count': len({x['sha256'] for x in files.values()}),
            'artifacts': files, 'common_foundation_storage_excluded': True,
            'byte_identical_files_at_different_paths_are_still_counted': True}


class MultiKRuntime:
    def __init__(self, strategy, specs, factory, sync=lambda: None):
        if strategy not in STRATEGIES:
            raise ValueError('Unknown transition strategy')
        self.strategy = strategy
        self.specs = {int(k): dict(v) for k, v in specs.items()}
        self.ks = tuple(sorted(self.specs))
        balanced_edges(self.ks, 0)
        if any(int(spec['latent_k']) != k for k, spec in self.specs.items()):
            raise ValueError('Canonical route K differs from its declared slot')
        self.factory, self.sync = factory, sync
        self.routes, self.active, self.current_k = {}, None, None
        self.parity_passed, self.canonical_predictions = False, {}
        self.prefix_contract = {}
        self.initial_loading_seconds = 0.
        self.initial_model_states = None

    @property
    def resident_route_count(self):
        return len(self.routes) if self.strategy == 'independent_hot' else int(self.active is not None)

    def initialize(self):
        started = time.perf_counter()
        if self.strategy in {'shared', 'posthoc', 'nested'}:
            identities = {(v['artifact_cache_relative_path'], v['artifact_sha256']) for v in self.specs.values()}
            if len(identities) != 1:
                raise ValueError('A single-checkpoint strategy cannot load three separate artifacts')
            self.active = self.factory(self.specs[max(self.ks)])
            self.current_k = max(self.ks)
            if self.strategy == 'nested':
                if self.active.family != 'PCA_FIXED_BOLT' or self.active.model.E.shape[1] != max(self.ks):
                    raise ValueError('Nested strategy requires a full-rank nested basis route')
                self.prefix_contract = {'changed_attribute': 'Route.latent_k', 'basis_storage': 'one full basis; E[:,:K] used by frozen deployment.py'}
            else:
                original = self.active.bundle['spec'].get('k_grid')
                if self.active.model.kmax != max(self.ks):
                    raise ValueError('Prefix strategy requires its genuine maximum-K parent')
                if self.strategy == 'shared' and tuple(original or ()) != self.ks:
                    raise ValueError('Shared strategy requires genuine registered multi-K training')
                if self.strategy == 'posthoc':
                    if original != [max(self.ks)]:
                        raise ValueError('Posthoc strategy requires a maximum-only trained parent')
                    self.active.model.k_grid = self.ks
                self.prefix_contract = {'changed_attribute': 'Route.latent_k',
                    'deployment_only_allowed_prefix_attribute': 'ForecastAdapter.k_grid' if self.strategy == 'posthoc' else None,
                    'saved_training_k_grid': original, 'runtime_allowed_k_grid': list(self.active.model.k_grid),
                    'equivalent_original_operation': 'ordered_pilot.predict_prefix: spec[k_grid]=sorted(set(trained_grid+[K])); encode(x,K), F(K), decode(x,z,future,K)',
                    'weights_or_saved_bundle_modified': False, 'allowed_for_measurement_only_after_canonical_parity': True}
        elif self.strategy == 'independent_hot':
            if len({(x['artifact_cache_relative_path'],x['artifact_sha256']) for x in self.specs.values()}) != 3 or any(x['family'] != 'B0' for x in self.specs.values()):
                raise ValueError('Independent B0 scope requires three genuine independent artifacts')
            self.routes = {k: self.factory(spec) for k, spec in self.specs.items()}
            self.current_k = max(self.ks)
            self.active = self.routes[self.current_k]
        else:
            if len({(x['artifact_cache_relative_path'],x['artifact_sha256']) for x in self.specs.values()}) != 3 or any(x['family'] != 'B0' for x in self.specs.values()):
                raise ValueError('Independent B0 scope requires three genuine independent artifacts')
            self.current_k = max(self.ks)
            self.active = self.factory(self.specs[self.current_k])
        self.sync()
        self.initial_loading_seconds = time.perf_counter() - started
        self.initial_model_states = self.model_states()

    def model_states(self):
        from networks import module_state_hash
        routes = self.routes if self.strategy == 'independent_hot' else {'single':self.active}
        return {str(key):module_state_hash(route.model) for key,route in routes.items()}

    def assert_model_states_unchanged(self):
        after=self.model_states()
        if after != self.initial_model_states:
            raise AssertionError('Runtime K changes modified saved E/P/D tensors')
        return {'before':self.initial_model_states,'after':after,'all_equal':True}

    def _timed(self, function):
        self.sync()
        start = time.perf_counter()
        value = function()
        self.sync()
        return value, time.perf_counter() - start

    def set_k(self, k):
        if k not in self.ks or self.active is None:
            raise ValueError('K is not registered or runtime is not initialized')
        record = {name: 0. for name in ('deactivate_seconds', 'release_seconds', 'artifact_load_seconds', 'prefix_configuration_seconds', 'activate_seconds')}
        factory_guard_before=float(getattr(self.factory,'guard_seconds',0.))
        self.sync()
        started = time.perf_counter()
        if k != self.current_k:
            _, record['deactivate_seconds'] = self._timed(self.active.deactivate)
            if self.strategy == 'independent_cold':
                def release():
                    self.active.close()
                    self.active = None
                    gc.collect()
                _, record['release_seconds'] = self._timed(release)
                self.active, record['artifact_load_seconds'] = self._timed(lambda: self.factory(self.specs[k]))
            elif self.strategy == 'independent_hot':
                self.active, record['prefix_configuration_seconds'] = self._timed(lambda: self.routes[k])
            else:
                def configure():
                    self.active.latent_k = k
                _, record['prefix_configuration_seconds'] = self._timed(configure)
            self.current_k = k
        _, record['activate_seconds'] = self._timed(self.active.activate)
        self.sync()
        record['switch_total_seconds'] = time.perf_counter() - started
        record['controller_guard_seconds_inside_switch']=float(getattr(self.factory,'guard_seconds',0.))-factory_guard_before
        return record

    def predict(self, physical, observed):
        result = self.active.predict_physical(physical, observed)
        expected = len(physical) * self.current_k
        receipt = self.active.last_receipt
        if receipt['F_logical_streams'] != expected or receipt['F_physical_streams'] != expected:
            raise AssertionError('Actual F streams must equal B*K; max-K slicing is forbidden')
        return result

    def verify_canonical(self, physical, observed, lease=None, atol=1e-5, rtol=1e-4):
        checks = []
        self.parity_passed = False
        for k in self.ks:
            if lease is not None:
                lease.check()
            canonical = self.factory(self.specs[k])
            try:
                if owner_boundary(lease,'after_canonical_route_construction_before_prediction') is not None:
                    raise BudgetExceeded('Canonical constructor returned after owner guard')
                expected = canonical.predict_physical(physical, observed)
                if owner_boundary(lease,'after_canonical_physical_prediction') is not None:
                    raise BudgetExceeded('Canonical prediction returned after owner guard')
                if canonical.last_receipt['F_logical_streams'] != len(physical) * k or canonical.last_receipt['F_physical_streams'] != len(physical) * k:
                    raise AssertionError('Canonical K route does not execute B*K streams')
                self.set_k(k)
                if owner_boundary(lease,'after_canonical_runtime_configuration') is not None:
                    raise BudgetExceeded('Canonical K configuration returned after owner guard')
                actual = self.predict(physical, observed)
                if owner_boundary(lease,'after_canonical_runtime_prediction') is not None:
                    raise BudgetExceeded('Canonical runtime prediction returned after owner guard')
                if not np.allclose(expected, actual, atol=atol, rtol=rtol):
                    raise AssertionError('Single runtime differs from its exact canonical K prediction')
                self.canonical_predictions[k] = expected.copy()
                checks.append({'K': k, 'canonical_run_id': self.specs[k]['run_id'],
                    'canonical_prediction_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
                    'runtime_prediction_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                    'max_abs': float(np.abs(expected - actual).max()), 'F_streams': len(physical) * k,
                    'atol': atol, 'rtol': rtol, 'passed': True})
            finally:
                canonical.close()
                del canonical
        self.parity_passed = True
        return {'all_passed': True, 'checks': checks, 'prefix_contract': self.prefix_contract,
                'scope': 'Actual physical CPU-input forecasts from canonical routes vs one runtime; saved forecasts not read'}

    def close(self):
        routes = list(self.routes.values()) if self.strategy == 'independent_hot' else [self.active]
        for route in routes:
            if route is not None:
                route.close()
        self.routes, self.active = {}, None
        gc.collect()


def transition_ledger(ks, warmups, timed_trials, rows=()):
    result=[]
    for phase,repeats in [('warmup',warmups),('timed',timed_trials)]:
        balanced_edges(tuple(ks),repeats)
        for a in ks:
            for b in ks:
                if a != b:
                    completed=sum(x['phase']==phase and x['from_K']==a and x['to_K']==b for x in rows)
                    within=sum(x['phase']==phase and x['from_K']==a and x['to_K']==b and x.get('resource_within_limit',True) for x in rows)
                    result.append({'phase':phase,'from_K':a,'to_K':b,'requested':repeats,
                                   'actual_complete':completed,'remaining':repeats-completed,
                                   'completed_within_resource':within,'completed_after_resource_guard':completed-within,
                                   'remaining_within_resource':repeats-within})
    return result


def owner_boundary(lease, boundary):
    if lease is None:
        return None
    try:
        lease.check()
        return None
    except (BudgetExceeded,ResourceBusy,OSError) as error:
        now=time.time()
        owner=getattr(lease,'lease',lease)
        deadline=getattr(owner,'deadline',None)
        overrun=max(0.,now-float(deadline)) if deadline is not None else None
        return {'boundary':boundary,'detected_unix':now,'exception_class':type(error).__name__,
                'owner_deadline_unix':deadline,'deadline_overrun_seconds':overrun,
                'state':'DEADLINE_OVERRUN' if overrun and overrun>0 else
                        ('RESOURCE_GUARD_TRIGGERED' if isinstance(error,BudgetExceeded) else 'OWNER_UNAVAILABLE')}


def mark_resource(result,event):
    if event is None:
        return
    result.setdefault('resource_events',[]).append(event)
    states={x['state'] for x in result['resource_events']}
    result['resource_status']='DEADLINE_OVERRUN' if 'DEADLINE_OVERRUN' in states else (
        'OWNER_UNAVAILABLE' if 'OWNER_UNAVAILABLE' in states else 'RESOURCE_GUARD_TRIGGERED')
    result['status']='PARTIAL_RESOURCE_LIMIT'
    result['exact_minimum_protocol_complete']=False
    result['stop_new_scopes']=True


def measure(runtime, physical, observed, warmups=10, timed_trials=20, lease=None, on_trial=None):
    if not runtime.parity_passed:
        raise ValueError('Canonical prediction/F-stream parity is required before measurement')
    rows, status, resource_events = [], 'COMPLETE_PROTOCOL', []
    attempt=None
    stop=False
    schedules = [('warmup', warmups), ('timed', timed_trials)]
    try:
        for phase, repeats in schedules:
            for repeat, position, source, target in balanced_edges(runtime.ks, repeats):
                attempt={'phase':phase,'repeat':repeat,'from_K':source,'to_K':target,'completed_numeric':False,'stage':'before_trial'}
                event=owner_boundary(lease,'before_trial')
                if event is not None:
                    resource_events.append(event);stop=True;break
                prepared = runtime.set_k(source)
                attempt['stage']='after_preparation'
                event=owner_boundary(lease,'after_preparation')
                if event is not None:
                    resource_events.append(event);stop=True;break
                runtime.sync()
                started = time.perf_counter()
                switching = runtime.set_k(target)
                switch_returned=time.perf_counter()
                attempt['stage']='after_transition'
                guard_started=time.perf_counter()
                event=owner_boundary(lease,'after_transition')
                controller_guard_seconds=time.perf_counter()-guard_started
                if event is not None:
                    resource_events.append(event);stop=True;break
                prediction_started=time.perf_counter()
                result = runtime.predict(physical, observed)
                runtime.sync()
                prediction_returned=time.perf_counter()
                whole = prediction_returned - started
                switch_segment=switch_returned-started
                prediction_segment=prediction_returned-prediction_started
                inside_switch_guard=switching['controller_guard_seconds_inside_switch']
                attempt['stage']='after_prediction_sync'
                event=owner_boundary(lease,'after_prediction_sync')
                expected = runtime.canonical_predictions[target]
                if not np.allclose(result, expected, atol=1e-5, rtol=1e-4):
                    raise AssertionError('Transition forecast no longer matches its canonical route')
                row = {'strategy': runtime.strategy, 'phase': phase, 'repeat': repeat, 'order_position': position,
                       'from_K': source, 'to_K': target, 'batch_size': len(physical),
                       'accuracy_run_id': runtime.specs[target]['run_id'],
                       'transition_plus_physical_seconds': whole,
                       'controller_guard_seconds_between_segments':controller_guard_seconds,
                       'controller_guard_seconds_inside_whole':controller_guard_seconds+inside_switch_guard,
                       'switch_execution_segment_seconds':switch_segment-inside_switch_guard,
                       'physical_prediction_segment_seconds':prediction_segment,
                       'other_clock_overhead_seconds':whole-switch_segment-controller_guard_seconds-prediction_segment,
                       'derived_seconds_minus_observed_controller_guard':max(0.,whole-controller_guard_seconds-inside_switch_guard),
                       'physical_pipeline_seconds': runtime.active.last_receipt['whole_seconds'],
                       'preparation_outside_timed_seconds': prepared['switch_total_seconds'],
                       'F_streams': runtime.active.last_receipt['F_logical_streams'],
                       'F_physical_streams': runtime.active.last_receipt['F_physical_streams'],
                       'prediction_sha256': hashlib.sha256(result.tobytes()).hexdigest(),
                       'numeric_prediction_valid':True,'resource_within_limit':event is None,
                       'resource_event_after_prediction':event,
                       'resident_route_count': runtime.resident_route_count, **switching}
                if on_trial is not None:
                    on_trial(row, len(rows))
                rows.append(row)
                attempt['completed_numeric']=True
                if event is not None:
                    resource_events.append(event);stop=True;break
            if stop:
                break
    except BudgetExceeded:
        status = 'PARTIAL_RESOURCE_LIMIT'
        resource_events.append(owner_boundary(lease,attempt['stage'] if attempt else 'before_trial') or
                               {'boundary':attempt['stage'] if attempt else 'before_trial','state':'RESOURCE_GUARD_TRIGGERED',
                                'exception_class':'BudgetExceeded','detected_unix':time.time(),'deadline_overrun_seconds':None})
    except Exception as error:
        status = 'FAILED_STRATEGY_AFTER_PARTIAL_MEASUREMENT'
        failure = type(error).__name__ + ': ' + str(error)
    ledger = transition_ledger(runtime.ks,warmups,timed_trials,rows)
    output={'status': status, 'rows': rows, 'transition_ledger': ledger,
            'numeric_execution_status':'COMPLETE_REQUESTED_TRIALS' if not any(x['remaining'] for x in ledger) else 'PARTIAL_NUMERIC_EXECUTION',
            'post_measure_validation_status':'NOT_RUN', 'resource_status':'WITHIN_OBSERVED_BOUNDARIES',
            'resource_events':[],'stop_new_scopes':False,'last_attempt':attempt,
            'unfinished_attempt':attempt if attempt and not attempt['completed_numeric'] else None,
            'failure': locals().get('failure'), 'exact_minimum_protocol_complete':
            status == 'COMPLETE_PROTOCOL' and warmups >= 10 and timed_trials >= 20 and not any(x['remaining'] for x in ledger)}
    for event in resource_events:
        mark_resource(output,event)
    mark_resource(output,owner_boundary(lease,'after_measurement_before_post_validation'))
    return output


def default_plan(dataset, channels, seed=92601):
    if dataset not in {'beijing', 'electricity'}:
        raise ValueError('This cost supplement is scoped to Beijing/Electricity')
    ks = sorted({math.ceil(channels / 4), math.ceil(channels / 2), channels})
    balanced_edges(tuple(ks), 0)
    shared = f'primary_{dataset}_E4_ORDERED_MULTI_K_k{channels}_s{seed}'
    independent = {str(k): f'primary_{dataset}_E1_B0_k{k}_s{seed}' for k in ks}
    return {'dataset': dataset, 'K_grid': ks, 'strategies': {
        'shared': {str(k): shared if k == channels else shared + f'__prefix_k{k}' for k in ks},
        'posthoc': {str(k): f'primary_{dataset}_E4_POSTHOC_PREFIX_k{k}_s{seed}' for k in ks},
        'nested': {str(k): f'primary_{dataset}_E4_NESTED_PCA_k{k}_s{seed}' for k in ks},
        'independent_hot': independent, 'independent_cold': dict(independent)},
        'status': 'PLAN_ONLY_NOT_EXECUTED', 'selection_on_T': False}


def _json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def _memory(runtime, pool, device):
    import psutil
    import torch
    routes = list(runtime.routes.values()) if runtime.strategy == 'independent_hot' else [runtime.active]
    modules = [x.model for x in routes if x is not None]
    if pool.foundation is not None:
        modules.append(getattr(pool.foundation, 'model', pool.foundation))
    seen, totals = set(), collections.Counter()
    for module in modules:
        if module is None:
            continue
        for value in list(module.parameters()) + list(module.buffers()):
            key = (str(value.device), value.data_ptr(), value.numel())
            if key not in seen:
                seen.add(key)
                totals[value.device.type] += value.numel() * value.element_size()
    return {'resident_route_count': runtime.resident_route_count, 'foundation_instances': pool.instances_created,
            'deduplicated_resident_weight_bytes_by_device': dict(totals),
            'RSS_bytes': psutil.Process().memory_info().rss,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0,
            'training_peak_included': False, 'fresh_process_claim': False}


def post_validation_failure(result, stage, error):
    result.setdefault('post_measure_validation_errors',[]).append({'stage':stage,'error_type':type(error).__name__,'reason':str(error)})
    result['post_measure_validation_status']='FAILED'
    result['exact_minimum_protocol_complete']=False
    if not result.get('stop_new_scopes'):
        result['status']='FAILED_POST_MEASURE_VALIDATION'


def _summaries(result):
    rows=[]
    for entry in result['transition_ledger']:
        if entry['phase']=='timed':
            group=[x for x in result['rows'] if x['phase']=='timed' and x['from_K']==entry['from_K'] and x['to_K']==entry['to_K']]
            rows.append(dict(entry,median_transition_plus_physical_seconds=statistics.median([x['transition_plus_physical_seconds'] for x in group]) if group else None,
                             median_physical_pipeline_seconds=statistics.median([x['physical_pipeline_seconds'] for x in group]) if group else None))
    return rows


def _durable_trial_check(folder,result):
    paths=sorted((folder/'actual_trials').glob('trial_*.json'))
    persisted=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
    if persisted != result['rows']:
        raise AssertionError('Durable numeric trial rows and final rows differ')
    with (folder/'TRIALS.csv').open(encoding='utf-8',newline='') as handle:
        csv_rows=list(csv.DictReader(handle))
    if len(csv_rows)!=len(persisted):
        raise AssertionError('TRIALS CSV count differs from durable numeric rows')
    expected=[{key:'' if value is None else str(value) for key,value in row.items()} for row in persisted]
    if csv_rows != expected:
        raise AssertionError('TRIALS CSV values differ from durable numeric rows')
    for entry in result['transition_ledger']:
        group=[row for row in persisted if row['phase']==entry['phase'] and row['from_K']==entry['from_K'] and row['to_K']==entry['to_K']]
        within=sum(row.get('resource_within_limit',True) for row in group)
        if (entry['actual_complete']!=len(group) or entry['remaining']!=entry['requested']-len(group)
                or entry['completed_within_resource']!=within or entry['remaining_within_resource']!=entry['requested']-within):
            raise AssertionError('Final transition ledger differs from durable numeric rows')
    return {'status':'PASS','actual_trials':len(persisted),'CSV_rows':len(csv_rows),
            'ledger_actual_complete':sum(x['actual_complete'] for x in result['transition_ledger'])}


def worker(args):
    import torch
    import data_contract as data
    import deployment
    import route_catalog
    from measure_costs import accuracy_binding, worker_lease_check
    lease = worker_lease_check(args.cache_root, args.device)
    torch.set_num_threads(2)
    plan_path, public, cache = Path(args.plan_json), Path(args.public_root), Path(args.cache_root)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    task = data.load_task(plan['dataset'], cache_dir=cache)
    if plan['dataset'] not in {'beijing', 'electricity'} or plan['K_grid'] != sorted({math.ceil(len(task['columns']) / 4), math.ceil(len(task['columns']) / 2), len(task['columns'])}):
        raise ValueError('Task/grid differs from predeclared E4 quarter/half/C scope')
    origin_values = data.origins(task['split_ranges']['V_select'], task['L'], task['H'], task['H'])
    if len(origin_values) < args.batch:
        raise ValueError('Insufficient distinct V past histories for requested batch')
    origin_values = origin_values[:args.batch]
    physical = np.stack([task['physical'][o-task['L']:o] for o in origin_values])
    observed = np.stack([task['observed'][o-task['L']:o] for o in origin_values])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    strategies = STRATEGIES if args.strategy == 'all' else (args.strategy,)
    results, stop_new_scopes = [], False
    sync = (lambda: torch.cuda.synchronize()) if args.device.startswith('cuda') else (lambda: None)
    for strategy in strategies:
        event = None if stop_new_scopes else owner_boundary(lease,'before_strategy')
        if stop_new_scopes or event is not None:
            stop_new_scopes=True
            results.append({'strategy': strategy, 'status': 'NOT_STARTED_RESOURCE_LIMIT',
                            'exact_minimum_protocol_complete':False,'numeric_execution_status':'NOT_STARTED',
                            'transition_ledger':transition_ledger(plan['K_grid'],args.warmups,args.timed_trials)})
            continue
        folder = output / strategy
        folder.mkdir(exist_ok=False)
        pool = deployment.FoundationPool(args.model, args.device)
        runtime, measurement_returned = None, False
        result={'strategy':strategy,'status':'NOT_STARTED','rows':[],
                'numeric_execution_status':'NOT_STARTED','post_measure_validation_status':'NOT_RUN',
                'resource_status':'WITHIN_OBSERVED_BOUNDARIES','resource_events':[],
                'stop_new_scopes':False,'exact_minimum_protocol_complete':False,
                'transition_ledger':transition_ledger(plan['K_grid'],args.warmups,args.timed_trials)}
        stage='route_initialization'
        def require_owner(boundary):
            event=owner_boundary(lease,boundary)
            if event is not None:
                mark_resource(result,event)
                raise BudgetExceeded('Owner boundary rejected further work')
        try:
            specs = {}
            for k, run_id in plan['strategies'][strategy].items():
                spec, size = route_catalog.build_route(run_id, task, cache, public)
                if size != 'small':
                    raise ValueError('All transition routes must use the same original small foundation')
                specs[int(k)] = spec
            storage = storage_inventory(specs, cache)
            def factory(spec):
                if lease is not None:
                    guard_started=time.perf_counter()
                    try:
                        lease.check()
                    finally:
                        factory.guard_seconds+=time.perf_counter()-guard_started
                return deployment.Route(spec, task, cache, args.model, args.device, pool)
            factory.guard_seconds=0.
            runtime = MultiKRuntime(strategy, specs, factory, sync)
            runtime.initialize()
            require_owner('after_runtime_initialization')
            bindings = {}
            for k in runtime.ks:
                canonical = factory(specs[k])
                try:
                    bindings[str(k)] = accuracy_binding(dict(specs[k], accuracy_run_id=specs[k]['run_id']), canonical, task, cache, public)
                finally:
                    canonical.close()
                    del canonical
                require_owner('after_canonical_binding_close')
            stage='canonical_parity'
            parity = runtime.verify_canonical(physical, observed, lease)
            require_owner('after_canonical_parity')
            if args.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats()
            trials = folder / 'actual_trials'
            trials.mkdir(exist_ok=False)
            def persist(row, index):
                _json_new(trials / f'trial_{index:06}.json', row)
            stage='measurement'
            result = measure(runtime, physical, observed, args.warmups, args.timed_trials, lease, persist)
            measurement_returned=True
            result['strategy']=strategy
            # Seal numeric execution before optional memory/hash/statistical validation.
            _json_new(folder/'NUMERIC_EXECUTION.json',dict(result,
                status='NUMERIC_TRIALS_RECORDED_PENDING_FINAL_VALIDATION',
                measured_component_status=result['status'],exact_minimum_protocol_complete=False,
                final_protocol_decision=False))
            stage='post_measure_metadata'
            result.update(dataset=task['name'],device=args.device,batch=args.batch,
                origins=list(map(int,origin_values)),L=task['L'],H=task['H'],C=len(task['columns']),
                physical_input_sha256=data.array_sha256(physical),observed_input_sha256=data.array_sha256(observed),
                data_sha256=task['data_sha256'],storage=storage,accuracy_bindings=bindings,parity=parity,
                initial_loading_seconds=runtime.initial_loading_seconds,
                cold_scope='Model reload; shared frozen F remains resident; OS disk/page cache not controlled' if strategy=='independent_cold' else None,
                timer_scope='Instrumented guarded K-transition plus full CPU physical pipeline; explicit boundary-check overhead is reported separately',
                substage_sum_equals_whole_claimed=False,row_timers_are_saved_forecast_read_cost=False,
                load_timer_includes='Existing Route constructor: artifact hashing/read, normalization/PCA regeneration/validation, state loading and device placement',
                allocation_cache_eviction_claimed=False,fresh_process_claim=args.strategy!='all',
                public_scope_separate_from_previous_warmed_predict_costs=True,T_accessed=False,
                GPU_owner_child_lease_validated=lease is not None,
                source_sha256={name:data.sha256_file(Path(__file__).with_name(name)) for name in ('measure_multik_transition.py','deployment.py','route_catalog.py','measure_costs.py','ordered_pilot.py','networks.py','resources.py')})
            checks=[('memory',lambda:_memory(runtime,pool,args.device)),
                    ('runtime_model_states',lambda:runtime.assert_model_states_unchanged() if strategy!='independent_cold' else
                        {'scope':'Different independent K checkpoints loaded; canonical per-K SHA contracts verified'}),
                    ('summaries',lambda:_summaries(result))]
            result['post_measure_validation_status']='PASS'
            for name,call in checks:
                if result['stop_new_scopes']:
                    result.setdefault('post_checks_skipped_resource',[]).append(name)
                    if result['post_measure_validation_status']=='PASS':result['post_measure_validation_status']='NOT_RUN_RESOURCE_LIMIT'
                    continue
                stage=name
                try:
                    result[name]=call()
                except Exception as error:
                    post_validation_failure(result,name,error)
                mark_resource(result,owner_boundary(lease,'after_post_validation_'+name))
        except BudgetExceeded:
            mark_resource(result,owner_boundary(lease,'returned_resource_limit_'+stage) or
                {'boundary':'returned_resource_limit_'+stage,'state':'RESOURCE_GUARD_TRIGGERED','detected_unix':time.time(),
                 'exception_class':'BudgetExceeded','deadline_overrun_seconds':None})
        except Exception as error:
            if measurement_returned:
                post_validation_failure(result,stage,error)
            else:
                reason=str(error)
                for path,replacement in ((str(cache),'PRIVATE_CACHE'),(str(public),'PUBLIC_CAMPAIGN'),(str(args.model),'PINNED_MODEL')):
                    reason=reason.replace(path,replacement)
                result.update(status='FAILED_STRATEGY_OR_CANONICAL_PARITY',error_type=type(error).__name__,reason=reason,
                              other_strategies_may_continue=True)
        finally:
            for name,close in [('runtime_close',runtime.close if runtime is not None else lambda:None),('foundation_close',pool.close)]:
                try:
                    close()
                    if name=='foundation_close':
                        result['backbone_close_status']='PASS_EXACT_HASH' if pool.instances_created else 'NO_FOUNDATION_INSTANTIATED'
                except Exception as error:
                    post_validation_failure(result,name,error)
            gc.collect()
            if pool.instances_created:
                try:
                    sync()
                except Exception as error:
                    post_validation_failure(result,'teardown_sync',error)
            mark_resource(result,owner_boundary(lease,'after_final_close_sync'))
        if result.get('rows'):
            try:
                with (folder/'TRIALS.csv').open('x',encoding='utf-8',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=list(result['rows'][0]));writer.writeheader();writer.writerows(result['rows'])
                result['durable_trial_integrity']=_durable_trial_check(folder,result)
            except Exception as error:
                post_validation_failure(result,'durable_trial_integrity',error)
        mark_resource(result,owner_boundary(lease,'before_final_completion_decision'))
        stop_new_scopes=stop_new_scopes or result['stop_new_scopes']
        _json_new(folder/'RESULT.json',result)
        results.append({key:result.get(key) for key in ('strategy','status','numeric_execution_status','resource_status',
                                                      'post_measure_validation_status','exact_minimum_protocol_complete')})
    _json_new(output/'BATCH_RECEIPT.json',{'status':'COMPLETE_REQUESTED_STRATEGIES' if all(x.get('exact_minimum_protocol_complete') for x in results) else 'PARTIAL_OR_FAILED',
              'results':results,'plan_sha256':data.sha256_file(plan_path),'source_sha256':data.sha256_file(Path(__file__)),
              'actual_trials_preserved_per_trial':True,'training_or_T_executed':False,
              'resource_limit_stopped_new_scopes':stop_new_scopes,'existing_379_CPU_suite_predates_this_module':True})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--make-plan', action='store_true')
    parser.add_argument('--dataset', choices=('beijing','electricity'))
    parser.add_argument('--channels', type=int)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--plan-json')
    parser.add_argument('--public-root')
    parser.add_argument('--cache-root')
    parser.add_argument('--model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch', type=int, choices=(1,8), default=1)
    parser.add_argument('--strategy', choices=STRATEGIES+('all',), default='all')
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--timed-trials', type=int, default=20)
    args = parser.parse_args()
    if args.make_plan:
        if args.worker or not args.dataset or not args.channels:
            parser.error('--make-plan needs dataset/channels and cannot execute a worker')
        _json_new(args.output, default_plan(args.dataset,args.channels))
    elif args.worker:
        if not all((args.plan_json,args.public_root,args.cache_root,args.model)):
            parser.error('Worker requires plan/public/cache/model paths')
        print(json.dumps(worker(args),ensure_ascii=False))
    else:
        parser.error('Explicit --make-plan or guarded --worker is required')


if __name__ == '__main__':
    main()
