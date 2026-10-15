"""FIT-target vs past-PCA output spaces, matched inverse D and frozen Bolt/native."""
from pathlib import Path
import argparse
import json
import hashlib
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
OLD = ROOT / 'research/smallk_failure_followup_20261014'
sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
DATASETS = {'robin': ('first_robin_v1', 4, 'val'),
            'beijing_aotizhongxin': ('first_beijing_v1', 2, 'eval')}
ROUTES = ('PCA_PINV', 'FIT_TARGET_PINV')
SEEDS = (92601, 92602)


def record(path):
    with Path(path).open('rb') as handle:
        return {'path': str(Path(path).resolve()), 'sha256': hashlib.file_digest(handle, 'sha256').hexdigest()}


def prior_file_records(source, *bindings):
    found = {}

    def collect(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                path = Path(value['path'])
                path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
                if path.name.casefold() in ('test.npz', 'jena_test.npz'):
                    raise ValueError('PROTECTED_TEST_REFERENCE_REJECTED_BEFORE_HASH')
                actual = record(path)
                if actual['sha256'] != value['sha256']:
                    raise ValueError('PRIOR_FILE_CHANGED:' + str(path))
                found[str(path)] = actual
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(source)
    for binding in bindings:
        collect(binding)
    return list(found.values())


def load_gates():
    source_path = HERE / 'runs/actual_projection_v2/RESULT.json'
    audit_path = HERE / 'audits/PROJECTION_RESULTS_AUDIT_v2.json'
    source = json.loads(source_path.read_text(encoding='utf-8-sig'))
    audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
    if source.get('status') != 'COMPLETE' or audit.get('status') != 'PASS':
        raise ValueError('ACTUAL_COMPLETE_SOURCE_AND_PASS_AUDIT_REQUIRED')
    binding = audit['producer']
    path = Path(binding['path'])
    path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    if path != source_path.resolve() or binding['sha256'] != record(source_path)['sha256']:
        raise ValueError('EXACT_ACCEPTED_SOURCE_AUDIT_BINDING_REQUIRED')
    expected = {'projection_states_checked': 4, 'native_fits_checked': 8,
                'checkpoint_states_checked': 40, 'score_field_checks': 1464,
                'saved_prediction_arrays_checked': 88, 'cost_cells_checked': 20,
                'recorded_timer_samples_checked': 120}
    if any(audit['totals'].get(key) != value for key,value in expected.items()):
        raise ValueError('FULL_ACCEPTED_SOURCE_COVERAGE_REQUIRED')
    if source.get('protected_TEST_accessed') is not False or audit.get('protected_TEST_accessed') is not False:
        raise ValueError('PROTECTED_TEST_SCOPE_REQUIRED')
    for name in ('PLAN_FIT_TARGET_SUBSPACE_KO_v1.md', 'FIT_TARGET_CONTINUATION_SCOPE_KO_v1.md'):
        if not (HERE/name).is_file():
            raise ValueError('FRESH_SCOPE_AND_PLAN_REQUIRED')
    return source, record(source_path), record(audit_path)


def matching_parent_inputs(dataset, parent, accepted):
    expected = {'fit', 'tune', DATASETS[dataset][2]}
    if parent.get('status') != 'COMPLETE' or set(parent.get('inputs', {})) != expected:
        raise ValueError('COMPLETE_PARENT_INPUT_CONTRACT_REQUIRED:' + dataset)
    if parent['inputs'] != accepted['inputs']:
        raise ValueError('EXACT_ACCEPTED_PARENT_INPUT_BINDING_REQUIRED:' + dataset)
    return parent['inputs']


def fresh_output_path(tag):
    if not isinstance(tag, str) or not tag or Path(tag).name != tag or tag in ('.', '..'):
        raise ValueError('SINGLE_FRESH_RUN_TAG_REQUIRED')
    output = (HERE / 'runs' / tag).resolve()
    if output.parent != (HERE / 'runs').resolve() or output.exists():
        raise ValueError('FRESH_OUTPUT_DIRECTORY_REQUIRED')
    return output


def isolated_path(failed_paths, dataset, route, stage, action, seed=None):
    try:
        return action()
    except Exception as error:
        row = {'dataset': dataset, 'route': route, 'stage': stage,
               'status': 'FAILED_PATH_PRESERVED',
               'failure': {'type': type(error).__name__, 'message': str(error)}}
        if seed is not None:
            row['seed'] = str(seed)
        failed_paths.append(row)
        return None


def execute(tag):
    source_projection, source_binding, audit_binding = load_gates()
    protected_prior = prior_file_records(source_projection, source_binding, audit_binding)
    output = fresh_output_path(tag)
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import projection_math_v1 as pm
    import native_head_v1 as nh
    from native_batch_contract_v2 import native_components
    import whole_fit_target_cost_v1 as wc
    import future_target_projection_v1 as ft
    import prepare_inputs as inputs
    import run_condition_probes as probes
    import matched_readout_contract_v1 as contract
    import cost_readout_v1 as old_cost

    meter = rt.Meter(tag, gpu=True)
    inputs.install_protected_test_guard()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output.parent.mkdir(exist_ok=True)
    output.mkdir(exist_ok=False)
    result = {'status': 'RUNNING', 'scientific_execution': True,
              'protected_TEST_accessed': False, 'independent_holdout': False,
              'new_backbone_training': False, 'datasets': {}, 'cost': {},
              'native_training_fits': 0, 'native_training_attempts': 0,
              'projection_fits': 0, 'projection_fit_attempts': 0,
              'reused_projection_states': 0, 'projection_conditions': 0,
              'failed_paths': [], 'source_parent_results': {},
              'source_projection': source_binding, 'source_projection_audit': audit_binding,
              'protected_prior_records': protected_prior,
              'scope_addendum': record(HERE/'FIT_TARGET_CONTINUATION_SCOPE_KO_v1.md'),
              'plan': record(HERE/'PLAN_FIT_TARGET_SUBSPACE_KO_v1.md'),
              'individual_exact_human_method_approval_claimed': False,
              'continuation_scope_is_documented_judgment': True,
              'matched_decoder_rule': 'pinv(E64cast32 interpreted64,rcond=1e-15)cast32',
              'encoder_fit_uses_FIT_true_future': {'PCA_PINV': False, 'FIT_TARGET_PINV': True},
              'source_snapshots': [], 'source_records': [],
              'authorization': {'path': str(rt.AUTH), 'sha256': rt.sha(rt.AUTH)},
              'continuation_manifest': {'path': str(rt.MANIFEST), 'sha256': rt.sha(rt.MANIFEST)},
              'hard_elapsed_or_cumulative_resource_caps': None,
              'forecast_contract': 'FP32 raw_last + (latent_future - encoded_history_last) @ decoder',
              'oracle_role': 'diagnostic_only; excluded from checkpoint/branch/method selection'}
    models, states, raw, original = {}, {}, {}, {}
    result['canonical_native_batch_size'] = 32
    result['original_LS_native_heads_reused_as_optimized_inverse_controls'] = False
    failure = None
    started = time.perf_counter()

    def save(name, arrays):
        meter.check()
        return probes.save_npz(output / (name + '.npz'),
                               {k: np.ascontiguousarray(v) for k, v in arrays.items()})

    def score(prediction, values, mask=None):
        observed = values['observed'] if mask is None else mask
        if not observed.any():
            return {'status': 'NO_OBSERVED_TARGETS', 'normalized': None, 'physical_per_channel_diagnostic': None}
        return {'normalized': nh.score_predictions(prediction, values['target'], observed),
                'physical_per_channel_diagnostic': nh.score_predictions(
                    prediction.astype(np.float64) * values['std'],
                    values['target'].astype(np.float64) * values['std'], observed)}

    def snapshot():
        directory = output / 'sources'
        directory.mkdir()
        for path in (Path(__file__), HERE/'projection_math_v1.py', HERE/'native_head_v1.py',
                     HERE/'whole_fit_target_cost_v1.py', HERE/'continuation_runtime_v1.py',
                     HERE/'future_target_projection_v1.py',
                     HERE/'PLAN_FIT_TARGET_SUBSPACE_KO_v1.md',
                     HERE/'FIT_TARGET_CONTINUATION_SCOPE_KO_v1.md',
                     HERE/'native_batch_contract_v2.py',
                     BASE/'cost_readout_v1.py', OLD/'run_condition_probes.py',
                     OLD/'prepare_inputs.py', OLD/'matched_readout_contract_v1.py'):
            target = directory / path.name
            with path.open('rb') as source, target.open('xb') as destination:
                shutil.copyfileobj(source, destination)
            result['source_snapshots'].append({'path': str(path), 'sha256': rt.sha(path), 'snapshot_path': str(target)})

    def load_split(dataset, split):
        parent = original[dataset]
        arrays, checked = probes.load_checked_npz(parent['inputs'][split])
        arrays = {key: np.ascontiguousarray(value) for key, value in arrays.items()}
        fit = raw[dataset].get('fit')
        mean, std = (arrays['mean'], arrays['std']) if fit is None else (fit['mean'], fit['std'])
        observed_contract = contract.validate_raw_split(dataset, split, arrays, mean, std)
        raw[dataset][split] = arrays
        result['datasets'][dataset]['inputs'][split] = parent['inputs'][split]
        result['datasets'][dataset]['data_contract'][split] = observed_contract
        result['source_records'].append(checked)
        return arrays

    def prepare_dataset(dataset, parent_name):
        parent_path = BASE/'runs'/parent_name/'RESULT.json'
        original[dataset] = probes.read(parent_path)
        matching_parent_inputs(dataset, original[dataset], source_projection['datasets'][dataset])
        parent_record = record(parent_path)
        result['source_parent_results'][dataset] = parent_record
        protected_prior.append(parent_record)
        for split in ('fit', 'tune'):
            values = load_split(dataset, split)
            actual = raw_predict(values['history'])
            parity = old_cost.parity(actual, values['raw_forecast'])
            result['datasets'][dataset]['raw_parity'][split] = parity
            if not parity['pass']:
                raise ValueError('RAW_BASELINE_PARITY:' + dataset + ':' + split)
            result['datasets'][dataset]['raw_scores'][split] = score(values['raw_forecast'], values)
        return True

    def prepare_projection(dataset, route, k, condition):
        meter.check({'dataset': dataset, 'route': route, 'stage': 'projection_fit'})
        fit = raw[dataset]['fit']
        if route == 'PCA_PINV':
            source_state_record = source_projection['datasets'][dataset]['projections']['PCA']['state_artifact']
            source_state_arrays, checked = probes.load_checked_npz(source_state_record)
            state = ft.pca_encoder_pinv_control(pm.ProjectionState.from_npz(source_state_arrays))
            condition['source_original_PCA_state'] = source_state_record
            result['source_records'].append(checked)
            result['reused_projection_states'] += 1
        else:
            result['projection_fit_attempts'] += 1
            state = ft.fit_target_subspace(fit['history'], fit['history_observed'],
                fit['target'], fit['observed'], fit['raw_last'], fit['origins'], k)
            result['projection_fits'] += 1
        condition['state_artifact'] = save(dataset+'_'+route+'_STATE', state.to_npz())
        condition['fit_metadata'] = state.metadata()
        result['projection_conditions'] += 1
        return state

    def bolt_split(dataset, route, condition, state, split):
        values = raw[dataset][split]
        prediction, latent, future = projected_predict(state, values['history'])
        condition['bolt']['artifacts'][split] = save(dataset+'_'+route+'_BOLT_'+split,
            {'prediction': prediction, 'latent_history': latent, 'latent_future': future, 'origins': values['origins']})
        condition['bolt']['scores'][split] = score(prediction, values)
        return prediction

    def train_seed(dataset, route, condition, state, seed):
        checkpoints = []
        entry = {'status': 'TRAINING', 'checkpoints': checkpoints, 'scores': {}, 'artifacts': {}}
        condition['native'][str(seed)] = entry
        def callback(update, checkpoint_state, metrics):
            temporary = nh.NativeForecaster(state)
            temporary.load_state_dict(checkpoint_state)
            temporary.eval().requires_grad_(False)
            tune_prediction = native_predict(temporary, raw[dataset]['tune']['history'])
            prefix = dataset+'_'+route+'_NATIVE_s'+str(seed)+'_u'+str(update)
            checkpoints.append({'update': update, 'tune': metrics,
                'state_sha256': nh.state_hash(checkpoint_state),
                'state_artifact': save(prefix+'_STATE', {name: value.numpy() for name, value in checkpoint_state.items()}),
                'tune_prediction_artifact': save(prefix+'_TUNE', {'prediction': tune_prediction,
                    'origins': raw[dataset]['tune']['origins']})})
        result['native_training_attempts'] += 1
        trained = nh.train_native(state, raw[dataset]['fit'], raw[dataset]['tune'], seed,
            progress_hook=lambda event: meter.check({'dataset': dataset, 'route': route, 'seed': seed, **event}),
            checkpoint_callback=callback)
        result['native_training_fits'] += 1
        model = trained['model']
        entry.update(metadata=trained['metadata'], learning_curve=trained['learning_curve'])
        for split in ('fit', 'tune'):
            values = raw[dataset][split]
            prediction, z, future = native_components(model, values['history'])
            entry['artifacts'][split] = save(dataset+'_'+route+'_NATIVE_s'+str(seed)+'_'+split,
                {'prediction': prediction, 'latent_history': z, 'latent_future': future,
                 'origins': values['origins']})
            entry['scores'][split] = score(prediction, values)
        entry['status'] = 'COMPLETE'
        print(json.dumps({'event': 'NATIVE_COMPLETE', 'dataset': dataset, 'route': route, 'seed': seed,
            'selected_update': entry['metadata']['selected_update'],
            'tune_mse': entry['scores']['tune']['normalized']['mse']}), flush=True)
        return model

    def bolt_future_perturbation(condition, state, tiny, changed):
        before = projected_predict(state, tiny['history'])[0]
        after = projected_predict(state, changed['history'])[0]
        if not np.array_equal(before, after):
            raise ValueError('FIXED_PROJECTION_FUTURE_LABEL_PERTURBATION')
        condition['future_label_perturbation'] = {
            'scope': 'Actual extracted two-window deployment payload; fixed fitted E/D/F, not whole raw-future replay',
            'target_and_future_mask_changed': True, 'bolt_exact_equal': True, 'native_exact_equal': {}}
        return True

    def native_development(dataset, route, seed, condition, model, development, tiny, changed):
        a = native_predict(model, tiny['history'])
        b = native_predict(model, changed['history'])
        if not np.array_equal(a, b):
            raise ValueError('FIXED_NATIVE_FUTURE_LABEL_PERTURBATION')
        condition.setdefault('future_label_perturbation', {}).setdefault('native_exact_equal', {})[str(seed)] = True
        values = raw[dataset][development]
        prediction, z, future = native_components(model, values['history'])
        entry = condition['native'][str(seed)]
        entry['artifacts'][development] = save(dataset+'_'+route+'_NATIVE_s'+str(seed)+'_'+development,
            {'prediction': prediction, 'latent_history': z, 'latent_future': future, 'origins': values['origins']})
        entry['scores'][development] = score(prediction, values)
        return prediction

    def oracle_split(dataset, route, condition, state, split):
        values = raw[dataset][split]
        complete = np.all(values['observed'], axis=2, keepdims=True)
        mask = np.broadcast_to(complete, values['observed'].shape).copy()
        safe_y = np.where(values['observed'], values['target'], np.float32(0.))
        future_y = state.encode(np.ascontiguousarray(safe_y))
        last = state.encode(values['history'])[:, -1]
        oracle = state.forecast(values['raw_last'], future_y, last)
        diagnostic = {'role': 'diagnostic_only', 'selection_used': False,
            'complete_target_positions': int(complete.sum()), 'total_target_positions': int(complete.size),
            'scores': score(oracle, values, mask),
            'raw_scores_same_mask': score(values['raw_forecast'], values, mask)}
        if split in condition['bolt']['artifacts']:
            bolt_saved, _ = probes.load_checked_npz(condition['bolt']['artifacts'][split])
            diagnostic['bolt_scores_same_mask'] = score(bolt_saved['prediction'], values, mask)
        else:
            diagnostic['bolt_scores_same_mask'] = {'status': 'SKIPPED_MISSING_BOLT_OUTPUT'}
        diagnostic['native_scores_same_mask'] = {}
        for seed, entry in condition['native'].items():
            if split in entry['artifacts']:
                saved, _ = probes.load_checked_npz(entry['artifacts'][split])
                diagnostic['native_scores_same_mask'][seed] = score(saved['prediction'], values, mask)
        condition['oracle'][split] = {**diagnostic, 'status': 'COMPLETE',
            'artifact': save(dataset+'_'+route+'_ORACLE_'+split,
                {'prediction': oracle, 'latent_future': future_y, 'evaluation_mask': mask, 'origins': values['origins']})}
        return True

    def cost_dataset(dataset, dev, predictors, predictions, metadata):
        physical_record = old_cost_result['dataset_input_sources'][dataset]['physical_bridge']
        physical, checked = probes.load_checked_npz(physical_record)
        result['source_records'].append(checked)
        old_cost.validate_physical_inputs(physical['history_physical'], physical['history_observed'],
            dev['history'], dev['mean'], dev['std'], dev['origins'])
        memory = lambda: {'resident_rss_bytes': psutil.Process().memory_info().rss,
                          'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated(),
                          'peak_reserved_vram_bytes': torch.cuda.max_memory_reserved()}
        result['datasets'][dataset]['physical_bridge'] = physical_record
        return wc.measure_cost(dataset, physical['history_physical'], physical['history_observed'],
            dev['mean'], dev['std'], dev['origins'], predictors, predictions, metadata,
            output/('cost_'+dataset), meter.check, torch.cuda.synchronize, memory)

    try:
        meter.protected_check()
        snapshot()
        old_cost_path = BASE/'cost_runs/first_readout24_v2/RESULT.json'
        old_cost_result = probes.read(old_cost_path)
        protected_prior.append(record(old_cost_path))
        engine = probes.import_old_engine()
        weight = probes.verified_file(old_cost_result['model_binary_before'])
        config = probes.verified_file(old_cost_result['model_config_before'])
        result.update(model_binary_before=weight, model_config_before=config)
        bolt = engine.ex.previous.FrozenBolt()
        result['F_before'] = probes.model_hashes(bolt)
        if result['F_before'] != old_cost_result['F_before']:
            raise ValueError('EXACT_ORIGINAL_FROZEN_BOLT_DUAL_HASH_REQUIRED')
        result['runtime_behavior_contract'] = old_cost.bolt_runtime_contract(engine, bolt, meter.check)
        bolt.model.eval().requires_grad_(False)

        def raw_predict(history):
            pieces = []
            with torch.no_grad():
                for index in range(0, len(history), 8):
                    meter.check()
                    pieces.append(bolt(engine.tensor(history[index:index+8])).detach().cpu().numpy().copy())
            return np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32)

        def projected_predict(state, history):
            latent = state.encode(history)
            future = raw_predict(latent)
            return state.forecast(history[:, -1], future, latent[:, -1]), latent, future

        def native_predict(model, history):
            return native_components(model, history)[0]

        with threadpool_limits(limits=2):
            for dataset, (parent_name, k, development) in DATASETS.items():
                row = {'status': 'RUNNING', 'K': k, 'inputs': {}, 'data_contract': {},
                       'projections': {}, 'raw_parity': {}, 'raw_scores': {}}
                result['datasets'][dataset] = row
                raw[dataset], states[dataset], models[dataset] = {}, {}, {}
                prepared = isolated_path(result['failed_paths'], dataset, None, 'dataset_fit_tune',
                    lambda: prepare_dataset(dataset, parent_name))
                if prepared is None:
                    row.update(status='FAILED_PATH_PRESERVED', failure=result['failed_paths'][-1])
                    result['cost'][dataset] = {'status': 'SKIPPED_INCOMPLETE_REQUIRED_ROUTES',
                        'reason': 'Dataset FIT/TUNE input binding or RAW parity failed',
                        'missing_routes': list(wc.ROUTES)}
                    continue
                for route in ROUTES:
                    condition = {'status': 'RUNNING', 'bolt': {'scores': {}, 'artifacts': {}},
                                 'native': {}, 'oracle': {}}
                    row['projections'][route] = condition
                    state = isolated_path(result['failed_paths'], dataset, route, 'projection_fit',
                        lambda: prepare_projection(dataset, route, k, condition))
                    if state is None:
                        condition.update(status='FAILED_PATH_PRESERVED', failure=result['failed_paths'][-1])
                        continue
                    states[dataset][route], models[dataset][route] = state, {}
                    for split in ('fit', 'tune'):
                        prediction = isolated_path(result['failed_paths'], dataset, route, 'bolt_'+split,
                            lambda: bolt_split(dataset, route, condition, state, split))
                        if prediction is None:
                            condition['bolt'].setdefault('failed_splits', {})[split] = result['failed_paths'][-1]
                    condition['bolt']['status'] = ('COMPLETE' if len(condition['bolt']['scores']) == 2
                                                   else 'PARTIAL_FAILED_PATHS')
                    for seed in SEEDS:
                        model = isolated_path(result['failed_paths'], dataset, route, 'native_fit',
                            lambda: train_seed(dataset, route, condition, state, seed), seed=seed)
                        if model is None:
                            condition['native'][str(seed)].update(status='FAILED_PATH_PRESERVED',
                                failure=result['failed_paths'][-1])
                            continue
                        models[dataset][route][seed] = model
                    condition['status'] = ('COMPLETE_FIT_TUNE_SEALED' if
                        condition['bolt']['status'] == 'COMPLETE' and len(models[dataset][route]) == 2
                        else 'PARTIAL_FIT_TUNE_SEALED')
                row['status'] = 'FIT_TUNE_PHASE_FINISHED'
            rt.write_new(output/'SELECTION_LOCK.json', {'status': 'SEALED_BEFORE_NEW_DEV_AND_ORACLE',
                'datasets': {dataset: {route: {seed: entry['metadata']['selected_update']
                    for seed, entry in values['native'].items() if entry.get('status') == 'COMPLETE'}
                    for route, values in row['projections'].items() if route in states[dataset]}
                    for dataset, row in result['datasets'].items()},
                'failed_paths_at_seal': list(result['failed_paths']),
                'selection_scope': 'Completed native FIT/TUNE seeds only; failures never become selected methods',
                'sealed_unix': time.time()})
            result['selection_lock'] = {'path': str(output/'SELECTION_LOCK.json'), 'sha256': rt.sha(output/'SELECTION_LOCK.json')}
            for dataset, (_, k, development) in DATASETS.items():
                row = result['datasets'][dataset]
                if not states[dataset]:
                    row['status'] = 'PARTIAL_COMPLETED' if row['inputs'] else 'FAILED_PATH_PRESERVED'
                    result['cost'][dataset] = {'status': 'SKIPPED_INCOMPLETE_REQUIRED_ROUTES',
                        'reason': 'No successfully serialized projection state for this dataset',
                        'missing_routes': list(wc.ROUTES[1:])}
                    continue
                dev = isolated_path(result['failed_paths'], dataset, None, 'development_input',
                    lambda: load_split(dataset, development))
                if dev is None:
                    row.update(status='PARTIAL_COMPLETED', development_failure=result['failed_paths'][-1])
                    result['cost'][dataset] = {'status': 'SKIPPED_INCOMPLETE_REQUIRED_ROUTES',
                        'reason': 'Matched development input failed validation', 'missing_routes': list(wc.ROUTES)}
                    continue
                row['raw_scores'][development] = score(dev['raw_forecast'], dev)
                predictions = {'RAW_BOLT': dev['raw_forecast']}
                predictors = {'RAW_BOLT': raw_predict}
                metadata = {'RAW_BOLT': {'source': 'original frozen Bolt', 'shared_resident_model': True}}
                for route, condition in row['projections'].items():
                    if route not in states[dataset]:
                        continue
                    state = states[dataset][route]
                    tiny = {key: value[:2].copy() for key, value in dev.items()
                            if value.ndim and len(value) == len(dev['history'])}
                    changed = {key: value.copy() for key, value in tiny.items()}
                    changed['target'][:] = np.float32(1e6)
                    changed['observed'][:] = False
                    perturbation = isolated_path(result['failed_paths'], dataset, route, 'bolt_future_label_check',
                        lambda: bolt_future_perturbation(condition, state, tiny, changed))
                    if perturbation is None:
                        condition['future_label_perturbation'] = {'status': 'FAILED_PATH_PRESERVED',
                            'failure': result['failed_paths'][-1], 'native_exact_equal': {}}
                    prediction = isolated_path(result['failed_paths'], dataset, route, 'bolt_'+development,
                        lambda: bolt_split(dataset, route, condition, state, development))
                    if prediction is not None:
                        predictions[route+'_BOLT'] = prediction
                        predictors[route+'_BOLT'] = lambda history, state=state: projected_predict(state, history)[0]
                        metadata[route+'_BOLT'] = {'projection_route': route, 'E_D_frozen': True}
                    else:
                        condition['bolt']['status'] = 'PARTIAL_FAILED_PATHS'
                        condition['bolt'].setdefault('failed_splits', {})[development] = result['failed_paths'][-1]
                    for seed, model in models[dataset][route].items():
                        prediction = isolated_path(result['failed_paths'], dataset, route, 'native_'+development,
                            lambda: native_development(dataset, route, seed, condition, model, development, tiny, changed), seed=seed)
                        if prediction is None:
                            condition['native'][str(seed)].update(status='PARTIAL_FAILED_PATHS',
                                development_failure=result['failed_paths'][-1])
                            continue
                        if seed == 92601:
                            predictions[route+'_NATIVE'] = prediction
                            predictors[route+'_NATIVE'] = lambda history, model=model: native_predict(model, history)
                            metadata[route+'_NATIVE'] = {'projection_route': route, 'native_seed': 92601,
                                                       'E_D_frozen': True, 'head_device': 'cpu'}
                    for split in raw[dataset]:
                        completed = isolated_path(result['failed_paths'], dataset, route, 'oracle_'+split,
                            lambda: oracle_split(dataset, route, condition, state, split))
                        if completed is None:
                            condition['oracle'][split] = {'status': 'FAILED_PATH_PRESERVED',
                                'role': 'diagnostic_only', 'selection_used': False, 'failure': result['failed_paths'][-1]}
                    failed_this_route = any(item['dataset'] == dataset and item['route'] == route
                                            for item in result['failed_paths'])
                    condition['status'] = 'PARTIAL_COMPLETED' if failed_this_route else 'COMPLETE'
                    print(json.dumps({'event': 'PROJECTION_PHASE_FINISHED', 'dataset': dataset, 'route': route,
                        'status': condition['status'],
                        'bolt_dev_mse': condition['bolt']['scores'].get(development, {}).get('normalized', {}).get('mse'),
                        'native_dev_mse': {seed: entry['scores'][development]['normalized']['mse']
                            for seed, entry in condition['native'].items() if development in entry['scores']}}), flush=True)
                missing = sorted(set(wc.ROUTES) - set(predictors))
                if missing:
                    result['cost'][dataset] = {'status': 'SKIPPED_INCOMPLETE_REQUIRED_ROUTES',
                        'reason': 'The fixed five-route benchmark cannot be replaced by a smaller cohort',
                        'missing_routes': missing, 'available_routes': sorted(predictors)}
                else:
                    cost = isolated_path(result['failed_paths'], dataset, None, 'whole_cost',
                        lambda: cost_dataset(dataset, dev, predictors, predictions, metadata))
                    if cost is None:
                        cost_result = output/('cost_'+dataset)/'RESULT.json'
                        result['cost'][dataset] = (probes.read(cost_result) if cost_result.exists() else
                            {'status': 'FAILED_PATH_PRESERVED', 'failure': result['failed_paths'][-1]})
                    else:
                        result['cost'][dataset] = cost
                row['status'] = ('COMPLETE' if len(row['projections']) == 2 and
                    all(condition['status'] == 'COMPLETE' for condition in row['projections'].values()) and
                    result['cost'][dataset].get('status') == 'COMPLETE' else 'PARTIAL_COMPLETED')
        result['F_after'] = probes.model_hashes(bolt)
        if result['F_before'] != result['F_after']:
            raise ValueError('FROZEN_BOLT_CHANGED')
        result['model_binary_after'] = probes.verified_file(weight)
        result['model_config_after'] = probes.verified_file(config)
        for file_row in result['source_snapshots']:
            if rt.sha(file_row['path']) != file_row['sha256'] or rt.sha(file_row['snapshot_path']) != file_row['sha256']:
                raise ValueError('EXECUTED_SOURCE_CHANGED')
        for file_row in protected_prior:
            if record(file_row['path']) != file_row:
                raise ValueError('PRIOR_FILE_CHANGED_DURING_RUN:' + file_row['path'])
        result['prior_source_results_caches_and_selection_locks_unchanged'] = True
        if rt.sha(result['selection_lock']['path']) != result['selection_lock']['sha256']:
            raise ValueError('NEW_SEALED_SELECTION_LOCK_CHANGED')
        meter.protected_check()
        completed_conditions = sum(condition.get('status') == 'COMPLETE'
            for row in result['datasets'].values() for condition in row['projections'].values())
        cost_rows = [row for cost in result['cost'].values() for row in cost.get('rows', [])
                     if row.get('status') == 'COMPLETE']
        result['completeness'] = {
            'expected': {'projection_fits': 2, 'reused_projection_states': 2,
                         'projection_conditions': 4, 'completed_projection_conditions': 4,
                         'native_training_fits': 8, 'cost_cells': 20, 'timer_samples': 120},
            'actual': {'projection_fits': result['projection_fits'],
                       'reused_projection_states': result['reused_projection_states'],
                       'projection_conditions': result['projection_conditions'],
                       'completed_projection_conditions': completed_conditions,
                       'native_training_fits': result['native_training_fits'],
                       'cost_cells': len(cost_rows),
                       'timer_samples': sum(row['repeats'] for row in cost_rows)}}
        complete = (not result['failed_paths'] and
                    result['completeness']['expected'] == result['completeness']['actual'] and
                    all(row['status'] == 'COMPLETE' for row in result['datasets'].values()))
        result['status'] = 'COMPLETE' if complete else 'PARTIAL_COMPLETED'
        result['full_scope_complete_claimed'] = complete
    except Exception as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
        result.update(status='FAILED_PRESERVED', failure=failure)
        raise
    finally:
        result['wall_seconds'] = time.perf_counter()-started
        result['resource_terminal'] = meter.close(result['status'], failure)
        rt.write_new(output/'RESULT.json', result)
        print(json.dumps({'status': result['status'], 'output': str(output/'RESULT.json'),
                          'native_training_fits': result['native_training_fits'],
                          'projection_fits': result['projection_fits'],
                          'projection_conditions': result['projection_conditions']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    execute(parser.parse_args().tag)
