"""Actual PCA/ForeCA, frozen Bolt/native, and diagnostic-only oracle comparison."""
from pathlib import Path
import argparse
import json
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


def execute(tag):
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import projection_math_v1 as pm
    import native_head_v1 as nh
    import whole_cost_v1 as wc
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
    output = HERE / 'runs' / tag
    output.parent.mkdir(exist_ok=True)
    output.mkdir(exist_ok=False)
    result = {'status': 'RUNNING', 'scientific_execution': True,
              'protected_TEST_accessed': False, 'independent_holdout': False,
              'new_backbone_training': False, 'datasets': {}, 'cost': {},
              'native_training_fits': 0, 'projection_fits': 0,
              'source_snapshots': [], 'source_records': [],
              'authorization': {'path': str(rt.AUTH), 'sha256': rt.sha(rt.AUTH)},
              'continuation_manifest': {'path': str(rt.MANIFEST), 'sha256': rt.sha(rt.MANIFEST)},
              'hard_elapsed_or_cumulative_resource_caps': None,
              'forecast_contract': 'FP32 raw_last + (latent_future - encoded_history_last) @ decoder',
              'oracle_role': 'diagnostic_only; excluded from checkpoint/branch/method selection'}
    models, states, raw, original = {}, {}, {}, {}
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
                     HERE/'whole_cost_v1.py', HERE/'continuation_runtime_v1.py',
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

    try:
        meter.protected_check()
        snapshot()
        old_cost_result = probes.read(BASE/'cost_runs/first_readout24_v2/RESULT.json')
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
            with torch.no_grad():
                return np.ascontiguousarray(model(torch.from_numpy(history),
                    torch.from_numpy(history[:, -1].copy())).cpu().numpy())

        with threadpool_limits(limits=2):
            for dataset, (parent_name, k, development) in DATASETS.items():
                original[dataset] = probes.read(BASE/'runs'/parent_name/'RESULT.json')
                if original[dataset]['status'] != 'COMPLETE':
                    raise ValueError('COMPLETE_PARENT_INPUTS_REQUIRED')
                result['datasets'][dataset] = {'status': 'RUNNING', 'K': k,
                    'inputs': {}, 'data_contract': {}, 'projections': {}, 'raw_parity': {}, 'raw_scores': {}}
                raw[dataset] = {}
                for split in ('fit', 'tune'):
                    values = load_split(dataset, split)
                    actual = raw_predict(values['history'])
                    parity = old_cost.parity(actual, values['raw_forecast'])
                    result['datasets'][dataset]['raw_parity'][split] = parity
                    if not parity['pass']:
                        raise ValueError('RAW_BASELINE_PARITY:' + dataset + ':' + split)
                    result['datasets'][dataset]['raw_scores'][split] = score(values['raw_forecast'], values)
                states[dataset], models[dataset] = {}, {}
                for route in ('PCA', 'FORECA'):
                    meter.check({'dataset': dataset, 'route': route, 'stage': 'projection_fit'})
                    condition = {'status': 'RUNNING', 'bolt': {'scores': {}, 'artifacts': {}}, 'native': {}, 'oracle': {}}
                    result['datasets'][dataset]['projections'][route] = condition
                    fit = raw[dataset]['fit']
                    state = pm.fit_projection(route, fit['history'], fit['history_observed'], k)
                    states[dataset][route] = state
                    condition['state_artifact'] = save(dataset+'_'+route+'_STATE', state.to_npz())
                    condition['fit_metadata'] = state.metadata()
                    result['projection_fits'] += 1
                    for split in ('fit', 'tune'):
                        values = raw[dataset][split]
                        prediction, latent, future = projected_predict(state, values['history'])
                        condition['bolt']['artifacts'][split] = save(dataset+'_'+route+'_BOLT_'+split,
                            {'prediction': prediction, 'latent_history': latent, 'latent_future': future, 'origins': values['origins']})
                        condition['bolt']['scores'][split] = score(prediction, values)
                    condition['bolt']['status'] = 'COMPLETE'
                    models[dataset][route] = {}
                    for seed in (92601, 92602):
                        checkpoints = []
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
                        trained = nh.train_native(state, raw[dataset]['fit'], raw[dataset]['tune'], seed,
                            progress_hook=lambda event: meter.check({'dataset': dataset, 'route': route, 'seed': seed, **event}),
                            checkpoint_callback=callback)
                        model = trained['model']
                        models[dataset][route][seed] = model
                        entry = {'status': 'COMPLETE', 'metadata': trained['metadata'],
                                 'learning_curve': trained['learning_curve'], 'checkpoints': checkpoints,
                                 'scores': {}, 'artifacts': {}}
                        condition['native'][str(seed)] = entry
                        for split in ('fit', 'tune'):
                            values = raw[dataset][split]
                            prediction = native_predict(model, values['history'])
                            with torch.no_grad():
                                z = model.projection.encode(torch.from_numpy(values['history']))
                                future = model.head(z).cpu().numpy()
                            entry['artifacts'][split] = save(dataset+'_'+route+'_NATIVE_s'+str(seed)+'_'+split,
                                {'prediction': prediction, 'latent_history': z.numpy(), 'latent_future': future,
                                 'origins': values['origins']})
                            entry['scores'][split] = score(prediction, values)
                        result['native_training_fits'] += 1
                        print(json.dumps({'event': 'NATIVE_COMPLETE', 'dataset': dataset, 'route': route, 'seed': seed,
                            'selected_update': entry['metadata']['selected_update'],
                            'tune_mse': entry['scores']['tune']['normalized']['mse']}), flush=True)
                    condition['status'] = 'COMPLETE_FIT_TUNE_SEALED'
            rt.write_new(output/'SELECTION_LOCK.json', {'status': 'SEALED_BEFORE_NEW_DEV_AND_ORACLE',
                'datasets': {dataset: {route: {seed: entry['metadata']['selected_update']
                    for seed, entry in values['native'].items()} for route, values in row['projections'].items()}
                    for dataset, row in result['datasets'].items()}, 'sealed_unix': time.time()})
            result['selection_lock'] = {'path': str(output/'SELECTION_LOCK.json'), 'sha256': rt.sha(output/'SELECTION_LOCK.json')}
            for dataset, (_, k, development) in DATASETS.items():
                row = result['datasets'][dataset]
                dev = load_split(dataset, development)
                row['raw_scores'][development] = score(dev['raw_forecast'], dev)
                predictions = {'RAW_BOLT': dev['raw_forecast']}
                predictors = {'RAW_BOLT': raw_predict}
                metadata = {'RAW_BOLT': {'source': 'original frozen Bolt', 'shared_resident_model': True}}
                for route, condition in row['projections'].items():
                    state = states[dataset][route]
                    tiny = {key: value[:2].copy() for key, value in dev.items()
                            if value.ndim and len(value) == len(dev['history'])}
                    changed = {key: value.copy() for key, value in tiny.items()}
                    changed['target'][:] = np.float32(1e6)
                    changed['observed'][:] = False
                    before = projected_predict(state, tiny['history'])[0]
                    after = projected_predict(state, changed['history'])[0]
                    if not np.array_equal(before, after):
                        raise ValueError('FIXED_PROJECTION_FUTURE_LABEL_PERTURBATION')
                    condition['future_label_perturbation'] = {
                        'scope': 'Actual extracted two-window deployment payload; fixed fitted E/D/F, not whole raw-future replay',
                        'target_and_future_mask_changed': True, 'bolt_exact_equal': True,
                        'native_exact_equal': {}}
                    for seed, model in models[dataset][route].items():
                        a = native_predict(model, tiny['history'])
                        b = native_predict(model, changed['history'])
                        if not np.array_equal(a, b):
                            raise ValueError('FIXED_NATIVE_FUTURE_LABEL_PERTURBATION')
                        condition['future_label_perturbation']['native_exact_equal'][str(seed)] = True
                    prediction, latent, future = projected_predict(state, dev['history'])
                    condition['bolt']['artifacts'][development] = save(dataset+'_'+route+'_BOLT_'+development,
                        {'prediction': prediction, 'latent_history': latent, 'latent_future': future, 'origins': dev['origins']})
                    condition['bolt']['scores'][development] = score(prediction, dev)
                    predictions[route+'_BOLT'] = prediction
                    predictors[route+'_BOLT'] = lambda history, state=state: projected_predict(state, history)[0]
                    metadata[route+'_BOLT'] = {'projection_route': route, 'E_D_frozen': True}
                    for seed, entry in condition['native'].items():
                        model = models[dataset][route][int(seed)]
                        prediction = native_predict(model, dev['history'])
                        with torch.no_grad():
                            z = model.projection.encode(torch.from_numpy(dev['history']))
                            future = model.head(z).cpu().numpy()
                        entry['artifacts'][development] = save(dataset+'_'+route+'_NATIVE_s'+seed+'_'+development,
                            {'prediction': prediction, 'latent_history': z.numpy(), 'latent_future': future, 'origins': dev['origins']})
                        entry['scores'][development] = score(prediction, dev)
                        if seed == '92601':
                            predictions[route+'_NATIVE'] = prediction
                            predictors[route+'_NATIVE'] = lambda history, model=model: native_predict(model, history)
                            metadata[route+'_NATIVE'] = {'projection_route': route, 'native_seed': 92601,
                                                       'E_D_frozen': True, 'head_device': 'cpu'}
                    for split, values in raw[dataset].items():
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
                        bolt_saved, _ = probes.load_checked_npz(condition['bolt']['artifacts'][split])
                        diagnostic['bolt_scores_same_mask'] = score(bolt_saved['prediction'], values, mask)
                        diagnostic['native_scores_same_mask'] = {}
                        for seed, native_entry in condition['native'].items():
                            saved, _ = probes.load_checked_npz(native_entry['artifacts'][split])
                            diagnostic['native_scores_same_mask'][seed] = score(saved['prediction'], values, mask)
                        condition['oracle'][split] = {**diagnostic, 'artifact': save(dataset+'_'+route+'_ORACLE_'+split,
                            {'prediction': oracle, 'latent_future': future_y, 'evaluation_mask': mask, 'origins': values['origins']})}
                    condition['status'] = 'COMPLETE'
                    print(json.dumps({'event': 'PROJECTION_COMPLETE', 'dataset': dataset, 'route': route,
                        'bolt_dev_mse': condition['bolt']['scores'][development]['normalized']['mse'],
                        'native_dev_mse': {seed: entry['scores'][development]['normalized']['mse']
                                           for seed, entry in condition['native'].items()}}), flush=True)
                physical_record = old_cost_result['dataset_input_sources'][dataset]['physical_bridge']
                physical, checked = probes.load_checked_npz(physical_record)
                result['source_records'].append(checked)
                old_cost.validate_physical_inputs(physical['history_physical'], physical['history_observed'],
                    dev['history'], dev['mean'], dev['std'], dev['origins'])
                memory = lambda: {'resident_rss_bytes': psutil.Process().memory_info().rss,
                                  'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated(),
                                  'peak_reserved_vram_bytes': torch.cuda.max_memory_reserved()}
                result['cost'][dataset] = wc.measure_cost(dataset, physical['history_physical'], physical['history_observed'],
                    dev['mean'], dev['std'], dev['origins'], predictors, predictions, metadata,
                    output/('cost_'+dataset), meter.check, torch.cuda.synchronize, memory)
                row['physical_bridge'] = physical_record
                row['status'] = 'COMPLETE'
        result['F_after'] = probes.model_hashes(bolt)
        if result['F_before'] != result['F_after']:
            raise ValueError('FROZEN_BOLT_CHANGED')
        result['model_binary_after'] = probes.verified_file(weight)
        result['model_config_after'] = probes.verified_file(config)
        for record in result['source_snapshots']:
            if rt.sha(record['path']) != record['sha256'] or rt.sha(record['snapshot_path']) != record['sha256']:
                raise ValueError('EXECUTED_SOURCE_CHANGED')
        meter.protected_check()
        result['status'] = 'COMPLETE'
    except Exception as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
        result.update(status='FAILED_PRESERVED', failure=failure)
        raise
    finally:
        result['wall_seconds'] = time.perf_counter()-started
        result['resource_terminal'] = meter.close(result['status'], failure)
        rt.write_new(output/'RESULT.json', result)
        print(json.dumps({'status': result['status'], 'output': str(output/'RESULT.json'),
                          'native_training_fits': result['native_training_fits'], 'projection_fits': result['projection_fits']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    execute(parser.parse_args().tag)
