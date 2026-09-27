"""Seal VAL selection, then perform the approved fixed Hog TEST evaluation."""
import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

from runtime_v4 import (CACHE, HERE, ROOT, Job, digest, environment_receipt,
                        load_data, save_json, source_receipt)

SEEDS = (92601, 92602)
PERIODS = ('test_a', 'test_b')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def receipt(path):
    path = resolve(path)
    return {'path': str(path), 'sha256': digest(path)}


def check_receipt(item):
    path = resolve(item['path'])
    if digest(path) != item['sha256']:
        raise ValueError(f'Sealed input changed: {path}')
    return path


def load_module(path, name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def validate_name(name):
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Use a unique alphanumeric/underscore/hyphen run name')


def neural_entry(row):
    result = dict(row)
    result['kind'] = 'neural'
    result['checkpoint'] = str(resolve(row['checkpoint']))
    check_receipt({'path': result['checkpoint'], 'sha256': row['checkpoint_sha256']})
    check_receipt(row['best_val_prediction'])
    if not math.isfinite(row['best_val_mse']):
        raise ValueError(f'Nonfinite VAL loss: {row["id"]}')
    result_path = check_receipt({'path': row['result_path'], 'sha256': row['result_sha256']})
    completed = read_json(result_path)
    if completed['status'] != 'complete':
        raise ValueError(f'Incomplete run cannot be sealed: {row["id"]}')
    for key in ('id', 'spec', 'checkpoint_sha256', 'best_val_prediction', 'best_val_mse', 'selected_epoch'):
        if completed[key] != row[key]:
            raise ValueError(f'Run receipt disagrees with completed result: {row["id"]}/{key}')
    return result


def cohort_selection(selection, models):
    candidates = {}
    for family in ('tsfm_res', 'tsfm_raw', 'linear_res', 'compress', 'lora'):
        candidates[family] = {}
        modes = ('current',) if family == 'lora' else ('current', 'fixed_ed')
        for mode in modes:
            members = [r for r in models if r['spec']['family'] == family and r['spec']['ed_mode'] == mode]
            expected = 1 if family == 'compress' and mode == 'fixed_ed' else 2
            if len(members) != expected:
                raise ValueError(f'{family}/{mode}: expected {expected} checkpoints')
            if expected == 2 and sorted(r['spec']['seed'] for r in members) != list(SEEDS):
                raise ValueError(f'{family}/{mode}: paired seeds missing')
            members.sort(key=lambda r: r['spec'].get('seed') or 0)
            candidates[family][mode] = {'run_ids': [r['id'] for r in members],
                                       'mean_best_val_mse': sum(r['best_val_mse'] for r in members) / expected}
    selected = {}
    for family, modes in candidates.items():
        mode = min(modes, key=lambda m: (modes[m]['mean_best_val_mse'], m != 'fixed_ed'))
        declared = selection['selected'][family]
        if declared['ed_mode'] != mode or set(declared['run_ids']) != set(modes[mode]['run_ids']):
            raise ValueError(f'{family}: declaration disagrees with frozen VAL selection rule')
        if not math.isclose(declared['mean_best_val_mse'], modes[mode]['mean_best_val_mse'], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f'{family}: declared VAL mean mismatch')
        selected[family] = dict(modes[mode], ed_mode=mode)
    if selection['res_selected_mode'] != selected['tsfm_res']['ed_mode']:
        raise ValueError('RES matched-mode declaration mismatch')
    return candidates, selected


def seal_evaluation(args):
    destination = HERE / 'evaluation_seal.json'
    if destination.exists() or (HERE / 'test_exposure.json').exists():
        raise FileExistsError('Never overwrite a seal or retroactively reseal exposed TEST')
    protocol = read_json(HERE / 'protocol.json')
    selection = read_json(args.selection)
    contract = read_json(args.data_contract)
    neural = selection['all_neural_runs']
    if len(neural) != 16:
        raise ValueError('All sixteen planned neural configurations must complete before TEST')
    expected_specs = {(s['family'], s['ed_mode'], s['seed']) for s in protocol['fits']}
    actual_specs = {(r['spec']['family'], r['spec']['ed_mode'], r['spec']['seed']) for r in neural}
    if expected_specs != actual_specs:
        raise ValueError('Neural candidate coverage differs from the approved protocol')
    models = [neural_entry(r) for r in neural]
    models += [neural_entry(selection['deterministic_models'][key]) for key in ('f0', 'compress_fixed_ed')]
    candidates, selected = cohort_selection(selection, models)
    shared = read_json(args.shared_selection)
    if shared['test_used_for_selection'] is not False:
        raise ValueError('SHARED selection must use VAL only')
    order = {float(p): i for i, p in enumerate(protocol['ridge_penalties'])}
    if len(shared['fits']) != 3 or {r['penalty'] for r in shared['fits']} != set(order):
        raise ValueError('All three approved ridge penalties must complete before TEST')
    shared_entry = min(shared['fits'], key=lambda r: (r['val_mse'], order[r['penalty']]))
    if shared_entry['id'] != shared['selected_id'] or shared_entry['penalty'] != shared['selected_penalty']:
        raise ValueError('SHARED selection differs from the approved VAL/tie rule')
    for fit in shared['fits']:
        check_receipt(fit['result_file'])
        if not math.isfinite(fit['val_mse']):
            raise ValueError(f'Nonfinite SHARED VAL loss: {fit["id"]}')
    shared_model = {'id': shared_entry['id'], 'kind': 'shared',
                    'spec': {'family': 'shared', 'dataset': 'hog', 'seed': None, 'ed_mode': None,
                             'penalty': shared_entry['penalty']},
                    'weights': shared_entry['weights'], 'best_val_mse': shared_entry['val_mse'],
                    'best_val_prediction': shared_entry['validation_predictions']}
    check_receipt(shared_model['weights'])
    check_receipt(shared_model['best_val_prediction'])
    models.append(shared_model)
    if len({r['id'] for r in models}) != 19:
        raise ValueError('Expected nineteen unique evaluation model identifiers')
    f0 = next(r['id'] for r in models if r['spec']['family'] == 'f0')
    roles = {family: value['run_ids'] for family, value in selected.items()}
    roles.update(f0=[f0], shared=[shared_model['id']])
    mode = selected['tsfm_res']['ed_mode']
    comparisons = []
    for family in ('tsfm_raw', 'linear_res', 'compress'):
        comparisons.append({'name': f'res_vs_{family}_at_res_mode',
                            'primary': family == 'tsfm_raw', 'left': roles['tsfm_res'],
                            'right': candidates[family][mode]['run_ids'], 'matching': f'RES-selected {mode}'})
    for family in ('tsfm_raw', 'linear_res', 'compress', 'lora', 'f0', 'shared'):
        comparisons.append({'name': f'selected_res_vs_selected_{family}', 'primary': False,
                            'left': roles['tsfm_res'], 'right': roles[family], 'matching': 'independent VAL selection'})
    gpu_ids = list(dict.fromkeys(identifier for family in ('f0', 'tsfm_res', 'tsfm_raw', 'linear_res', 'lora', 'compress')
                                for identifier in roles[family]))
    cpu_ids = roles['linear_res'] + roles['shared']
    if len(gpu_ids) not in (10, 11) or len(cpu_ids) != 3:
        raise ValueError('Unexpected approved cost cohort size')
    sealed = {'schema_version': 1, 'status': 'sealed', 'sealed_utc': time.time(),
              'selection': receipt(args.selection), 'shared_selection': receipt(args.shared_selection),
              'data_contract': receipt(args.data_contract), 'data_receipts': {k: contract[k] for k in ('trainval', 'test')},
              'protocol': receipt(HERE / 'protocol.json'), 'plan': receipt(HERE / 'PLAN.md'),
              'channels': protocol['selected_columns'], 'splits': protocol['splits'],
              'origins': contract['origins']['lists'],
              'origin_counts': protocol['expected_origins'], 'metric': protocol['primary_metric'],
              'bootstrap': protocol['bootstrap'], 'models': models, 'candidates': candidates,
              'selected': selected, 'roles': roles, 'comparisons': comparisons,
              'cost': dict(protocol['cost'], gpu_model_ids=gpu_ids, cpu_model_ids=cpu_ids),
              'source': source_receipt(), 'environment': environment_receipt(),
              'test_opened_during_seal': False,
              'scope': 'One fixed new Hog group; pretraining disjointness is not certified.'}
    save_json(destination, sealed)
    return {'seal': str(destination), 'sha256': digest(destination), 'models': len(models)}


def load_seal():
    path = HERE / 'evaluation_seal.json'
    sealed = read_json(path)
    if sealed.get('status') != 'sealed':
        raise ValueError('Evaluation requires a completed fixed seal')
    for key in ('selection', 'shared_selection', 'data_contract', 'protocol', 'plan'):
        check_receipt(sealed[key])
    for model in sealed['models']:
        if model['kind'] == 'neural':
            check_receipt({'path': model['checkpoint'], 'sha256': model['checkpoint_sha256']})
            check_receipt({'path': model['result_path'], 'sha256': model['result_sha256']})
        else:
            check_receipt(model['weights'])
        check_receipt(model['best_val_prediction'])
    return sealed


def shared_predict(inputs, row):
    with np.load(check_receipt(row['weights']), allow_pickle=False) as saved:
        weight, bias = saved['weight'], saved['bias']
    if weight.shape != (512, 48) or bias.shape != (48,):
        raise ValueError('SHARED saved affine dimensions differ from L512/H48')
    prediction = np.einsum('blc,lh->bhc', inputs.astype(np.float64), weight.astype(np.float64)) + bias[None, :, None]
    return prediction.astype(np.float32)


def error_terms(prediction, target, observed):
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError('Prediction, target and observation mask shapes differ')
    if not np.isfinite(prediction).all() or not np.isfinite(target[observed]).all():
        raise ValueError('Predictions and observed targets must be finite')
    delta = np.where(observed, prediction.astype(np.float64) - target.astype(np.float64), 0)
    return {'squared': (delta * delta).sum(axis=1), 'absolute': np.abs(delta).sum(axis=1),
            'count': observed.sum(axis=1)}


def scores_from_terms(terms, index=None):
    if index is None:
        index = np.arange(len(terms['count']))
    counts = terms['count'][index].sum(axis=0)
    squared = terms['squared'][index].sum(axis=0)
    absolute = terms['absolute'][index].sum(axis=0)
    valid = counts > 0
    channel_mse = [float(squared[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
    channel_mae = [float(absolute[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
    return {'status': 'available' if valid.all() else 'unavailable_missing_channel_targets',
            'mse': float(np.mean(channel_mse)) if valid.all() else None,
            'mae': float(np.mean(channel_mae)) if valid.all() else None,
            'channel_mse': channel_mse, 'channel_mae': channel_mae, 'channel_target_count': counts.tolist(),
            'channel_squared_error_sum': squared.tolist(), 'channel_absolute_error_sum': absolute.tolist(),
            'missing_channel_indices': np.flatnonzero(~valid).tolist()}


def pair_ids(ids, models):
    if len(ids) == 1:
        return ids * 2
    by_seed = {models[i]['spec']['seed']: i for i in ids}
    if set(by_seed) != set(SEEDS):
        raise ValueError('Comparison must pair both fixed seeds or broadcast one deterministic model')
    return [by_seed[s] for s in SEEDS]


def compare_terms(comparison, terms, models, period_indices, job):
    left_ids, right_ids = (pair_ids(comparison[k], models) for k in ('left', 'right'))
    count = terms[left_ids[0]]['count']
    for identifier in left_ids + right_ids:
        if not np.array_equal(terms[identifier]['count'], count):
            raise ValueError('All comparisons must use the same observed targets')
    left = np.mean([terms[i]['squared'] for i in left_ids], axis=0)
    right = np.mean([terms[i]['squared'] for i in right_ids], axis=0)

    def values(index):
        denominator = count[index].sum(axis=0)
        if np.any(denominator == 0):
            return None
        a = float(np.mean(left[index].sum(axis=0) / denominator))
        b = float(np.mean(right[index].sum(axis=0) / denominator))
        return {'left_mse': a, 'reference_mse': b, 'delta_mse': a - b,
                'relative_to_reference_pct': 100 * (a - b) / b if b > 0 else None}

    output = dict(comparison, direction='TSFM_RES minus reference; negative favors RES',
                  left_seed_ids=left_ids, right_seed_ids=right_ids,
                  deterministic_reference_broadcast=len(comparison['right']) == 1, periods={})
    for label in (*PERIODS, 'combined'):
        strata = [period_indices[label]] if label != 'combined' else list(period_indices.values())
        full_index = np.concatenate(strata)
        full = values(full_index)
        seed_values = []
        for seed, left_id, right_id in zip(SEEDS, left_ids, right_ids):
            a, b = scores_from_terms(terms[left_id], full_index)['mse'], scores_from_terms(terms[right_id], full_index)['mse']
            seed_values.append({'seed': seed, 'left_id': left_id, 'right_id': right_id,
                                'left_mse': a, 'reference_mse': b, 'delta_mse': a - b if a is not None and b is not None else None,
                                'relative_to_reference_pct': 100 * (a - b) / b if a is not None and b is not None and b > 0 else None})
        rng = np.random.default_rng(9262026)
        draws = []
        unavailable = 0
        for draw in range(2000):
            if draw % 250 == 0:
                job.check_limits()
            selected = []
            for stratum in strata:
                n = len(stratum)
                starts = rng.integers(0, n - 7 + 1, int(np.ceil(n / 7)))
                index = np.concatenate([np.arange(start, start + 7) for start in starts])[:n]
                selected.append(stratum[index])
            value = values(np.concatenate(selected))
            if value is None or value['relative_to_reference_pct'] is None:
                unavailable += 1
            else:
                draws.append(value)
        interval = None
        if full is not None and unavailable == 0:
            interval = {key: np.quantile([r[key] for r in draws], [0.025, 0.975]).tolist()
                        for key in ('delta_mse', 'relative_to_reference_pct')}
        output['periods'][label] = {'full': full, 'paired_seed_deltas': seed_values,
                                    'conditional_block95': interval, 'draws': 2000,
                                    'unavailable_draws': unavailable,
                                    'interval_status': 'available' if interval is not None else 'unavailable_without_dropping_channels',
                                    'strata': [label] if label != 'combined' else list(PERIODS)}
    output['interval_scope'] = 'Non-circular 7-origin blocks within each period, common resamples across channels/models; two trained seeds fixed. Does not include training or selection uncertainty.'
    return output


def evaluate_sealed(args, job):
    sealed = load_seal()
    exposure_path = HERE / 'test_exposure.json'
    seal_receipt = receipt(HERE / 'evaluation_seal.json')
    previous_result = None
    if exposure_path.exists():
        if not args.correction_reason or args.reuse_evaluation is None:
            raise FileExistsError('TEST exposed: require a documented correction and previous result/partial file to inspect and reuse completed predictions')
        previous = read_json(exposure_path)
        if previous['seal_sha256'] != seal_receipt['sha256']:
            raise ValueError('Cannot correct against a different TEST selection seal')
        previous_result = read_json(args.reuse_evaluation)
        if previous_result['seal']['sha256'] != seal_receipt['sha256']:
            raise ValueError('Previous evaluation does not belong to this frozen seal')
        for entry in previous_result['models'].values():
            check_receipt(entry['prediction'])
    else:
        if args.correction_reason or args.reuse_evaluation or args.recompute_ids:
            raise ValueError('A correction reason is only appropriate after recorded exposure')
        save_json(exposure_path, {'schema_version': 1, 'first_exposed_utc': time.time(),
                                 'status': 'exposed',
                                 'seal_sha256': seal_receipt['sha256'], 'seal': seal_receipt,
                                 'evaluation_name': args.name, 'command': sys.argv,
                                 'scope': 'Written before the first protected TEST archive is opened; immutable first exposure record.'})
    result = {'schema_version': 1, 'status': 'running', 'name': args.name, 'seal': seal_receipt,
              'first_exposure': receipt(exposure_path), 'correction_reason': args.correction_reason or None,
              'reuse_evaluation': receipt(args.reuse_evaluation) if args.reuse_evaluation else None,
              'explicit_recompute_ids': args.recompute_ids,
              'source': source_receipt(), 'environment': environment_receipt(), 'channels': sealed['channels'],
              'origins': {p: sealed['origins'][p] for p in PERIODS}, 'models': {},
              'roles': sealed['roles'], 'metric': sealed['metric'], 'comparisons': []}
    save_json(HERE / f'{args.name}_partial.json', result)
    data = load_data(include_test=True)
    times = np.asarray(data['times']).astype('datetime64[ns]')
    origins_by_period = {period: np.asarray(data[f'{period}_origins'], dtype=np.int64) for period in PERIODS}
    for period, origins in origins_by_period.items():
        if len(origins) != 40 or len(np.unique(origins)) != 40 or np.any(np.diff(origins) != 24):
            raise ValueError(f'{period}: sealed origin count/stride mismatch')
        if origins.tolist() != sealed['origins'][period]:
            raise ValueError(f'{period}: origins changed from the pre-exposure seal')
        if np.any(origins < 512) or np.any(origins + 48 > len(times)):
            raise ValueError(f'{period}: context/target slice falls outside the archive')
        start, end = (np.datetime64(value, 'ns') for value in sealed['splits'][period])
        if (np.any(times[origins] < start) or np.any(times[origins + 47] >= end)
                or np.any(times[origins + 47] - times[origins] != np.timedelta64(47, 'h'))):
            raise ValueError(f'{period}: a full 48-hour target crosses its sealed period bounds')
    if origins_by_period['test_a'][-1] + 48 > origins_by_period['test_b'][0]:
        raise ValueError('Period target intervals overlap')
    origins = np.concatenate(list(origins_by_period.values()))
    period_indices = {'test_a': np.arange(40), 'test_b': np.arange(40, 80)}
    target = np.stack([data['x'][o:o + 48] for o in origins])
    observed = np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool)
    if target.shape != (80, 48, 32):
        raise ValueError('Unexpected sealed TEST shape')
    if data['columns'].astype(str).tolist() != sealed['channels']:
        raise ValueError('TEST channel identities/order differ from the pre-exposure seal')
    prediction_dir = CACHE / f'{args.name}_predictions'
    prediction_dir.mkdir(parents=True, exist_ok=False)
    module = load_module(HERE / 'model_v4.py', '_v4_evaluation_model')
    models = {r['id']: r for r in sealed['models']}
    if not set(args.recompute_ids).issubset(models):
        raise ValueError('Correction requested an unknown model ID')
    terms = {}
    for identifier, row in models.items():
        job.check_limits()
        reused = previous_result is not None and identifier in previous_result['models'] and identifier not in args.recompute_ids
        if reused:
            prediction_receipt = previous_result['models'][identifier]['prediction']
            with np.load(check_receipt(prediction_receipt), allow_pickle=False) as archive:
                if not np.array_equal(archive['origins'], origins):
                    raise ValueError('Stored correction predictions have different TEST origins')
                prediction = archive['prediction'].copy()
        elif row['kind'] == 'shared':
            inputs = np.stack([data['x'][o - 512:o] for o in origins])
            prediction = shared_predict(inputs, row)
            del inputs
        else:
            model = module.restore_model(resolve(row['checkpoint']), device='cuda').eval()
            predictions = []
            for offset in range(0, len(origins), 4):
                job.check_limits()
                inputs = torch.as_tensor(np.stack([data['x'][o - 512:o] for o in origins[offset:offset + 4]]),
                                         dtype=torch.float32, device='cuda')
                predictions.append(model(inputs).float().cpu().numpy())
                del inputs
            prediction = np.concatenate(predictions)
            del predictions, model
            gc.collect()
            torch.cuda.synchronize()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
            if torch.cuda.memory_allocated():
                raise RuntimeError('Evaluation retains a prior model or output on CUDA')
        terms[identifier] = error_terms(prediction, target, observed)
        if not reused:
            prediction_path = prediction_dir / f'{identifier}.npz'
            np.savez_compressed(prediction_path, prediction=prediction, origins=origins)
            prediction_receipt = receipt(prediction_path)
        scores = {period: scores_from_terms(terms[identifier], index) for period, index in period_indices.items()}
        scores['combined'] = scores_from_terms(terms[identifier])
        result['models'][identifier] = {'spec': row['spec'], 'kind': row['kind'], 'prediction': prediction_receipt,
                                         'reused_prediction_after_documented_correction': reused,
                                         'scores': scores}
        del prediction
        save_json(HERE / f'{args.name}_partial.json', result)
        job.heartbeat(f'{args.name}: evaluated {identifier}; {len(result["models"])}/19 models')
    result['role_scores'] = {}
    for role, ids in sealed['roles'].items():
        result['role_scores'][role] = {}
        for period in (*PERIODS, 'combined'):
            scores = [result['models'][i]['scores'][period] for i in ids]
            available = all(s['mse'] is not None for s in scores)
            result['role_scores'][role][period] = {'run_ids': ids, 'count': len(ids),
                'mse': float(np.mean([s['mse'] for s in scores])) if available else None,
                'mae': float(np.mean([s['mae'] for s in scores])) if available else None,
                'scope': 'Mean of individual model channel-macro losses; deterministic fit counted once; not an ensemble.'}
    for comparison in sealed['comparisons']:
        result['comparisons'].append(compare_terms(comparison, terms, models, period_indices, job))
        job.heartbeat(f'{args.name}: paired interval {comparison["name"]}')
    result.update(status='complete', completed_utc=time.time(),
                  combined_definition='Within each seed/channel, sum TEST-A/B errors and observed counts before channel mean; then average individual seed losses.',
                  new_test_model_predictions=sum(not r['reused_prediction_after_documented_correction'] for r in result['models'].values()),
                  test_reselection=False)
    save_json(HERE / f'{args.name}.json', result)
    return {'result': str(HERE / f'{args.name}.json'), 'models': len(models)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--seal', action='store_true')
    operation.add_argument('--evaluate', action='store_true')
    parser.add_argument('--selection', type=Path, default=HERE / 'selected.json')
    parser.add_argument('--shared-selection', type=Path, default=HERE / 'shared_ridge_selection.json')
    parser.add_argument('--data-contract', type=Path, default=HERE / 'data_contract.json')
    parser.add_argument('--name', required=True)
    parser.add_argument('--reserve-seconds', type=float, required=True)
    parser.add_argument('--correction-reason', default='')
    parser.add_argument('--reuse-evaluation', type=Path)
    parser.add_argument('--recompute-ids', nargs='*', default=[])
    args = parser.parse_args()
    validate_name(args.name)
    paths = [HERE / f'{args.name}{suffix}.json' for suffix in ('', '_partial', '_attempt')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Preserve existing evaluation attempts; use a new name')
    attempt = {'status': 'running', 'started_utc': time.time(), 'command': sys.argv,
               'source': source_receipt(), 'environment': environment_receipt()}
    save_json(paths[-1], attempt)
    try:
        with Job(args.name, category='cpu_analysis' if args.seal else 'gpu', reserve_seconds=args.reserve_seconds) as job:
            if args.seal:
                outcome = seal_evaluation(args)
            else:
                global np, torch
                import numpy as np
                import torch
                from threadpoolctl import threadpool_limits
                torch.set_num_threads(4)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                with threadpool_limits(limits=4), torch.inference_mode():
                    outcome = evaluate_sealed(args, job)
            attempt.update(status='complete', outcome=outcome)
            print(json.dumps(outcome), flush=True)
    except BaseException:
        attempt.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        attempt['ended_utc'] = time.time()
        save_json(paths[-1], attempt)


if __name__ == '__main__':
    main()
