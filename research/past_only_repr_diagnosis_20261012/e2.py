"""E2: matched multi-horizon direct ridge with self/current-cross/lagged-cross inputs."""
import argparse
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from common import (Budget, CACHE, H, HERE, L, LAMBDAS, Ridge, access_guard, ahash,
                    load_data, masks, physical_score, read, score, sha, windows, write)
from synthetic import CONDITIONS, SYNTHETIC_SEEDS, make_record, noiseless_alignment

MODELS = ('SELF_LAG', 'SELF_LAG_PLUS_CROSS_NOW', 'ALL_CHANNEL_LAG', 'PERSISTENCE')
MAX_LAG = 32


def feature_block(history, model, channel=None):
    lagged = np.asarray(history)[:, -(np.arange(MAX_LAG+1)+1), :]
    if model == 'ALL_CHANNEL_LAG':
        return lagged.reshape(len(history), -1)
    own = lagged[:, :, channel]
    if model == 'SELF_LAG':
        return own
    if model == 'SELF_LAG_PLUS_CROSS_NOW':
        other = [c for c in range(history.shape[-1]) if c != channel]
        return np.concatenate((own, lagged[:, 0, other]), 1)
    raise ValueError(model)


def fit_candidates(history, targets, target_masks, model, output_scale, budget):
    channels = targets['fit'].shape[-1]
    groups = []
    if model == 'ALL_CHANNEL_LAG':
        candidates = [(None, list(range(H*channels)))]
    else:
        candidates = [(c, list(range(c, H*channels, channels))) for c in range(channels)]
    fit_flat = targets['fit'].reshape(len(targets['fit']), -1)
    mf = target_masks['fit'].reshape(len(fit_flat), -1)
    for c, columns in candidates:
        grouped = {}
        for j in columns:
            grouped.setdefault(mf[:, j].tobytes(), []).append(j)
        features = {split: feature_block(h, model, c) for split, h in history.items()}
        groups.extend((c, js, mf[:, js[0]], features) for js in grouped.values())
    rows = []
    predictions = []
    selected_fits = []
    for lam in LAMBDAS:
        out = {split: np.zeros(targets[split].shape, dtype=np.float64) for split in ('tune', 'val')}
        flat_out = {split: p.reshape(len(p), -1) for split, p in out.items()}
        fits = []
        for c, js, valid, features in groups:
            if valid.sum() < 2:
                raise ValueError(f'Fewer than 2 observed FIT values for outputs {js}')
            ridge = Ridge(lam).fit(features['fit'][valid], fit_flat[valid][:, js])
            for split in out:
                flat_out[split][:, js] = ridge.predict(features[split])
            fits.append((c, js, valid, ridge))
        tune = score(out['tune']*output_scale, targets['tune']*output_scale, target_masks['tune'])
        normalized = score(out['tune'], targets['tune'], target_masks['tune'])
        rows.append({'lambda': lam, 'tune_physical_macro_mse': tune['mse'],
                     'tune_physical_macro_mae': tune['mae'], 'tune_normalized_macro_mse': normalized['mse']})
        predictions.append(out)
        selected_fits.append(fits)
        budget.check()
    chosen = min(range(len(rows)), key=lambda i: rows[i]['tune_physical_macro_mse'])
    info = {'selected_lambda': LAMBDAS[chosen], 'selected_grid_index': chosen, 'lambda_grid': rows,
            'selection': 'TUNE physical channel macro MSE; first grid value wins exact ties',
            'fit_groups': len(groups), 'feature_scaler_fit': 'FIT_ONLY_PER_OBSERVED_OUTPUT_GROUP',
            'ridge_objective': 'observed-output mean SSE + lambda ||w||^2; intercept unpenalized',
            'alpha_conversion': 'n_observed * lambda', 'target_space': 'FIT-standardized raw channels',
            'feature_count_per_output': MAX_LAG+1 if model == 'SELF_LAG' else
                (MAX_LAG+channels if model == 'SELF_LAG_PLUS_CROSS_NOW' else (MAX_LAG+1)*channels),
            'nominal_regression_coefficients': H*channels*(MAX_LAG+1 if model == 'SELF_LAG' else
                MAX_LAG+channels if model == 'SELF_LAG_PLUS_CROSS_NOW' else (MAX_LAG+1)*channels),
            'intercepts': H*channels}
    return predictions[chosen], info, predictions, selected_fits[chosen]


def overlap_summary(origins):
    target_times = np.asarray(origins)[:, None]+np.arange(H)
    unique, count = np.unique(target_times, return_counts=True)
    return {'origins': len(origins), 'horizon_rows': int(target_times.size),
            'unique_target_timestamps': len(unique), 'minimum_overlap': int(count.min()),
            'maximum_overlap': int(count.max()), 'overlap_count_histogram': {
                str(int(k)): int(v) for k, v in zip(*np.unique(count, return_counts=True))},
            'interpretation': 'Origins and overlapping horizon rows are not independent sample counts'}


def regime_score(data, pred, indices=None, target_time_condition=None):
    split = 'val'
    target = windows(data, split, True)
    valid = masks(data, split)
    if target_time_condition is not None:
        target_timestamps = data['origins'][split][:, None]+np.arange(H)
        valid = valid & target_time_condition(target_timestamps)[:, :, None]
    if indices is not None:
        pred, target, valid = pred[indices], target[indices], valid[indices]
    result = score(pred*data['std']+data['mean'], target*data['std']+data['mean'], valid)
    result['normalized'] = score(pred, target, valid)
    return result


def save_npz(path, arrays):
    path = Path(path).resolve()
    if not path.is_relative_to(CACHE):
        raise ValueError('Science arrays must remain in new campaign cache')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)
    return {'path': str(path), 'sha256': sha(path), 'array_hashes': {k: ahash(v) for k, v in arrays.items()}}


def run_record(data, tag, budget, synthetic_arrays=None):
    started = time.perf_counter()
    history = {s: windows(data, s) for s in ('fit', 'tune', 'val')}
    targets = {s: windows(data, s, True) for s in history}
    target_masks = {s: masks(data, s) for s in history}
    source_before = data['source_hash']
    outputs, predictions, states = {}, {}, {}
    blocks = np.array_split(np.arange(len(data['origins']['val'])), 4)
    for model in MODELS:
        budget.check()
        if model == 'PERSISTENCE':
            selected = {s: np.repeat(history[s][:, -1:, :], H, 1) for s in ('tune', 'val')}
            info, candidate_predictions, fits = {'selected_lambda': None, 'fitted_coefficients': 0,
                'input': 'last observed causal-filled channel value repeated 48 times'}, [], []
        else:
            selected, info, candidate_predictions, fits = fit_candidates(
                history, targets, target_masks, model, data['std'], budget)
        predictions[model] = selected['val']
        result = {'method': model, 'tune': physical_score(data, selected['tune'], 'tune'),
                  'val': physical_score(data, selected['val'], 'val'), 'fit': info,
                  'val_blocks': [regime_score(data, selected['val'], idx) for idx in blocks]}
        if 'metadata' in data:
            metadata = data['metadata']
            change = metadata['change_timestamp_effective']
            result['receiver_horizon_mse'] = score(
                selected['val'][:, :, 3:]*data['std'][3:]+data['mean'][3:],
                targets['val'][:, :, 3:]*data['std'][3:]+data['mean'][3:],
                target_masks['val'][:, :, 3:])
            result['each_receiver_each_horizon'] = {
                data['columns'][c]: {'mse': [float(np.mean(((selected['val'][:, h, c]-targets['val'][:, h, c])*data['std'][c])**2)) for h in range(H)],
                                    'mae': [float(np.mean(np.abs((selected['val'][:, h, c]-targets['val'][:, h, c])*data['std'][c]))) for h in range(H)]}
                for c in range(3, 6)}
            if change is not None:
                result['before_change'] = regime_score(data, selected['val'], target_time_condition=lambda t: t < change)
                result['after_change'] = regime_score(data, selected['val'], target_time_condition=lambda t: t >= change)
                result['regime_scoring'] = 'Mask each target timestamp; a crossing horizon contributes separately to its actual regime'
        arrays = {'prediction_val': selected['val'], 'prediction_tune': selected['tune']}
        for i, p in enumerate(candidate_predictions):
            arrays[f'lambda_{i}_prediction_val'] = p['val']
            arrays[f'lambda_{i}_prediction_tune'] = p['tune']
        for i, (channel, js, valid, ridge) in enumerate(fits):
            states[f'{model}_{i}_outputs'] = np.asarray(js, dtype=np.int64)
            states[f'{model}_{i}_valid_fit_rows'] = valid
            states[f'{model}_{i}_feature_mean'] = ridge.mean
            states[f'{model}_{i}_feature_std'] = ridge.std
            states[f'{model}_{i}_target_mean'] = ridge.ym
            states[f'{model}_{i}_train_standardized'] = ridge.train
            states[f'{model}_{i}_coefficient' if ridge.coef is not None else f'{model}_{i}_dual'] = ridge.coef if ridge.coef is not None else ridge.dual
        result['arrays'] = save_npz(CACHE/'e2'/tag/f'{model}.npz', arrays)
        outputs[model] = result
        print(f'E2 {tag} {model} MSE={result["val"]["mse"]:.9g} MAE={result["val"]["mae"]:.9g}', flush=True)
    self_result, all_result = outputs['SELF_LAG'], outputs['ALL_CHANNEL_LAG']
    direction = [a['mse'] < s['mse'] for a, s in zip(all_result['val_blocks'], self_result['val_blocks'])]
    lag_signal = all_result['val']['mse'] < self_result['val']['mse'] and sum(direction) >= 3
    gate = 'LAG_SIGNAL_MSE_ONLY' if lag_signal and all_result['val']['mae'] >= self_result['val']['mae'] else 'LAG_SIGNAL' if lag_signal else 'NO_LINEAR_32LAG_SIGNAL'
    paired = {}
    for model, pred in predictions.items():
        mask = target_masks['val']
        e = (pred-targets['val'])*data['std']
        e_self = (predictions['SELF_LAG']-targets['val'])*data['std']
        counts = mask.sum(1)
        for metric, values in (('mse', e**2-e_self**2), ('mae', np.abs(e)-np.abs(e_self))):
            per_channel = np.divide(np.where(mask, values, 0).sum(1), counts,
                                    out=np.zeros(counts.shape, dtype=float), where=counts > 0)
            paired[f'{model}_origin_paired_{metric}_difference_from_SELF'] = per_channel.sum(1)/(counts > 0).sum(1)
    shared = {'mean': data['mean'], 'std': data['std'], **paired}
    for s in history:
        shared[f'{s}_origins'] = data['origins'][s]
        shared[f'{s}_target'] = targets[s]
        shared[f'{s}_mask'] = target_masks[s]
        shared[f'{s}_target_timestamps'] = data['origins'][s][:, None]+np.arange(H)
        shared[f'{s}_actual_target_times'] = data['times'][shared[f'{s}_target_timestamps']]
    if synthetic_arrays is not None:
        shared.update(synthetic_arrays)
    shared_receipt = save_npz(CACHE/'e2'/tag/'shared_targets_and_ridge_states.npz', {**shared, **states})
    result = {'status': 'COMPLETE', 'tag': tag, 'dataset': data['dataset'], 'models': outputs,
              'LAG_SIGNAL': gate, 'all_channel_lower_mse_in_blocks': direction,
              'block_origin_bounds': [[int(data['origins']['val'][idx[0]]), int(data['origins']['val'][idx[-1]])] for idx in blocks],
              'source': data['source'], 'source_hash': source_before, 'source_hash_after': sha(data['source']) if Path(data['source']).is_file() else source_before,
              'normalization_fit_scope': data['normalization_fit_scope'], 'L': L, 'H': H,
              'lags': list(range(MAX_LAG+1)), 'lag_reference': 'last observed o-1; feature lag 0 is X[o-1]',
              'target_reference': 'first future is X[o], last future is X[o+47]',
              'channels': data['columns'], 'interval_seconds': data['interval_seconds'],
              'bounds': {'fit': data['fit_bounds'], 'tune': data['tune_bounds'], 'val': data['val_bounds']},
              'split_samples': {s: overlap_summary(data['origins'][s]) for s in history},
              'origin_hashes': {s: ahash(data['origins'][s]) for s in history},
              'target_hashes': {s: ahash(targets[s]) for s in history},
              'mask_hashes': {s: ahash(target_masks[s]) for s in history},
              'shared_arrays': shared_receipt, 'synthetic': data.get('metadata'),
              'cpu_wall_seconds': time.perf_counter()-started, 'optimizer_steps': 0,
              'scope': 'NEW_FIT; full legal development VAL; repeated exposure, not independent confirmation',
              'scale_note': 'Physical per-channel MSE/MAE and within-dataset macro; units differ by channel; normalized companion retained; no cross-dataset pooling',
              'gate_note': 'Prespecified computation branch, not significance/effect certification',
              'source_pins': {'e2.py': sha(HERE/'e2.py'), 'synthetic.py': sha(HERE/'synthetic.py'), 'common.py': sha(HERE/'common.py')}}
    if result['source_hash_after'] != source_before:
        raise ValueError('TRAIN/VAL source bytes changed during execution')
    write(HERE/'e2'/f'{tag}.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--datasets', nargs='*', choices=('jena', 'robin'), default=['jena', 'robin'])
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--e1-gate', required=True)
    args = parser.parse_args()
    if not args.tag.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Tag must be a filename-safe identifier')
    access_guard()
    gate_path = Path(args.e1_gate)
    gate = read(gate_path)
    if gate.get('status') not in ('PASS', 'VERIFIED') and not (
            gate.get('status') == 'COMPLETE' and gate.get('e2_allowed') is True):
        raise ValueError('E1 gate must pass, or explicitly permit E2 despite a path-specific held result')
    manifest = {'purpose': 'Does other-channel history add prediction value beyond self history?',
                'simple_alternative': 'SELF_LAG and persistence', 'varying_factor': 'available observed lag/current channels',
                'fixed': 'origin/target/mask/preprocessing/48-horizon/5lambda TUNE selection per comparison',
                'improvement': 'LAG_SIGNAL computed only by prespecified macro MSE and 3/4 blocks',
                'no_difference_or_worse': 'No linear evidence in lag0..32 range; not all nonlinear relationships absent',
                'stop': 'TEST access/split-mask mismatch/NaN/resource cap/invalid noiseless alignment',
                'e1_gate': str(gate_path), 'e1_gate_sha256': sha(gate_path),
                'datasets': args.datasets, 'synthetic': args.synthetic, 'synthetic_seeds': SYNTHETIC_SEEDS,
                'CPU_threads': 2, 'gpu': False, 'science_started_unix': time.time()}
    write(HERE/'e2'/f'MANIFEST_{args.tag}.json', manifest)
    with threadpool_limits(limits=2), Budget(phase=1, gpu=False, label=f'E2_{args.tag}') as budget:
        alignment = noiseless_alignment()
        write(HERE/'e2'/f'SYNTHETIC_UNIT_{args.tag}.json', alignment)
        results = []
        for dataset in args.datasets:
            results.append(run_record(load_data(dataset, frozen=False), f'{args.tag}_{dataset}', budget))
        if args.synthetic:
            for condition in CONDITIONS:
                for seed in SYNTHETIC_SEEDS:
                    data, arrays = make_record(condition, seed)
                    results.append(run_record(data, f'{args.tag}_{condition}_{seed}', budget, arrays))
        write(HERE/'e2'/f'SUMMARY_{args.tag}.json', {'status': 'COMPLETE', 'manifest': manifest,
              'records': [{'tag': r['tag'], 'path': str(HERE/'e2'/f'{r["tag"]}.json'), 'sha256': sha(HERE/'e2'/f'{r["tag"]}.json'),
                           'LAG_SIGNAL': r['LAG_SIGNAL']} for r in results], 'synthetic_alignment': alignment['status'],
              'budget_receipt': str(budget.path)})


if __name__ == '__main__':
    main()
