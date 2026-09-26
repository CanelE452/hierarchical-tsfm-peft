import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import make_model, pca_basis
from run import CACHE, HERE, GPUJob, digest, evaluate, load_data, save_json, score_arrays, source_receipt


def select_initial():
    specs = json.loads((HERE / 'initial_specs.json').read_text())
    rows = []
    for spec in specs:
        path = HERE / 'runs' / spec['id'] / 'result.json'
        row = json.loads(path.read_text())
        assert row['status'] == 'complete'
        rows.append(row)
    selection = {}
    for arm in sorted({row['arm'] for row in rows}):
        arm_rows = [r for r in rows if r['arm'] == arm]
        lrs = sorted({r['lr'] for r in arm_rows})
        mean_val = {lr: float(np.mean([r['best_val_mse'] for r in arm_rows if r['lr'] == lr])) for lr in lrs}
        best_lr = min(lrs, key=lambda lr: (mean_val[lr], lr))
        selection[arm] = {'lr': best_lr, 'mean_best_validation': mean_val[best_lr], 'lr_scores': mean_val, 'fits': [r['id'] for r in arm_rows if r['lr'] == best_lr]}
    return selection


def paired_block_effect(predictions, references, target, mask, block=7):
    if len(references) not in (1, len(predictions)):
        raise ValueError('Pair the same training seeds, or use one fixed reference')
    def losses(pred):
        return (pred.astype(np.float64) - target) ** 2 * mask
    model_terms = np.mean([losses(p) for p in predictions], axis=0)
    ref_terms = np.mean([losses(p) for p in references], axis=0)
    counts = mask.astype(np.float64)
    def gain(index):
        denom = counts[index].sum(axis=(0, 1))
        if not np.all(denom > 0):
            raise ValueError('A time-block draw has no targets for a selected channel; CI unavailable')
        m = np.mean(model_terms[index].sum(axis=(0, 1)) / denom)
        r = np.mean(ref_terms[index].sum(axis=(0, 1)) / denom)
        if r <= 0:
            raise ValueError('Relative effect undefined for a zero-loss reference')
        return float(100 * (r - m) / r)
    n = len(target)
    if not 1 <= block <= n:
        raise ValueError('Block length must fit the evaluation period')
    rng = np.random.default_rng(9262026)
    draws = []
    for _ in range(2000):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        index = np.concatenate([np.arange(block) + s for s in starts])[:n]
        draws.append(gain(index))
    seed_gains = []
    for i, pred in enumerate(predictions):
        ref = references[0 if len(references) == 1 else i]
        m = score_arrays(pred, target, mask)['mse']
        r = score_arrays(ref, target, mask)['mse']
        seed_gains.append(100 * (r - m) / r)
    return {'gain_pct': gain(np.arange(n)), 'paired_seed_gain_pct': seed_gains, 'period_halves_gain_pct': [gain(np.arange(0, n // 2)), gain(np.arange(n // 2, n))], 'conditional_block95_pct': np.quantile(draws, [0.025, 0.975]).tolist(), 'block_origins': block, 'draws': len(draws), 'scope': 'paired non-circular moving time-block resampling, all channels together; trained seeds fixed, development reuse and selection uncertainty not represented'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='initial')
    parser.add_argument('--selection', help='optional prewritten validation-only selection JSON')
    args = parser.parse_args()
    output = HERE / f'{args.name}_development.json'
    if output.exists():
        raise RuntimeError('Never overwrite completed evaluation')
    selection = json.loads(Path(args.selection).read_text()) if args.selection else select_initial()
    selection_path = HERE / f'{args.name}_selection.json'
    save_json(selection_path, selection)
    data = load_data('electricity')
    local = CACHE / f'{args.name}_evaluation'
    local.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], {}
    with GPUJob(args.name + '_development_evaluation') as job:
        basis = pca_basis(data['x'][:int(data['train_end'])], 8)
        f0 = make_model('f0', basis)
        f0_scores, f0_pred = evaluate(f0, data, data['dev_origins'], job, 'F0 development')
        predictions['f0'] = [f0_pred]
        np.savez_compressed(local / 'f0.npz', prediction=f0_pred, origins=data['dev_origins'])
        rows.append({'arm': 'f0', 'fit': None, 'seed': None, 'scores': f0_scores})
        del f0
        torch.cuda.empty_cache()
        for arm, selected in selection.items():
            predictions[arm] = []
            for fit_id in selected['fits']:
                result = json.loads((HERE / 'runs' / fit_id / 'result.json').read_text())
                saved = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)
                spec = saved['spec']
                model = make_model(spec['arm'], saved['basis'], residual_rank=spec.get('residual_rank', 8))
                model.restore_adapter(saved['state'])
                scores, pred = evaluate(model, data, data['dev_origins'], job, fit_id + ' development')
                predictions[arm].append(pred)
                np.savez_compressed(local / f'{fit_id}.npz', prediction=pred, origins=data['dev_origins'])
                rows.append({'arm': arm, 'fit': fit_id, 'seed': spec['seed'], 'scores': scores, 'prediction_sha256': digest(local / f'{fit_id}.npz')})
                del model
                torch.cuda.empty_cache()
    baselines = json.loads((HERE / 'linear_baselines.json').read_text())
    for arm, record in baselines['selected'].items():
        with np.load(CACHE / 'baselines' / f'{arm}_{record["ridge_penalty"]:g}_dev.npz') as archive:
            predictions[arm] = [archive['prediction']]
        rows.append({'arm': arm, 'fit': None, 'seed': None, 'scores': record['scores']['dev']})
    with np.load(CACHE / 'baselines' / 'seasonal_dev.npz') as archive:
        predictions['seasonal'] = [archive['prediction']]
    rows.append({'arm': 'seasonal', 'fit': None, 'seed': None, 'scores': baselines['seasonal']['dev']})
    target = np.stack([data['x'][o:o + 48] for o in data['dev_origins']])
    mask = np.stack([data['finite'][o:o + 48] for o in data['dev_origins']])
    effects = {ref: paired_block_effect(predictions['residual'], forecasts, target, mask) for ref, forecasts in predictions.items() if ref != 'residual'}
    report = {'status': 'complete', 'scope': 'Electricity reused development; no protected Bull scores', 'source': source_receipt(), 'selection': selection, 'selection_sha256': digest(selection_path), 'data_sha256': digest(data['_path']), 'rows': rows, 'residual_vs': effects}
    save_json(output, report)
    print(json.dumps({'mean_dev_mse': {arm: float(np.mean([r['scores']['mse'] for r in rows if r['arm'] == arm])) for arm in predictions}, 'residual_vs': effects}), flush=True)


if __name__ == '__main__':
    main()
