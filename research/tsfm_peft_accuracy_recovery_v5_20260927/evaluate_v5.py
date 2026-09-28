"""Development evaluation of frozen VAL selections and saved historical controls."""
import argparse
import csv
import gc
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from runtime_v5 import CACHE, HERE, ROOT, V4, Job, digest, load_data, save_json, source_receipt


def load_module(name, path):
    if name in sys.modules:
        assert Path(sys.modules[name].__file__).resolve() == Path(path).resolve()
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.append(str(V4))
METRICS = load_module('_v5_reused_v4_metrics', V4 / 'evaluate_v4.py')
METRICS.np = np
SEEDS = (92601, 92602)


def baseline_entries(dataset):
    c1 = ROOT / '.cache/tsfm_peft_development_20260926'
    c2 = ROOT / '.cache/tsfm_peft_followup_v2_20260926'
    c4 = ROOT / '.cache/tsfm_peft_fresh_group_v4_20260927'
    rows = []
    for role in ('old_res', 'old_raw', 'f0', 'lora'):
        for seed in ((None,) if role == 'f0' else SEEDS):
            if dataset == 'hog':
                suffix = {'old_res': 'tsfm_res_fixed_ed', 'old_raw': 'tsfm_raw_fixed_ed', 'f0': 'f0', 'lora': 'lora'}[role]
                name = f'v4_hog_{suffix}' + (f'_{seed}' if seed else '')
                paths = {'combined': c4 / 'v4_test_01_predictions' / f'{name}.npz'}
            elif dataset == 'bull':
                if role in ('old_res', 'old_raw'):
                    arm = 'residual' if role == 'old_res' else 'raw_bypass'
                    name = f'v2_bull_{arm}_fixed_ed_{seed}'
                    folder = c2 / ('first_modes_evaluation' if role == 'old_res' else 'all_modes_evaluation')
                else:
                    name = 'f0' if role == 'f0' else f'bull_final_lora_{seed}_0.0001'
                    folder = c1 / 'protected_evaluation'
                paths = {period: folder / f'{period}_{name}.npz' for period in ('e1', 'e2')}
            else:
                if role in ('old_res', 'old_raw'):
                    arm = 'residual' if role == 'old_res' else 'raw_bypass'
                    name = f'revision1_rank32_{arm}_{seed}_0.001'
                elif role == 'f0':
                    name = 'f0'
                else:
                    name = f'{"extended120" if seed == 92601 else "initial"}_lora_{seed}_0.0001'
                paths = {'dev': c1 / 'revision1_rank32_evaluation' / f'{name}.npz'}
            rows.append({'id': name, 'role': role, 'spec': {'seed': seed}, 'predictions': paths})
    return rows


def saved_predictions(row, periods, all_origins):
    predictions, origins, receipts = [], [], {}
    manifest = json.loads((HERE / 'reuse_manifest.json').read_text())
    expected_hashes = {}
    for dataset in manifest['datasets'].values():
        for model in dataset['models'].values():
            for item in model.get('evaluation_predictions', {}).values():
                expected_hashes[str((ROOT / item['path']).resolve())] = item['sha256']
    for period, path in row['predictions'].items():
        actual = digest(path)
        wanted = row.get('prediction_sha256') or expected_hashes[str(path.resolve())]
        if actual != wanted:
            raise ValueError(f'Prediction receipt changed: {path}')
        with np.load(path, allow_pickle=False) as archive:
            pred, origin = archive['prediction'], archive['origins']
        expected_origins = all_origins if period == 'combined' else periods[period]
        if not np.array_equal(origin, expected_origins):
            raise ValueError(f'Historical origin identity/order mismatch: {path}')
        predictions.append(pred)
        origins.append(origin)
        receipts[period] = {'path': str(path.relative_to(ROOT)), 'sha256': actual}
    if not np.array_equal(np.concatenate(origins), all_origins):
        raise ValueError('Concatenated historical periods differ')
    return np.concatenate(predictions), receipts


def predict(args):
    import torch
    from model_v5 import restore_model
    from train_v5 import configure, evaluate
    configure()
    target = HERE / f'{args.name}_predictions.json'
    if target.exists():
        raise FileExistsError(target)
    with Job(args.name + '_gpu', 'gpu', reserve_seconds=300) as job, torch.inference_mode():
        data = load_data(args.dataset, evaluation=True)
        periods = {p: data[k] for p, k in data['evaluation_periods'].items()}
        origins = np.concatenate(list(periods.values()))
        entries = []
        for run in args.runs:
            result_path = HERE / 'runs' / run / 'result.json'
            result = json.loads(result_path.read_text())
            assert result['status'] == 'complete' and result['spec']['dataset'] == args.dataset
            assert digest(result['checkpoint']) == result['checkpoint_sha256']
            model = restore_model(result['checkpoint'], 'cuda')
            score, prediction = evaluate(model, data, origins, job, run + ' development')
            path = CACHE / 'evaluations' / args.name / f'{run}.npz'
            if path.exists():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, prediction=prediction, origins=origins)
            entries.append({'id': run, 'role': 'modified_' + result['spec']['variant'], 'spec': result['spec'],
                            'predictions': {'combined': str(path)}, 'result_path': str(result_path),
                            'result_sha256': digest(result_path), 'checkpoint_sha256': result['checkpoint_sha256'],
                            'prediction_sha256': digest(path), 'combined_score': score})
            del model
            gc.collect()
            torch.cuda.empty_cache()
        first = json.loads((HERE / 'runs' / args.runs[0] / 'result.json').read_text())
        if first['spec']['method'] == 'level':
            if args.dataset == 'hog':
                diagnostic = json.loads((HERE / 'v5_hog_diagnosis_01/diagnosis.json').read_text())
                source = diagnostic['arrays']
                assert digest(source['path']) == source['sha256']
                with np.load(source['path'], allow_pickle=False) as archive:
                    assert np.array_equal(archive['origins'], origins)
                    prediction = archive['level_only_prediction']
                reference_source = source
            else:
                assert digest(first['initial_checkpoint']) == first['initial_checkpoint_sha256']
                model = restore_model(first['initial_checkpoint'], 'cuda')
                _, prediction = evaluate(model, data, origins, job, 'LEVEL_ONLY fit0')
                reference_source = {'path': first['initial_checkpoint'], 'sha256': first['initial_checkpoint_sha256']}
                del model
                gc.collect()
                torch.cuda.empty_cache()
            path = CACHE / 'evaluations' / args.name / 'level_only.npz'
            np.savez_compressed(path, prediction=prediction, origins=origins)
            entries.append({'id': f'{args.dataset}_level_only', 'role': 'level_only', 'spec': {'seed': None},
                            'predictions': {'combined': str(path)}, 'prediction_sha256': digest(path),
                            'fit_count': 0, 'reference_source': reference_source})
        save_json(target, {'dataset': args.dataset, 'entries': entries, 'source': source_receipt(),
                           'scope': 'Exposed development, no new independent confirmation'})


def aggregate(args):
    path = HERE / f'{args.name}.json'
    if path.exists():
        raise FileExistsError(path)
    with Job(args.name + '_aggregate', 'cpu_analysis', reserve_seconds=300) as job:
        data = load_data(args.dataset, evaluation=True)
        periods = {p: data[k] for p, k in data['evaluation_periods'].items()}
        origins = np.concatenate(list(periods.values()))
        indices, offset = {}, 0
        for period, values in periods.items():
            indices[period] = np.arange(offset, offset + len(values))
            offset += len(values)
        METRICS.PERIODS = tuple(periods)
        target = np.stack([data['x'][o:o+48] for o in origins])
        mask = np.stack([data['finite'][o:o+48] for o in origins])
        new = json.loads((HERE / f'{args.name}_predictions.json').read_text())
        rows = baseline_entries(args.dataset) + new['entries']
        models, terms, roles = {}, {}, {}
        for row in rows:
            row['predictions'] = {p: Path(f) for p, f in row['predictions'].items()}
            prediction, receipt = saved_predictions(row, periods, origins)
            term = METRICS.error_terms(prediction, target, mask)
            scores = {p: METRICS.scores_from_terms(term, ix) for p, ix in indices.items()}
            scores['combined'] = METRICS.scores_from_terms(term)
            models[row['id']] = {'spec': row['spec'], 'role': row['role'], 'scores': scores, 'predictions': receipt}
            terms[row['id']] = term
            roles.setdefault(row['role'], []).append(row['id'])
        for role in ('modified_res', 'modified_raw'):
            assert sorted(models[i]['spec']['seed'] for i in roles[role]) == list(SEEDS)
        role_scores = {}
        for role, ids in roles.items():
            role_scores[role] = {p: {metric: float(np.mean([models[i]['scores'][p][metric] for i in ids]))
                                      for metric in ('mse', 'mae')}
                                 for p in (*periods, 'combined')}
        comparisons = []
        pairs = [('modified_res', r) for r in ('old_res', 'modified_raw', 'f0', 'lora')]
        pairs += [('modified_raw', 'old_raw')]
        if 'level_only' in roles:
            pairs.append(('modified_res', 'level_only'))
        for left, right in pairs:
            comparison = METRICS.compare_terms({'name': f'{left}_minus_{right}', 'left': roles[left], 'right': roles[right]},
                                               terms, models, indices, job)
            comparison['direction'] = f'{left} minus {right}; negative favors {left}'
            comparisons.append(comparison)
        result = {'status': 'complete', 'dataset': args.dataset, 'evaluation_scope': 'exposed development',
                  'data_sha256': digest(data['_path']), 'columns': data['columns'].tolist(),
                  'period_origins': {p: o.tolist() for p, o in periods.items()}, 'roles': roles,
                  'models': models, 'role_scores': role_scores, 'comparisons': comparisons,
                  'aggregation': 'Pool per-channel observed errors/counts across periods; channel macro, then mean seed losses. No ensemble.',
                  'source': source_receipt()}
        save_json(path, result)
        with (HERE / f'{args.name}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['dataset','role','run','seed','period','mse','mae'])
            for run, model in models.items():
                for period, score in model['scores'].items():
                    writer.writerow([args.dataset,model['role'],run,model['spec']['seed'],period,score['mse'],score['mae']])
        print(json.dumps(role_scores), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['hog','bull','electricity'])
    parser.add_argument('--name', required=True)
    parser.add_argument('--runs', nargs='+')
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    if not args.aggregate_only:
        predict(args)
    aggregate(args)
