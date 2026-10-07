"""Sealed S1 TEST forecasts, canonical scoring, and paired temporal intervals."""
import csv
import gc
import json
import os
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from model_readout import ARMS, Readout, load_backbone
from prepare_readout import receipt, require_s1, seal_sources, verify_receipt
from runtime_readout import (CACHE, DATASETS, HERE, ROOT, SEEDS, array_hash, batches,
                             configure, digest, inputs, load_data, read_json,
                             resolve, save_json)


PERIODS = ('test_a', 'test_b', 'combined')
PAIRS = (('OUTPUT_AFFINE', 'F0'), ('NATIVE_HEAD', 'F0'),
         ('NATIVE_HEAD', 'OUTPUT_AFFINE'), ('OUTPUT_AFFINE', 'MSE_LORA'),
         ('NATIVE_HEAD', 'MSE_LORA'), ('OUTPUT_AFFINE', 'LEVEL'), ('NATIVE_HEAD', 'LEVEL'))
COORDINATES = 'original TRAIN-standardized channels after native inverse normalization'
_verified_json = {}


def _stop(message):
    raise RuntimeError('STOP_DEBUG: ' + message)


def _json_receipt(entry):
    path = resolve(entry['path'])
    stamp = (entry['sha256'], path.stat().st_size, path.stat().st_mtime_ns)
    old = _verified_json.get(str(path))
    if old is None or old[0] != stamp:
        verify_receipt(entry)
        _verified_json[str(path)] = (stamp, read_json(path))
    return _verified_json[str(path)][1]


def load_selection(budget):
    """Validate the joint pre-TEST seal and return its twelve selected runs."""
    require_s1(budget)
    seal_sources()
    seal = read_json(HERE / 'selection_seal.json')
    if (not seal.get('all_trainval_selection_complete_before_test')
            or not seal.get('global_choice_sealed_before_test')
            or seal.get('selected_models') != 12 or seal.get('test_predictions_at_sealing') != 0):
        _stop('The joint twelve-model TRAIN/VAL choice seal is incomplete')
    for entry in seal['artifacts']:
        verify_receipt(entry)
    selected = read_json(HERE / 'selection.json')
    if selected.get('fits_considered') != 24 or selected.get('selected_models') != 12:
        _stop('Selection must consider all twenty-four fixed fits')
    choices = seal.get('choices')
    if choices is None or set(choices) != set(DATASETS):
        _stop('Joint seal must contain all three dataset choices')
    runs, fit_ids = [], set()
    for dataset in DATASETS:
        if set(choices[dataset]) != set(ARMS):
            _stop('Joint seal must contain both arms: ' + dataset)
        for arm in ARMS:
            choice = choices[dataset][arm]
            original = selected['choices'][dataset][arm]
            lr = choice['lr']
            if lr != original['lr'] or choice != original:
                _stop('Seal choices disagree with selection.json')
            candidates = original['candidates']
            if [row['learning_rate'] for row in candidates] != [1e-3, 1e-4]:
                _stop('Fixed LR candidate ordering changed')
            means = []
            for candidate in candidates:
                members = candidate['runs']
                if [row['seed'] for row in members] != list(SEEDS):
                    _stop('Both fixed seeds are required for every LR')
                for run in members:
                    result = _json_receipt(run['result'])
                    spec = result['spec']
                    if (result.get('status') != 'complete' or not result.get('frozen_unchanged')
                            or spec['dataset'] != dataset or spec['arm'] != arm
                            or spec['seed'] != run['seed'] or spec['learning_rate'] != candidate['learning_rate']
                            or result['checkpoint'] != run['checkpoint']
                            or result['best_val_mse'] != run['best_val_mse']
                            or result['selected_epoch'] != run['selected_epoch'] or run['id'] in fit_ids):
                        _stop('Incomplete or inconsistent fixed fit: ' + run['id'])
                    verify_receipt(run['checkpoint'])
                    fit_ids.add(run['id'])
                mean = float(np.mean([row['best_val_mse'] for row in members]))
                if not np.isfinite(mean) or mean != candidate['mean_seed_val_mse']:
                    _stop('Mean seed VAL loss changed')
                means.append(mean)
            winner = min(range(2), key=lambda index: means[index])
            if lr != candidates[winner]['learning_rate'] or original['mean_seed_val_mse'] != means[winner]:
                _stop('The sealed LR differs from the fixed VAL rule')
            if [row['seed'] for row in choice['runs']] != list(SEEDS):
                _stop('Selected seeds changed')
            for run in choice['runs']:
                runs.append(dict(run, dataset=dataset, arm=arm, lr=lr))
    if len(fit_ids) != 24 or len(runs) != 12:
        _stop('Expected twenty-four fits and twelve selected models')
    budget.check()
    return seal, runs


def error_terms(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape or prediction.ndim != 3:
        _stop('Prediction/target/mask must share [origin,48,original_channel] axes')
    if prediction.shape[1] != 48 or mask.dtype != np.bool_:
        _stop('Horizon or observed-mask dtype changed')
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        _stop('Nonfinite forecast or observed target')
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    return {'squared': np.square(error).sum(axis=1), 'absolute': np.abs(error).sum(axis=1),
            'count': mask.sum(axis=1, dtype=np.int64)}


def aggregate_terms(terms, index):
    count = terms['count'][index].sum(axis=0)
    if np.any(count <= 0):
        _stop('A fixed evaluation channel is unobserved; channels cannot be dropped')
    squared = terms['squared'][index].sum(axis=0)
    absolute = terms['absolute'][index].sum(axis=0)
    return {'mse': float(np.mean(squared / count)), 'mae': float(np.mean(absolute / count)),
            'channel_mse': (squared / count).tolist(), 'channel_mae': (absolute / count).tolist(),
            'squared_error_sums': squared.tolist(), 'absolute_error_sums': absolute.tolist(),
            'target_counts': count.tolist(), 'target_count': int(count.sum()), 'origins': len(index)}


def mean_seed_scores(members):
    first = members[0]
    if any(row['target_counts'] != first['target_counts'] or row['origins'] != first['origins'] for row in members):
        _stop('Seed scores must use identical observed targets')
    result = dict(first)
    for key in ('mse', 'mae', 'channel_mse', 'channel_mae', 'squared_error_sums', 'absolute_error_sums'):
        if key in first:
            value = np.mean([row[key] for row in members], axis=0)
            result[key] = value.tolist() if value.ndim else float(value)
    result['aggregation'] = 'arithmetic mean of individual seed losses; no forecast ensemble'
    return result


def bootstrap_weights(indices, block, salt, draws=2000, seed=9262026):
    """Literal v11/v12 noncircular block RNG order, including final truncation."""
    total = len(indices['combined'])
    weights = {}
    for period_index, period in enumerate(PERIODS[:2]):
        index, n = indices[period], len(indices[period])
        if n < block:
            _stop('A TEST period is shorter than its prespecified block')
        rng = np.random.default_rng(np.random.SeedSequence([seed, salt, period_index]))
        matrix = np.zeros((draws, total), dtype=np.int32)
        for draw in range(draws):
            starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
            local = np.concatenate([np.arange(start, start + block) for start in starts])[:n]
            matrix[draw] = np.bincount(index[local], minlength=total)
        weights[period] = matrix
    weights['combined'] = weights['test_a'] + weights['test_b']
    return weights


def paired_comparisons(dataset, method_terms, indices, protocol, budget):
    config = protocol['bootstrap']
    weights = bootstrap_weights(indices, config['blocks'][dataset], config['dataset_salts'][dataset],
                                config['draws'], config['seed'])
    expected = method_terms['F0'][str(SEEDS[0])]['count']
    for members in method_terms.values():
        if set(members) != {str(seed) for seed in SEEDS}:
            _stop('Paired methods must preserve the two seed identities')
        if any(not np.array_equal(row['count'], expected) for row in members.values()):
            _stop('Paired methods/seeds have different origin-channel masks')
    summaries, draws = {}, {}
    for period, matrix in weights.items():
        denominator = matrix @ expected
        if np.any(denominator <= 0):
            _stop('A bootstrap draw has an unobserved fixed channel')
        draws[period], summaries[period] = {}, {}
        for method, members in method_terms.items():
            budget.check()
            summaries[period][method] = mean_seed_scores([aggregate_terms(members[str(seed)], indices[period]) for seed in SEEDS])
            draws[period][method] = {}
            for metric, term in (('mse', 'squared'), ('mae', 'absolute')):
                seed_losses = [np.mean((matrix @ members[str(seed)][term]) / denominator, axis=1) for seed in SEEDS]
                draws[period][method][metric] = np.mean(seed_losses, axis=0)
    comparisons = {}
    for left, right in PAIRS:
        item = {'candidate': left, 'reference': right, 'direction': 'candidate minus reference; negative favors candidate', 'periods': {}}
        for period in PERIODS:
            item['periods'][period] = {}
            for metric in ('mse', 'mae'):
                candidate, reference = summaries[period][left][metric], summaries[period][right][metric]
                a, b = draws[period][left][metric], draws[period][right][metric]
                difference = candidate - reference
                item['periods'][period][metric] = {'candidate': candidate, 'reference': reference,
                    'absolute_difference': difference,
                    'relative_change_percent': 100. * difference / reference if reference > 0 else None,
                    'conditional_block95_absolute': np.percentile(a - b, [2.5, 97.5]).tolist(),
                    'conditional_block95_relative_percent': np.percentile(100. * (a - b) / b, [2.5, 97.5]).tolist() if np.all(b > 0) else None}
        comparisons[left + '__vs__' + right] = item
    return comparisons


def _data_for_test(dataset):
    binding = read_json(HERE / 'data_contract.json')['source_contracts'][dataset]
    contract = _json_receipt(binding['source'])
    verify_receipt(binding['test'])
    data = load_data(dataset, include_test=True)
    if digest(data['_path']) != binding['test']['sha256']:
        _stop('TEST loader returned different canonical data')
    x, observed = data['x'], data['finite']
    if x.ndim != 2 or x.shape[1] != binding['c'] or x.dtype != np.float32 or observed.shape != x.shape or observed.dtype != np.bool_:
        _stop('Canonical standardized input/mask shape or dtype changed')
    if not np.isfinite(x).all() or data['columns'].tolist() != binding['columns']:
        _stop('Canonical imputed contexts or channel ordering changed')
    origins_by_period = {}
    for period in PERIODS[:2]:
        value = data[period + '_origins']
        literal = np.asarray(contract['origins'][period]['values'], dtype=np.int64)
        if value.dtype != np.int64 or value.ndim != 1 or not np.array_equal(value, literal) or array_hash(value) != contract['origins'][period]['values_sha256']:
            _stop('Literal TEST origins changed: ' + dataset + '/' + period)
        if np.any(value < 512) or np.any(value + 48 > len(x)) or np.any(np.diff(value) <= 0):
            _stop('TEST origins exceed canonical context/target bounds')
        origins_by_period[period] = value
    origins = np.concatenate([origins_by_period[period] for period in PERIODS[:2]])
    if len(np.unique(origins)) != len(origins):
        _stop('TEST periods overlap')
    n_a = len(origins_by_period['test_a'])
    indices = {'test_a': np.arange(n_a), 'test_b': np.arange(n_a, len(origins)), 'combined': np.arange(len(origins))}
    target = np.stack([x[int(origin):int(origin) + 48] for origin in origins])
    mask = np.stack([observed[int(origin):int(origin) + 48] for origin in origins])
    provenance = {'canonical_test': binding['test'], 'source_contract': binding['source'],
                  'columns': binding['columns'], 'coordinates': COORDINATES,
                  'origins_sha256': array_hash(origins), 'target_sha256': array_hash(target),
                  'target_mask_sha256': array_hash(mask),
                  'model_inputs': 'only canonical x[origin-512:origin]; no mask, target, basis, or future statistics',
                  'normalization': binding['normalization'], 'new_standardization': False}
    return data, origins, target, mask, indices, provenance


def read_prediction(entry, origins, shape, key='prediction', origins_key='origins'):
    path = verify_receipt(entry)
    with np.load(path, allow_pickle=False) as archive:
        saved_origins, value = archive[origins_key], archive[key]
        if saved_origins.dtype != np.int64 or not np.array_equal(saved_origins, origins):
            _stop('Stored forecast origin values/order differ: ' + str(path))
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            _stop('Stored forecast shape/dtype/finite contract changed: ' + str(path))
    return value


def _predict(run, data, origins, provenance, selection_receipt, manifest, budget):
    path = CACHE / 'evaluation' / run['dataset'] / run['arm'] / ('seed' + str(run['seed'])) / 'prediction.npz'
    previous = manifest['models'].get(run['id'])
    if previous is not None:
        if (previous['checkpoint'] != run['checkpoint'] or previous['selection_seal'] != selection_receipt
                or previous['scoring_provenance'] != provenance):
            _stop('Existing TEST receipt belongs to a different choice or canonical data')
        return previous, read_prediction(previous['prediction'], origins, (len(origins), 48, data['_C']))
    if path.exists():
        _stop('Unreceipted TEST forecast requires inspection: ' + str(path))
    budget.check()
    saved = torch.load(verify_receipt(run['checkpoint']), map_location='cpu', weights_only=True)
    spec = saved['spec']
    result = _json_receipt(run['result'])
    if (saved.get('schema') != 'tsfm_readout_s1_selected_v1' or spec != result['spec']
            or saved['binding'] != result['binding'] or saved['epoch'] != run['selected_epoch']
            or saved['val_mse'] != run['best_val_mse'] or saved['frozen_hash'] != result['frozen_parameters_and_all_buffers_hash']):
        _stop('Selected checkpoint specification/state binding changed')
    backbone = load_backbone(run['dataset'], device='cuda')
    model = Readout(backbone, data['_C'], run['arm']).eval()
    started = time.perf_counter()
    try:
        if model.frozen_hash() != saved['frozen_hash']:
            _stop('Restored frozen parent parameters/buffers differ from selected fit')
        model.load_trainable_state(saved['state'])
        budget.tick('test')
        values = []
        with torch.no_grad():
            for batch_origins in batches(origins, 4):
                budget.tick('prefix')
                budget.tick('head', 2 if run['arm'] == 'OUTPUT_AFFINE' else 1)
                prediction = model(inputs(data, batch_origins).to('cuda')).detach().cpu().numpy()
                if prediction.shape != (len(batch_origins), 48, data['_C']) or prediction.dtype != np.float32 or not np.isfinite(prediction).all():
                    _stop('New TEST forecast violates its canonical shape/dtype')
                values.append(prediction)
                budget.check()
        prediction = np.concatenate(values)
        if model.frozen_hash() != saved['frozen_hash']:
            _stop('TEST inference changed frozen parameters or buffers')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
        with temporary.open('xb') as handle:
            np.savez(handle, prediction=prediction, origins=origins)
        os.replace(temporary, path)
        record = {'id': run['id'], 'dataset': run['dataset'], 'method': run['arm'], 'seed': run['seed'],
                  'learning_rate': run['lr'], 'selected_epoch': run['selected_epoch'], 'checkpoint': run['checkpoint'],
                  'spec': spec, 'selection_seal': selection_receipt, 'prediction': receipt(path),
                  'prediction_array_sha256': array_hash(prediction), 'origins_sha256': array_hash(origins),
                  'scoring_provenance': provenance, 'prediction_seconds': time.perf_counter() - started,
                  'new_test_model_instances': 1, 'frozen_parameters_and_all_buffers_unchanged': True}
        manifest['models'][run['id']] = record
        save_json(HERE / 'prediction_manifest.json', manifest)
        return record, prediction
    finally:
        del model, backbone, saved
        gc.collect()
        torch.cuda.empty_cache()
        budget.check()


def _pointer(value, pointer):
    for key in pointer.strip('/').split('/'):
        value = value[key.replace('~1', '/').replace('~0', '~')]
    return value


def _canonical_score(score, origin_count):
    result = dict(score)
    if 'target_counts' not in result:
        result['target_counts'] = result['channel_target_count']
    result['origins'] = result.get('origins', origin_count)
    for current, historical in (('squared_error_sums', 'channel_squared_error_sum'),
                                ('absolute_error_sums', 'channel_absolute_error_sum')):
        if current not in result and historical in result:
            result[current] = result[historical]
    return result


def _source_scores(row, dataset, provenance, indices, counts):
    evaluation = _json_receipt(row['source_evaluation'])
    unit = evaluation.get('datasets', {}).get(dataset, evaluation)
    if 'data_arrays' in unit:
        if unit['data_arrays']['sha256'] != provenance['canonical_test']['sha256'] or unit['columns'] != provenance['columns']:
            _stop('Historical scores use a different canonical data/channel contract')
    else:
        found = False
        for path, sha in evaluation.get('source', {}).items():
            if 'data_contract' not in path:
                continue
            source_path = resolve(path)
            if digest(source_path) != sha:
                _stop('Historical data contract source changed')
            contract = read_json(source_path)
            if contract.get('test', {}).get('sha256') == provenance['canonical_test']['sha256']:
                columns = contract.get('dataset_info', {}).get('columns') or contract.get('dataset', {}).get('selected_columns')
                found = columns == provenance['columns']
        if not found:
            _stop('Historical scores lack the exact canonical TEST binding')
    scores = {period: _canonical_score(score, len(indices[period]))
              for period, score in _pointer(evaluation, row['source_pointer']).items() if period in PERIODS}
    for period in PERIODS:
        score = scores[period]
        if score['target_counts'] != counts[period] or score['origins'] != len(indices[period]):
            _stop('Historical observed target counts/origins differ')
        if any(score[metric] != row['expected_period_scores'][period][metric] for metric in ('mse', 'mae')):
            _stop('Historical score receipt differs from the reuse manifest')
    return scores, evaluation


def _baseline_terms(dataset, origins, target, mask, indices, provenance, budget):
    scaling = read_json(ROOT / 'research/level_bolt_backbone_scaling_v1_20261006/reuse_manifest.json')
    manifest = _json_receipt(scaling['original_manifest'])
    terms, receipts, native = {}, {}, []
    counts = {period: mask[ids].sum(axis=(0, 1), dtype=np.int64).tolist() for period, ids in indices.items()}
    for method in ('F0', 'MSE_LORA', 'LEVEL'):
        members = [row for row in manifest['units'][dataset]['rows'] if row['method'] == method]
        if (method == 'F0' and len(members) != 1) or (method != 'F0' and sorted(row['seed'] for row in members) != list(SEEDS)):
            _stop('Exact paired reference seed set changed: ' + method)
        terms[method], receipts[method] = {}, []
        for row in members:
            budget.check()
            if row.get('checkpoint') is not None:
                verify_receipt(row['checkpoint'])
            _, evaluation = _source_scores(row, dataset, provenance, indices, counts)
            entry = row['test_prediction']
            if entry['axes'] != ['origin', 'horizon', 'original_channel']:
                _stop('Paired baseline forecast axes changed')
            value = read_prediction(entry, origins, target.shape, entry['key'], entry['origins_key'])
            term = error_terms(value, target, mask)
            for seed in SEEDS if method == 'F0' else (row['seed'],):
                terms[method][str(seed)] = term
            receipts[method].append({'id': row['id'], 'seed': row['seed'], 'checkpoint': row['checkpoint'],
                                     'prediction': entry, 'source_evaluation': row['source_evaluation'],
                                     'canonical_test': provenance['canonical_test'], 'target_mask_sha256': provenance['target_mask_sha256'],
                                     'coordinates': COORDINATES, 'origins_sha256': array_hash(origins)})
            if method == 'MSE_LORA' and not native:
                unit = evaluation['datasets'][dataset]
                family = 'native' if dataset == 'peacock_education' else 'lora_native'
                native_models = [model for model in unit['models'].values() if model['family'] == family]
                if sorted(model['seed'] for model in native_models) != list(SEEDS):
                    _stop('Historical native-quantile LoRA seeds changed')
                for model in native_models:
                    checkpoint = model.get('checkpoint')
                    if checkpoint is None:
                        checkpoint = receipt(model['sources']['native_checkpoint'])
                        if checkpoint['sha256'] != model['sources']['native_checkpoint_sha256']:
                            _stop('Historical native LoRA checkpoint changed')
                    verify_receipt(checkpoint)
                    verify_receipt(model['prediction'])
                    for period in PERIODS:
                        score = model['periods'][period]
                        if score['target_counts'] != counts[period] or score['origins'] != len(indices[period]):
                            _stop('Native-quantile reference uses different targets')
                    native.append({'id': model['id'], 'seed': model['seed'], 'periods': model['periods'],
                                   'checkpoint': checkpoint, 'prediction': model['prediction'],
                                   'source_evaluation': row['source_evaluation'],
                                   'scope': 'historical native quantile loss; distinct from point MSE adaptation'})
    return terms, receipts, native


def _accuracy_row(dataset, method, run_id, seed, aggregation, period, score, reused, source, scope):
    return {'dataset': dataset, 'method': method, 'id': run_id, 'seed': seed,
            'aggregation': aggregation, 'period': period, 'mse': score['mse'], 'mae': score['mae'],
            'target_count': sum(score['target_counts']), 'origins': score['origins'],
            'target_counts': json.dumps(score['target_counts']), 'reused': reused,
            'source': source, 'evidence_scope': scope}


def _stored_accuracy(dataset, indices, mask, native):
    reuse = read_json(HERE / 'reuse_manifest.json')
    accuracy, rows = {}, []
    scope = 'historical stored result on exposed development targets; no new fit or reselection'
    counts = {period: mask[ids].sum(axis=(0, 1), dtype=np.int64).tolist() for period, ids in indices.items()}
    for source in reuse['stored_score_receipts']:
        path = verify_receipt(source)
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['dataset'] != dataset or ('original_method' in row and row['method'] not in ('BOLT_MINI_F0', 'BOLT_TINY_F0')):
                    continue
                period = row['period']
                score = {'mse': float(row['mse']), 'mae': float(row['mae']),
                         'target_counts': json.loads(row['target_counts']), 'origins': int(row['origins'])}
                if (period not in indices or score['target_counts'] != counts[period]
                        or score['origins'] != len(indices[period]) or not np.isfinite([score['mse'], score['mae']]).all()):
                    _stop('Stored baseline accuracy target/score contract changed')
                aggregate = row['aggregation'] != 'individual'
                if aggregate:
                    accuracy.setdefault(row['method'], {})[period] = score | {'aggregation': row['aggregation'], 'source': source}
                rows.append(_accuracy_row(dataset, row['method'], row['id'], row['seed'],
                                          'mean_seed' if aggregate else 'individual', period, score, True, source['path'], scope))
    for period in PERIODS:
        for member in native:
            rows.append(_accuracy_row(dataset, 'NATIVE_LORA_QUANTILE', member['id'], member['seed'],
                        'individual', period, member['periods'][period], True, member['source_evaluation']['path'], member['scope']))
        mean = mean_seed_scores([member['periods'][period] for member in native])
        accuracy.setdefault('NATIVE_LORA_QUANTILE', {})[period] = mean
        rows.append(_accuracy_row(dataset, 'NATIVE_LORA_QUANTILE', ';'.join(member['id'] for member in native), '',
                                  'mean_seed', period, mean, True, native[0]['source_evaluation']['path'], native[0]['scope']))
    for anchor in reuse['stored_score_anchors']:
        if anchor['dataset'] == dataset and anchor['method'] in accuracy:
            score = accuracy[anchor['method']]['combined']
            if score['mse'] != float(anchor['mse']) or score['mae'] != float(anchor['mae']):
                _stop('Stored combined score differs from its audited anchor')
    return accuracy, rows


def _write_csv(path, rows):
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def evaluate_all(budget):
    """Run at most twelve selected TEST instances, then jointly report all datasets."""
    try:
        return _evaluate_all(budget)
    except Exception as error:
        if 'STOP_RESOURCE' in str(error) or 'STOP_DEBUG' in str(error):
            raise
        raise RuntimeError('STOP_DEBUG: evaluation contract failure: ' + str(error)) from error


def _evaluate_all(budget):
    configure()
    seal, runs = load_selection(budget)
    protocol = read_json(HERE / 'evaluation_protocol.json')
    if (protocol['datasets'] != list(DATASETS) or protocol['methods'] != list(ARMS)
            or protocol['periods'] != list(PERIODS) or protocol['bootstrap']['draws'] != 2000
            or protocol['bootstrap']['seed'] != 9262026
            or protocol['bootstrap']['blocks'] != {'robin': 7, 'peacock_education': 7, 'jena': 42}
            or protocol['bootstrap']['dataset_salts'] != {'robin': 0, 'peacock_education': 0, 'jena': 1}
            or protocol['comparisons'] != [left + ' minus ' + right for left, right in PAIRS]):
        _stop('The fixed evaluation/bootstrap protocol changed')
    selection_receipt = receipt(HERE / 'selection_seal.json')
    binding = {'selection_seal': selection_receipt, 'evaluation_protocol': receipt(HERE / 'evaluation_protocol.json'),
               'evaluator': receipt(HERE / 'evaluate_readout.py'), 'data_contract': receipt(HERE / 'data_contract.json'),
               'reuse_manifest': receipt(HERE / 'reuse_manifest.json')}
    manifest_path = HERE / 'prediction_manifest.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else {
        'schema': 'tsfm_readout_s1_predictions_v1', 'status': 'predicting', 'binding': binding, 'models': {}}
    if manifest['binding'] != binding or not set(manifest['models']).issubset({run['id'] for run in runs}):
        _stop('Existing TEST prediction manifest belongs to different sealed choices')
    exposure_path = HERE / 'exposure.json'
    if exposure_path.exists():
        if read_json(exposure_path)['selection_seal'] != selection_receipt:
            _stop('TEST exposure belongs to a different choice seal')
    else:
        save_json(exposure_path, {'schema': 'tsfm_readout_s1_test_exposure_v1', 'started_utc': time.time(),
                  'selection_seal': selection_receipt, 'all_trainval_selection_complete_before_test': True,
                  'evidence_scope': 'previously exposed development targets; no protected independent confirmation'})
    result = {'schema': 'tsfm_readout_s1_evaluation_v1', 'status': 'scoring', 'binding': binding,
              'accuracy': {}, 'datasets': {}, 'models': {}, 'comparisons': {},
              'primary_metric': 'masked channel-macro MSE', 'secondary_metric': 'masked channel-macro MAE',
              'combined_definition': 'pool A/B channel error sums and observed counts, then divide and average channels',
              'seed_aggregation': 'arithmetic mean of individual seed losses; never averaged forecasts',
              'independent_confirmation': False, 'evidence_scope': 'exposed development targets',
              'bootstrap': protocol['bootstrap'] | {'selection_uncertainty_included': False,
                    'period_boundaries_crossed': False, 'scope': 'paired temporal resampling conditional on selected seeds; no independent-domain claims'}}
    accuracy_rows, comparison_rows = [], []
    with threadpool_limits(limits=4):
        for dataset in DATASETS:
            budget.check()
            data, origins, target, mask, indices, provenance = _data_for_test(dataset)
            method_terms = {arm: {} for arm in ARMS}
            for run in (run for run in runs if run['dataset'] == dataset):
                record, prediction = _predict(run, data, origins, provenance, selection_receipt, manifest, budget)
                terms = error_terms(prediction, target, mask)
                method_terms[run['arm']][str(run['seed'])] = terms
                scores = {period: aggregate_terms(terms, index) for period, index in indices.items()}
                result['models'][run['id']] = record | {'periods': scores}
                for period, score in scores.items():
                    accuracy_rows.append(_accuracy_row(dataset, run['arm'], run['id'], run['seed'], 'individual', period,
                                                      score, False, record['prediction']['path'], 'new S1 selected readout'))
                budget.heartbeat({'phase': 'test_predictions', 'dataset': dataset, 'completed_models': len(manifest['models']), 'selected_models': 12})
            baseline_terms, baseline_receipts, native = _baseline_terms(dataset, origins, target, mask, indices, provenance, budget)
            method_terms.update(baseline_terms)
            accuracy, stored_rows = _stored_accuracy(dataset, indices, mask, native)
            accuracy_rows.extend(stored_rows)
            for arm in ARMS:
                accuracy[arm] = {period: mean_seed_scores([aggregate_terms(method_terms[arm][str(seed)], index) for seed in SEEDS])
                                 for period, index in indices.items()}
                ids = ';'.join(run['id'] for run in runs if run['dataset'] == dataset and run['arm'] == arm)
                for period, score in accuracy[arm].items():
                    accuracy_rows.append(_accuracy_row(dataset, arm, ids, '', 'mean_seed', period, score, False,
                                                      'prediction_manifest.json', 'new S1 selected readout'))
            comparisons = paired_comparisons(dataset, method_terms, indices, protocol, budget)
            result['accuracy'][dataset] = accuracy
            result['comparisons'][dataset] = comparisons
            result['datasets'][dataset] = {'scoring_provenance': provenance, 'paired_reference_receipts': baseline_receipts,
                'native_quantile_reference_receipts': [{key: value for key, value in member.items() if key != 'periods'} for member in native],
                'period_origin_counts': {period: len(index) for period, index in indices.items()}}
            for comparison in comparisons.values():
                for period, metrics in comparison['periods'].items():
                    for metric, values in metrics.items():
                        absolute, relative = values['conditional_block95_absolute'], values['conditional_block95_relative_percent']
                        comparison_rows.append({'dataset': dataset, 'candidate': comparison['candidate'], 'reference': comparison['reference'],
                            'period': period, 'metric': metric, 'candidate_loss': values['candidate'], 'reference_loss': values['reference'],
                            'difference': values['absolute_difference'], 'relative_change_percent': values['relative_change_percent'],
                            'block95_difference_low': absolute[0], 'block95_difference_high': absolute[1],
                            'block95_relative_low': None if relative is None else relative[0],
                            'block95_relative_high': None if relative is None else relative[1],
                            'draws': 2000, 'block_origins': protocol['bootstrap']['blocks'][dataset],
                            'scope': 'paired temporal resampling conditional on selected seed losses'})
            del data, prediction, target, mask, method_terms, baseline_terms
            gc.collect()
            budget.check()
    if len(manifest['models']) != 12:
        _stop('Joint reporting requires all twelve selected TEST predictions')
    manifest['status'] = 'complete'
    save_json(manifest_path, manifest)
    _write_csv(HERE / 'accuracy.csv', accuracy_rows)
    _write_csv(HERE / 'comparisons.csv', comparison_rows)
    result.update(status='complete', prediction_manifest=receipt(manifest_path),
                  tables={'accuracy': receipt(HERE / 'accuracy.csv'), 'comparisons': receipt(HERE / 'comparisons.csv')},
                  selected_test_models=12, completed_utc=time.time())
    save_json(HERE / 'evaluation.json', result)
    budget.check()
    return result
