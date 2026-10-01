"""Independent saved-array arithmetic checks; no fitting or model execution."""
import argparse
from collections import defaultdict
import csv
import traceback

import numpy as np

from runtime_v15 import (HERE, DATASETS, SEEDS, Job, artifact, budget_snapshot, digest,
                         load_data, protocol, read_json, resolve, save_json)
from evaluate_v15 import NEW, CONTROLS, ROLES, PERIODS, LAMBDAS


def close(actual, expected, name, atol=1e-10, rtol=1e-9):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(name)


def receipts(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            artifact(value['path'], value['sha256'])
        for item in value.values():
            receipts(item)
    elif isinstance(value, list):
        for item in value:
            receipts(item)


def metrics(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape or not np.isfinite(prediction).all():
        raise AssertionError('Original prediction/target/mask dimensions or finiteness')
    if not np.isfinite(target[mask]).all():
        raise AssertionError('Nonfinite observed target')
    count = mask.sum((0, 1))
    if np.any(count <= 0):
        raise AssertionError('An evaluation channel has no observed target')
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    squared, absolute, signed = (value.sum((0, 1)) for value in (error * error, np.abs(error), error))
    return {'mse': np.mean(squared / count), 'mae': np.mean(absolute / count),
            'signed_mean_error': np.mean(signed / count),
            'channel_mse': squared / count, 'channel_mae': absolute / count, 'channel_bias': signed / count,
            'squared_error_sums': squared, 'absolute_error_sums': absolute,
            'signed_error_sums': signed, 'target_counts': count, 'origins': len(prediction)}


def read_prediction(receipt, origins):
    artifact(receipt['path'], receipt['sha256'])
    with np.load(resolve(receipt['path']), allow_pickle=False) as arrays:
        if not np.array_equal(arrays['origins'], origins):
            raise AssertionError('Original time-origin order changed')
        return arrays['prediction'].copy()


def fits_and_seal(job):
    from fit_v15 import fit_specs, result_for
    from evaluate_v15 import selected
    specs = fit_specs()
    choice = selected()
    seal = read_json(HERE / 'selection_seal.json')
    exposure = read_json(HERE / 'test_exposure.json')
    ledger = read_json(HERE / 'ledger.json')['jobs']
    assert len(specs) == 48 and set(choice['selected']) == set(DATASETS)
    assert exposure['seal_sha256'] == digest(HERE / 'selection_seal.json')
    assert seal['created_utc'] <= exposure['first_new_evaluation_utc']
    assert seal['all_fits_selected_before_new_evaluation'] is True
    for name, sha in seal['files'].items():
        artifact(HERE / name, sha)
    results = []
    for spec in specs:
        result = result_for(spec)
        row = ledger[result['ledger_job_id']]
        assert row['status'] == 'complete' and row['category'] == 'coefficient_fit'
        assert row['metadata']['fit_id'] == spec['id'] and row['metadata']['dataset'] == spec['dataset']
        assert row['updates'] == 0 and row['ended_utc'] <= seal['created_utc']
        assert result['test_used'] is False and result['selection_eligible'] is True
        assert result['parent_job_id'] == row['metadata']['parent_job_id']
        parent_job = ledger[result['parent_job_id']]
        assert parent_job['category'] == 'cpu_analysis' and parent_job['status'] == 'complete'
        assert parent_job['started_utc'] <= row['started_utc'] <= row['ended_utc'] <= parent_job['ended_utc']
        assert parent_job['ended_utc'] <= seal['created_utc']
        receipts(result['artifacts'])
        receipt = read_json(resolve(result['artifacts']['receipt']['path']))
        assert receipt['no_test_data_loaded'] is True and receipt['optimizer_updates'] == receipt['model_runs'] == 0
        for name, sha in receipt['source_files'].items():
            artifact(HERE / name, sha)
        receipts(receipt['data'])
        receipts(receipt['selector'])
        receipts(receipt['feature_cache'])
        receipts(receipt['parent_cache'])
        for relative, sha in row['source'].items():
            snapshot = row['source_snapshots'][relative]
            artifact(snapshot, sha)
        results.append(result)
    attempts = [row for row in ledger if row['category'] == 'coefficient_fit']
    assert 48 <= len(attempts) <= 52
    assert sum(bool(row['metadata'].get('technical_retry')) for row in attempts) <= 4
    assert all(row['status'] in ('complete', 'failed') and row['updates'] == 0
               and row['ended_utc'] <= seal['created_utc'] for row in attempts)
    assert len({row['ledger_job_id'] for row in results}) == 48
    for row in attempts:
        if row['status'] == 'failed':
            replacements = [r for r in attempts if r['metadata']['fit_id'] == row['metadata']['fit_id']
                            and r['status'] == 'complete' and r['metadata'].get('technical_retry')]
            assert len(replacements) == 1 and row['ended_utc'] <= replacements[0]['started_utc']
    job.heartbeat('All coefficient attempts, fixed selection and source snapshots verified')
    objective = verify_train_objectives(results, job)
    return {'canonical_completed_fits': 48, 'actual_attempts': len(attempts),
            'technical_retries': sum(bool(row['metadata'].get('technical_retry')) for row in attempts),
            'optimizer_updates': 0, 'all_attempts_before_joint_seal': True,
            'selection_rule': 'Common mean two-seed VAL; exact ties parent, lambda0, lambda0.001, lambda0.1',
            'objective_verification': objective, 'independent_reproduction': False}


def verify_train_objectives(results, job):
    cache = read_json(HERE / 'cache_manifest.json')['datasets']
    direct = read_json(HERE / 'direct_cache_refs.json')['datasets']
    selectors = read_json(HERE / 'selector_manifest.json')['datasets']
    checked = 0
    for dataset, (c, k, _) in DATASETS.items():
        data = load_data(dataset)
        train_o, val_o = data['train_origins'], data['val_origins']
        train_y = np.stack([data['x'][o:o + 48] for o in train_o]).astype(np.float64)
        train_mask = np.stack([data['finite'][o:o + 48] for o in train_o]).astype(bool)
        val_y = np.stack([data['x'][o:o + 48] for o in val_o]).astype(np.float64)
        val_mask = np.stack([data['finite'][o:o + 48] for o in val_o]).astype(bool)
        counts = train_mask.sum((0, 1))
        assert np.all(counts > 0) and np.isfinite(train_y[train_mask]).all()
        for family in NEW:
            kind = 'raw' if family == 'raw_free' else 'pca'
            e = np.asarray(selectors[dataset]['E_' + kind], dtype=np.float32)
            f_train = read_prediction(cache[dataset][kind]['train'], train_o)
            f_val = read_prediction(cache[dataset][kind]['val'], val_o)
            for seed in SEEDS:
                parent_train = read_prediction(direct[dataset][str(seed)]['train'], train_o)
                parent_val = read_prediction(direct[dataset][str(seed)]['val'], val_o)
                z_train32 = f_train - parent_train @ e
                z_val32 = f_val - parent_val @ e
                z = z_train32.reshape(-1, k).astype(np.float64)
                residual = (train_y - parent_train.astype(np.float64)).reshape(-1, c)
                observed = train_mask.reshape(-1, c)
                scale = float(np.mean(np.sum(z * z, axis=1) / k))
                gram, rhs, target_square = [], [], []
                for channel in range(c):
                    a, b = z[observed[:, channel]], residual[observed[:, channel], channel]
                    gram.append(a.T @ a / counts[channel])
                    rhs.append(a.T @ b / counts[channel])
                    target_square.append(float(np.mean(b * b)))
                parent_score = metrics(parent_val, val_y, val_mask)
                members = [r for r in results if r['dataset'] == dataset and r['family'] == family and r['seed'] == seed]
                assert len(members) == 3 and {r['lambda'] for r in members} == set(LAMBDAS)
                for result in members:
                    with np.load(resolve(result['artifacts']['weights']['path']), allow_pickle=False) as archive:
                        w, w64 = archive['W'].copy(), archive['W64'].copy()
                        close(archive['train_counts'], counts, 'Exact all-TRAIN observed counts')
                    assert w.shape == w64.shape == (k, c) and w.dtype == np.float32 and w64.dtype == np.float64
                    close(w, w64.astype(np.float32), 'Exact FP32 coefficient export', atol=0, rtol=0)
                    objective = read_json(resolve(result['artifacts']['objective']['path']))
                    diagnostic = objective['diagnostics']
                    close(diagnostic['train_feature_scale_s_E'], scale, 'TRAIN innovation scale', atol=1e-12, rtol=1e-10)
                    close(diagnostic['target_counts_by_channel'], counts, 'TRAIN counts')
                    data_loss, penalty_loss, normal = [], [], []
                    penalty = result['lambda']
                    for channel in range(c):
                        wc = w64[:, channel]
                        data_loss.append(float(wc @ gram[channel] @ wc - 2 * wc @ rhs[channel] + target_square[channel]))
                        penalty_loss.append(float(penalty * scale * np.dot(wc, wc)))
                        derivative = gram[channel] @ wc - rhs[channel] + penalty * scale * wc
                        normal.append(float(np.max(np.abs(derivative))))
                        if scale != 0:
                            assert normal[-1] <= 1e-8 * (1 + np.max(np.abs(rhs[channel])))
                    if scale == 0:
                        assert np.count_nonzero(w64) == 0 and diagnostic['policy'] == 'zero_feature_scale_declared_w0'
                    close(diagnostic['data_loss_by_channel'], data_loss, 'Independent innovation residual objective', atol=1e-9, rtol=1e-8)
                    close(diagnostic['penalty_loss_by_channel'], penalty_loss, 'Independent normalized ridge penalty')
                    close(diagnostic['objective_by_channel'], np.asarray(data_loss) + penalty_loss, 'Independent total objective', atol=1e-9, rtol=1e-8)
                    close(diagnostic['objective_macro'], np.mean(np.asarray(data_loss) + penalty_loss), 'Macro TRAIN objective', atol=1e-9, rtol=1e-8)
                    close(diagnostic['normal_equation_residual_maxabs_by_channel'], normal, 'Independent regularized normal residual', atol=1e-8, rtol=1e-7)
                    prediction = parent_val + z_val32 @ w
                    actual = metrics(prediction, val_y, val_mask)
                    for key in ('mse', 'mae', 'signed_mean_error', 'channel_mse', 'channel_mae', 'channel_bias', 'target_counts'):
                        close(result['val_score'][key], actual[key], 'Exported FP32 VAL selection score')
                        close(result['parent_val_score'][key], parent_score[key], 'Frozen DIRECT VAL fallback score')
                    checked += 1
                job.heartbeat(f'{dataset}/{family}/{seed}: innovation TRAIN objective and exported VAL scores checked')
    assert checked == 48
    return {'checked_fits': checked, 'uses_all_train_origins_and_original_masks': True,
            'innovation': 'FP32 F0(XE)-S(X)E, then FP64 training objective',
            'coefficient_resolves': 0, 'minimum_norm_solver_not_rerun': True,
            'val_arithmetic': 'FP32 matmul; channel losses accumulated in FP64'}


def selector_and_cache(job):
    from evaluate_v15 import trainval_cache_binding
    selector = read_json(HERE / 'selector_manifest.json')
    cache = read_json(HERE / 'cache_manifest.json')
    seal = read_json(HERE / 'selection_seal.json')
    exposure = read_json(HERE / 'test_exposure.json')
    ledger = read_json(HERE / 'ledger.json')['jobs']
    if trainval_cache_binding() != seal['trainval_cache']:
        raise AssertionError('TRAIN/VAL cache records changed after the seal')
    total = 0
    for dataset, (c, k, _) in DATASETS.items():
        data = load_data(dataset)
        row = selector['datasets'][dataset]
        e = np.asarray(row['E_raw'], dtype=np.float32)
        u = np.asarray(row['E_pca'], dtype=np.float32)
        indices = row['indices']
        assert len(set(indices)) == k and all(0 <= i < c for i in indices)
        assert row['performance_used_for_selection'] is False and row['target_filling'] is False
        close(e, np.eye(c, dtype=np.float32)[:, indices], 'Actual coordinate selector', atol=0, rtol=0)
        close(u, data['pca_basis'], 'Canonical original PCA basis', atol=0, rtol=0)
        assert row['C'] == c and row['K'] == k
        raw_check = cache['raw_coordinate_checks'][dataset]
        assert raw_check['parity']['passed'] is True and raw_check['indices'] == indices
        assert raw_check['selector_sha256'] == digest(HERE / 'selector_manifest.json')
        assert raw_check['origins'] == data['val_origins'][:4].tolist()
        if row['W_interp'] is None:
            assert row['interpolation_rank'] < k
        else:
            assert row['interpolation_rank'] == k
            w = np.asarray(row['W_interp'], dtype=np.float32)
            close(w @ e, np.eye(k), 'Fixed interpolation selector identity', atol=1e-5, rtol=1e-4)
        for kind in ('raw', 'pca'):
            for split in ('train', 'val', 'test'):
                original = data if split != 'test' else load_data(dataset, include_test=True)
                origins = (np.concatenate([original['test_a_origins'], original['test_b_origins']])
                           if split == 'test' else original[split + '_origins'])
                values = read_prediction(cache['datasets'][dataset][kind][split], origins)
                assert values.shape == (len(origins), 48, k) and values.dtype == np.float32
                assert np.isfinite(values).all()
                entry = cache['datasets'][dataset][kind][split]
                assert entry['replay_parity']['passed'] is True and entry['chunk_parity']['passed'] is True
                assert entry['seed_independent'] is True
                cache_job = ledger[entry['job_id']]
                assert cache_job['category'] == 'gpu' and cache_job['status'] == 'complete' and cache_job['updates'] == 0
                if split == 'test':
                    assert exposure['first_new_evaluation_utc'] <= cache_job['ended_utc']
                    assert seal['created_utc'] <= cache_job['started_utc']
                else:
                    assert cache_job['ended_utc'] <= seal['created_utc']
                binding = entry['binding']
                assert binding['dataset'] == dataset and binding['space'] == kind and binding['split'] == split
                assert binding['selector_sha256'] == digest(HERE / 'selector_manifest.json')
                receipts(binding['data'])
                for name, sha in binding['files'].items():
                    artifact(HERE / name, sha)
                frozen = cache['frozen_state_checks'][dataset][entry['job_label']]
                assert frozen['passed'] is True and frozen['all_weights_frozen_and_no_grad'] is True
                assert frozen['before'] == frozen['after'] and frozen['job_id'] == entry['job_id']
                total += 1
        job.heartbeat(dataset + ': selector and complete origin-ordered frozen caches verified')
    return {'latent_cache_arrays': total, 'train_only_selector': True, 'original_pca_basis': True,
            'new_model_runs': 0, 'qr_or_coefficient_fits_repeated': False}


def saved_evaluation(job):
    value = read_json(HERE / 'evaluation01.json')
    assert value['status'] == 'complete' and value['independent_confirmation'] is False
    assert set(value['datasets']) == set(DATASETS) and value['prescribed_instances'] == 84
    receipts(value['prediction_manifest'])
    receipts(value['selection_seal'])
    selectors = read_json(HERE / 'selector_manifest.json')['datasets']
    csv_scores, actual_instances, unavailable = {}, 0, []
    for dataset, unit in value['datasets'].items():
        data = load_data(dataset, include_test=True)
        origins = np.concatenate([data['test_a_origins'], data['test_b_origins']])
        n = len(data['test_a_origins'])
        periods = {'test_a': np.arange(n), 'test_b': np.arange(n, len(origins)), 'combined': np.arange(len(origins))}
        target = np.stack([data['x'][o:o + 48] for o in origins])
        mask = np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool)
        assert len(unit['models']) == 21 and set(unit['role_summary']) == set(ROLES)
        assert unit['columns'] == list(map(str, data['columns']))
        assert unit['bootstrap'] == {'draws': 2000, 'seed': 9262026,
            'block_origins': protocol()['units'][dataset]['block_origins'],
            'period_boundaries_crossed': False, 'paired_channels_methods_and_seeds': True,
            'selection_uncertainty_included': False}
        by_family = defaultdict(list)
        seed_members = defaultdict(list)
        for identifier, row in unit['models'].items():
            assert identifier == row['id'] and row['dataset'] == dataset
            seed_members[row['family']].append(row['seed'])
            if row['status'] == 'unavailable':
                assert row['family'] == 'raw_interp' and selectors[dataset]['W_interp'] is None
                assert selectors[dataset]['interpolation_rank'] < selectors[dataset]['K']
                assert row['periods'] == {} and row.get('unavailable_reason')
                unavailable.append(identifier)
                continue
            assert row['status'] == 'complete'
            prediction = read_prediction(row['prediction'], origins)
            scores = {period: metrics(prediction[index], target[index], mask[index]) for period, index in periods.items()}
            for period, score in scores.items():
                for key, expected in score.items():
                    close(row['periods'][period][key], expected, identifier + '/' + period + '/' + key)
                csv_scores[(dataset, identifier, period)] = score
            for key in ('squared_error_sums', 'absolute_error_sums', 'signed_error_sums', 'target_counts'):
                close(scores['combined'][key], scores['test_a'][key] + scores['test_b'][key], 'Pooled periods, original target counts')
            by_family[row['family']].append(scores)
            actual_instances += 1
            if row['reused']:
                assert row['reference_score_replay']['passed'] is True
                source = row['source_evaluation']
                artifact(source['path'], source['sha256'])
                old = read_json(resolve(source['path']))['datasets'][dataset]['models'][row['source_model_id']]
                for period in PERIODS:
                    for metric in ('mse', 'mae', 'signed_mean_error'):
                        close(scores[period][metric], old['periods'][period][metric], 'Original reference score replay')
            else:
                parent = read_prediction(row['direct_prediction'], origins)
                latent = read_prediction(row['latent_prediction'], origins)
                selector = selectors[dataset]
                kind = 'raw' if row['family'].startswith('raw') else 'pca'
                e = np.asarray(selector['E_' + kind], dtype=np.float32)
                if row['family'] in NEW:
                    artifact(row['weights']['path'], row['weights']['sha256'])
                    with np.load(resolve(row['weights']['path']), allow_pickle=False) as arrays:
                        w = arrays['W'].copy()
                elif row['family'] == 'raw_interp':
                    w = np.asarray(selector['W_interp'], dtype=np.float32)
                else:
                    w = e.T.copy()
                assert w.dtype == np.float32 and w.shape == (selector['K'], selector['C'])
                zero = bool(np.count_nonzero(w) == 0)
                expected = parent.copy() if zero else parent + (latent - parent @ e) @ w
                close(prediction, expected, 'Saved FP32 decoder arithmetic', atol=1e-6, rtol=0)
                assert row['zero_decoder'] == zero
                if zero:
                    close(prediction, parent, 'Exact direct fallback', atol=0, rtol=0)
        for family in ROLES:
            expected = [None] if family == 'f0' else list(SEEDS)
            assert sorted(seed_members[family], key=str) == sorted(expected, key=str)
            summary = unit['role_summary'][family]
            members = by_family[family]
            if not members:
                assert family == 'raw_interp' and summary['status'] == 'unavailable' and summary['periods'] == {}
                continue
            assert len(members) == len(expected) and summary['status'] == 'available'
            for period in PERIODS:
                for metric in ('mse', 'mae', 'signed_mean_error', 'channel_mse', 'channel_mae', 'channel_bias'):
                    close(summary['periods'][period][metric],
                          np.mean([member[period][metric] for member in members], axis=0), 'Mean seed losses, not ensemble')
        for item in unit['comparisons'].values():
            if item['status'] == 'unavailable':
                assert 'raw_interp' in (item['candidate'], item['reference']) and selectors[dataset]['W_interp'] is None
                continue
            for period in PERIODS:
                for metric in ('mse', 'mae'):
                    a = unit['role_summary'][item['candidate']]['periods'][period][metric]
                    b = unit['role_summary'][item['reference']]['periods'][period][metric]
                    reported = item['periods'][period][metric]
                    close([reported['candidate'], reported['reference'], reported['absolute_difference']], [a, b, a - b], 'Paired point estimates')
                    close(reported['relative_change_percent'], 100 * (a - b) / b, 'Relative change percent')
                    for field in ('conditional_block95_absolute', 'conditional_block95_relative_percent'):
                        ci = np.asarray(reported[field])
                        assert ci.shape == (2,) and np.isfinite(ci).all() and ci[0] <= ci[1]
        job.heartbeat(dataset + ': independent saved-array metrics and decoder composition verified')
    with (HERE / 'evaluation01_models.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(csv_scores) == actual_instances * 3
    for row in rows:
        expected = csv_scores[(row['dataset'], row['id'], row['period'])]
        for metric in ('mse', 'mae', 'signed_mean_error'):
            close(float(row[metric]), expected[metric], 'Model CSV metric')
    with (HERE / 'evaluation01_channels.csv').open(encoding='utf-8', newline='') as stream:
        channels = list(csv.DictReader(stream))
    assert len(channels) == sum(len(value['datasets'][dataset]['columns']) for dataset, _, _ in csv_scores)
    for row in channels:
        expected = csv_scores[(row['dataset'], row['id'], row['period'])]
        index = value['datasets'][row['dataset']]['columns'].index(row['channel'])
        for key, field in (('mse', 'channel_mse'), ('mae', 'channel_mae'), ('signed_mean_error', 'channel_bias'), ('target_count', 'target_counts')):
            close(float(row[key]), expected[field][index], 'Channel CSV metric')
    assert actual_instances == value['scored_instances'] and actual_instances + len(unavailable) == 84
    return {'prescribed_instances': 84, 'scored_instances': actual_instances, 'unavailable_fixed_controls': unavailable,
            'model_period_rows': len(rows), 'seed_mean_not_ensemble': True, 'new_model_runs': 0,
            'bootstrap_intervals_recomputed': False, 'scope': 'Arithmetic self-check, not independent reproduction'}


def costs_and_report(job):
    evaluation = read_json(HERE / 'evaluation01.json')
    summary = read_json(HERE / 'cost_summary.json')
    assert summary['status'] in ('complete', 'budget_stopped')
    assert summary['accuracy_dependent_dataset_filtering'] is False
    complete_scope = summary['status'] == 'complete'
    receipts(summary['binding'])
    binding = read_json(resolve(summary['binding']['path']))
    for name, sha in binding['files'].items():
        artifact(HERE / name, sha)
    assert binding['settings'] == protocol()['cost']
    ledger = read_json(HERE / 'ledger.json')['jobs']
    counts, missing, seen_parity = {}, {}, set()
    stopping = None
    available_receipt = None
    report_gpu_status = summary['gpu']['status']
    for device, section, required_count in (('cuda', 'gpu', 612), ('cpu', 'cpu', 48)):
        part = summary[section]
        partial = device == 'cuda' and not complete_scope
        expected_status = 'budget_stopped' if partial else 'complete'
        assert part is not None and part['status'] == expected_status
        raw = read_json(HERE / f'cost_{device}_rows.json')
        planned = read_json(HERE / f'cost_{device}_grid.json')
        assert raw['status'] == expected_status and raw['expected_rows'] == len(planned['rows']) == required_count
        measured_count = len(raw['rows'])
        if partial:
            assert 0 < measured_count < required_count
        else:
            assert measured_count == required_count
        assert raw['binding_sha256'] == planned['binding_sha256'] == summary['binding']['sha256']
        expected = set()
        for dataset, (c, k, _) in DATASETS.items():
            unit = evaluation['datasets'][dataset]
            roles = ('direct',) if device == 'cpu' else NEW + ('direct', 'a', 'f0', 'full_mse', 'native')
            for identifier, model in unit['models'].items():
                family = model['family']
                if family not in roles:
                    continue
                if family == 'direct':
                    configs = [(1, 'direct', None), (4, 'direct', None)]
                elif family in ('f0', 'full_mse', 'native'):
                    configs = [(1, 'full', c), (4, 'full', 4*c), (1, 'chunk_k', k),
                               (4, 'chunk_4k', 4*k), (4, 'chunk_k', k)]
                else:
                    configs = [(1, 'full', k), (4, 'full', 4*k), (4, 'chunk_k', k)]
                for block in range(3):
                    for batch, execution, chunk in configs:
                        expected.add((dataset, identifier, family, model['seed'], block, None, batch, execution, chunk))
                    if device == 'cuda' and family == 'f0':
                        for sentinel in ('pre', 'post'):
                            for batch in (1, 4):
                                expected.add((dataset, identifier, family, model['seed'], block, sentinel, batch, 'full', batch*c))
        identity_keys = ('dataset', 'id', 'family', 'seed', 'block', 'sentinel', 'batch', 'execution', 'chunk_rows')
        planned_keys = [tuple(row[key] for key in identity_keys) for row in planned['rows']]
        assert len(set(planned_keys)) == required_count and set(planned_keys) == expected
        assert all(row['planned_sequence'] == index for index, row in enumerate(planned['rows']))
        actual, grouped, sentinels = set(), defaultdict(list), {}
        for index, (row, planned_row) in enumerate(zip(raw['rows'], planned['rows'])):
            assert row['planned_sequence'] == index and all(row[key] == value for key, value in planned_row.items())
            key = tuple(row[key] for key in identity_keys)
            assert key not in actual and row['status'] == 'complete' and row['device'] == device
            actual.add(key)
            assert row['binding_sha256'] == summary['binding']['sha256']
            cost_job = ledger[row['job_id']]
            assert cost_job['category'] == ('gpu' if device == 'cuda' else 'cpu_analysis')
            assert cost_job['status'] == 'complete' and cost_job['metadata']['v15_cost_campaign'] is True
            assert cost_job['updates'] == 0
            assert row['passes'] == 3 == binding['settings']['passes']
            assert row['warmup_calls'] == 10 == binding['settings']['warmups']
            times = np.asarray(row['seconds_per_24_origins'])
            assert times.shape == (3,) and np.isfinite(times).all() and np.all(times > 0)
            close(row['milliseconds_per_origin']['median'], np.median(times) * 1000 / 24, 'Median timing of complete workload')
            close(row['origins_per_second']['median'], np.median(24 / times), 'Median complete-workload throughput')
            model = row['model']
            assert model['merge_export_parity']['pass'] is True
            assert all(item['parity']['pass'] for item in model['configuration_parity'])
            if row['family'] in NEW:
                assert model['cached_online_parity']['pass'] is True
                c, k, _ = DATASETS[row['dataset']]
                assert model['fitted_decoder_coefficients'] == c*k and model['historical_direct_fit_parameters'] == 24624
            if device == 'cpu':
                assert model['cpu_gpu_parity']['pass'] is True
                assert row['peak_allocated_bytes'] is None and row['peak_reserved_bytes'] is None
            else:
                assert 0 < row['peak_allocated_bytes'] <= row['peak_reserved_bytes']
                assert row['resident_after_load']['allocated_bytes'] <= row['peak_allocated_bytes']
            if model.get('replay'):
                assert model['replay']['pass'] is True
            receipt = model['parity_receipt']
            if receipt['sha256'] not in seen_parity:
                receipts(receipt)
                saved = read_json(resolve(receipt['path']))
                assert saved['binding_sha256'] == summary['binding']['sha256']
                receipts(saved['prediction'])
                seen_parity.add(receipt['sha256'])
            if row['sentinel'] is None:
                grouped[(row['dataset'], row['id'], row['batch'], row['execution'], row['chunk_rows'])].append(row)
            else:
                sentinels[(row['dataset'], row['block'], row['batch'], row['sentinel'])] = row
        assert actual == set(planned_keys[:measured_count]) and len(actual) == measured_count
        counts[device] = measured_count
        missing[device] = planned['rows'][measured_count:]
        if partial:
            guard = read_json(HERE / 'cost_guard.json')
            assert guard['status'] == 'budget_stopped' and guard['reason'] == 'V15 cost time cap: preserve partial rows'
            assert guard['complete_rows'] == part['complete_rows'] == measured_count
            assert guard['expected_rows'] == part['expected_rows'] == required_count
            assert part['complete_frontier'] is False
            assert not part.get('instances') and not part.get('groups') and not part.get('sentinels')
            campaign_jobs = [row for row in ledger if row['category'] == 'gpu'
                             and row['metadata'].get('v15_cost_campaign')]
            assert campaign_jobs and all(row['status'] in ('complete', 'failed') for row in campaign_jobs)
            assert campaign_jobs[-1]['status'] == 'complete'
            spent = sum(row['elapsed_s'] for row in campaign_jobs)
            cap = binding['settings']['reserve_seconds']
            assert cap - 30 <= guard['elapsed_seconds'] <= spent <= cap
            stopping = {'guard': artifact(HERE / 'cost_guard.json'), 'reason': guard['reason'],
                        'guard_elapsed_seconds': guard['elapsed_seconds'], 'ledger_cost_gpu_seconds': spent,
                        'fixed_cost_cap_seconds': cap, 'job_ids': [row['id'] for row in campaign_jobs],
                        'unmeasured_rows': len(missing[device]),
                        'scope': 'Saved exact grid prefix checked; no missing blocks or measurements imputed'}
            job.heartbeat(f'{device}: {measured_count}/{required_count} exact-prefix rows verified; remaining scope incomplete')
            available_path = HERE / 'cost_available_summary.json'
            if not available_path.exists():
                continue
            available = read_json(available_path)
            assert available['status'] == 'partial_cost' and available['complete_campaign'] is False
            for name, path in (('original_cost_summary', HERE / 'cost_summary.json'),
                               ('raw_gpu', HERE / 'cost_cuda_rows.json'),
                               ('grid', HERE / 'cost_cuda_grid.json'), ('guard', HERE / 'cost_guard.json')):
                assert available[name] == artifact(path)
            assert set(available['source_files']) == {'summarize_available_cost_v15.py', 'cost_v15.py', 'runtime_v15.py'}
            for name, sha in available['source_files'].items():
                artifact(HERE / name, sha)
            receipts(available['source'])
            aggregate_job = ledger[available['job_id']]
            assert aggregate_job['category'] == 'cpu_analysis' and aggregate_job['status'] == 'complete'
            assert aggregate_job['updates'] == 0 and aggregate_job['metadata']['new_measurements'] == 0
            assert aggregate_job['metadata']['model_runs'] == 0 and aggregate_job['metadata']['fits'] == 0
            complete_datasets = [dataset for dataset in DATASETS
                                 if all(key in actual for key in planned_keys if key[0] == dataset)]
            assert available['complete_datasets'] == complete_datasets
            assert available['complete_rows'] == measured_count
            assert available['summarized_rows'] == sum(row['dataset'] in complete_datasets for row in raw['rows'])
            assert available['omitted_dataset_rows'] == {
                dataset: sum(row['dataset'] == dataset for row in raw['rows'])
                for dataset in DATASETS if dataset not in complete_datasets}
            assert available['missing_grid_rows'] == missing[device]
            part = available['gpu']
            assert part['status'] == 'complete_for_listed_datasets'
            assert part['complete_datasets'] == complete_datasets
            report_gpu_status = part['status']
            grouped = {key: rows for key, rows in grouped.items() if key[0] in complete_datasets}
            sentinels = {key: row for key, row in sentinels.items() if key[0] in complete_datasets}
            available_receipt = artifact(available_path)
            stopping['aggregate_datasets'] = complete_datasets
            stopping['omitted_dataset_rows'] = available['omitted_dataset_rows']
            stopping['available_summary'] = available_receipt
        else:
            assert actual == expected
        instances = {}
        for item in part['instances']:
            key = (item['dataset'], item['id'], item['batch'], item['execution'], item['chunk_rows'])
            group = grouped.pop(key)
            assert len(group) == 3 and {row['block'] for row in group} == {0, 1, 2}
            close(item['median_block_median_ms'], np.median([r['milliseconds_per_origin']['median'] for r in group]), 'Three-block latency median')
            close(item['median_block_median_origins_s'], np.median([r['origins_per_second']['median'] for r in group]), 'Three-block throughput median')
            for field in ('peak_allocated_bytes', 'peak_reserved_bytes'):
                assert item[field] == [r[field] for r in group]
            instances[key] = item
        assert not grouped
        expected_groups = {(row['dataset'], row['family'], row['batch'], row['execution'], row['chunk_rows'])
                           for row in instances.values()}
        actual_groups = set()
        for group in part['groups']:
            group_key = (group['dataset'], group['family'], group['batch'], group['execution'], group['chunk_rows'])
            assert group_key not in actual_groups
            actual_groups.add(group_key)
            items = [row for row in instances.values() if row['dataset'] == group['dataset']
                     and row['family'] == group['family'] and row['batch'] == group['batch']
                     and row['execution'] == group['execution'] and row['chunk_rows'] == group['chunk_rows']]
            assert len(items) == (1 if group['family'] == 'f0' else 2)
            close(group['mean_seed_median_ms'], np.mean([row['median_block_median_ms'] for row in items]), 'Cost seed aggregation')
            close(group['mean_seed_median_origins_s'], np.mean([row['median_block_median_origins_s'] for row in items]), 'Cost seed throughput aggregation')
            for field in ('allocated', 'reserved'):
                values = [v for item in items for v in item['peak_' + field + '_bytes'] if v is not None]
                mean_key = 'mean_seed_median_peak_' + field + '_bytes'
                range_key = field + '_range_bytes'
                if values:
                    close(group[mean_key], np.mean([np.median(item['peak_' + field + '_bytes']) for item in items]), 'Cost seed memory aggregation')
                    close(group[range_key], [min(values), max(values)], 'Cost memory range')
                else:
                    assert group[mean_key] is None and group[range_key] is None
        assert actual_groups == expected_groups
        for row in part['sentinels']:
            key = (row['dataset'], row['block'], row['batch'])
            pre, post = [sentinels.pop(key + (position,))['milliseconds_per_origin']['median'] for position in ('pre', 'post')]
            close([row['pre_ms'], row['post_ms'], row['post_relative_change']], [pre, post, post/pre - 1], 'Sentinel drift')
        assert not sentinels
        job.heartbeat(device + ': saved rows, parity and declared complete-dataset block aggregation verified')
    report_path = HERE / 'report_values.json'
    assert report_path.exists(), 'Saved-scope verification requires the generated report'
    saved = read_json(report_path)
    assert saved['status'] == ('complete' if complete_scope else 'partial_cost')
    assert saved['cost_status']['complete_for_report'] is complete_scope
    assert saved['cost_status']['top_level'] == summary['status']
    assert saved['cost_status']['gpu'] == report_gpu_status
    assert saved['cost_status']['cpu'] == summary['cpu']['status']
    if not complete_scope:
        assert saved['frontier_claim']['made'] is False
    if available_receipt is not None:
        assert saved['cost_available_summary'] == available_receipt
    assert set(saved['source_files']) == {'report_v15.py', 'evaluate_v15.py', 'cost_v15.py',
                                        'runtime_v15.py', 'PLAN.md', 'protocol.json'}
    for name, sha in saved['source_files'].items():
        artifact(HERE / name, sha)
    assert set(saved['tables']) == {'comparison', 'paired_differences', 'cost_comparison'}
    assert set(saved['figures']) == {'accuracy_by_period', 'accuracy_cost'}
    assert saved['roles'] == list(ROLES) and saved['periods'] == list(PERIODS)
    receipts(saved)
    with resolve(saved['tables']['comparison']['path']).open(encoding='utf-8', newline='') as stream:
        report_rows = list(csv.DictReader(stream))
    assert len(report_rows) == saved['accuracy_rows'] == len(DATASETS)*len(ROLES)*len(PERIODS)
    for row in report_rows:
        unit = evaluation['datasets'][row['dataset']]['role_summary']
        expected = unit[row['family']]['periods'][row['period']]
        reference = unit['full_mse']['periods'][row['period']]
        for metric in ('mse', 'mae', 'signed_mean_error'):
            close(float(row[metric]), expected[metric], 'Report accuracy CSV metric')
        for metric in ('mse', 'mae'):
            close(float(row['relative_' + metric + '_to_full_mse_percent']),
                  100*expected[metric]/reference[metric], 'Report relative score denominator')
    report = {'status': 'complete_sources_receipts_and_accuracy_checked', 'receipt': artifact(report_path),
              'scope': 'Saved report sources, figure bytes and accuracy table; scientific interpretation remains a separate review'}
    return {'cost_status': summary['status'], 'complete_scope': complete_scope,
            'saved_cost_rows': counts, 'missing_planned_rows': missing, 'budget_stop': stopping,
            'all_four_datasets_preserved': True,
            'cached_online_parity': True, 'new_model_runs': 0, 'benchmark_repeats': 0, 'report': report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', required=True, type=float)
    parser.add_argument('--scope', choices=('accuracy', 'all'), default='all')
    args = parser.parse_args()
    actions = [('fits_and_seal', fits_and_seal), ('selector_and_cache', selector_and_cache), ('saved_evaluation', saved_evaluation)]
    if args.scope == 'all':
        actions.append(('costs_and_report', costs_and_report))
    checks = []
    with Job('cpu_check', args.job, reserve_s=args.reserve_s,
             metadata={'purpose': 'v15 saved-artifact verification', 'model_runs': 0, 'fit_count': 0}) as job:
        for name, action in actions:
            try:
                checks.append({'name': name, 'status': 'passed', 'details': action(job)})
            except Exception:
                checks.append({'name': name, 'status': 'failed', 'error': traceback.format_exc()})
        budget = budget_snapshot()
        limits = budget['limits']
        assert budget['real_fit_attempts'] <= limits['real_fit'] and budget['technical_reserve_attempts'] <= limits['technical_reserve']
        assert budget['neural_fit_attempts'] == 0
        assert not [row for row in budget['active_jobs'] if row['id'] != job.id]
        for category in ('gpu', 'cpu_analysis', 'cpu_check'):
            assert budget[category + '_seconds'] <= limits[category + '_seconds']
        assert budget['storage_bytes'] <= limits['storage_bytes']
        passed = all(row['status'] == 'passed' for row in checks)
        complete_cost = any(row['name'] == 'costs_and_report' and row['status'] == 'passed'
                            and row['details']['complete_scope'] for row in checks)
        result = {'status': 'passed' if passed else 'failed', 'scope': args.scope, 'checks': checks,
                  'complete_campaign_verified': bool(passed and args.scope == 'all' and complete_cost),
                  'verification_scope': 'Saved artifacts only; missing planned costs remain incomplete',
                  'job_id': job.id, 'model_runs': 0, 'coefficient_fits': 0, 'optimizer_updates': 0,
                  'independent_reproduction': False, 'budget_during_check': budget}
        path = HERE / ('final_checks.json' if args.scope == 'all' else 'accuracy_checks.json')
        if path.exists():
            prior = read_json(path)
            save_json(path.with_name(path.stem + f'_prior_job_{prior["job_id"]}.json'), prior)
        save_json(path, result)
    result['budget_after_check'] = budget_snapshot()
    save_json(path, result)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
