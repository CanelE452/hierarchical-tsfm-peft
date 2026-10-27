"""Read-only historical V-array audit; writes only this campaign's two reports."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


NAME = 'comparative_ablation_20261025_v1'
KT = 'k_tau_rule_20261023_v1'
SEARCH = 'b2_dataset_search_20261019_v1'
BROAD = 'research/informs_broad_validation_20261015_v1'
EXPECTED = {'jena': (6, 10482, 12579), 'ETTh1': (4, 8710, 10452),
            'ETTh2_safe17400': (4, 8700, 10440), 'appliances': (10, 9867, 11841)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--asset', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    source, asset = args.source.resolve(), args.asset.resolve()
    output = repo / 'reports' / NAME
    destinations = [output / 'G1_REUSE_AUDIT.json', output / 'G1_RUN_INVENTORY.csv']
    if any(p.exists() for p in destinations):
        raise FileExistsError('Refusing to overwrite sealed audit outputs')
    accessed, failures, provenance_gaps = set(), [], []

    def sha(path):
        accessed.add(str(path))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def read(path):
        accessed.add(str(path))
        return json.loads(path.read_text(encoding='utf8'))

    def csv_read(path):
        accessed.add(str(path))
        with path.open(encoding='utf8', newline='') as stream:
            return list(csv.DictReader(stream))

    def resolve(binding, cache):
        kind, relative = binding['reference'].split('://', 1)
        root = {'repo': source, 'asset': asset, 'cache': cache}[kind]
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Binding escapes its registered root')
        return path

    historical_data = {x['dataset']: x for x in read(source / 'reports' / KT / 'CURRENT_DATA_FINGERPRINTS.json')}
    f_sha = read(source / 'reports' / KT / 'CURRENT_ENVIRONMENT.json')['F_SHA']
    runs = csv_read(source / 'reports' / KT / 'RUNS.csv')
    if len(runs) != 40 or {(x['dataset'], int(x['seed'])) for x in runs} != {
            (d, s) for d in EXPECTED for s in range(92601, 92611)}:
        raise ValueError('Expected exactly four datasets times ten registered seeds')
    legacy = read(source / '.cache/b2_revin_20261018_v1/VERIFIED_ROWS.json')
    inventory, reference_arrays, core_checks = [], {}, {}

    def score(path, expected_mse, expected_mae, dataset):
        if 'V_select' not in path.name and ('initialization_predictions' not in str(path)
                and '/costs/' not in path.as_posix()):
            raise ValueError('Only explicitly mapped V prediction archives are allowed')
        accessed.add(str(path))
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in
                      ('pred_normalized', 'target_normalized', 'observed', 'origins', 'channel_ids')}
            schema = list(archive.files)
        pred, target, mask = (values[k] for k in ('pred_normalized', 'target_normalized', 'observed'))
        start, end = historical_data[dataset]['split_ranges']['V_select']
        valid = (pred.shape == target.shape == mask.shape and mask.dtype == np.bool_
                 and pred.shape[1] == 48 and np.all(values['origins'] >= start)
                 and np.all(values['origins'] + 48 <= end)
                 and np.all(np.isfinite(pred[mask])) and np.all(np.isfinite(target[mask])))
        if not valid:
            raise ValueError('Prediction archive violates V shape/mask/information boundary')
        error = pred.astype(np.float64) - target.astype(np.float64)
        count = mask.sum(axis=(0, 1))
        if np.any(count == 0):
            raise ValueError('Empty observation channel')
        mse = float(np.mean(np.where(mask, error * error, 0).sum(axis=(0, 1)) / count))
        mae = float(np.mean(np.where(mask, np.abs(error), 0).sum(axis=(0, 1)) / count))
        aligned = True
        if dataset in reference_arrays:
            aligned = all(np.array_equal(values[k], reference_arrays[dataset][k]) for k in
                          ('target_normalized', 'observed', 'origins', 'channel_ids'))
        else:
            reference_arrays[dataset] = values
        return dict(mse_recomputed=mse, mae_recomputed=mae,
                    mse_delta=abs(mse - expected_mse), mae_delta=abs(mae - expected_mae),
                    metrics_pass=max(abs(mse - expected_mse), abs(mae - expected_mae)) <= 1e-9,
                    aligned_V_targets=aligned, shape=list(pred.shape), archive_keys=schema,
                    origin_min=int(values['origins'].min()), origin_max=int(values['origins'].max()))

    for record in runs:
        d, seed, k = record['dataset'], int(record['seed']), int(record['K'])
        reused = record['reused'] == 'True'
        if not reused:
            result_path = source / '.cache' / KT / 'results' / (record['job_id'] + '.json')
            result = read(result_path)['result']
            row = result['rows'][0]
            cache = source / '.cache' / KT / 'datasets' / d
            receipt_path = source / 'reports' / KT / 'datasets' / d / 'runs' / record['job_id'] / 'cap3/BASE_FIT_RECEIPT.json'
        elif d == 'jena':
            result_path = source / '.cache/b2_revin_20261018_v1/VERIFIED_ROWS.json'
            row = next(x for x in legacy if x.get('run_id') == f'B2_jena_REVIN_OFF_s{seed}_cap3')
            cache = source / '.cache/adapter_diagnostics_20261017_v2'
            receipt_path = source / 'reports/adapter_diagnostics_20261017_v2/runs' / f'init_PCA_s{seed}' / 'BASE_FIT_RECEIPT.json'
        else:
            result_path = source / 'reports' / SEARCH / 'datasets' / d / 'VERIFIED_ROWS.json'
            row = next(x for x in read(result_path) if x.get('run_id') == f'B2_{d}_K{k}_OFF_s{seed}')
            cache = source / '.cache' / SEARCH / 'datasets' / d
            receipt_path = source / 'reports' / SEARCH / 'datasets' / d / 'runs' / row['run_id'] / 'cap3/BASE_FIT_RECEIPT.json'
        receipt = read(receipt_path)
        contract = read(receipt_path.with_name('SOURCE_CONTRACT.json'))
        pred_path = resolve(row['prediction_binding'], cache)
        bundle_path = resolve(row['bundle_binding'], cache)
        item = dict(dataset=d, method_id='OURS', seed=seed, K=k, reused=reused,
                    job_id=record['job_id'], source_run_id=row['run_id'],
                    result_path=str(result_path), receipt_path=str(receipt_path),
                    prediction_path=str(pred_path), bundle_path=str(bundle_path),
                    prediction_sha256=sha(pred_path), bundle_sha256=sha(bundle_path),
                    selected_epoch=receipt['selected_epoch'],
                    normalization_fit_end=receipt['normalization_fit_end'])
        item.update(score(pred_path, float(record['mse']), float(record['mae']), d))
        item['prediction_hash_pass'] = item['prediction_sha256'] == row['prediction_binding']['sha256']
        item['bundle_hash_pass'] = item['bundle_sha256'] == row['bundle_binding']['sha256'] == receipt['artifact_sha256']
        item['contract_pass'] = (contract['model_sha'] == f_sha
            and contract['data_sha'] == historical_data[d]['data_sha256']
            and (row['L'], row['H'], k) == (512, 48, EXPECTED[d][0])
            and receipt['search_fit_end'] == EXPECTED[d][1]
            and receipt['refit_fit_end'] == receipt['normalization_fit_end'] == EXPECTED[d][2])
        for name, expected in contract.get('sources', contract.get('files', {})).items():
            if '/' not in name:
                relative = f'{BROAD}/{name}'
                actual = sha(repo / relative)
                core_checks[relative] = dict(expected=expected, actual=actual, matches=expected == actual)
        if not all(item[key] for key in ('metrics_pass', 'aligned_V_targets', 'prediction_hash_pass', 'bundle_hash_pass', 'contract_pass')):
            failures.append(dict(job_id=item['job_id'], reason='V array or historical contract check failed'))
        inventory.append(item)

    baseline = csv_read(source / 'reports' / KT / 'BASELINE_RECALCULATION.csv')
    for d in EXPECTED:
        old = next(x for x in baseline if x['dataset'] == d and x['family'] == 'RAW')
        if d == 'jena':
            row = next(x for x in legacy if x.get('family') == 'RAW' and x.get('dataset') == d)
            cache = source / '.cache/b2_revin_20261018_v1'
            path = resolve(row['prediction_binding'], cache)
            expected_hash = row['prediction_binding']['sha256']
            receipt_path = source / '.cache/b2_revin_20261018_v1/VERIFIED_ROWS.json'
            provenance_gaps.append(dict(dataset=d, method_id='RAW', field='historical F state',
                value=row['model_sha'], evidence=str(receipt_path),
                required='Resolve historical F identity or demonstrate permitted current Raw V inference parity'))
        else:
            receipt_path = source / 'reports' / SEARCH / 'datasets' / d / 'costs' / f'Raw_{d}_DEFAULT128_B8.json'
            row = read(receipt_path)
            cache = source / '.cache' / SEARCH / 'datasets' / d
            path = cache / row['prediction_reference'].split('://', 1)[1]
            expected_hash = row['prediction_file_sha256']
        item = dict(dataset=d, method_id='RAW', seed=92601, K=None, reused=True,
                    job_id=f'RAW_{d}', prediction_path=str(path), receipt_path=str(receipt_path),
                    prediction_sha256=sha(path))
        item.update(score(path, float(old['MSE_mean']), float(old['MAE_mean']), d))
        item['prediction_hash_pass'] = item['prediction_sha256'] == expected_hash
        if not all(item[key] for key in ('metrics_pass', 'aligned_V_targets', 'prediction_hash_pass')):
            failures.append(dict(job_id=item['job_id'], reason='Raw array rescore, alignment or hash failed'))
        inventory.append(item)

    if not all(x['matches'] for x in core_checks.values()):
        failures.append(dict(reason='Current training/evaluation core differs from historical source hash'))
    summary = dict(status='FAIL' if failures else 'PARTIAL_PENDING_CURRENT_RUNTIME_AND_DATA',
        created_utc=datetime.now(timezone.utc).isoformat(), source_worktree=str(source),
        current_worktree=str(repo), numeric_scope='Only 44 explicitly mapped V prediction archives',
        T_numeric_values_read=0, training_executed=False, count_ours=40, count_raw=4,
        ours_new=34, ours_historical=6, failures=failures, provenance_gaps=provenance_gaps,
        max_mse_delta=max(x['mse_delta'] for x in inventory),
        max_mae_delta=max(x['mae_delta'] for x in inventory), core_hash_checks=core_checks,
        remaining_checks=['Current loaded frozen F_SHA', 'Current pre-T data identities and split/norm contract'],
        historical_selected_epochs=[{k:x[k] for k in ('dataset', 'seed', 'selected_epoch')}
                                    for x in inventory if x['method_id'] == 'OURS' and x['reused']],
        opened_files=sorted(accessed), runs=inventory)
    output.mkdir(parents=True, exist_ok=True)
    with destinations[0].open('x', encoding='utf8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    fields = list(dict.fromkeys(k for x in inventory for k in x))
    with destinations[1].open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)
    print(json.dumps({k:summary[k] for k in ('status','count_ours','count_raw','ours_new',
                     'ours_historical','max_mse_delta','max_mae_delta','failures','provenance_gaps')}))


if __name__ == '__main__':
    main()
