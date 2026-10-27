"""CPU-only, complete-case affine output-space diagnostics for saved V forecasts."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from research.comparative_ablation_20261025_v1 import analysis, runtime
from . import common


REFERENCES = ('RAW', 'C1', 'C2', 'C3', 'OURS', 'A1')
ALGORITHM_VERSION = 'affine-output-floor-v1'
FLOOR_FILES = ('D1_DECODER_FLOOR.csv', 'D1_DECODER_FLOOR_SUMMARY.csv', 'D0_D1_VALIDATION.json',
               'D0_SUBSPACE_FLOOR.csv', 'D0_MEAN_SHIFT_CHANNELS.csv')
TOLERANCE_DESCRIPTION = (
    'All projections and squared-error reductions use FP64. The SVD rank cutoff is '
    'max(W.shape)*eps64*s_max on the saved FP32 weights cast to FP64. Prediction '
    'space residual RMS must not exceed 32*eps32*(K+2) times the RMS scale of the '
    'prediction, bias, and abs(minimum-norm latent)@abs(W)+abs(bias). This allows '
    'FP32 dot-product/bias rounding with a conservative factor 32. The lower-bound '
    'MSE tolerance is 2*sqrt(floor_MSE*prediction_space_residual_MSE) + '
    'prediction_space_residual_MSE + 64*eps64*max(floor_MSE, actual_MSE), derived '
    'from Cauchy-Schwarz on the measured off-space prediction residual. '
    'Scientific violations raise ValueError; no bound or rank is tuned to outcomes.'
)


def row_space(weight):
    """Return an orthonormal basis for a K-by-C matrix's row space."""
    weight = np.asarray(weight, np.float64)
    if weight.ndim != 2 or not np.isfinite(weight).all() or min(weight.shape) < 1:
        raise ValueError('Decoder weights must be a finite nonempty K-by-C matrix')
    left, singular, right = np.linalg.svd(weight, full_matrices=False)
    cutoff = float(max(weight.shape) * np.finfo(np.float64).eps * singular[0])
    rank = int((singular > cutoff).sum())
    return dict(basis=right[:rank].T, singular_values=singular, rank=rank,
                rank_tolerance=cutoff, left=left[:, :rank])


def outside(values, bias, basis):
    centered = np.asarray(values, np.float64) - np.asarray(bias, np.float64)
    return centered - (centered @ basis) @ basis.T


def complete_cohort(values):
    analysis.validate_arrays(values)
    active = np.asarray(values['observed']).all(axis=-1)
    if not active.any():
        raise ValueError('No fully observed V target points remain')
    return (np.asarray(values['target_normalized'], np.float64)[active],
            np.asarray(values['pred_normalized'], np.float64)[active],
            dict(total_points=int(active.size), complete_points=int(active.sum()),
                 excluded_points=int((~active).sum()), excluded_fraction=float((~active).mean())))


def affine_diagnostic(values, weight, bias):
    target, prediction, cohort = complete_cohort(values)
    weight, bias = np.asarray(weight, np.float64), np.asarray(bias, np.float64)
    if weight.shape[1] != target.shape[1] or bias.shape != (target.shape[1],) or not np.isfinite(bias).all():
        raise ValueError('Decoder channels or finite bias differ from the V archive')
    space = row_space(weight)
    basis = space['basis']
    floor = float(np.square(outside(target, bias, basis)).mean())
    actual = float(np.square(prediction - target).mean())
    residual = float(np.square(outside(prediction, bias, basis)).mean())
    rank = space['rank']
    latent = ((prediction - bias) @ basis / space['singular_values'][:rank]) @ space['left'].T
    operation_scale = np.abs(latent) @ np.abs(weight) + np.abs(bias)
    scale = max(float(np.sqrt(np.square(array).mean())) for array in (prediction, bias, operation_scale))
    residual_limit = float((32 * np.finfo(np.float32).eps * (weight.shape[0] + 2) * scale) ** 2)
    tolerance = float(2 * np.sqrt(floor * residual) + residual
                      + 64 * np.finfo(np.float64).eps * max(floor, actual))
    if actual + tolerance < floor:
        raise ValueError(f'Affine lower-bound violation: actual={actual}, floor={floor}, tolerance={tolerance}')
    if residual > residual_limit:
        raise ValueError(f'Prediction outside saved decoder space: residual={residual}, limit={residual_limit}')
    return dict(status='PASS', **cohort, rank=rank, rank_tolerance=space['rank_tolerance'],
                singular_values=space['singular_values'].tolist(), floor_MSE=floor,
                actual_same_cohort_MSE=actual, floor_fraction=floor / actual if actual else None,
                prediction_space_residual_MSE=residual, prediction_space_residual_limit=residual_limit,
                lower_bound_tolerance=tolerance)


def _csv_new(path, rows, empty_fields=()):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(empty_fields)
    with Path(path).open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def _input_manifest(results, methods, include_d0):
    manifest = dict(algorithm_version=ALGORITHM_VERSION, algorithm_source_sha256=common.sha(__file__),
                    methods=list(methods), include_d0=include_d0,
                    reference_runs_sha256=common.sha(common.OLD_REPORT / 'RUNS.csv'), results=[])
    for result in results:
        job = result['job']
        candidates = [root / 'jobs' / job['job_id'] / 'RESULT.json' for root in (common.OLD_REPORT, common.REPORT)]
        paths = [path for path in candidates if path.is_file()]
        if len(paths) != 1:
            raise ValueError('Expected one saved input RESULT: ' + ', '.join(map(str, candidates)))
        path = paths[0]
        if common.read(path) != result:
            raise ValueError('Input RESULT object differs from saved bytes: ' + str(path))
        entry = dict(job_id=job['job_id'], result_path=str(path), result_sha256=common.sha(path),
                     result_content_digest=_digest(result), files=[])
        if job['method_id'] in ('C1', *methods):
            sources = [('prediction', result['row']['prediction_path'], result['row']['prediction_sha256']),
                       ('bundle', result['bundle_path'], result['bundle_sha256']),
                       ('selection', result['selection_path'], result['selection_sha256'])]
            for kind, filename, expected in sources:
                source = Path(filename)
                absent_allowed = kind == 'bundle' and job['method_id'] in ('C2', 'C3')
                actual = common.sha(source) if source.is_file() else None
                if (actual is None and not absent_allowed) or (actual is not None and actual != expected):
                    raise ValueError('Input ' + kind + ' missing or SHA changed: ' + str(source))
                entry['files'].append(dict(kind=kind, path=str(source), sha256=actual, expected_sha256=expected))
        manifest['results'].append(entry)
    return manifest


def _csv_types(rows):
    return [{key: type(value).__name__ for key, value in row.items()} for row in rows]


def _typed_csv(path, row_types):
    with Path(path).open(encoding='utf8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(row_types):
        raise ValueError('Sealed floor row count differs: ' + str(path))
    parsed = []
    for row, types in zip(rows, row_types):
        converted = {}
        for key, kind in types.items():
            value = row[key]
            if kind == 'NoneType':
                if value != '':
                    raise ValueError('Sealed null field differs: ' + str(path))
                converted[key] = None
            elif kind == 'bool':
                if value not in ('True', 'False'):
                    raise ValueError('Sealed boolean field differs: ' + str(path))
                converted[key] = value == 'True'
            elif kind in ('list', 'dict'):
                converted[key] = json.loads(value)
            elif kind in ('int', 'float', 'str'):
                converted[key] = {'int': int, 'float': float, 'str': str}[kind](value)
            else:
                raise ValueError('Unknown sealed field type in ' + str(path) + ': ' + kind)
        parsed.append(converted)
    return parsed


def _reuse(output, names, manifest):
    receipt_path = output / 'D0_D1_VALIDATION.json'
    missing = [output / name for name in names if not (output / name).is_file()]
    unexpected = [output / name for name in FLOOR_FILES if name not in names and (output / name).exists()]
    if missing or unexpected:
        raise FileExistsError('Partial or incompatible sealed floor output: '
                              + ', '.join(map(str, missing + unexpected)))
    receipt = common.read(receipt_path)
    if (receipt.get('status') != 'PASS' or receipt.get('input_manifest') != manifest
            or receipt.get('input_digest') != _digest(manifest)):
        raise ValueError('Sealed floor input/algorithm mismatch: ' + str(receipt_path))
    expected_csv = set(names) - {'D0_D1_VALIDATION.json'}
    if set(receipt.get('artifact_sha256', {})) != expected_csv or set(receipt.get('csv_types', {})) != expected_csv:
        raise ValueError('Sealed floor artifact manifest differs: ' + str(receipt_path))
    tables = {}
    for name in names:
        if name == 'D0_D1_VALIDATION.json':
            continue
        path = output / name
        if common.sha(path) != receipt['artifact_sha256'][name]:
            raise ValueError('Sealed floor artifact SHA changed: ' + str(path))
        tables[name] = _typed_csv(path, receipt['csv_types'][name])
    return dict(d0=tables.get('D0_SUBSPACE_FLOOR.csv', []), d1=tables['D1_DECODER_FLOOR.csv'],
                d1_summary=tables['D1_DECODER_FLOOR_SUMMARY.csv'], validation=receipt,
                artifacts={name: str(output / name) for name in names})


def _check_no_t(dataset, access):
    field = 'new_T_numeric_values_parsed' if dataset == 'jena' else 'T_numeric_values_parsed'
    if field not in access or access[field] != 0:
        raise ValueError('Missing or nonzero bounded-loader T receipt: ' + dataset + '/' + field)
    if dataset == 'jena':
        members = access.get('members')
        if not isinstance(members, list) or not members:
            raise ValueError('Missing Jena member-level bounded-loader receipts')
        for member in members:
            if 'tail_numeric_values_parsed' not in member or member['tail_numeric_values_parsed'] != 0:
                raise ValueError('Missing or nonzero Jena member tail receipt: ' + str(member.get('member')))


def _reference_rows(results):
    with (common.OLD_REPORT / 'RUNS.csv').open(encoding='utf8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['seed'] = int(row['seed'])
        for field in ('extended', 'cap_selected'):
            row[field] = row[field] == 'True'
        for field in ('MSE', 'MAE', 'MASE', 'R2', 'MAPE_nonzero_pct', 'MAPE_far_from_zero_pct', 'nearzero_fraction'):
            row[field] = float(row[field]) if row.get(field) else None
    lookup = {row['job_id']: row for row in rows}
    for result in results:
        lookup[result['row']['job_id']] = dict(result['row'])
    rows = list(lookup.values())
    summaries = analysis.choose_primary(rows)
    return {row['job_id']: row for row in rows}, {(row['dataset'], row['method_id']): row for row in summaries}


def _bound_arrays(result, task, scales, modules, reference=None):
    row = result['row']
    if row['normalization_fit_end'] != scales['fit_end']:
        raise ValueError('Saved prediction normalization was not fitted on A60')
    values = analysis.load_arrays(row, reference)
    for key in ('mean', 'std'):
        if key not in values or not np.array_equal(values[key], scales[key]):
            raise ValueError('Saved prediction A60 scaler mismatch: ' + key)
    expected = modules['data'].origins(task['split_ranges']['V_select'], task['L'], task['H'], task['H'])
    if not np.array_equal(values['origins'], expected):
        raise ValueError('Saved origins differ from the full registered V evaluation grid')
    bound = analysis.bind_physical_targets(values, task, scales)
    analysis.metrics(bound, scales, expected_mse=row['MSE'], expected_mae=row['MAE'])
    return bound


def _bundle(result, task, scales):
    import torch
    path = Path(result['bundle_path'])
    if not any(path.resolve().is_relative_to(root.resolve()) for root in (common.OLD_CACHE, common.CACHE)):
        raise ValueError('Selected bundle is outside the known campaign caches')
    if common.sha(path) != result['bundle_sha256']:
        raise ValueError('Selected bundle SHA changed: ' + str(path))
    seal_path = Path(result['selection_path'])
    if common.sha(seal_path) != result['selection_sha256']:
        raise ValueError('Saved selection seal SHA changed')
    seal = common.read(seal_path)
    if seal['status'] != 'SEALED' or seal['job'] != result['job'] or seal['V_used'] is not False:
        raise ValueError('Saved selection was not independently sealed before V')
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    if (bundle['spec'] != result['spec'] or bundle['contract'] != result['contract']
            or bundle['contract']['data_sha'] != task['data_sha256']
            or bundle['selection_sha256'] != result['selection_sha256']
            or bundle['selected_search_epoch'] != result['row']['selected_epoch']
            or bundle['refit_summary'] != result['refit_summary']
            or bundle['refit_summary']['status'] != 'SEGMENT_COMPLETE'
            or result['selection_sealed_before_V'] is not True):
        raise ValueError('Selected bundle is not the recorded fresh A60 refit')
    normalization = bundle['normalization']
    if normalization['fit_end'] != scales['fit_end'] or normalization['train_end'] != scales['fit_end']:
        raise ValueError('Selected bundle was not freshly normalized on A60')
    for key in ('mean', 'std'):
        if not np.array_equal(normalization[key], scales[key]):
            raise ValueError('Selected bundle A60 scaler mismatch: ' + key)
    return bundle


def _tensor(value):
    return value.detach().cpu().numpy()


def _d0(dataset, task, scales, modules, result, values, summaries):
    k = runtime.KS[dataset]
    pca = modules['data'].initialize_pca(task, scales['fit_end'], k, scales['mean'], scales['std'])
    bundle = _bundle(result, task, scales)
    if (not np.array_equal(_tensor(bundle['model']['E']), pca['E'])
            or not np.array_equal(_tensor(bundle['model']['center']), pca['pca_center'])):
        raise ValueError('C1 saved affine subspace differs from exact registered A60 PCA')
    weight, center = pca['E'].T.astype(np.float64), pca['pca_center'].astype(np.float64)
    diagnostic = affine_diagnostic(values, weight, center)
    basis = row_space(weight)['basis']
    filled = modules['data'].causal_forward_fill(task['physical'][:scales['fit_end']],
                                                task['observed'][:scales['fit_end']], scales['mean'])
    normalized = ((filled - scales['mean']) / scales['std']).astype(np.float64)
    a60_mean = normalized.mean(axis=0)
    a60_floor = float(np.square(outside(normalized, center, basis)).mean())
    target, _, _ = complete_cohort(values)
    vmean = target.mean(axis=0)
    centered_v = target - vmean
    singular_v = np.linalg.svd(centered_v, compute_uv=False)
    optimal_floor = float(np.square(singular_v[k:]).sum() / target.size)
    v_energy = float(np.square(target - center).mean())
    mean_shift = vmean - a60_mean
    shift_outside = outside(mean_shift, np.zeros_like(center), basis)
    center_shift = vmean - center
    center_shift_outside = outside(center_shift, np.zeros_like(center), basis)
    row = dict(dataset=dataset, K=k, C=target.shape[1], status='PASS',
               A60_fit_end=scales['fit_end'], A60_rows=len(normalized),
               A60_explained_variance=pca['explained_variance'], A60_floor_MSE=a60_floor,
               V_floor_MSE=diagnostic['floor_MSE'], V_energy_MSE=v_energy,
               V_energy_outside_ratio=diagnostic['floor_MSE'] / v_energy if v_energy else None,
               V_optimal_rankK_floor_MSE=optimal_floor,
               V_mean_shift_vs_A60_causalfilled_mean_outside_MSE=float(np.square(shift_outside).mean()),
               V_mean_shift_vs_A60_observed_scaler_mean_outside_MSE=float(np.square(outside(vmean, np.zeros_like(center), basis)).mean()),
               V_mean_shift_vs_PCA_center_outside_MSE=float(np.square(center_shift_outside).mean()),
               C1_actual_same_cohort_MSE=diagnostic['actual_same_cohort_MSE'],
               C1_floor_fraction=diagnostic['floor_fraction'],
               **{key: diagnostic[key] for key in ('total_points', 'complete_points', 'excluded_points',
                    'excluded_fraction', 'rank', 'rank_tolerance', 'singular_values',
                    'prediction_space_residual_MSE', 'prediction_space_residual_limit', 'lower_bound_tolerance')})
    for method in REFERENCES:
        summary = summaries[dataset, method]
        row[f'ref_{method}_MSE'] = summary['MSE_median']
        row[f'ref_{method}_n'] = summary['n']
        row[f'ref_{method}_extended'] = summary['extended']
    scaler_shift_outside = outside(vmean, np.zeros_like(center), basis)
    shifts = [dict(dataset=dataset, channel_id=str(name), A60_causalfilled_normalized_mean=float(a60_mean[i]),
                   A60_observed_scaler_normalized_mean=0.,
                   PCA_center=float(center[i]), V_complete_case_mean=float(vmean[i]),
                   V_minus_A60_causalfilled_mean=float(mean_shift[i]),
                   V_minus_A60_causalfilled_mean_outside=float(shift_outside[i]),
                   V_minus_A60_observed_scaler_mean=float(vmean[i]),
                   V_minus_A60_observed_scaler_mean_outside=float(scaler_shift_outside[i]),
                   V_minus_PCA_center=float(center_shift[i]),
                   V_minus_PCA_center_outside=float(center_shift_outside[i]))
              for i, name in enumerate(task['columns'])]
    return row, shifts


def summarize_d1(rows, methods):
    summaries = []
    for dataset in analysis.DATASETS:
        for method in methods:
            selected = [row for row in rows if row['dataset'] == dataset and row['method_id'] == method
                        and row['is_primary']]
            passed = [row for row in selected if row['status'] == 'PASS']
            skipped = [row for row in selected if row['status'] == 'SKIPPED_MISSING_BUNDLE']
            complete = len(passed) == 10 and not skipped
            row = dict(dataset=dataset, method_id=method, n=len(passed), expected_n=10,
                       complete=complete, skipped_n=len(skipped),
                       status='PASS' if complete else 'INCOMPLETE' if selected else 'NO_COMPLETE_RESULTS',
                       aggregation='median of primary per-job values; floor_fraction is median of ratios')
            for key in ('floor_MSE', 'actual_same_cohort_MSE', 'floor_fraction', 'excluded_fraction',
                        'prediction_space_residual_MSE'):
                finite = [item[key] for item in passed if item.get(key) is not None]
                row[key + '_median'] = float(np.median(finite)) if finite else None
            summaries.append(row)
    return summaries


def run(output_dir, methods=('C2', 'C3', 'A4'), include_d0=True, results=None):
    sys.dont_write_bytecode = True
    if len(set(methods)) != len(methods) or any(method not in ('C2', 'C3', 'A4') for method in methods):
        raise ValueError('D1 methods must be a unique subset of C2, C3, A4')
    output = Path(output_dir).resolve()
    if not output.is_relative_to(common.REPORT.resolve()):
        raise ValueError('Floor outputs must be within the new follow-up report namespace')
    names = ['D1_DECODER_FLOOR.csv', 'D1_DECODER_FLOOR_SUMMARY.csv', 'D0_D1_VALIDATION.json']
    if include_d0:
        names += ['D0_SUBSPACE_FLOOR.csv', 'D0_MEAN_SHIFT_CHANNELS.csv']
    preserved_before = common.guard_preserved()
    results = common.all_results() if results is None else list(results)
    if any(result.get('status') != 'COMPLETE' for result in results):
        raise ValueError('D0/D1 requires COMPLETE results only')
    if len({result['job']['job_id'] for result in results}) != len(results):
        raise ValueError('Duplicate RESULT job identity')
    manifest = _input_manifest(results, methods, include_d0)
    if any((output / name).exists() for name in FLOOR_FILES):
        return _reuse(output, names, manifest)
    common.install_write_guard()
    primary_rows, ref_summaries = _reference_rows(results)
    modules = runtime.load_modules()
    d0_rows, d1_rows, shifts, datasets = [], [], [], []
    for dataset in analysis.DATASETS:
        task = runtime.load_task(dataset)
        _check_no_t(dataset, task['_campaign_access'])
        scales = analysis.training_scales(task, runtime.FIT_ENDS[dataset][1])
        c1 = [result for result in results if result['job']['method_id'] == 'C1'
              and result['job']['dataset'] == dataset]
        if len(c1) != 1:
            raise ValueError('Exactly one saved C1 job per dataset is required')
        reference = _bound_arrays(c1[0], task, scales, modules)
        if include_d0:
            d0, channel_shifts = _d0(dataset, task, scales, modules, c1[0], reference, ref_summaries)
            d0_rows.append(d0)
            shifts.extend(channel_shifts)
        for result in results:
            job = result['job']
            if job['dataset'] != dataset or job['method_id'] not in methods:
                continue
            values = _bound_arrays(result, task, scales, modules, reference)
            base = dict(job, is_primary=primary_rows[job['job_id']]['is_primary'],
                        bundle_path=result['bundle_path'], bundle_sha256=result['bundle_sha256'],
                        prediction_path=result['row']['prediction_path'],
                        prediction_sha256=result['row']['prediction_sha256'])
            if not Path(result['bundle_path']).is_file() and job['method_id'] in ('C2', 'C3'):
                d1_rows.append(dict(base, status='SKIPPED_MISSING_BUNDLE'))
                continue
            bundle = _bundle(result, task, scales)
            if job['method_id'] in ('C2', 'C3'):
                weight, bias = _tensor(bundle['model']['decoder.weight']).T, _tensor(bundle['model']['decoder.bias'])
            else:
                weight, bias = _tensor(bundle['model']['W']), _tensor(bundle['model']['b'])
            if weight.shape != (job['K'], len(task['columns'])):
                raise ValueError('Saved decoder does not have the registered K/C shape')
            d1_rows.append(dict(base, **affine_diagnostic(values, weight, bias)))
        datasets.append(dict(dataset=dataset, data_sha256=task['data_sha256'], A60_fit_end=scales['fit_end'],
                             task_access=task['_campaign_access'],
                             C1_prediction_sha256=c1[0]['row']['prediction_sha256']))
    d1_summary = summarize_d1(d1_rows, methods)
    validation = dict(status='PASS', training_executed=False, foundation_loaded=False,
                      T_numeric_values_parsed=0, include_d0=include_d0, methods=list(methods),
                      cohort='origin × horizon points with observed.all(-1); all channels equally weighted',
                      numerical_tolerance=TOLERANCE_DESCRIPTION, datasets=datasets,
                      D0_datasets=len(d0_rows), D1_jobs=len(d1_rows),
                      D1_missing_bundles=[row['job_id'] for row in d1_rows if row['status'] != 'PASS'],
                      preserved_before=preserved_before, preserved_after=common.guard_preserved(),
                      algorithm_version=ALGORITHM_VERSION, input_manifest=manifest, input_digest=_digest(manifest))
    if _input_manifest(results, methods, include_d0) != manifest:
        raise ValueError('Floor input changed during computation: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    tables = {'D1_DECODER_FLOOR.csv': d1_rows, 'D1_DECODER_FLOOR_SUMMARY.csv': d1_summary}
    if include_d0:
        tables.update({'D0_SUBSPACE_FLOOR.csv': d0_rows, 'D0_MEAN_SHIFT_CHANNELS.csv': shifts})
    for name, rows in tables.items():
        _csv_new(output / name, rows, ('dataset', 'method_id', 'job_id', 'status'))
    validation['artifact_sha256'] = {name: common.sha(output / name) for name in tables}
    validation['csv_types'] = {name: _csv_types(rows) for name, rows in tables.items()}
    common.write_new(output / 'D0_D1_VALIDATION.json', validation)
    return dict(d0=d0_rows, d1=d1_rows, d1_summary=d1_summary, validation=validation,
                artifacts={name: str(output / name) for name in names})
