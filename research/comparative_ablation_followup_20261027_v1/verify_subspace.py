"""Independent FP64 least-squares check of final A4 decoder-floor reports.

This verifier never imports floors, creates a model, loads the foundation model,
or trains. Only a successful, complete check may create its one exclusive receipt.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
from research.comparative_ablation_20261025_v1 import runtime
from . import common


OUTPUT_NAME = 'FINAL_SUBSPACE_INDEPENDENT_CHECK.json'
ABS_TOL = 1e-10
REL_TOL = 1e-9
ALLOWED_UNFINISHED = {'NOT_RUN_BUDGET', 'FAILED', 'NOT_RUN_FAMILY_FAILURE',
                      'FAILED_ONCE_RETRY_NOT_RUN_BUDGET'}


def _close(actual, reported, label):
    if actual is None:
        if reported not in (None, ''):
            raise ValueError('Expected undefined value: ' + label)
        return 0.
    reported = float(reported)
    if not np.isfinite([actual, reported]).all() or abs(actual - reported) > ABS_TOL + REL_TOL * abs(reported):
        raise ValueError(f'Independent numeric mismatch {label}: actual={actual}, reported={reported}')
    return float(abs(actual - reported))


def _read_csv(path):
    with Path(path).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def _no_t(dataset, access):
    key = 'new_T_numeric_values_parsed' if dataset == 'jena' else 'T_numeric_values_parsed'
    if key not in access or access[key] != 0:
        raise ValueError('Missing or nonzero T access evidence: ' + dataset + '/' + key)
    if dataset == 'jena':
        if not access.get('members'):
            raise ValueError('Jena bounded-loader member receipts are absent')
        for member in access['members']:
            if member.get('tail_numeric_values_parsed') != 0:
                raise ValueError('Missing or nonzero Jena tail access: ' + str(member.get('member')))


def _validate_arrays(values, task, mean, std):
    p, y, mask = (values[key] for key in ('pred_normalized', 'target_normalized', 'observed'))
    if (p.dtype != np.float32 or y.dtype != np.float32 or mask.dtype != np.bool_
            or p.ndim != 3 or p.shape != y.shape or p.shape != mask.shape
            or p.shape[1:] != (task['H'], len(task['columns']))
            or not np.isfinite(p[mask]).all() or not np.isfinite(y[mask]).all()):
        raise ValueError('Saved FP32 prediction, target, mask, or shape contract differs')
    start, end = task['split_ranges']['V_select']
    expected_origins = np.arange(max(start, task['L']), end - task['H'] + 1, task['H'], dtype=np.int64)
    origins = values['origins']
    if (origins.shape != (p.shape[0],) or not np.issubdtype(origins.dtype, np.integer)
            or not np.array_equal(origins, expected_origins)
            or not np.array_equal(values['channel_ids'], np.asarray(task['columns']))
            or not np.array_equal(values['mean'], mean) or not np.array_equal(values['std'], std)):
        raise ValueError('Saved registered origins, channels, or A60 scaler differ')
    indices = origins[:, None] + np.arange(task['H'])
    if indices.size == 0 or int(indices.max()) >= task['split_ranges']['T_final'][0]:
        raise ValueError('Saved V target coordinates reach protected T or are empty')
    physical = np.asarray(task['physical'], np.float32)[indices]
    bound_mask = np.asarray(task['observed'], bool)[indices]
    normalized = np.where(bound_mask, (physical - mean) / std, 0).astype(np.float32)
    if not np.array_equal(mask, bound_mask) or not np.array_equal(y[mask], normalized[mask]):
        raise ValueError('Saved normalized target/mask differs from bounded physical V truth')
    counts = mask.sum((0, 1))
    if np.any(counts == 0):
        raise ValueError('A saved V channel has no observations')
    error = np.where(mask, p.astype(np.float64) - y.astype(np.float64), 0)
    return dict(MSE=float((np.square(error).sum((0, 1)) / counts).mean()),
                MAE=float((np.abs(error).sum((0, 1)) / counts).mean()))


def _least_squares_metrics(target, prediction, observed, weight, bias):
    """Use an independently solved affine least-squares residual, not a projector."""
    target, prediction = np.asarray(target, np.float64), np.asarray(prediction, np.float64)
    observed = np.asarray(observed)
    weight, bias = np.asarray(weight, np.float64), np.asarray(bias, np.float64)
    if (target.ndim != 3 or target.shape != prediction.shape or target.shape != observed.shape
            or observed.dtype != np.bool_ or weight.ndim != 2
            or weight.shape[1] != target.shape[-1] or bias.shape != (target.shape[-1],)
            or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
        raise ValueError('Independent least-squares axes or decoder parameters differ')
    complete = observed.all(axis=-1)
    if not complete.any():
        raise ValueError('No complete-channel target points remain')
    y, p = target[complete], prediction[complete]
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite complete-case predictions or targets')
    coefficients, _, rank, singular = np.linalg.lstsq(weight.T, (y - bias).T, rcond=None)
    target_residual = (y - bias).T - weight.T @ coefficients
    prediction_coefficients, _, prediction_rank, _ = np.linalg.lstsq(weight.T, (p - bias).T, rcond=None)
    if rank != prediction_rank:
        raise ValueError('Least-squares rank unexpectedly depends on the right-hand side')
    prediction_residual = (p - bias).T - weight.T @ prediction_coefficients
    floor = float(np.square(target_residual).mean())
    actual = float(np.square(p - y).mean())
    residual = float(np.square(prediction_residual).mean())
    operation_size = np.abs(prediction_coefficients.T) @ np.abs(weight) + np.abs(bias)
    scale = max(float(np.sqrt(np.square(array).mean())) for array in (p, bias, operation_size))
    residual_limit = float((32 * np.finfo(np.float32).eps * (weight.shape[0] + 2) * scale) ** 2)
    bound_tolerance = float(2 * np.sqrt(floor * residual) + residual
                           + 64 * np.finfo(np.float64).eps * max(floor, actual))
    if residual > residual_limit:
        raise ValueError('Saved prediction lies outside the independent decoder least-squares space')
    if actual + bound_tolerance < floor:
        raise ValueError('Independent affine lower-bound violation')
    return dict(rank=int(rank), singular_values=singular.tolist(), floor_MSE=floor,
                actual_same_cohort_MSE=actual, floor_fraction=floor / actual if actual else None,
                total_points=int(complete.size), complete_points=int(complete.sum()),
                excluded_points=int((~complete).sum()), excluded_fraction=float((~complete).mean()),
                prediction_space_residual_MSE=residual, prediction_space_residual_limit=residual_limit,
                lower_bound_tolerance=bound_tolerance)


def _summaries(checks, reported_rows):
    summaries = []
    for dataset in common.DATASETS:
        group = [row for row in checks if row['dataset'] == dataset]
        candidates = [row for row in reported_rows if row['dataset'] == dataset and row['method_id'] == 'A4']
        if len(candidates) != 1:
            raise ValueError('Expected one final A4 floor summary: ' + dataset)
        reported = candidates[0]
        n = len(group)
        if int(reported['n']) != n or int(reported['skipped_n']) != 0 or (reported['complete'] == 'True') != (n == 10):
            raise ValueError('Final A4 summary completeness differs: ' + dataset)
        result = dict(dataset=dataset, n=n, expected_n=10)
        for key in ('floor_MSE', 'actual_same_cohort_MSE', 'floor_fraction', 'excluded_fraction', 'prediction_space_residual_MSE'):
            finite = [row[key] for row in group if row[key] is not None]
            value = float(np.median(finite)) if finite else None
            result[key + '_median'] = value
            result[key + '_median_abs_diff'] = _close(value, reported[key + '_median'], dataset + '/' + key + '_median')
        summaries.append(result)
    return summaries


def _execution_scope(execution, jobs):
    status = execution.get('status')
    if status in ('STARTED', 'STOPPED'):
        raise RuntimeError('Execution state requires root action before verification: ' + status)
    if status not in ('COMPLETE', 'PARTIAL'):
        return dict(status='INCOMPLETE', reason='Execution has not reached terminal COMPLETE/PARTIAL',
                    execution_status=status)
    complete, omitted = [], []
    statuses = execution.get('run_statuses', {})
    for job in jobs:
        job_id = job['job_id']
        state = statuses.get(job_id)
        if state in ('STARTED', 'STOPPED', 'FAILED_ONCE_CONTROLLER_STOPPED'):
            raise RuntimeError('A4 job state requires root action before verification: ' + job_id + '/' + state)
        if state in ('RUNNING', 'FAILED_ONCE'):
            return dict(status='INCOMPLETE', reason='A4 execution or registered retry remains unresolved',
                        execution_status=status, job_id=job_id, job_status=state)
        if state == 'COMPLETE':
            complete.append(job_id)
        elif state in ALLOWED_UNFINISHED:
            omitted.append(dict(job_id=job_id, status=state))
        else:
            raise ValueError('Missing or unregistered terminal A4 status: ' + job_id + '/' + str(state))
    if status == 'COMPLETE' and omitted:
        raise ValueError('COMPLETE execution contradicts unfinished A4 jobs')
    return dict(status='READY', execution_status=status, complete_job_ids=complete,
                not_completed_jobs=omitted, partial_scope=bool(omitted))


def _match_completed_scope(scope, result_ids, csv_ids):
    expected = set(scope['complete_job_ids'])
    missing = expected - set(result_ids)
    missing_csv = expected - set(csv_ids)
    omitted = {row['job_id'] for row in scope['not_completed_jobs']}
    contradictory = (set(result_ids) | set(csv_ids)) & omitted
    unexpected_csv = set(csv_ids) - expected
    if missing or missing_csv or contradictory or unexpected_csv:
        raise ValueError('Completed final scope mismatch: '
                         + json.dumps(dict(missing_COMPLETE_RESULT=sorted(missing),
                                           missing_COMPLETE_CSV=sorted(missing_csv),
                                           noncomplete_with_result_or_csv=sorted(contradictory),
                                           unexpected_CSV=sorted(unexpected_csv))))


def run(check_only=False):
    output = common.REPORT / OUTPUT_NAME
    if not check_only and output.exists():
        raise FileExistsError('Independent verification receipt already exists: ' + str(output))
    preserved_before = common.guard_preserved()
    common.install_write_guard()
    common.validate_admission()
    watched = {}

    def watch(path, expected=None):
        path = Path(path).resolve()
        actual = common.sha(path)
        if expected is not None and actual != expected:
            raise ValueError('Verification input SHA changed: ' + str(path))
        if str(path) in watched and watched[str(path)] != actual:
            raise ValueError('Verification input changed during inspection: ' + str(path))
        watched[str(path)] = actual
        return actual

    jobs = [job for job in common.selected_jobs() if job['method_id'] == 'A4']
    execution_path = common.REPORT / 'EXECUTION_COMPLETION.json'
    if not execution_path.is_file():
        return dict(status='INCOMPLETE', missing=[str(execution_path)], reason='Execution is not finalized',
                    jobs_checked=0, expected_jobs=40, receipt_written=False)
    execution = common.read(execution_path)
    scope = _execution_scope(execution, jobs)
    if scope['status'] == 'INCOMPLETE':
        return dict(scope, jobs_checked=0, expected_jobs=40, receipt_written=False)
    watch(execution_path)
    required = [common.REPORT / name for name in ('REPORTING_COMPLETION.json', 'REPORTING_INPUTS.json',
                'D0_D1_VALIDATION.json', 'D1_DECODER_FLOOR.csv', 'D1_DECODER_FLOOR_SUMMARY.csv')]
    if not required[0].is_file():
        return dict(status='INCOMPLETE', missing=[str(required[0])], jobs_checked=0, expected_jobs=40,
                    T_numeric_values_parsed=0, training_executed=False, receipt_written=False)
    receipt = common.read(required[0])
    if receipt.get('status') != 'COMPLETE' or receipt.get('stage') != 'final':
        return dict(status='INCOMPLETE', reason='Final reporting is not COMPLETE', path=str(required[0]),
                    jobs_checked=0, expected_jobs=40, receipt_written=False)
    missing = [str(path) for path in required[1:] if not path.is_file()]
    if missing:
        raise ValueError('Completed final report is missing required floor artifacts: ' + ', '.join(missing))
    for name, expected in receipt['outputs'].items():
        path = (common.REPORT / name).resolve()
        if not path.is_relative_to(common.REPORT.resolve()):
            raise ValueError('Final reporting artifact escapes its namespace: ' + str(path))
        if not path.is_file():
            missing.append(str(path))
        else:
            watch(path, expected)
    if missing:
        raise ValueError('Completed final report is missing sealed output artifacts: ' + ', '.join(missing))
    watch(required[0])
    for path in required[1:]:
        if path.name not in receipt['outputs']:
            raise ValueError('Required floor artifact is absent from final seal: ' + str(path))
    inputs = common.read(common.REPORT / 'REPORTING_INPUTS.json')
    reported_scope = _execution_scope(inputs['execution_status'], jobs)
    if reported_scope != scope:
        raise ValueError('Final reporting execution snapshot differs from terminal A4 run statuses')
    missing_results = [str(common.OLD_REPORT / 'jobs' / row['job_id'] / 'RESULT.json')
                       for row in inputs['jobs']
                       if not (common.OLD_REPORT / 'jobs' / row['job_id'] / 'RESULT.json').is_file()]
    if missing_results:
        raise ValueError('Completed final report is missing sealed RESULT inputs: ' + ', '.join(missing_results))
    all_results = common.all_results()
    digest = hashlib.sha256(json.dumps(all_results, sort_keys=True, ensure_ascii=False,
                                       separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()
    if digest != receipt['input_results_sha256']:
        raise ValueError('Final report RESULT snapshot differs from current COMPLETE results')
    if inputs.get('stage') != 'final' or inputs['input_results_sha256'] != digest:
        raise ValueError('Final report input snapshot does not match its completion seal')
    for result in all_results:
        path = common.OLD_REPORT / 'jobs' / result['job']['job_id'] / 'RESULT.json'
        if common.read(path) != result:
            raise ValueError('RESULT changed after reading the final input snapshot: ' + str(path))
        watch(path)
    prior_path = common.REPORT / 'preflight' / 'D0_D1_INDEPENDENT_CHECK.json'
    if not prior_path.is_file():
        raise ValueError('Prior independent verification receipt is missing: ' + str(prior_path))
    prior = common.read(prior_path)
    if prior.get('status') != 'PASS' or prior.get('jobs_checked') != 84 or prior.get('T_numeric_values_parsed') != 0:
        raise ValueError('Prior independent C1/C2/C3 verification is not a complete PASS: ' + str(prior_path))
    watch(prior_path)
    floor_receipt = common.read(common.REPORT / 'D0_D1_VALIDATION.json')
    if floor_receipt.get('status') != 'PASS' or floor_receipt.get('T_numeric_values_parsed') != 0:
        raise ValueError('Final floor validation did not pass its no-T contract')
    for name in ('D1_DECODER_FLOOR.csv', 'D1_DECODER_FLOOR_SUMMARY.csv'):
        watch(common.REPORT / name, floor_receipt['artifact_sha256'][name])
    reported = [row for row in _read_csv(common.REPORT / 'D1_DECODER_FLOOR.csv') if row['method_id'] == 'A4']
    if len({row['job_id'] for row in reported}) != len(reported):
        raise ValueError('Duplicate A4 job in final decoder-floor CSV')
    by_id = {row['job_id']: row for row in reported}
    if set(by_id) - {job['job_id'] for job in jobs}:
        raise ValueError('Final floor CSV contains an unregistered A4 job')
    completed = {result['job']['job_id']: result for result in all_results}
    _match_completed_scope(scope, completed, by_id)
    available = set(scope['complete_job_ids'])
    modules = runtime.load_modules() if available else None
    tasks, references, checks = {}, {}, []
    import torch
    for job in jobs:
        job_id, dataset = job['job_id'], job['dataset']
        if job_id not in available:
            continue
        result, row = completed[job_id], by_id[job_id]
        if (result['status'] != 'COMPLETE' or result['job'] != job or row['status'] != 'PASS'
                or row['is_primary'] != 'True' or row['extended'] != 'False'
                or row['dataset'] != dataset or int(row['seed']) != job['seed'] or int(row['K']) != job['K']):
            raise ValueError('Final A4 result/CSV job identity differs: ' + job_id)
        result_path = common.OLD_REPORT / 'jobs' / job_id / 'RESULT.json'
        watch(result_path)
        _no_t(dataset, result['task_access'])
        if dataset not in tasks:
            task = runtime.load_task(dataset)
            _no_t(dataset, task['_campaign_access'])
            endpoint = runtime.FIT_ENDS[dataset][1]
            if task['split_ranges']['A_stop'][1] != endpoint or len(task['physical']) != task['split_ranges']['T_final'][0]:
                raise ValueError('Bounded task A60/T endpoint mismatch: ' + dataset)
            mean, std, _ = modules['data'].fit_statistics(task, endpoint)
            tasks[dataset] = task, endpoint, mean, std
        task, endpoint, mean, std = tasks[dataset]
        saved = result['row']
        if any(saved[key] != job[key] for key in ('job_id', 'dataset', 'method_id', 'K', 'seed', 'extended')):
            raise ValueError('RESULT row identity differs from registered A4 job: ' + job_id)
        if saved['normalization_fit_end'] != endpoint:
            raise ValueError('Saved A4 forecast is not A60-normalized: ' + job_id)
        for path_key, sha_key, owner in [('prediction_path', 'prediction_sha256', saved),
                                         ('bundle_path', 'bundle_sha256', result),
                                         ('selection_path', 'selection_sha256', result)]:
            path = Path(owner[path_key]).resolve()
            root = common.OLD_REPORT if path_key == 'selection_path' else common.OLD_CACHE
            if not path.is_relative_to(root.resolve()):
                raise ValueError('Saved A4 artifact escapes registered namespace: ' + str(path))
            watch(path, owner[sha_key])
            if path_key != 'selection_path' and (row[path_key] != owner[path_key] or row[sha_key] != owner[sha_key]):
                raise ValueError('Final floor CSV uses a different saved artifact: ' + job_id)
        with np.load(saved['prediction_path'], allow_pickle=False) as archive:
            values = {key: archive[key] for key in ('pred_normalized', 'target_normalized', 'observed',
                                                   'origins', 'channel_ids', 'mean', 'std')}
        global_metrics = _validate_arrays(values, task, mean, std)
        for metric in ('MSE', 'MAE'):
            _close(global_metrics[metric], saved[metric], job_id + '/recorded_' + metric)
        if dataset in references:
            for key in ('target_normalized', 'observed', 'origins', 'channel_ids', 'mean', 'std'):
                if not np.array_equal(values[key], references[dataset][key]):
                    raise ValueError('A4 seed cohorts differ: ' + dataset + '/' + key)
        else:
            references[dataset] = values
        seal = common.read(result['selection_path'])
        bundle = torch.load(result['bundle_path'], map_location='cpu', weights_only=False)
        norm = bundle['normalization']
        if (seal.get('status') != 'SEALED' or seal['job'] != job or seal['V_used'] is not False
                or result['selection_sealed_before_V'] is not True
                or bundle['spec'] != result['spec'] or bundle['contract'] != result['contract']
                or bundle['contract']['data_sha'] != task['data_sha256']
                or bundle['selection_sha256'] != result['selection_sha256']
                or bundle['selected_search_epoch'] != saved['selected_epoch']
                or bundle['refit_summary'] != result['refit_summary']
                or bundle['refit_summary']['status'] != 'SEGMENT_COMPLETE'
                or norm['fit_end'] != endpoint or norm['train_end'] != endpoint
                or not np.array_equal(norm['mean'], mean) or not np.array_equal(norm['std'], std)):
            raise ValueError('Selected A4 bundle is not the sealed fresh A60 refit: ' + job_id)
        weight = bundle['model']['W'].detach().cpu().numpy()
        bias = bundle['model']['b'].detach().cpu().numpy()
        if weight.shape != (job['K'], len(task['columns'])):
            raise ValueError('A4 decoder K/C shape differs: ' + job_id)
        metrics = _least_squares_metrics(values['target_normalized'], values['pred_normalized'], values['observed'], weight, bias)
        if any(int(row[key]) != metrics[key] for key in ('rank', 'total_points', 'complete_points', 'excluded_points')):
            raise ValueError('Independent rank/cohort counts differ: ' + job_id)
        differences = {key + '_abs_diff': _close(metrics[key], row[key], job_id + '/' + key)
                       for key in ('floor_MSE', 'actual_same_cohort_MSE', 'floor_fraction', 'excluded_fraction')}
        checks.append(dict(job_id=job_id, dataset=dataset, seed=job['seed'], status='PASS', **metrics, **differences))
    summaries = _summaries(checks, _read_csv(common.REPORT / 'D1_DECODER_FLOOR_SUMMARY.csv'))
    for path, expected in watched.items():
        if common.sha(path) != expected:
            raise ValueError('Verification input changed during independent computation: ' + path)
    preserved_after = common.guard_preserved()
    result = dict(status='PASS_AVAILABLE_SCOPE' if scope['partial_scope'] else 'PASS',
                  independent_method='FP64 numpy.linalg.lstsq(W.T, (target-b).T), rcond=None',
                  scope='Every terminal COMPLETE A4 selected A60 decoder; explicitly unfinished jobs are listed and were not verified',
                  execution_status=scope['execution_status'], partial_scope=scope['partial_scope'],
                  jobs_checked=len(checks), expected_jobs=40, available_complete_jobs=len(available),
                  not_completed_jobs=scope['not_completed_jobs'], checks=checks, summaries=summaries,
                  max_abs_floor_difference=max((row['floor_MSE_abs_diff'] for row in checks), default=None),
                  max_abs_actual_difference=max((row['actual_same_cohort_MSE_abs_diff'] for row in checks), default=None),
                  max_abs_ratio_difference=max((row['floor_fraction_abs_diff'] for row in checks), default=None),
                  comparison_tolerance=dict(absolute=ABS_TOL, relative=REL_TOL,
                      rationale='Fixed strict tolerance for two FP64 least-squares/projection reductions and decimal CSV round trips'),
                  source_sha256=common.sha(__file__), input_files_sha256=watched,
                  prior_independent_check=dict(path=str(prior_path), sha256=watched[str(prior_path.resolve())],
                                               jobs_checked=prior['jobs_checked'],
                                               max_abs_floor_difference=prior['max_abs_floor_difference']),
                  preservation_before=preserved_before, preservation_after=preserved_after,
                  T_numeric_values_parsed=0, training_executed=False, foundation_loaded=False,
                  receipt_written=not check_only)
    if not check_only:
        common.write_new(output, result)
    return result


def self_test():
    import unittest

    class SyntheticTests(unittest.TestCase):
        def metrics(self, target, prediction, weight, bias, observed=None):
            target, prediction = np.asarray(target, np.float32)[None], np.asarray(prediction, np.float32)[None]
            observed = np.ones(target.shape, bool) if observed is None else np.asarray(observed, bool)[None]
            return _least_squares_metrics(target, prediction, observed, weight, bias)

        def test_affine_offset_rank_deficiency_and_scale(self):
            y, p = np.asarray([[9., 14., 17.], [3., 8., 9.]]), np.asarray([[9., 11., 13.], [3., 11., 13.]])
            w, b = np.asarray([[2., 0., 0.], [4., 0., 0.]]), np.asarray([7., 11., 13.])
            for scale in (.125, 1., 32.):
                row = self.metrics(y * scale, p * scale, w * scale, b * scale)
                self.assertEqual(row['rank'], 1)
                self.assertAlmostEqual(row['floor_MSE'], 25 / 3 * scale ** 2)
                self.assertAlmostEqual(row['floor_fraction'], 1.)

        def test_partial_mask_excludes_whole_point(self):
            row = self.metrics([[2., 3.], [800., 1000.], [4., 5.]], [[2., 0.], [-900., -1000.], [4., 0.]],
                               [[1., 0.]], [0., 0.], [[True, True], [True, False], [True, True]])
            self.assertEqual(row['complete_points'], 2)
            self.assertEqual(row['excluded_points'], 1)
            self.assertAlmostEqual(row['floor_MSE'], 8.5)

        def test_impossible_below_floor_prediction_stops(self):
            with self.assertRaisesRegex(ValueError, 'outside|lower-bound'):
                self.metrics([[3., 4.], [5., 6.]], [[3., 4.], [5., 6.]], [[1., 0.]], [0., 0.])

        def test_fp32_affine_rounding(self):
            rng = np.random.default_rng(852)
            w, b = rng.normal(size=(3, 8)).astype(np.float32), rng.normal(size=8).astype(np.float32)
            latent = rng.normal(size=(64, 3)).astype(np.float32)
            row = self.metrics(rng.normal(size=(64, 8)), latent @ w + b, w, b)
            self.assertLess(row['prediction_space_residual_MSE'], row['prediction_space_residual_limit'])
            self.assertGreaterEqual(row['actual_same_cohort_MSE'] + row['lower_bound_tolerance'], row['floor_MSE'])

        def test_target_identity_scaler_origins_and_no_t(self):
            physical = np.asarray([[9., 19.], [11., 21.], [9., 21.], [11., 19.],
                                   [12., 23.], [10., 24.], [14., 20.], [15., 25.]], np.float32)
            mean, std = np.asarray([10., 20.], np.float32), np.ones(2, np.float32)
            y = ((physical[4:] - mean) / std).reshape(2, 2, 2)
            task = dict(physical=physical, observed=np.ones(physical.shape, bool), columns=['a', 'b'], L=2, H=2,
                        split_ranges={'V_select': (4, 8), 'T_final': (8, 10)})
            values = dict(target_normalized=y, pred_normalized=y + 2, observed=np.ones(y.shape, bool),
                          origins=np.asarray([4, 6]), channel_ids=np.asarray(['a', 'b']), mean=mean, std=std)
            self.assertEqual(_validate_arrays(values, task, mean, std)['MSE'], 4.)
            for key, value in [('mean', mean + 1), ('origins', np.asarray([4, 5])), ('channel_ids', np.asarray(['b', 'a'])),
                               ('target_normalized', y + 1), ('observed', np.zeros(y.shape, bool))]:
                with self.assertRaises(ValueError):
                    _validate_arrays(dict(values, **{key: value}), task, mean, std)
            _no_t('jena', dict(new_T_numeric_values_parsed=0, members=[dict(tail_numeric_values_parsed=0)]))
            with self.assertRaises(ValueError):
                _no_t('jena', dict(T_numeric_values_parsed=0))

        def test_report_comparison_rejects_changed_value(self):
            self.assertEqual(_close(.5, '.5', 'ratio'), 0)
            with self.assertRaisesRegex(ValueError, 'numeric mismatch'):
                _close(.5, '.6', 'ratio')

        def test_summary_compares_median_of_ratios(self):
            checks, reported = [], []
            for dataset in common.DATASETS:
                checks.extend(dict(dataset=dataset, floor_MSE=floor, actual_same_cohort_MSE=actual,
                                   floor_fraction=floor / actual, excluded_fraction=0., prediction_space_residual_MSE=0.)
                              for floor, actual in ((1., 2.), (2., 100.), (3., 3.)))
                reported.append(dict(dataset=dataset, method_id='A4', n='3', skipped_n='0', complete='False',
                                     floor_MSE_median='2.', actual_same_cohort_MSE_median='3.', floor_fraction_median='.5',
                                     excluded_fraction_median='0.', prediction_space_residual_MSE_median='0.'))
            summaries = _summaries(checks, reported)
            self.assertEqual(summaries[0]['floor_fraction_median'], .5)
            reported[0]['floor_fraction_median'] = str(2 / 3)
            with self.assertRaisesRegex(ValueError, 'numeric mismatch'):
                _summaries(checks, reported)

        def test_terminal_partial_accepts_only_documented_available_scope(self):
            jobs = [{'job_id': name} for name in ('a', 'b', 'c')]
            execution = dict(status='PARTIAL', run_statuses={'a': 'COMPLETE', 'b': 'NOT_RUN_BUDGET', 'c': 'FAILED'})
            scope = _execution_scope(execution, jobs)
            self.assertEqual(scope['status'], 'READY')
            self.assertEqual(scope['complete_job_ids'], ['a'])
            self.assertTrue(scope['partial_scope'])
            self.assertEqual(len(scope['not_completed_jobs']), 2)
            _match_completed_scope(scope, {'a'}, {'a'})
            for result_ids, csv_ids in [({'a'}, set()), (set(), {'a'}), ({'a', 'b'}, {'a'}), ({'a'}, {'a', 'c'})]:
                with self.assertRaisesRegex(ValueError, 'scope mismatch'):
                    _match_completed_scope(scope, result_ids, csv_ids)
            with self.assertRaisesRegex(ValueError, 'unregistered terminal'):
                _execution_scope(dict(status='PARTIAL', run_statuses={'a': 'COMPLETE'}), jobs)
            with self.assertRaisesRegex(ValueError, 'contradicts unfinished'):
                _execution_scope(dict(execution, status='COMPLETE'), jobs)

        def test_live_and_stopped_execution_cannot_be_sealed(self):
            jobs = [{'job_id': 'a'}]
            self.assertEqual(_execution_scope(dict(status='RUNNING'), jobs)['status'], 'INCOMPLETE')
            self.assertEqual(_execution_scope(dict(status='PARTIAL', run_statuses={'a': 'RUNNING'}), jobs)['status'], 'INCOMPLETE')
            for status in ('STARTED', 'STOPPED'):
                with self.assertRaisesRegex(RuntimeError, 'root action'):
                    _execution_scope(dict(status=status), jobs)
            with self.assertRaisesRegex(RuntimeError, 'root action'):
                _execution_scope(dict(status='PARTIAL', run_statuses={'a': 'FAILED_ONCE_CONTROLLER_STOPPED'}), jobs)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests))
    return result.wasSuccessful()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true', help='Read and verify final data; print without creating a receipt')
    mode.add_argument('--self-test', action='store_true', help='Run pure synthetic checks only, without loading campaign data')
    args = parser.parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1
    try:
        result = run(check_only=args.check_only)
    except Exception as exc:
        print(json.dumps(dict(status='FAILED', error=f'{type(exc).__name__}: {exc}', receipt_written=False), ensure_ascii=False), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result['status'] in ('PASS', 'PASS_AVAILABLE_SCOPE') else 2


if __name__ == '__main__':
    raise SystemExit(main())
