"""Preregistration-gated spectra using the unchanged training data functions.

Importing this module performs no file access, PCA or tau-to-K computation.
Only the campaign controller may call ``compute_spectra`` after committing the
preregistration. Existing artifact paths are never reused or overwritten.
"""
import csv
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from research.b2_revin_20261018_v1 import contracts


NAME = 'k_tau_rule_20261023_v1'
DATASETS = ('jena', 'ETTh1', 'ETTh2_safe17400', 'appliances')
TAUS = (.7, .8, .9, .95)
EXISTING_K = dict(jena=(3, 6, 12), ETTh1=(1, 2, 4),
                  ETTh2_safe17400=(1, 2, 4), appliances=(4, 7, 14))
SPECTRUM_FIELDS = ('dataset', 'fit_scope', 'k', 'eigenvalue', 'cumulative')
K_FIELDS = ('dataset', 'C', 'existing_K_candidates', 'tau', 'K_A_fit', 'K_A60',
            'cumulative_A_fit', 'cumulative_A60', 'compression_A_fit', 'compression_A60')
OUTPUTS = ('SPECTRUM.csv', 'K_TAU.csv', 'SPECTRUM_VALIDATION.json')


def _cumulative(eigenvalues):
    values = np.asarray(eigenvalues, dtype=np.float64)
    if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(np.diff(values) > 0)):
        raise ValueError('A finite nonnegative descending 1D eigenvalue spectrum is required')
    total = values.sum()
    if not math.isfinite(float(total)) or total <= 0:
        raise ValueError('Eigenvalue spectrum must have positive finite total variance')
    return np.cumsum(values) / total


def select_k(eigenvalues, tau):
    """Return the smallest k attaining tau; trailing zero eigenvalues are valid."""
    if not math.isfinite(float(tau)) or not 0 < tau <= 1:
        raise ValueError('tau must be finite and in (0, 1]')
    cumulative = _cumulative(eigenvalues)
    # At tau=1, different floating-point summation orders can put the last
    # cumulative value a few ulps below one; the full rank still explains all.
    return min(len(cumulative), int(np.searchsorted(cumulative, tau, side='left')) + 1)


def _git(repo, *args):
    result = subprocess.run(['git', *args], cwd=repo, capture_output=True, check=False)
    if result.returncode != 0:
        raise ValueError('Preregistration git evidence failed: ' + ' '.join(args)
                         + ': ' + result.stderr.decode('utf-8', errors='replace').strip())
    return result.stdout


def _verify_preregistration(repo, report, receipt):
    commit = receipt.get('commit', '')
    expected = receipt.get('preregistration_sha256', '')
    if (re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', str(commit)) is None
            or re.fullmatch(r'[0-9a-f]{64}', str(expected)) is None):
        raise ValueError('A full preregistration commit and SHA256 receipt are required')
    path = report / 'PREREGISTRATION.json'
    current = path.read_bytes()
    relative = path.relative_to(repo).as_posix()
    committed = _git(repo, 'show', commit + ':' + relative)
    if (hashlib.sha256(current).hexdigest() != expected
            or hashlib.sha256(committed).hexdigest() != expected or committed != current):
        raise ValueError('Preregistration committed/current/receipt bytes or SHA differ')
    _git(repo, 'merge-base', '--is-ancestor', commit, 'HEAD')
    commit_unix = int(_git(repo, 'show', '-s', '--format=%ct', commit).strip())
    if receipt.get('commit_unix') != commit_unix or not 0 < commit_unix <= time.time():
        raise ValueError('Preregistration commit time does not match its receipt or precede computation')
    return dict(status='PASS', commit=commit, preregistration_sha256=expected,
                commit_unix=commit_unix, ancestor_of_HEAD=True,
                committed_bytes_equal_current=True, checked_unix=time.time())


def _source_snapshot(contexts, *, enforce):
    snapshots = {}
    for dataset in DATASETS:
        ctx, entries = contexts[dataset], {}
        repo = Path(ctx['repo']).resolve()
        expected = ctx['scientific_source_hashes']
        data_path = Path(ctx['modules']['data'].__file__).resolve()
        if not data_path.is_relative_to(repo) or data_path.relative_to(repo).as_posix() not in expected:
            raise ValueError('Training data function source must be registered inside the repository')
        for name, digest in expected.items():
            path = (repo / name).resolve()
            if not path.is_relative_to(repo):
                raise ValueError('Registered scientific source escapes repository')
            actual = contracts.sha(path)
            if enforce and actual != digest:
                raise ValueError('Registered scientific source hash mismatch: ' + name)
            entries[name] = actual
        snapshots[dataset] = entries
    return snapshots


def _bounds(dataset, ctx):
    task = ctx['task']
    if task.get('dataset') != dataset or (task.get('L'), task.get('H')) != (512, 48):
        raise ValueError('G3 task identity or L512/H48 changed')
    previous, ranges = 0, task['split_ranges']
    for key in ('A_fit', 'A_stop', 'M_cal', 'V_select', 'T_final'):
        pair = ranges.get(key)
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(type(value) is not int for value in pair)
                or pair[0] != previous or pair[1] <= pair[0]):
            raise ValueError('G3 chronological split boundary changed: ' + key)
        previous = pair[1]
    rows, channels = task['physical'].shape
    if (rows != ranges['V_select'][1] or rows != ranges['T_final'][0]
            or task['observed'].shape != task['physical'].shape
            or channels != len(task['columns']) or channels < 1):
        raise ValueError('G3 task must materialize exactly the approved prefix before T')
    fit_end, refit_end = ranges['A_fit'][1], ranges['A_stop'][1]
    if fit_end < task['L'] + task['H']:
        raise ValueError('G3 A_fit must contain complete registered training horizons')
    module = ctx['modules']['data']
    if not all(callable(getattr(module, name, None)) for name in ('fit_statistics', 'initialize_pca')):
        raise ValueError('G9 actual training data functions are missing')
    return dict(dataset=dataset, data_sha256=task['data_sha256'], C=channels,
        A_fit=dict(rows_half_open=[0, fit_end], fit_end=fit_end,
            last_row_inclusive=fit_end - 1, selects_main_K=True),
        A60=dict(rows_half_open=[0, refit_end], fit_end=refit_end,
            last_row_inclusive=refit_end - 1, selects_main_K=False),
        V_rows_half_open=list(ranges['V_select']), T_rows_half_open=list(ranges['T_final']),
        L=task['L'], H=task['H'], materialized_prefix_rows=rows,
        A_fit_training_origins_inclusive=[task['L'], fit_end - task['H']],
        A_fit_latest_training_target_end_exclusive=fit_end,
        PCA_uses_standardized_raw_rows=True, PCA_uses_forecast_targets=False,
        V_numeric_rows_used_for_PCA=0, T_numeric_rows_used_for_PCA=0,
        explicit_proof='Both fit_end bounds precede V/T; unchanged training functions slice task rows [:fit_end].')


def _crosscheck(disclosure, computed, bounds):
    checks, errors = [], []
    records = disclosure.get('scalar_storage', {}).get('records', [])
    if not isinstance(records, list):
        raise ValueError('G2 prior scalar records must use the disclosed list schema')
    for dataset in DATASETS:
        selected = [row for row in records if row.get('dataset') == dataset]
        if not selected:
            checks.append(dict(dataset=dataset, status='NO_EXISTING_SCALAR_DISCLOSED'))
        for record in selected:
            k = record.get('K')
            valid = (type(k) is int and 1 <= k <= bounds[dataset]['C']
                     and record.get('fit_end') == bounds[dataset]['A60']['fit_end'])
            if not valid:
                errors.append(dict(dataset=dataset, K=k, error='Prior scalar K/fit_end differs from A60'))
                continue
            previous = float(record['explained_variance'])
            actual = float(computed[dataset]['A60'][k - 1])
            difference = abs(actual - previous)
            passed = math.isfinite(previous) and difference <= 1e-6
            check = dict(dataset=dataset, fit_scope='A60', K=k, fit_end=record['fit_end'],
                previous_explained_variance=previous, current_cumulative=actual,
                absolute_difference=difference, tolerance=1e-6, status='PASS' if passed else 'FAIL',
                scalar_provenance=record.get('path'), raw_bundle_opened=False)
            checks.append(check)
            if not passed:
                errors.append(dict(dataset=dataset, K=k, error='Existing A60 explained_variance mismatch',
                                   absolute_difference=difference))
    return dict(status='FAIL' if errors else 'PASS', checks=checks, errors=errors,
                historical_binary_arrays_opened=False)


def _new_csv(path, fields, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def compute_spectra(contexts, report_root, prereg_commit_receipt):
    """Compute A_fit/A60 spectra only after byte-bound ancestor commit checks.

    All four contexts come from ``data.build_context``. On a calculation/G2
    failure, completed raw spectrum rows and a failed validation receipt remain
    available. A new invocation refuses any existing requested output path.
    """
    if set(contexts) != set(DATASETS):
        raise ValueError('Exactly the four preregistered dataset contexts are required')
    repo = Path(contexts[DATASETS[0]]['repo']).resolve()
    report = Path(report_root).resolve()
    if (report != repo / 'reports' / NAME
            or any(Path(ctx['repo']).resolve() != repo for ctx in contexts.values())):
        raise ValueError('Spectrum outputs must use the shared campaign report namespace')
    for name in OUTPUTS:
        if (report / name).exists():
            raise FileExistsError('Existing spectrum artifact must not be overwritten: ' + str(report / name))
    validation = dict(status='FAILED', created_unix=None, G1=dict(status='NOT_CHECKED'),
        G2=dict(status='NOT_CHECKED', errors=[]), G3=dict(status='NOT_CHECKED'),
        source_hashes_pre={}, source_hashes_post={}, errors=[], T_access=False)
    spectra, ktau, computed = [], [], {}
    preflight_complete = False
    try:
        validation['G1'] = _verify_preregistration(repo, report, prereg_commit_receipt)
        validation['source_hashes_pre'] = _source_snapshot(contexts, enforce=True)
        bounds = {dataset: _bounds(dataset, contexts[dataset]) for dataset in DATASETS}
        validation['G3'] = dict(status='PASS', datasets=bounds)
        disclosure_path = report / 'PRIOR_PCA_DISCLOSURE.json'
        disclosure = contracts.read(disclosure_path)
        validation['prior_scalar_disclosure_sha256'] = contracts.sha(disclosure_path)
        preflight_complete = True
        validation['first_PCA_call_not_before_unix'] = time.time()
        for dataset in DATASETS:
            ctx, task = contexts[dataset], contexts[dataset]['task']
            module, channels = ctx['modules']['data'], bounds[dataset]['C']
            computed[dataset] = {}
            for scope in ('A_fit', 'A60'):
                fit_end = bounds[dataset][scope]['fit_end']
                mean, std, _ = module.fit_statistics(task, fit_end)
                initialized = module.initialize_pca(task, fit_end, 1, mean, std)
                values = np.asarray(initialized['eigenvalues'], dtype=np.float64)
                raw_rows = [dict(dataset=dataset, fit_scope=scope, k=k,
                    eigenvalue=float(value), cumulative='') for k, value in enumerate(values.reshape(-1), 1)]
                spectra.extend(raw_rows)
                cumulative = _cumulative(values)
                if len(values) != channels or initialized.get('fit_end') != fit_end:
                    raise ValueError('G2 training PCA eigenvalue count or fit_end differs from the task')
                for row, value in zip(raw_rows, cumulative):
                    row['cumulative'] = float(value)
                computed[dataset][scope] = cumulative
            # select_k receives the original spectra, not V scores or the A60 K.
            values_by_scope = {scope: [row['eigenvalue'] for row in spectra
                if row['dataset'] == dataset and row['fit_scope'] == scope] for scope in ('A_fit', 'A60')}
            for tau in TAUS:
                k_fit = select_k(values_by_scope['A_fit'], tau)
                k_refit = select_k(values_by_scope['A60'], tau)
                ktau.append(dict(dataset=dataset, C=channels,
                    existing_K_candidates=';'.join(map(str, EXISTING_K[dataset])), tau=tau,
                    K_A_fit=k_fit, K_A60=k_refit,
                    cumulative_A_fit=float(computed[dataset]['A_fit'][k_fit - 1]),
                    cumulative_A60=float(computed[dataset]['A60'][k_refit - 1]),
                    compression_A_fit='NO_COMPRESSION' if k_fit == channels else 'COMPRESSED',
                    compression_A60='NO_COMPRESSION' if k_refit == channels else 'COMPRESSED'))
        validation['G2'] = _crosscheck(disclosure, computed, bounds)
        if validation['G2']['errors']:
            raise ValueError('G2 existing scalar crosscheck failed; preserve raw spectra and stop')
        validation['source_hashes_post'] = _source_snapshot(contexts, enforce=False)
        validation['core_hashes_unchanged'] = validation['source_hashes_pre'] == validation['source_hashes_post']
        if not validation['core_hashes_unchanged']:
            raise ValueError('Scientific source hashes changed during spectrum computation')
        validation['status'] = 'COMPLETE'
    except BaseException as exc:
        validation['errors'].append(f'{type(exc).__name__}: {exc}')
        if preflight_complete:
            try:
                validation['source_hashes_post'] = _source_snapshot(contexts, enforce=False)
                validation['core_hashes_unchanged'] = validation['source_hashes_pre'] == validation['source_hashes_post']
            except Exception as post_error:
                validation['errors'].append('Post-failure source audit: ' + str(post_error))
        raise
    finally:
        validation['created_unix'], validation['created_utc'] = time.time(), contracts.utc()
        if preflight_complete:
            _new_csv(report / 'SPECTRUM.csv', SPECTRUM_FIELDS, spectra)
            _new_csv(report / 'K_TAU.csv', K_FIELDS, ktau)
            validation['output_creation_evidence'] = {
                name: dict(sha256=contracts.sha(report / name),
                    modified_unix=(report / name).stat().st_mtime,
                    created_after_commit=(report / name).stat().st_mtime >= validation['G1']['commit_unix'])
                for name in ('SPECTRUM.csv', 'K_TAU.csv')}
            contracts.sealed_json(report / 'SPECTRUM_VALIDATION.json', validation)
        else:
            # A unique failure receipt does not claim that spectra were computed.
            contracts.sealed_json(report / f'SPECTRUM_ERROR_{time.time_ns()}.json', validation)
    return dict(status='COMPLETE', K_main={row['dataset']: row['K_A_fit']
        for row in ktau if row['tau'] == .9}, spectra_rows=spectra,
        k_tau_rows=ktau, validation=validation)
