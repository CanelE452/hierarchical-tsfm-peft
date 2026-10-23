"""Synthetic public-table fixtures; no campaign data or model is loaded."""
import csv
import json

import numpy as np
import pytest

from research.k_tau_rule_20261023_v1 import analysis


def write_csv(path, rows, columns=None):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def public_tables(tmp_path):
    baseline, ktau, runs = [], [], []
    for dataset in analysis.DATASETS:
        baseline.append(dict(dataset=dataset, family='RAW', K=8, MSE_mean=1.,
                             MSE_min=1., MSE_max=1., MAE_mean=.5))
        for k, mse in [(1, .9), (2, .8), (4, .7)]:
            if dataset == 'ETTh2_safe17400':
                mse += .5
            baseline.append(dict(dataset=dataset, family='B2', K=k, MSE_mean=mse,
                                 MSE_min=mse-.01, MSE_max=mse+.01, MAE_mean=.4))
        ktau.append(dict(dataset=dataset, tau=.9, K_A_fit=2, K_A60=4, C=8))
        for seed in analysis.SEEDS:
            runs.append(dict(dataset=dataset, K=2, seed=seed, mse=.6, mae=.4,
                             reused=seed < 92603, role='main', job_id=f'{dataset}_{seed}',
                             search_seconds=2., refit_seconds=1.))
    write_csv(tmp_path/'BASELINE_RECALCULATION.csv', baseline)
    write_csv(tmp_path/'K_TAU.csv', ktau)
    write_csv(tmp_path/'RUNS.csv', runs)
    return tmp_path, baseline, runs


def report_row(root, dataset):
    with (root/'SUMMARY.csv').open(encoding='utf-8') as handle:
        return next(row for row in csv.DictReader(handle) if row['dataset'] == dataset)


def test_fixed_K_linear_quantiles_loss_and_figures(public_tables):
    root, _, runs = public_tables
    values = [.8, .8, .2, .2, .2, .2, .2, .2, .2, .2]
    for row, value in zip([r for r in runs if r['dataset'] == 'jena'], values):
        row['mse'] = value
    write_csv(root/'RUNS.csv', runs)
    decision = analysis.build_report(root)
    row = report_row(root, 'jena')
    assert decision['decision'] == 'ADOPT'
    assert row['K_main'] == '2'  # A60 or V scores never replace A_fit K.
    assert float(row['Q1']) == pytest.approx(np.percentile(values, 25, method='linear'))
    assert float(row['Q3']) == pytest.approx(np.percentile(values, 75, method='linear'))
    assert float(row['loss']) == pytest.approx(.1)
    assert row['limitation'] == 'LIMITATION'
    assert row['success'] == 'True'
    for name in ['FIG_K_CURVE.png', 'FIG_K_CURVE.svg', 'FIG_BOXPLOT.png', 'FIG_BOXPLOT.svg']:
        assert (root/name).stat().st_size > 100
    text = (root/'RESULTS_KO.md').read_text(encoding='utf-8')
    assert '신뢰구간' in text and '반복 개발' in text


@pytest.mark.parametrize(('failures', 'expected'), [(1, 'ADOPT_CONDITIONAL'), (2, 'REPLAN'), (3, 'REPLAN')])
def test_strict_median_and_three_dataset_decision(public_tables, failures, expected):
    root, _, runs = public_tables
    failed = analysis.ELIGIBLE_DATASETS[:failures]
    for row in runs:
        if row['dataset'] in failed or row['dataset'] == 'ETTh2_safe17400':
            row['mse'] = 1.  # Exact Raw tie is failure; ETTh2 does not enter verdict.
    write_csv(root/'RUNS.csv', runs)
    decision = analysis.build_report(root)
    assert decision['decision'] == expected
    assert len(decision['failed_datasets']) == failures
    assert report_row(root, 'ETTh2_safe17400')['decision_scope'] == 'DESCRIPTIVE_ONLY'


def test_comparison_rows_override_historical_baseline(public_tables):
    root, _, runs = public_tables
    for k, value in [(1, .3), (2, .6), (4, .4)]:
        for seed in analysis.SEEDS[:2]:
            runs.append(dict(dataset='jena', K=k, seed=seed, mse=value, mae=.2,
                             reused=False, role='comparison', job_id=f'compare_{k}_{seed}',
                             search_seconds=1., refit_seconds=1.))
    write_csv(root/'RUNS.csv', runs)
    analysis.build_report(root)
    row = report_row(root, 'jena')
    assert float(row['loss']) == pytest.approx(.3)
    assert row['baseline_source'] == 'CURRENT_COMPARISON'


def test_better_main_loss_zero_and_no_limit_at_exact_IQR(public_tables):
    root, baseline, runs = public_tables
    for row in baseline:
        if row['dataset'] == 'jena' and row['family'] == 'B2':
            row.update(MSE_mean=.6, MSE_min=.6, MSE_max=.6)
    write_csv(root/'BASELINE_RECALCULATION.csv', baseline)
    analysis.build_report(root)
    row = report_row(root, 'jena')
    assert float(row['loss']) == 0.
    assert row['limitation'] == ''
    for row in runs:
        if row['dataset'] == 'jena':
            row['mse'] = .5
    write_csv(root/'RUNS.csv', runs)
    analysis.build_report(root)
    row = report_row(root, 'jena')
    assert float(row['loss']) == 0.
    assert row['main_improves_historical_best'] == 'True'


def test_missing_seed_and_incomplete_comparison_forbid_adopt(public_tables):
    root, _, runs = public_tables
    runs = [r for r in runs if not (r['dataset'] == 'jena' and r['seed'] == 92610)]
    runs.append(dict(dataset='appliances', K=1, seed=92601, mse=.2, mae=.1,
                     reused=False, role='comparison', job_id='partial_compare',
                     search_seconds=1., refit_seconds=1.))
    write_csv(root/'RUNS.csv', runs)
    decision = analysis.build_report(root)
    assert decision['decision'] == 'PARTIAL'
    assert decision['complete'] is False
    assert decision['datasets']['jena']['missing_seeds'] == [92610]
    assert report_row(root, 'appliances')['loss'] == ''
    stopped = analysis.build_report(root, stop_reason='budget limit', elapsed_seconds=123.)
    assert stopped['decision'] == 'STOPPED'
    assert stopped['stop_reason'] == 'budget limit'


def test_G4_failed_plan_requires_comparisons_even_before_first_row(public_tables):
    root, _, runs = public_tables
    (root/'RUN_PLAN.json').write_text(json.dumps(dict(
        G4={'jena': dict(reused_historical_results=False)})), encoding='utf-8')
    decision = analysis.build_report(root)
    row = report_row(root, 'jena')
    assert decision['decision'] == 'PARTIAL'
    assert decision['complete'] is False
    assert row['n'] == '10'
    assert row['comparison_complete'] == 'False'
    assert row['baseline_source'] == 'CURRENT_COMPARISON'
    assert row['loss'] == ''
    assert row['success'] == ''
    for k, value in [(1, .3), (2, .6), (4, .4)]:
        for seed in analysis.SEEDS[:2]:
            runs.append(dict(dataset='jena', K=k, seed=seed, mse=value, mae=.2,
                             reused=False, role='comparison', job_id=f'compare_{k}_{seed}',
                             search_seconds=1., refit_seconds=1.))
    write_csv(root/'RUNS.csv', runs)
    complete = analysis.build_report(root)
    assert complete['decision'] == 'ADOPT'
    assert complete['datasets']['jena']['complete'] is True
    assert complete['datasets']['jena']['loss'] == pytest.approx(.3)


def test_stop_before_spectrum_is_reportable(tmp_path):
    decision = analysis.build_report(tmp_path, stop_reason='G1 gate')
    assert decision['decision'] == 'STOPPED'
    assert all(report_row(tmp_path, d)['n'] == '0' for d in analysis.DATASETS)


@pytest.mark.parametrize('mutation', ['duplicate_seed', 'extra_seed', 'wrong_K'])
def test_invalid_main_identity_is_rejected(public_tables, mutation):
    root, _, runs = public_tables
    if mutation == 'duplicate_seed':
        runs.append(dict(runs[0]))
    elif mutation == 'extra_seed':
        runs[0]['seed'] = 99999
    else:
        runs[0]['K'] = 4
    write_csv(root/'RUNS.csv', runs)
    with pytest.raises(ValueError):
        analysis.build_report(root)
    assert not (root/'DECISION.json').exists()
