"""Independent public-only audit. Stdlib statistics; no data/model/PCA access.

Exit codes: PASS=0, FAIL=1, INCOMPLETE=2. --replay calls only the frozen
public-table report renderer, after a complete independent PASS, in cache.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile
import time


NAME = 'k_tau_rule_20261023_v1'
PRIVATE = Path(__file__).resolve().parent
REPO = PRIVATE.parent.parent
DATASETS = ('jena', 'ETTh1', 'ETTh2_safe17400', 'appliances')
PRIMARY = ('jena', 'ETTh1', 'appliances')
SEEDS = tuple(range(92601, 92611))
GRID = {'jena': (3, 6, 12), 'ETTh1': (1, 2, 4),
        'ETTh2_safe17400': (1, 2, 4), 'appliances': (4, 7, 14)}
PUBLIC = ('RUNS.csv', 'K_TAU.csv', 'BASELINE_RECALCULATION.csv',
          'RUN_PLAN.json', 'SUMMARY.csv', 'DECISION.json')


class IncompleteInputs(Exception):
    pass


def read_json(path):
    def reject_nonfinite(token):
        raise ValueError('Nonfinite JSON scalar: ' + token)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject_nonfinite)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite public scalar')
    return result


def boolean(value):
    if value is True or value == 'True':
        return True
    if value is False or value == 'False':
        return False
    raise ValueError('Explicit bool required: ' + repr(value))


def linear_quantile(values, fraction):
    """Manual percentile(method=linear), without NumPy or report helpers."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean(values):
    return math.fsum(values) / len(values)


def equivalent(actual, expected):
    if expected is None:
        return actual in (None, '')
    if isinstance(expected, bool):
        return actual is expected or actual == str(expected)
    if isinstance(expected, (int, float)):
        try:
            return math.isclose(number(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
        except (ValueError, TypeError):
            return False
    return actual == expected


def audit(report, *, expected_reused=6, expected_new=34):
    report = Path(report).resolve()
    errors, missing, checked, source_hashes = [], [], [], {}

    def check(condition, label, detail=None):
        if not condition:
            errors.append(dict(check=label, detail=detail))

    def fields(actual, expected, label):
        for key, value in expected.items():
            check(key in actual and equivalent(actual.get(key), value), label + '.' + key,
                  dict(expected=value, actual=actual.get(key)))

    def load(path, kind='json'):
        if not path.is_file():
            missing.append(str(path.relative_to(report)))
            return None
        source_hashes[str(path.relative_to(report))] = digest(path)
        return read_csv(path) if kind == 'csv' else read_json(path)

    expected_stats, observed_counts, candidate_verdict = {}, {}, None
    figure_presence = {}
    try:
        inputs = {name: load(report / name, 'csv' if name.endswith('.csv') else 'json') for name in PUBLIC}
        plan, ktau, baselines, rows = (inputs[name] for name in
            ('RUN_PLAN.json', 'K_TAU.csv', 'BASELINE_RECALCULATION.csv', 'RUNS.csv'))
        if any(value is None for value in (plan, ktau, baselines, rows)):
            raise IncompleteInputs('Required public inputs are not yet available')
        selected = {}
        for row in ktau:
            if number(row['tau']) != .9:
                continue
            d = row['dataset']
            check(d in DATASETS and d not in selected, 'one_primary_tau_row', d)
            selected[d] = dict(K=int(row['K_A_fit']), K_A60=int(row['K_A60']), C=int(row['C']))
            check(1 <= selected[d]['K'] <= selected[d]['C'], 'K_within_C', d)
        check(set(selected) == set(DATASETS), 'exact_four_K_choices')
        check(plan['K_main'] == {d: row['K'] for d, row in selected.items()}, 'plan_matches_A_fit_K')
        old, raw = {d: {} for d in DATASETS}, {}
        for row in baselines:
            d = row['dataset']
            check(d in DATASETS and row['row_policy'] == 'DEFAULT128', 'baseline_dataset_policy', d)
            if d not in DATASETS:
                continue
            mse = {int(k): number(v) for k, v in json.loads(row['seed_MSE_json']).items()}
            mae = {int(k): number(v) for k, v in json.loads(row['seed_MAE_json']).items()}
            expected_ids = {92601} if row['family'] == 'RAW' else set(SEEDS[:2])
            check(set(mse) == expected_ids and set(mae) == expected_ids, 'baseline_fixed_seeds', d)
            for metric, values in (('MSE', mse), ('MAE', mae)):
                fields(row, {metric + '_mean': mean(list(values.values())),
                    metric + '_min': min(values.values()), metric + '_max': max(values.values())},
                    d + '.historical.' + metric)
            check(int(row['n']) == len(expected_ids), 'baseline_n', d)
            if row['family'] == 'RAW':
                check(d not in raw, 'unique_Raw', d)
                raw[d] = number(row['MSE_mean'])
            else:
                k = int(row['K'])
                check(row['family'] == 'B2' and k in GRID[d] and k not in old[d], 'historical_grid', d)
                old[d][k] = dict(mean=mean(list(mse.values())), mse=mse, mae=mae)
        check(set(raw) == set(DATASETS), 'four_Raw_baselines')
        for d in DATASETS:
            check(set(old[d]) == set(GRID[d]), 'exact_three_historical_candidates', d)
        eligible = {d for d in DATASETS if any(v['mean'] < raw[d] for v in old[d].values())}
        check(eligible == set(PRIMARY), 'G7_exact_three_primary_datasets', sorted(eligible))
        expected_jobs, expected_reuse = {}, {}
        for d in DATASETS:
            k = selected[d]['K']
            compatible = plan['G4'][d]['reused_historical_results'] is True
            for seed in SEEDS:
                identity = (d, 'main', k, seed)
                (expected_reuse if compatible and k in GRID[d] and seed in SEEDS[:2]
                 else expected_jobs)[identity] = None
            if not compatible:
                for width in GRID[d]:
                    for seed in SEEDS[:2]:
                        expected_jobs[(d, 'comparison', width, seed)] = None
        actual_jobs, actual_reuse = {}, {}
        for entry in plan['jobs']:
            job, role = entry['job'], entry['role']
            identity = (job['cohort'], role, int(job['K']), int(job['seed']))
            check(identity not in actual_jobs, 'unique_plan_job', identity)
            fields(job, dict(family='B2', revin=False, init_kind='PCA', device='cuda'), 'plan_recipe')
            actual_jobs[identity] = job['job_id']
            check(re.fullmatch(r'[A-Za-z0-9_-]+', job['job_id']) is not None, 'safe_plan_job_id')
        for row in plan['reused']:
            identity = (row['dataset'], row['role'], int(row['K']), int(row['seed']))
            check(identity not in actual_reuse, 'unique_plan_reuse', identity)
            actual_reuse[identity] = row['job_id']
        check(set(actual_jobs) == set(expected_jobs), 'job_plan_matches_G4_seed_K_rules')
        check(set(actual_reuse) == set(expected_reuse), 'reuse_plan_matches_G4_seed_K_rules')
        n_new = sum(identity[1] == 'main' for identity in expected_jobs)
        if expected_reused is not None:
            check(len(expected_reuse) == expected_reused, 'campaign_expected_reuse6', len(expected_reuse))
        if expected_new is not None:
            check(n_new == expected_new, 'campaign_expected_new34', n_new)
        observed = {}
        for row in rows:
            d, role, k, seed = row['dataset'], row['role'], int(row['K']), int(row['seed'])
            identity = (d, role, k, seed)
            check(identity not in observed, 'unique_RUNS_identity', identity)
            check(identity in expected_jobs or identity in expected_reuse, 'RUNS_in_exact_plan', identity)
            check(d in DATASETS and role in ('main', 'comparison') and seed in SEEDS, 'registered_RUNS_axes', identity)
            row = dict(row, K=k, seed=seed, mse=number(row['mse']), mae=number(row['mae']),
                       reused=boolean(row['reused']))
            observed[identity] = row
            reused = identity in expected_reuse
            check(row['reused'] is reused, 'reuse_flag', identity)
            check(row['job_id'] == (actual_reuse if reused else actual_jobs).get(identity), 'job_id_binding', identity)
            if (identity not in expected_jobs and identity not in expected_reuse
                    or row['job_id'] != (actual_reuse if reused else actual_jobs).get(identity)
                    or re.fullmatch(r'[A-Za-z0-9_-]+', row['job_id']) is None):
                check(False, 'unsafe_or_unplanned_receipt_path', identity)
                continue
            if reused:
                fields(row, dict(mse=old[d][k]['mse'][seed], mae=old[d][k]['mae'][seed]), 'reused_exact_historical')
                continue
            directory = report / 'datasets' / d
            result = load(directory / 'jobs' / row['job_id'] / 'RESULT.json')
            seal_path = directory / 'jobs' / row['job_id'] / 'SELECTION.json'
            seal = load(seal_path)
            search = load(directory / 'runs' / row['job_id'] / 'cap3' / 'SEARCH_RESULT.json')
            if any(value is None for value in (result, seal, search)):
                continue
            fields(result, dict(status='COMPLETE', run_id=row['job_id'], cap=3,
                                selection_sealed_before_V=True), 'receipt')
            check(len(result['rows']) == 1, 'one_receipt_row', row['job_id'])
            scored = result['rows'][0]
            fields(scored, dict(dataset=d, K=k, seed=seed, MSE=row['mse'], MAE=row['mae'],
                independent_FP64='PASS', metric_accumulation_dtype='float64', prediction_dtype='float32',
                evaluation='V_select', fit='fresh_A60', family='B2', revin=False), 'scored_receipt')
            fields(row, dict(bundle_sha256=result['bundle_binding']['sha256'],
                selection_sealed_before_V=True, selected_epoch=result['fit_receipt']['selected_epoch'],
                search_seconds=result['search_summary']['seconds'],
                refit_seconds=result['refit_summary']['seconds']), 'RUNS_scalar_bindings')
            fields(seal, dict(status='SEALED', cap=3, V_used=False, selection_split='A_stop',
                             job_id=row['job_id'], cohort=d), 'selection_seal')
            fields(search, dict(status='COMPLETE', cap=3, V_used=False, selection_split='A_stop'), 'search_receipt')
            fields(search['spec'], dict(dataset=d, latent_k=k, seed=seed, family='B2', revin=False,
                init_kind='PCA', epochs=[.25, 1., 3.]), 'search_scientific_spec')
            check(result['selection_decision_binding']['sha256'] == digest(seal_path), 'selection_seal_SHA', row['job_id'])
            curve = [(number(item['epoch']), number(item['astop_mse'])) for item in search['curves']]
            check([e for e, _ in curve] == [0., .25, 1., 3.], 'cap3_fixed_checkpoints', row['job_id'])
            epoch, mse = min(curve, key=lambda item: (item[1], item[0]))
            fields(search, dict(selected_epoch=epoch), 'search_selected_epoch')
            for actual in (seal, search['summary'], result['fit_receipt']):
                fields(actual, dict(selected_epoch=epoch, selected_astop_mse=mse), 'earlier_tie_selection')
            checked.append(row['job_id'])
        for identity in set(expected_jobs) | set(expected_reuse):
            if identity not in observed:
                missing.append('RUNS:' + ':'.join(map(str, identity)))
        observed_counts = dict(main=sum(key[1] == 'main' for key in observed),
            comparison=sum(key[1] == 'comparison' for key in observed),
            reused=sum(row['reused'] for key, row in observed.items() if key[1] == 'main'),
            new=sum(not row['reused'] for key, row in observed.items() if key[1] == 'main'),
            completed_json_receipts_checked=len(checked))
        for d in DATASETS:
            main = {key[3]: row for key, row in observed.items() if key[0] == d and key[1] == 'main'}
            values = [row['mse'] for row in main.values()]
            candidates = {k: v['mean'] for k, v in old[d].items()}
            required = plan['G4'][d]['reused_historical_results'] is False
            comparison_rows = {key: row for key, row in observed.items() if key[0] == d and key[1] == 'comparison'}
            comparison_complete = True
            if required:
                for k in GRID[d]:
                    matched = [comparison_rows[(d, 'comparison', k, s)]['mse'] for s in SEEDS[:2]
                               if (d, 'comparison', k, s) in comparison_rows]
                    if len(matched) != 2:
                        comparison_complete = False
                    else:
                        candidates[k] = mean(matched)
            complete = set(main) == set(SEEDS) and comparison_complete
            first_two = mean([main[s]['mse'] for s in SEEDS[:2]]) if all(s in main for s in SEEDS[:2]) else None
            q1, q3 = (linear_quantile(values, q) for q in (.25, .75)) if values else (None, None)
            iqr = q3 - q1 if values else None
            loss = first_two - min([first_two, *candidates.values()]) if first_two is not None and comparison_complete else None
            med = statistics.median(values) if values else None
            success = med < raw[d] if complete else None
            expected_stats[d] = dict(dataset=d, K_main=selected[d]['K'], C=selected[d]['C'],
                K_A60=selected[d]['K_A60'], n=len(values), median=med,
                mean=mean(values) if values else None, Q1=q1, Q3=q3, IQR=iqr,
                min=min(values) if values else None, max=max(values) if values else None,
                MAE_mean=mean([row['mae'] for row in main.values()]) if values else None,
                Raw=raw[d], first_two_mean=first_two, loss=loss,
                limitation='LIMITATION' if loss is not None and iqr is not None and loss > iqr else '',
                main_improves_historical_best=first_two < min(v['mean'] for v in old[d].values()) if first_two is not None else None,
                success=success, complete=complete, baseline_source='CURRENT_COMPARISON' if required else 'HISTORICAL',
                comparison_complete=comparison_complete, decision_scope='PRIMARY' if d in PRIMARY else 'DESCRIPTIVE_ONLY',
                status='SUCCESS' if success is True else 'FAIL' if success is False else 'PARTIAL')
        summary, decision = inputs['SUMMARY.csv'], inputs['DECISION.json']
        if summary is not None:
            check(len(summary) == 4 and {row['dataset'] for row in summary} == set(DATASETS), 'four_SUMMARY_rows')
            for row in summary:
                fields(row, expected_stats[row['dataset']], 'SUMMARY.' + row['dataset'])
        complete = all(value['complete'] for value in expected_stats.values())
        failed = [d for d in PRIMARY if expected_stats[d]['success'] is False]
        candidate_verdict = ('ADOPT' if not failed else 'ADOPT_CONDITIONAL' if len(failed) == 1 else 'REPLAN') if complete else None
        if decision is not None:
            stopped = bool(decision.get('stop_reason'))
            fields(decision, dict(complete=complete, eligible_datasets=list(PRIMARY), failed_datasets=failed,
                decision='STOPPED' if stopped else candidate_verdict if complete else 'PARTIAL',
                status='STOPPED' if stopped else 'COMPLETE' if complete else 'PARTIAL'), 'DECISION')
            check(set(decision['datasets']) == set(DATASETS), 'four_DECISION_datasets')
            for d in DATASETS:
                fields(decision['datasets'][d], expected_stats[d], 'DECISION.datasets.' + d)
                fields(decision['datasets'][d], dict(
                    missing_seeds=[s for s in SEEDS if (d, 'main', selected[d]['K'], s) not in observed],
                    seed_ids=sorted(key[3] for key in observed if key[0] == d and key[1] == 'main'),
                    comparison_seed_rows=sum(key[0] == d and key[1] == 'comparison' for key in observed)), 'DECISION.seed_details')
            if stopped:
                missing.append('Campaign stopped: ' + str(decision['stop_reason']))
        if not complete:
            missing.append('Exact four datasets x ten main seeds and any required comparisons are incomplete')
        for kind in ('K_CURVE', 'BOXPLOT'):
            for extension in ('png', 'svg'):
                name = f'FIG_{kind}.{extension}'
                path = report / name
                figure_presence[name] = path.is_file() and path.stat().st_size > 100
                if complete and not figure_presence[name]:
                    missing.append('Missing/nonempty figure: ' + name)
        for relative, before in source_hashes.items():
            check(digest(report / relative) == before, 'public_input_stable_during_audit', relative)
    except IncompleteInputs as exc:
        missing.append(str(exc))
    except Exception as exc:
        errors.append(dict(check='invalid_public_artifact', detail=f'{type(exc).__name__}: {exc}'))
    return dict(status='FAIL' if errors else 'INCOMPLETE' if missing else 'PASS',
        created_unix=time.time(), method='Independent stdlib median, manual linear quartiles, fsum/n, fixed-seed two-mean loss',
        report_root=str(report), errors=errors, incomplete_reasons=missing,
        observed_counts=observed_counts, expected_reused=expected_reused, expected_new=expected_new,
        recomputed=expected_stats, candidate_verdict=candidate_verdict,
        checked_completed_receipts=checked, public_input_sha256=source_hashes,
        figure_presence_only_no_image_bytes_read=figure_presence,
        binary_model_or_data_files_opened=False, PCA_or_training_called=False,
        first_two_seed_loss_distinct_from_ten_seed_distribution=True)


def replay(report, result, destination):
    """Render only copied public tables after a complete independent PASS."""
    if result['status'] != 'PASS':
        raise ValueError('Replay requires a complete independent PASS')
    destination = Path(destination).resolve()
    if not destination.is_relative_to(PRIVATE) or destination.exists():
        raise ValueError('Replay requires a fresh directory inside the campaign cache')
    destination.mkdir(parents=True)
    for name in ('RUNS.csv', 'K_TAU.csv', 'BASELINE_RECALCULATION.csv', 'RUN_PLAN.json'):
        shutil.copyfile(report / name, destination / name)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    from research.k_tau_rule_20261023_v1.analysis import build_report
    canonical = read_json(report / 'DECISION.json')
    build_report(destination, stop_reason=canonical.get('stop_reason'),
                 elapsed_seconds=canonical.get('elapsed_seconds'))
    same_summary = read_csv(report / 'SUMMARY.csv') == read_csv(destination / 'SUMMARY.csv')
    actual = read_json(destination / 'DECISION.json')
    same_decision = {k: v for k, v in actual.items() if k != 'elapsed_seconds'} == {
        k: v for k, v in canonical.items() if k != 'elapsed_seconds'}
    figures = {name: all((root / name).is_file() and (root / name).stat().st_size > 100
                        for root in (report, destination))
               for kind in ('K_CURVE', 'BOXPLOT') for name in (f'FIG_{kind}.png', f'FIG_{kind}.svg')}
    result['public_only_replay'] = dict(destination=str(destination), summary_equal=same_summary,
                                      decision_equal_ignoring_elapsed_seconds=same_decision, figures=figures)
    if not same_summary or not same_decision or not all(figures.values()):
        result['errors'].append(dict(check='public_only_report_replay', detail=result['public_only_replay']))
        result['status'] = 'FAIL'


def self_test():
    """Hand-calculated synthetic public artifacts; never reads the campaign."""
    fixture_parent = PRIVATE / 'final_verifier_fixtures'
    fixture_parent.mkdir(exist_ok=True)
    tests = []
    assert linear_quantile([4., 1., 3., 2.], .25) == 1.75
    assert linear_quantile([4., 1., 3., 2.], .75) == 3.25
    tests.append('manual_linear_quartile_interpolation')

    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def fixture(root, failures=()):
        ks = dict(jena=6, ETTh1=4, ETTh2_safe17400=4, appliances=10)
        channels = dict(jena=21, ETTh1=7, ETTh2_safe17400=7, appliances=28)
        plan = dict(K_main=ks, G4={d: dict(reused_historical_results=True) for d in DATASETS},
                    jobs=[], reused=[])
        rows, baseline, tau, summaries, details = [], [], [], [], {}
        for d in DATASETS:
            raw = .1 if d == 'ETTh2_safe17400' else 1.
            old_means = {}
            for family, k, pair in [('RAW', channels[d], [raw])] + [
                ('B2', k, [.2, .4] if k == ks[d] else
                    ([.2, .3] if k == next(width for width in GRID[d] if width != ks[d]) else [.5, .5]))
                for index, k in enumerate(GRID[d])]:
                if family == 'B2':
                    old_means[k] = sum(pair) / 2
                mse = {str(92601+i): value for i, value in enumerate(pair)}
                mae = {seed: value / 2 for seed, value in mse.items()}
                baseline.append(dict(dataset=d, row_policy='DEFAULT128', family=family, K=k,
                    n=len(pair), seed_MSE_json=json.dumps(mse), seed_MAE_json=json.dumps(mae),
                    MSE_mean=sum(pair)/len(pair), MSE_min=min(pair), MSE_max=max(pair),
                    MAE_mean=sum(mae.values())/len(pair), MAE_min=min(mae.values()), MAE_max=max(mae.values())))
            for threshold in (.7, .8, .9, .95):
                tau.append(dict(dataset=d, tau=threshold, C=channels[d], K_A_fit=ks[d], K_A60=ks[d]))
            tail = 1.2 if d in failures else .6
            for seed in SEEDS:
                mse = .2 if seed == 92601 else .4 if seed == 92602 else tail
                reused = d != 'appliances' and seed in SEEDS[:2]
                identifier = f'fixture_{d}_{seed}'
                row = dict(dataset=d, K=ks[d], seed=seed, mse=mse, mae=mse/2,
                    reused=reused, role='main', job_id=identifier, search_seconds=2., refit_seconds=3.,
                    selected_epoch=.25, bundle_sha256='b' * 64, selection_sealed_before_V=True)
                rows.append(row)
                if reused:
                    plan['reused'].append(row)
                    continue
                plan['jobs'].append(dict(role='main', job=dict(job_id=identifier, cohort=d, K=ks[d],
                    seed=seed, family='B2', revin=False, init_kind='PCA', device='cuda')))
                directory = root / 'datasets' / d
                seal_path = directory / 'jobs' / identifier / 'SELECTION.json'
                seal = dict(status='SEALED', cap=3, V_used=False, selection_split='A_stop',
                    job_id=identifier, cohort=d, selected_epoch=.25, selected_astop_mse=.3)
                write_json(seal_path, seal)
                scored = dict(dataset=d, K=ks[d], seed=seed, MSE=mse, MAE=mse/2,
                    independent_FP64='PASS', metric_accumulation_dtype='float64', prediction_dtype='float32',
                    evaluation='V_select', fit='fresh_A60', family='B2', revin=False)
                write_json(directory / 'jobs' / identifier / 'RESULT.json', dict(status='COMPLETE',
                    run_id=identifier, cap=3, selection_sealed_before_V=True, rows=[scored],
                    fit_receipt=dict(selected_epoch=.25, selected_astop_mse=.3),
                    bundle_binding=dict(sha256='b' * 64), selection_decision_binding=dict(sha256=digest(seal_path)),
                    search_summary=dict(seconds=2.), refit_summary=dict(seconds=3.)))
                write_json(directory / 'runs' / identifier / 'cap3' / 'SEARCH_RESULT.json',
                    dict(status='COMPLETE', cap=3, V_used=False, selection_split='A_stop', selected_epoch=.25,
                         spec=dict(dataset=d, latent_k=ks[d], seed=seed, family='B2', revin=False,
                                   init_kind='PCA', epochs=[.25, 1., 3.]),
                         summary=dict(selected_epoch=.25, selected_astop_mse=.3),
                         curves=[dict(epoch=e, astop_mse=m) for e, m in ((0., .5), (.25, .3), (1., .3), (3., .4))]))
            # Eight equal tail values make the three quartiles hand-calculable.
            loss = .3 - min(.3, *old_means.values())
            success = tail < raw
            summary = dict(dataset=d, K_main=ks[d], C=channels[d], K_A60=ks[d], n=10,
                median=tail, mean=(.2 + .4 + 8*tail)/10, Q1=tail, Q3=tail, IQR=0.,
                min=.2, max=tail, MAE_mean=(.2 + .4 + 8*tail)/20, Raw=raw,
                first_two_mean=.3, loss=loss, limitation='LIMITATION' if loss > 0 else '',
                main_improves_historical_best=.3 < min(old_means.values()), success=success, complete=True,
                baseline_source='HISTORICAL', comparison_complete=True,
                decision_scope='PRIMARY' if d in PRIMARY else 'DESCRIPTIVE_ONLY',
                status='SUCCESS' if success else 'FAIL')
            summaries.append(summary)
            details[d] = dict(summary, missing_seeds=[], seed_ids=list(SEEDS), comparison_seed_rows=0)
        write_csv(root / 'RUNS.csv', rows)
        write_csv(root / 'K_TAU.csv', tau)
        write_csv(root / 'BASELINE_RECALCULATION.csv', baseline)
        write_csv(root / 'SUMMARY.csv', summaries)
        write_json(root / 'RUN_PLAN.json', plan)
        failed = [d for d in PRIMARY if d in failures]
        write_json(root / 'DECISION.json', dict(complete=True, eligible_datasets=list(PRIMARY),
            failed_datasets=failed, datasets=details, status='COMPLETE', stop_reason=None,
            decision='ADOPT' if not failed else 'ADOPT_CONDITIONAL' if len(failed) == 1 else 'REPLAN'))
        for kind in ('K_CURVE', 'BOXPLOT'):
            for extension in ('png', 'svg'):
                (root / f'FIG_{kind}.{extension}').write_bytes(b'synthetic existence-only placeholder\n' * 4)

    with tempfile.TemporaryDirectory(dir=fixture_parent) as temp:
        parent = Path(temp)
        for index, failures in enumerate(((), ('jena',), ('jena', 'ETTh1'))):
            root = parent / str(index)
            root.mkdir()
            fixture(root, failures)
            result = audit(root)
            assert result['status'] == 'PASS', result['errors']
            assert result['observed_counts']['reused'] == 6 and result['observed_counts']['new'] == 34
            tests.append('hand_calculated_' + result['candidate_verdict'])
        root = parent / 'mutations'
        root.mkdir()
        for mutation in ('all10_loss', 'duplicate_seed', 'receipt_mismatch', 'stale_complete', 'incomplete',
                         'missing_receipt_field', 'stopped_complete', 'missing_figure', 'nonfinite_receipt'):
            fixture(root)
            if mutation == 'all10_loss':
                rows = read_csv(root / 'SUMMARY.csv')
                rows[0]['loss'] = .54 - .25
                write_csv(root / 'SUMMARY.csv', rows)
            elif mutation in ('duplicate_seed', 'stale_complete', 'incomplete'):
                rows = read_csv(root / 'RUNS.csv')
                if mutation == 'duplicate_seed':
                    rows.append(dict(rows[0]))
                else:
                    rows = [r for r in rows if not (r['dataset'] == 'jena' and int(r['seed']) == 92603)]
                write_csv(root / 'RUNS.csv', rows)
                if mutation == 'incomplete':
                    (root / 'SUMMARY.csv').unlink()
                    (root / 'DECISION.json').unlink()
            elif mutation == 'stopped_complete':
                decision = read_json(root / 'DECISION.json')
                decision.update(decision='STOPPED', status='STOPPED', stop_reason='synthetic budget cutoff')
                write_json(root / 'DECISION.json', decision)
            elif mutation == 'missing_figure':
                (root / 'FIG_BOXPLOT.svg').unlink()
            else:
                path = root / 'datasets' / 'jena' / 'jobs' / 'fixture_jena_92603' / 'RESULT.json'
                value = read_json(path)
                if mutation == 'receipt_mismatch':
                    value['rows'][0]['MSE'] += .01
                elif mutation == 'nonfinite_receipt':
                    value['rows'][0]['MSE'] = float('nan')
                else:
                    del value['rows']
                write_json(path, value)
            result = audit(root)
            expected = 'INCOMPLETE' if mutation in ('incomplete', 'stopped_complete', 'missing_figure') else 'FAIL'
            assert result['status'] == expected, (mutation, result)
            tests.append(mutation + '_denied')
        matched = parent / 'matched_comparisons'
        matched.mkdir()
        fixture(matched)
        plan = read_json(matched / 'RUN_PLAN.json')
        rows = read_csv(matched / 'RUNS.csv')
        plan['G4']['jena']['reused_historical_results'] = False
        plan['reused'] = [row for row in plan['reused'] if row['dataset'] != 'jena']

        def clone_receipt(row):
            directory = matched / 'datasets' / 'jena'
            template = 'fixture_jena_92603'
            identifier = row['job_id']
            seal = read_json(directory / 'jobs' / template / 'SELECTION.json')
            seal['job_id'] = identifier
            seal_path = directory / 'jobs' / identifier / 'SELECTION.json'
            write_json(seal_path, seal)
            result = read_json(directory / 'jobs' / template / 'RESULT.json')
            result['run_id'] = identifier
            result['rows'][0].update(K=int(row['K']), seed=int(row['seed']),
                                     MSE=number(row['mse']), MAE=number(row['mae']))
            result['selection_decision_binding']['sha256'] = digest(seal_path)
            write_json(directory / 'jobs' / identifier / 'RESULT.json', result)
            search = read_json(directory / 'runs' / template / 'cap3' / 'SEARCH_RESULT.json')
            search['spec'].update(latent_k=int(row['K']), seed=int(row['seed']))
            write_json(directory / 'runs' / identifier / 'cap3' / 'SEARCH_RESULT.json', search)
            plan['jobs'].append(dict(role=row['role'], job=dict(job_id=identifier, cohort='jena',
                K=int(row['K']), seed=int(row['seed']), family='B2', revin=False, init_kind='PCA', device='cuda')))

        for row in rows:
            if row['dataset'] == 'jena' and int(row['seed']) in SEEDS[:2]:
                row['reused'] = False
                clone_receipt(row)
        for width, mse in ((3, .1), (6, .3), (12, .2)):
            for seed in SEEDS[:2]:
                row = dict(dataset='jena', K=width, seed=seed, mse=mse, mae=mse/2,
                    reused=False, role='comparison', job_id=f'matched_{width}_{seed}',
                    selected_epoch=.25, search_seconds=2., refit_seconds=3.,
                    bundle_sha256='b' * 64, selection_sealed_before_V=True)
                rows.append(row)
                clone_receipt(row)
        summary = read_csv(matched / 'SUMMARY.csv')
        summary[0].update(loss=.2, baseline_source='CURRENT_COMPARISON')
        decision = read_json(matched / 'DECISION.json')
        decision['datasets']['jena'].update(loss=.2, baseline_source='CURRENT_COMPARISON', comparison_seed_rows=6)
        write_csv(matched / 'RUNS.csv', rows)
        write_csv(matched / 'SUMMARY.csv', summary)
        write_json(matched / 'RUN_PLAN.json', plan)
        write_json(matched / 'DECISION.json', decision)
        result = audit(matched, expected_reused=4, expected_new=36)
        assert result['status'] == 'PASS', result['errors']
        assert math.isclose(result['recomputed']['jena']['loss'], .2)
        tests.append('G4_failed_matched_comparisons_override_old_baselines')
        write_csv(matched / 'RUNS.csv', [r for r in rows if r['role'] == 'main'])
        (matched / 'SUMMARY.csv').unlink()
        (matched / 'DECISION.json').unlink()
        result = audit(matched, expected_reused=4, expected_new=36)
        assert result['status'] == 'INCOMPLETE' and result['recomputed']['jena']['loss'] is None
        assert result['recomputed']['jena']['comparison_complete'] is False
        tests.append('G4_failed_zero_comparisons_cannot_use_old_loss')
    return dict(status='PASS', synthetic_checks=tests, campaign_artifacts_read=False,
                PCA_or_training_called=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=REPO / 'reports' / NAME)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-reused', type=int, default=6)
    parser.add_argument('--expected-new', type=int, default=34)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--replay-dir', type=Path, default=PRIVATE / 'report_replay')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return 0
    result = audit(args.report, expected_reused=args.expected_reused, expected_new=args.expected_new)
    if args.replay and result['status'] == 'PASS':
        try:
            replay(args.report.resolve(), result, args.replay_dir)
        except Exception as exc:
            result['status'] = 'FAIL'
            result['errors'].append(dict(check='replay', detail=f'{type(exc).__name__}: {exc}'))
    elif args.replay:
        result['public_only_replay'] = dict(status='SKIPPED_UNTIL_INDEPENDENT_PASS')
    output = args.output or args.report / 'FINAL_REPORT_VALIDATION.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], output=str(output), counts=result['observed_counts'],
                          errors=len(result['errors']), incomplete=len(result['incomplete_reasons'])), ensure_ascii=False))
    return {'PASS': 0, 'FAIL': 1, 'INCOMPLETE': 2}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
