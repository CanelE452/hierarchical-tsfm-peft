"""Boundary, primary-selection, archive-isolation and integrity tests."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from research.comparative_ablation_20261025_v1 import analysis
from research.comparative_ablation_followup_20261027_v1 import bundle, reporting


def summaries(dataset, a4, n=10):
    return [dict(dataset=dataset, method_id='OURS', n=10, complete=True, MSE_median=1.0),
            dict(dataset=dataset, method_id='A4', n=n, complete=n == 10, MSE_median=a4)]


def a4_check(dataset, a4, n=10):
    rows = reporting.prediction_checks(summaries(dataset, a4, n), [])
    return next(row for row in rows if row['kind'] == 'A4_EFFECT' and row['dataset'] == dataset)


@pytest.mark.parametrize('dataset', ['ETTh1', 'ETTh2_safe17400'])
def test_ett_prediction_includes_fifty_percent_boundary(dataset):
    assert a4_check(dataset, 1.5)['status'] == 'MATCH'
    assert a4_check(dataset, 1.499)['status'] == 'NOT_MATCH'
    assert a4_check(dataset, 1.5, n=9)['status'] == 'INCOMPLETE'


@pytest.mark.parametrize('dataset', ['jena', 'appliances'])
def test_small_effect_prediction_is_strict_and_not_a_decision(dataset):
    assert a4_check(dataset, 1.099)['status'] == 'MATCH'
    assert a4_check(dataset, 1.1)['status'] == 'NOT_MATCH'
    assert a4_check(dataset, .9)['status'] == 'NOT_MATCH'
    assert a4_check(dataset, .899)['status'] == 'NOT_MATCH'
    assert 'decision' not in a4_check(dataset, 1.099)


def floor_rows(ratios):
    return [dict(dataset='ETTh1', method_id='C2', seed=92601+i, is_primary=True,
                 status='PASS', floor_fraction=value, floor_MSE=value*(i+1),
                 actual_same_cohort_MSE=i+1) for i, value in enumerate(ratios)]


def d1_check(rows):
    checks = reporting.prediction_checks([], rows)
    return next(row for row in checks if row['kind'] == 'D1_RATIO' and row['dataset'] == 'ETTh1'
                and row['method_id'] == 'C2')


def test_d1_uses_median_per_job_ratio_and_fixed_half_boundary():
    rows = floor_rows([0.0]*5 + [1.0]*5)
    check = d1_check(rows)
    assert check['median_per_job_floor_fraction'] == .5
    assert check['status'] == 'MATCH'
    # Ratio of the medians is different and must not be substituted.
    assert (rows[4]['floor_MSE'] + rows[5]['floor_MSE']) / 2 / 5.5 != .5
    assert d1_check(floor_rows([.499]*10))['status'] == 'NOT_MATCH'
    assert d1_check(rows[:9])['status'] == 'INCOMPLETE'


def test_d1_missing_bundle_excluded_and_duplicate_seed_rejected():
    rows = floor_rows([.8]*10)
    rows[0]['status'] = 'SKIPPED_MISSING_BUNDLE'
    assert d1_check(rows)['n'] == 9
    assert d1_check(rows)['status'] == 'INCOMPLETE'
    rows = floor_rows([.8]*10)
    rows[-1]['seed'] = rows[0]['seed']
    with pytest.raises(ValueError, match='Duplicate'):
        d1_check(rows)


def test_interim_prediction_receives_existing_c2_c3_ten_seed_floors(tmp_path, monkeypatch):
    from research.comparative_ablation_followup_20261027_v1 import floors

    report, old_report = tmp_path / 'followup', tmp_path / 'original'
    old_report.mkdir()
    (old_report / 'G1_REUSE_AUDIT.json').write_text('{}', encoding='utf8')
    inputs = [dict(status='COMPLETE', job=dict(job_id=f'{method}_{dataset}_{seed}',
        method_id=method, dataset=dataset, seed=seed, extended=False))
        for method in ('C2', 'C3', 'A4') for dataset in analysis.DATASETS for seed in range(92601, 92611)]
    # A concurrently completed extension must still be excluded from the interim snapshot.
    inputs.append(dict(status='COMPLETE', job=dict(job_id='A1_jena_ext10', method_id='A1',
        dataset='jena', seed=92601, extended=True)))
    monkeypatch.setattr(reporting.common, 'REPORT', report)
    monkeypatch.setattr(reporting.common, 'OLD_REPORT', old_report)
    monkeypatch.setattr(reporting.common, 'guard_preserved', lambda: dict(status='PASS'))
    monkeypatch.setattr(reporting.common, 'install_write_guard', lambda: None)
    monkeypatch.setattr(reporting.common, 'all_results', lambda: inputs)
    monkeypatch.setattr(reporting, '_prepare_matplotlib', lambda: None)
    monkeypatch.setattr(analysis, 'GENERATED_FILES', ())
    monkeypatch.setattr(analysis, 'report', lambda *args: {})
    synthetic_summaries = [row for dataset in analysis.DATASETS for row in summaries(dataset, 1.6)]
    monkeypatch.setattr(reporting, '_read_csv', lambda path: synthetic_summaries if path.name == 'SUMMARY.csv' else [])
    observed = {}

    def fake_floors(output, methods, include_d0, results):
        observed.update(methods=methods, include_d0=include_d0,
                        jobs=[result['job'] for result in results])
        evidence = [dict(result['job'], is_primary=True, status='PASS', floor_fraction=.65)
                    for result in results if result['job']['method_id'] in methods]
        return dict(d1=evidence)

    class PredictionProbeComplete(Exception):
        pass

    def stop_before_figures(*args):
        raise PredictionProbeComplete()

    monkeypatch.setattr(floors, 'run', fake_floors)
    monkeypatch.setattr(reporting, '_figure', stop_before_figures)
    with pytest.raises(PredictionProbeComplete):
        reporting.run('interim', status=dict(status='RUNNING'))
    checks = json.loads((report / 'interim_A4' / 'PREDICTION_CHECK.json').read_text(encoding='utf8'))
    comparisons = [row for row in checks if row['kind'] == 'D1_RATIO' and row['method_id'] in ('C2', 'C3')]
    assert observed['methods'] == ('C2', 'C3', 'A4') and observed['include_d0'] is False
    assert len(observed['jobs']) == 120 and not any(job['extended'] for job in observed['jobs'])
    assert len(comparisons) == 4
    assert all(row['n'] == 10 and row['status'] == 'MATCH' for row in comparisons)


def primary_rows(extension_count):
    rows = []
    for extended, count in ((False, 10), (True, extension_count)):
        for i in range(count):
            rows.append(dict(dataset='jena', method_id='A1', seed=92601+i,
                extended=extended, cap_selected=True, MSE=.5 if extended else 1.,
                job_id=f'A1_jena_{i}' + ('_ext10' if extended else '')))
    analysis.choose_primary(rows)
    return rows


def test_updated_plot_adopts_extension_only_at_all_ten_seeds():
    incomplete = reporting.primary_figure_rows(primary_rows(9))
    assert len(incomplete) == 10
    assert not any(row['extended'] for row in incomplete)
    complete = reporting.primary_figure_rows(primary_rows(10))
    assert len(complete) == 10
    assert all(row['extended'] for row in complete)
    assert {row['seed'] for row in complete} == reporting.SEEDS


@pytest.mark.parametrize('name', ['.cache/x/RESULT.json', 'reports/checkpoints/model.json',
    'reports/raw_arrays/V.csv', 'research/__pycache__/x.pyc', 'reports/model.pt',
    'reports/predictions.npz', 'reports/predictions.npy', 'reports/inside.zip',
    'reports/selected_bundle.json', 'reports/prediction_array.csv'])
def test_bundle_excludes_private_arrays_checkpoints_and_nested_archives(name):
    assert not bundle.public_file(Path(name))


@pytest.mark.parametrize('name', ['reports/RUNS.csv', 'reports/RESULT.json',
    'reports/FIG_ABLATION_UPDATED.svg', 'research/reporting.py', 'reports/PREREGISTRATION.md'])
def test_bundle_accepts_public_evidence(name):
    assert bundle.public_file(Path(name))


def test_collection_includes_only_selected_new_job_public_reports(tmp_path, monkeypatch):
    report = tmp_path / 'reports' / bundle.common.NAME
    old_report = tmp_path / 'reports' / bundle.common.OLD_NAME
    instruction = tmp_path / 'instructions.md'
    monkeypatch.setattr(bundle.common, 'REPO', tmp_path)
    monkeypatch.setattr(bundle.common, 'REPORT', report)
    monkeypatch.setattr(bundle.common, 'OLD_REPORT', old_report)
    monkeypatch.setattr(bundle.common, 'SOURCE_DOCUMENT', instruction)
    monkeypatch.setattr(bundle.common, 'selected_jobs', lambda: [dict(job_id='new_job')])
    for path in (report / 'RESULTS_KO.md', report / 'PROGRESS.json', report / 'old.zip',
                 report / 'predictions.npz', old_report / 'PREREGISTRATION.md',
                 old_report / 'jobs' / 'old_job' / 'RESULT.json',
                 old_report / 'jobs' / 'new_job' / 'RESULT.json',
                 old_report / 'jobs' / 'new_job' / 'selected_bundle.pt', instruction):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('evidence', encoding='utf8')
    names = set(bundle.collect_members())
    assert f'reports/{bundle.common.NAME}/RESULTS_KO.md' in names
    assert f'reports/{bundle.common.OLD_NAME}/PREREGISTRATION.md' in names
    assert f'reports/{bundle.common.OLD_NAME}/jobs/new_job/RESULT.json' in names
    assert not any('old_job' in name or 'PROGRESS' in name or name.endswith(('.zip', '.npz', '.pt')) for name in names)


def archive_fixture(tmp_path, payload=b'public numerical results\n'):
    name = 'reports/example.csv'
    manifest = dict(files=[dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())])
    manifest_path = tmp_path / bundle.MANIFEST
    manifest_path.write_text(json.dumps(manifest), encoding='utf8')
    target = tmp_path / 'test.zip'
    with zipfile.ZipFile(target, 'x') as archive:
        archive.writestr(name, payload)
        archive.writestr('reports/' + bundle.common.NAME + '/' + bundle.MANIFEST, manifest_path.read_bytes())
    return target, manifest_path


def test_archive_verifies_every_entry_including_manifest(tmp_path):
    target, manifest = archive_fixture(tmp_path)
    assert bundle.verify_archive(target, manifest)['all_entry_sha256'] == 'PASS'
    manifest.write_text(manifest.read_text(encoding='utf8') + '\n', encoding='utf8')
    with pytest.raises(ValueError, match='SHA/size mismatch'):
        bundle.verify_archive(target, manifest)


def test_archive_rejects_unmanifested_entry(tmp_path):
    target, manifest = archive_fixture(tmp_path)
    with zipfile.ZipFile(target, 'a') as archive:
        archive.writestr('unlisted.txt', 'unexpected')
    with pytest.raises(ValueError, match='entry set'):
        bundle.verify_archive(target, manifest)


def test_sealed_artifact_replay_never_overwrites(tmp_path):
    path = tmp_path / 'RESULTS_KO.md'
    reporting._put(path, 'same\n')
    reporting._put(path, 'same\n')
    with pytest.raises(FileExistsError):
        reporting._put(path, 'different\n')
    assert path.read_text(encoding='utf8') == 'same\n'


def test_documents_keep_incomplete_and_scope_caveats_with_registered_labels(tmp_path):
    rows = []
    for dataset in analysis.DATASETS:
        for method in ('RAW', 'C1', 'C2', 'C3', 'OURS', 'A1', 'A2'):
            n = 1 if method in ('RAW', 'C1') else 10
            for i in range(n):
                rows.append(dict(dataset=dataset, method_id=method, seed=92601+i,
                    extended=False, cap_selected=method == 'A1', MSE=1.0,
                    job_id=f'{dataset}_{method}_{i}'))
    summary = analysis.choose_primary(rows)
    decision = analysis.decisions(rows, summary)
    checks = reporting.prediction_checks(summary, [])
    reporting._documents(tmp_path, summary, decision, checks,
        dict(d0=[], d1_summary=[]), dict(status='PARTIAL', elapsed_seconds=3600,
            remaining_seconds=0, stop_reason='deadline'), [], 'final')
    report = (tmp_path / 'RESULTS_KO.md').read_text(encoding='utf8')
    slide = (tmp_path / 'SLIDE_NUMBERS_UPDATED.md').read_text(encoding='utf8')
    assert 'INCOMPLETE' in report and 'UNCLEAR_PROVISIONAL' in report
    assert 'NOT_RUN_SCOPE' in report and 'undertrained_at_10' in report
    assert '주 집계 (jena·ETTh1·appliances' in report
    assert '경과=3600.0초 (1.00시간)' in report
    assert 'P contribution verdict = UNCLEAR_PROVISIONAL' in slide
    assert 'Whole-D contribution verdict = INCOMPLETE' in slide
