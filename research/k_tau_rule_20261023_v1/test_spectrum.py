"""Synthetic spectrum fixtures only; no campaign arrays, model or PCA loads."""
import csv
from types import SimpleNamespace

import numpy as np
import pytest

from research.b2_revin_20261018_v1 import contracts
from research.k_tau_rule_20261023_v1 import spectrum


@pytest.mark.parametrize(('values', 'tau', 'expected'), [
    ([5., 3., 2.], .8, 2), ([5., 3., 2.], .5, 1),
    ([5., 3., 2.], 1., 3), ([5., 3., 2.], .800001, 3),
    ([1., 0., 0.], .9, 1),
])
def test_select_k_smallest_exact_threshold(values, tau, expected):
    assert spectrum.select_k(values, tau) == expected


@pytest.mark.parametrize('values', [[], [0., 0.], [1., 2.], [2., -1.],
                                   [float('nan'), 1.], [[2., 1.]]])
def test_select_k_invalid_spectra_fail(values):
    with pytest.raises(ValueError):
        spectrum.select_k(values, .9)


@pytest.mark.parametrize('tau', [0., -1., 1.1, float('nan')])
def test_select_k_invalid_threshold_fails(tau):
    with pytest.raises(ValueError):
        spectrum.select_k([2., 1.], tau)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    report = tmp_path / 'reports' / spectrum.NAME
    report.mkdir(parents=True)
    prereg = report / 'PREREGISTRATION.json'
    prereg.write_bytes(b'{"synthetic_fixture":true}\n')
    receipt = dict(commit='c' * 40, preregistration_sha256=contracts.sha(prereg), commit_unix=100)
    calls, git_calls, contexts, records = [], [], {}, []
    source = tmp_path / 'research' / 'fixture_data_contract.py'
    source.parent.mkdir()
    source.write_text('# Synthetic fixture module; no source execution.\n')

    def git_run(argv, **kwargs):
        git_calls.append((argv, kwargs))
        assert argv[0] == 'git' and kwargs['cwd'] == tmp_path
        if argv[1] == 'show' and argv[2].startswith('c' * 40 + ':'):
            output = b'{"synthetic_fixture":true}\n'
        elif argv[1:3] == ['show', '-s']:
            output = b'100\n'
        elif argv[1:3] == ['merge-base', '--is-ancestor']:
            output = b''
        else:
            pytest.fail(f'Unexpected git operation: {argv}')
        return SimpleNamespace(returncode=0, stdout=output, stderr=b'')

    monkeypatch.setattr(spectrum.subprocess, 'run', git_run)
    for dataset in spectrum.DATASETS:
        task = dict(dataset=dataset, L=512, H=48, columns=list('abcd'),
            physical=SimpleNamespace(shape=(8000, 4)), observed=SimpleNamespace(shape=(8000, 4)),
            data_sha256='d' * 64,
            split_ranges=dict(A_fit=[0, 5000], A_stop=[5000, 6000],
                M_cal=[6000, 7000], V_select=[7000, 8000], T_final=[8000, 10000]))

        def statistics(actual_task, fit_end, *, expected=task):
            assert actual_task is expected
            calls.append((expected['dataset'], 'statistics', fit_end))
            return np.zeros(4), np.ones(4), np.ones(4)

        def pca(actual_task, fit_end, k, mean, std, *, expected=task):
            assert actual_task is expected and k == 1
            assert np.array_equal(mean, np.zeros(4)) and np.array_equal(std, np.ones(4))
            calls.append((expected['dataset'], 'pca', fit_end))
            return dict(eigenvalues=np.array([6., 3., 1., 0.]) if fit_end == 5000
                        else np.array([4., 3., 2., 1.]), fit_end=fit_end)

        module = SimpleNamespace(__file__=str(source), fit_statistics=statistics, initialize_pca=pca)
        contexts[dataset] = dict(repo=tmp_path, task=task, modules={'data': module},
            scientific_source_hashes={source.relative_to(tmp_path).as_posix(): contracts.sha(source)})
        records.append(dict(dataset=dataset, K=1, fit_end=6000, explained_variance=.4))
    contracts.sealed_json(report / 'PRIOR_PCA_DISCLOSURE.json', dict(scalar_storage=dict(records=records)))
    return contexts, report, receipt, calls, git_calls


def test_same_functions_task_fit_ends_and_gates(campaign):
    contexts, report, receipt, calls, git_calls = campaign
    result = spectrum.compute_spectra(contexts, report, receipt)
    assert result['K_main'] == {dataset: 2 for dataset in spectrum.DATASETS}
    assert len(calls) == 16
    for dataset in spectrum.DATASETS:
        assert [call[1:] for call in calls if call[0] == dataset] == [
            ('statistics', 5000), ('pca', 5000), ('statistics', 6000), ('pca', 6000)]
    validation = contracts.read(report / 'SPECTRUM_VALIDATION.json')
    assert validation['G1']['status'] == 'PASS' and validation['G2']['status'] == 'PASS'
    assert validation['G3']['status'] == 'PASS'
    assert validation['source_hashes_pre'] == validation['source_hashes_post']
    assert validation['created_unix'] >= receipt['commit_unix']
    assert git_calls
    with (report / 'K_TAU.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    row = next(r for r in rows if r['dataset'] == 'jena' and float(r['tau']) == .95)
    assert row['K_A_fit'] == '3' and row['K_A60'] == '4'
    assert row['compression_A60'] == 'NO_COMPRESSION'
    with (report / 'SPECTRUM.csv').open(newline='') as handle:
        assert len(list(csv.DictReader(handle))) == 32


def test_preregistration_sha_mismatch_fails_before_any_pca(campaign):
    contexts, report, receipt, calls, _ = campaign
    receipt['preregistration_sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='preregistration|Preregistration'):
        spectrum.compute_spectra(contexts, report, receipt)
    assert calls == [] and not (report / 'SPECTRUM.csv').exists()


@pytest.mark.parametrize('operation', ['show', 'ancestry', 'time'])
def test_invalid_commit_evidence_fails_before_pca(campaign, monkeypatch, operation):
    contexts, report, receipt, calls, _ = campaign
    real = spectrum.subprocess.run

    def bad(argv, **kwargs):
        result = real(argv, **kwargs)
        if operation == 'show' and argv[1] == 'show' and ':' in argv[2]:
            result.stdout = b'{"different_commit_bytes":true}'
        elif operation == 'ancestry' and argv[1] == 'merge-base':
            result.returncode = 1
        elif operation == 'time' and argv[1:3] == ['show', '-s']:
            result.stdout = b'101\n'
        return result

    monkeypatch.setattr(spectrum.subprocess, 'run', bad)
    with pytest.raises(ValueError):
        spectrum.compute_spectra(contexts, report, receipt)
    assert calls == []


@pytest.mark.parametrize('filename', ['SPECTRUM.csv', 'K_TAU.csv', 'SPECTRUM_VALIDATION.json'])
def test_existing_artifacts_are_never_overwritten(campaign, filename):
    contexts, report, receipt, calls, _ = campaign
    path = report / filename
    path.write_bytes(b'existing bytes')
    with pytest.raises(FileExistsError):
        spectrum.compute_spectra(contexts, report, receipt)
    assert path.read_bytes() == b'existing bytes' and calls == []


def test_scalar_mismatch_preserves_new_raw_spectrum_and_failed_validation(campaign):
    contexts, report, receipt, calls, _ = campaign
    path = report / 'PRIOR_PCA_DISCLOSURE.json'
    value = contracts.read(path)
    value['scalar_storage']['records'][0]['explained_variance'] = .45
    contracts.atomic_json(path, value)
    with pytest.raises(ValueError, match='G2'):
        spectrum.compute_spectra(contexts, report, receipt)
    assert calls and (report / 'SPECTRUM.csv').is_file()
    validation = contracts.read(report / 'SPECTRUM_VALIDATION.json')
    assert validation['G2']['status'] == 'FAIL' and validation['G2']['errors']


def test_bad_boundary_fails_before_any_pca(campaign):
    contexts, report, receipt, calls, _ = campaign
    contexts['jena']['task']['split_ranges']['A_stop'] = [5000, 7500]
    with pytest.raises(ValueError, match='G3|boundary|chronological'):
        spectrum.compute_spectra(contexts, report, receipt)
    assert calls == []


def test_source_hash_mismatch_fails_before_any_pca(campaign):
    contexts, report, receipt, calls, _ = campaign
    contexts['jena']['scientific_source_hashes']['research/fixture_data_contract.py'] = '0' * 64
    with pytest.raises(ValueError, match='source|Source'):
        spectrum.compute_spectra(contexts, report, receipt)
    assert calls == []


def test_source_change_during_computation_fails_and_preserves_rows(campaign):
    contexts, report, receipt, calls, _ = campaign
    module = contexts['jena']['modules']['data']
    original = module.initialize_pca

    def changed(*args):
        result = original(*args)
        with open(module.__file__, 'a', encoding='utf-8') as handle:
            handle.write('# fixture source mutation\n')
        return result

    module.initialize_pca = changed
    with pytest.raises(ValueError, match='source hashes changed'):
        spectrum.compute_spectra(contexts, report, receipt)
    validation = contracts.read(report / 'SPECTRUM_VALIDATION.json')
    assert validation['core_hashes_unchanged'] is False
    assert validation['status'] == 'FAILED' and (report / 'SPECTRUM.csv').is_file()


def test_scalar_difference_within_tolerance_passes(campaign):
    contexts, report, receipt, calls, _ = campaign
    path = report / 'PRIOR_PCA_DISCLOSURE.json'
    value = contracts.read(path)
    value['scalar_storage']['records'][0]['explained_variance'] = .4000009
    contracts.atomic_json(path, value)
    result = spectrum.compute_spectra(contexts, report, receipt)
    assert result['validation']['G2']['status'] == 'PASS'
