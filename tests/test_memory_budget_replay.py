import csv
import hashlib
import importlib
import json
import sys

import pytest


def modules():
    return (importlib.import_module('research.memory_budget_20261018_v1.analysis'),
            importlib.import_module('research.memory_budget_20261018_v1.replay'))


def portable_receipt(identifier, *, device='cuda', stage='confirm', batch=1):
    count = 20 if stage == 'confirm' else 3
    warmups = 10 if stage == 'confirm' else 2
    pid = 100 if device == 'cuda' else 200
    times = [.01 + index * .001 for index in range(count)]
    trials = []
    for phase, repeats in [('warmup', warmups), ('timed', count)]:
        for index in range(repeats):
            trials.append(dict(phase=phase, trial=index, index=index, pid=pid,
                whole_seconds=times[index % count],
                F_logical_streams=batch * 6 if device == 'cuda' else 0,
                F_physical_streams=batch * 6 if device == 'cuda' else 0,
                F_calls=1 if device == 'cuda' else 0,
                peak_allocated_MiB=(200. + index % 3 if phase == 'timed' else 400.) if device == 'cuda' else None,
                peak_reserved_MiB=(230. + index if phase == 'timed' else 500.) if device == 'cuda' else None,
                CPU_RSS_MiB=100. + index, process_GPU_MiB=None))
    return dict(schema_version=1, status='COMPLETE', facet_id=identifier,
        process=dict(pid=pid, create_time=1., started_unix=1., finished_unix=5., wall_seconds=4.),
        summary=dict(facet_id=identifier, status='COMPLETE', stage=stage,
            accuracy_run_id='parent_' + identifier, dataset='jena',
            family='B2' if device == 'cuda' else 'RAW_RIDGE', seed=92601,
            B=batch, batch_size=batch, C=21, K=6, L=512, H=48,
            row_microbatch=batch * 6 if device == 'cuda' else None,
            effective_row_microbatch=batch * 6 if device == 'cuda' else None,
            is_original=True, device=device,
            memory_layer='CUDA_ALLOCATED_TENSORS' if device == 'cuda' else 'CPU_RSS_SAMPLED',
            checkpoint_sha256='a' * 64, prediction_sha256='b' * 64,
            parity_status='PASS', MSE=1., MAE=.5, warmups=warmups, timed_trials=count,
            CPU_RSS_sampled_peak_MiB=150., process_GPU_sampled_peak_MiB=None,
            process_GPU_sampling_status='UNAVAILABLE' if device == 'cuda' else 'NOT_APPLICABLE_CPU'),
        trials=trials, parity=dict(status='PASS', full_V_origins=43,
            original_metrics=dict(MSE=1., MAE=.5, channel_sse=[1.], channel_sae=[.5], channel_counts=[1]),
            independent_metrics=dict(MSE=1., MAE=.5, channel_sse=[1.], channel_sae=[.5], channel_counts=[1])),
        private_receipt_sha256='c' * 64, worker_source_sha256='d' * 64)


def fixture_report(tmp_path, *, receipts=None):
    analysis, _ = modules()
    report = tmp_path / 'published'
    directory = report / 'receipts'
    directory.mkdir(parents=True)
    if receipts is None:
        receipts = [portable_receipt('gpu', batch=8), portable_receipt('ridge', device='cpu')]
    paths = []
    for value in receipts:
        path = directory / (value['facet_id'] + '_' + value['summary']['stage'] + '.receipt.json')
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')
        paths.append(path)
    analysis.export_receipts(paths, report, make_figures=False)
    return report


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


def test_csv_independent_replay_uses_timed_trials_and_batch_units(tmp_path):
    report = fixture_report(tmp_path)
    proof = modules()[1].verify_csv(report)
    assert proof['status'] == 'PASS'
    assert proof['verified_facets'] == 2
    assert proof['raw_trial_rows'] == 60
    gpu = next(row for row in read_csv(report / 'MEMORY_COST_SUMMARY.csv') if row['device'] == 'cuda')
    assert float(gpu['median_batch_ms']) == pytest.approx(19.5)
    assert float(gpu['IQR_batch_ms']) == pytest.approx(9.5)
    assert float(gpu['p95_batch_ms']) == pytest.approx(28.05)
    assert float(gpu['median_ms_per_origin']) == pytest.approx(19.5 / 8)
    assert float(gpu['origins_per_second']) == pytest.approx(8 * 1000 / 19.5)
    assert float(gpu['peak_allocated_MiB']) == 202.
    assert float(gpu['peak_reserved_MiB']) == 249.


@pytest.mark.parametrize('field', ['median_batch_ms', 'IQR_batch_ms', 'p95_batch_ms',
    'median_ms_per_origin', 'origins_per_second', 'peak_allocated_MiB', 'peak_reserved_MiB'])
def test_csv_replay_rejects_each_modified_summary_statistic(tmp_path, field):
    report = fixture_report(tmp_path)
    path = report / 'MEMORY_COST_SUMMARY.csv'
    rows = read_csv(path)
    gpu = next(row for row in rows if row['device'] == 'cuda')
    gpu[field] = str(float(gpu[field]) + .5)
    write_csv(path, rows)
    with pytest.raises(ValueError, match='independent statistic differs: ' + field):
        modules()[1].verify_csv(report)


@pytest.mark.parametrize('phase', ['warmup', 'timed'])
def test_csv_replay_rejects_missing_actual_repetition(tmp_path, phase):
    report = fixture_report(tmp_path)
    path = report / 'MEMORY_COST_TRIALS.csv'
    rows = read_csv(path)
    removed = next(row for row in rows if row['device'] == 'cuda' and row['phase'] == phase)
    rows.remove(removed)
    write_csv(path, rows)
    with pytest.raises(ValueError, match='repeat coverage differs'):
        modules()[1].verify_csv(report)


def test_csv_replay_rejects_mixed_process_trials(tmp_path):
    report = fixture_report(tmp_path)
    path = report / 'MEMORY_COST_TRIALS.csv'
    rows = read_csv(path)
    rows[0]['pid'] = '999'
    write_csv(path, rows)
    with pytest.raises(ValueError, match='recorded isolated worker'):
        modules()[1].verify_csv(report)


def test_csv_cpu_context_has_no_cuda_memory_and_no_gpu_speed_comparison(tmp_path):
    report = fixture_report(tmp_path)
    cpu = next(row for row in read_csv(report / 'MEMORY_COST_SUMMARY.csv') if row['device'] == 'cpu')
    assert cpu['peak_allocated_MiB'] == cpu['peak_reserved_MiB'] == ''
    pareto = read_csv(report / 'PARETO_TABLE.csv')
    assert {row['comparison_scope'] for row in pareto if row['device'] == 'cpu'} == {'CPU_CONTEXT'}
    assert not any(row['device'] == 'cpu' and row['comparison_scope'] in ('TSFM', 'GPU_PRACTICAL') for row in pareto)
    assert modules()[1].verify_csv(report)['status'] == 'PASS'


def test_csv_keeps_screen_and_confirm_of_one_facet_distinct(tmp_path):
    report = fixture_report(tmp_path, receipts=[portable_receipt('same', stage='screen'),
                                               portable_receipt('same', stage='confirm')])
    proof = modules()[1].verify_csv(report)
    assert proof['verified_facets'] == 2
    assert proof['raw_trial_rows'] == 35


def test_public_replay_needs_only_portable_receipts_and_preserves_source(tmp_path, monkeypatch):
    report = fixture_report(tmp_path)
    before = tree_bytes(report)
    monkeypatch.setitem(sys.modules, 'torch', None)
    proof = modules()[1].replay(report, tmp_path / 'fresh_replay', figures=False)
    assert proof['status'] == 'PASS'
    assert proof['verified_facets'] == 2
    assert tree_bytes(report) == before
    assert len(proof['products']) == 7
    for item in proof['products']:
        content = (tmp_path / 'fresh_replay' / item['file']).read_bytes()
        assert item['sha256'] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize('destination', ['same', 'child', 'ancestor'])
def test_public_replay_rejects_source_overlapping_destinations(tmp_path, destination):
    report = fixture_report(tmp_path)
    before = tree_bytes(report)
    output = {'same': report, 'child': report / 'nested', 'ancestor': report.parent}[destination]
    with pytest.raises(ValueError, match='separate from published evidence'):
        modules()[1].replay(report, output, figures=False)
    assert tree_bytes(report) == before


def test_public_replay_rejects_existing_output_without_overwriting_it(tmp_path):
    report = fixture_report(tmp_path)
    output = tmp_path / 'existing'
    output.mkdir()
    sentinel = output / 'MEMORY_COST_SUMMARY.csv'
    sentinel.write_bytes(b'preserve existing user content\n')
    before = tree_bytes(output)
    with pytest.raises(ValueError, match='fresh|exist|empty'):
        modules()[1].replay(report, output, figures=False)
    assert tree_bytes(output) == before


def test_public_replay_detects_published_csv_corruption(tmp_path):
    report = fixture_report(tmp_path)
    path = report / 'ACCURACY_RECHECK.csv'
    path.write_bytes(path.read_bytes() + b'corrupted extra row\n')
    with pytest.raises(ValueError, match='Public replay bytes differ: ACCURACY_RECHECK.csv'):
        modules()[1].replay(report, tmp_path / 'fresh_replay', figures=False)


def test_public_replay_rejects_missing_portable_receipts(tmp_path):
    report = tmp_path / 'empty_report'
    report.mkdir()
    with pytest.raises(ValueError, match='Portable receipts required'):
        modules()[1].replay(report, tmp_path / 'fresh_replay', figures=False)
