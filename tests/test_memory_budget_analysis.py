import csv
import importlib
import json

import pytest


def analysis():
    return importlib.import_module('research.memory_budget_20261018_v1.analysis')


def row(identifier, *, family='B2', mse=1., memory=200., latency=20.,
        chunk=6, original=False, dataset='jena', batch=1, stage='confirm',
        status='CONFIRMED', device='cuda', parity='PASS', run_id='b2_s1'):
    return dict(facet_id=identifier, accuracy_run_id=run_id, dataset=dataset,
        family=family, seed=92601, B=batch, batch_size=batch, C=21, K=6,
        L=512, H=48, stage=stage, status=status, device=device,
        memory_layer='CUDA_ALLOCATED_TENSORS' if device == 'cuda' else 'CPU_RSS_SAMPLED',
        MSE=mse, MAE=mse / 2, peak_allocated_MiB=memory,
        peak_reserved_MiB=memory + 10, median_batch_ms=latency,
        row_microbatch=chunk, effective_row_microbatch=chunk, original=original,
        parity_status=parity, checkpoint_sha256='a' * 64,
        CPU_RSS_sampled_peak_MiB=memory if device == 'cpu' else 100.,
        timed_peak_allocated_min_MiB=memory,
        timed_peak_allocated_max_MiB=memory)


def receipt(identifier='one', *, stage='screen', times=(.01, .02, .03),
            memory=(200., 200., 200.), family='B2', device='cuda'):
    summary = row(identifier, stage=stage, family=family, device=device,
                  status='COMPLETE', memory=max(memory), original=True)
    summary.update(warmups=2, timed_trials=len(times), median_batch_ms=999.,
                   IQR_batch_ms=999., p95_batch_ms=999.)
    trials = [dict(phase='timed', trial=i, whole_seconds=value,
                  peak_allocated_MiB=memory[i] if device == 'cuda' else None,
                  peak_reserved_MiB=memory[i] + 10 if device == 'cuda' else None,
                  F_logical_streams=6 if family == 'B2' else 0,
                  F_physical_streams=6 if family == 'B2' else 0,
                  F_calls=1 if family == 'B2' else 0,
                  pid=100, CPU_RSS_MiB=100.)
              for i, value in enumerate(times)]
    trials = [dict(trials[0], phase='warmup', trial=i) for i in range(2)] + trials
    return dict(schema_version=1, status='COMPLETE', facet_id=identifier,
        process=dict(pid=100, create_time=1., wall_seconds=4.), summary=summary,
        trials=trials, parity=dict(status='PASS', full_V_origins=43,
            original_metrics=dict(MSE=1., MAE=.5),
            independent_metrics=dict(MSE=1.00000001, MAE=.50000001),
            metric_deltas=dict(MSE=999., MAE=999.)), accuracy=dict(run_id='b2_s1'))


def test_schedule_clamps_and_deduplicates_without_origin_padding():
    m = analysis()
    assert m.schedule(6, 1, 6) == [6, 1]
    assert m.schedule(21, 8, 6) == [128, 48, 1]
    assert m.schedule(8, 8, 8) == [64, 1]
    assert m.schedule(None, 8, 8) == [None]


def test_schedule_phase_a_exact_44_and_phase_b_common_reference_exact_20():
    m = analysis()
    counts = sum(len(m.schedule(w, b, k)) for w, k in
        [(6, 6)] * 4 + [(21, 6)] * 2 + [(8, 8)] * 2 + [(32, 8), (None, 8)]
        for b in (1, 8))
    assert counts == 44
    assert sum(len(m.schedule(w, b, 6)) for w in (3, 3, 12, 12) for b in (1, 8)) == 20


@pytest.mark.parametrize('args', [(0, 1, 6), (6, 2, 6), (6, 1, 0), (True, 1, 6)])
def test_schedule_rejects_unregistered_or_nonpositive_axes(args):
    with pytest.raises(ValueError):
        analysis().schedule(*args)


def test_confirmation_original_always_and_selection_ignores_accuracy():
    rows = [row('original', original=True, memory=250., latency=40., stage='screen'),
            row('fast', memory=230., latency=20., stage='screen'),
            row('small', chunk=1, memory=190., latency=30., stage='screen'),
            row('worse', memory=240., latency=25., mse=.01, stage='screen')]
    first = analysis().select_confirmations(rows)
    for value in rows:
        value['MSE'] = 1000. - value['MSE']
    second = analysis().select_confirmations(rows)
    assert {r['facet_id'] for r in first if r['confirm_selected']} == {'original', 'fast', 'small'}
    assert [(r['facet_id'], r['confirm_selected']) for r in first] == [
        (r['facet_id'], r['confirm_selected']) for r in second]


def test_expensive_row_one_is_screened_but_original_never_excluded():
    result = analysis().select_confirmations([
        row('original', original=True, chunk=1, latency=5000., stage='screen'),
        row('other', chunk=1, latency=5000., memory=190., stage='screen')])
    assert result[0]['confirm_selected'] is True
    assert result[1]['confirm_selected'] is False
    assert result[1]['confirmation_reason'] == 'SCREENED_COST_EXCESS'


def test_invalid_parity_cannot_be_a_confirmation_candidate():
    output = analysis().select_confirmations([
        row('bad', original=True, stage='screen', parity='FAIL')])
    assert output[0]['confirm_selected'] is False
    assert output[0]['confirmation_reason'] == 'INVALID_PARITY'


def test_pareto_uses_three_axes_and_preserves_ties():
    rows = [row('a', mse=1., memory=200., latency=20.),
            row('b', mse=1., memory=200., latency=20.),
            row('dominated', mse=1.1, memory=201., latency=21.),
            row('tradeoff', mse=.8, memory=250., latency=25.)]
    result = [r for r in analysis().pareto_rows(rows) if r['comparison_scope'] == 'TSFM']
    assert {r['facet_id'] for r in result if r['frontier']} == {'a', 'b', 'tradeoff'}
    bad = next(r for r in result if r['facet_id'] == 'dominated')
    assert set(bad['dominating_facet_ids']) == {'a', 'b'}


def test_pareto_never_compares_batch_dataset_device_layer_or_screen():
    rows = [row('a'), row('batch8', batch=8, mse=.1, memory=1., latency=1.),
            row('pems', dataset='pems32', mse=.1, memory=1., latency=1.),
            row('cpu', family='RAW_RIDGE', device='cpu', mse=.1, memory=1., latency=1.),
            row('screen', stage='screen', status='SCREENED', mse=.1, memory=1., latency=1.),
            row('badparity', parity='FAIL', mse=.1, memory=1., latency=1.)]
    result = analysis().pareto_rows(rows)
    assert next(r for r in result if r['facet_id'] == 'a' and r['comparison_scope'] == 'TSFM')['frontier']
    assert not any(r['facet_id'] in ('screen', 'badparity') for r in result)
    assert not any(r['facet_id'] == 'cpu' and r['comparison_scope'] != 'CPU_CONTEXT' for r in result)


def test_direct_g_can_dominate_practical_without_changing_tsfm_front():
    result = analysis().pareto_rows([
        row('b2'), row('g', family='DIRECT_G', mse=.1, memory=10., latency=1.)])
    b2 = {r['comparison_scope']: r for r in result if r['facet_id'] == 'b2'}
    assert b2['TSFM']['frontier'] is True
    assert b2['GPU_PRACTICAL']['frontier'] is False


def test_phase_b_requires_frontier_memory_separation_and_whole_budget():
    m = analysis()
    rows = [row('full', original=True, memory=220., latency=10.),
            row('one', chunk=1, memory=200., latency=20.)]
    kwargs = dict(remaining_seconds=5000., estimated_training_seconds=1000.,
                  estimated_measurement_seconds=1000.)
    assert m.phase_b_decision(rows, **kwargs)['status'] == 'PROCEED'
    assert m.phase_b_decision(rows, **dict(kwargs, remaining_seconds=3799.))['reason'] == 'INSUFFICIENT_FULL_PHASE_B_BUDGET'
    rows[1]['timed_peak_allocated_max_MiB'] = 225.
    assert m.phase_b_decision(rows, **kwargs)['reason'] == 'NO_REPEATED_TRIAL_MEMORY_SEPARATION'


def test_phase_b_does_not_use_pems_only_or_dominated_jena_frontier():
    m = analysis()
    kwargs = dict(remaining_seconds=10000., estimated_training_seconds=1000.,
                  estimated_measurement_seconds=1000.)
    rows = [row('pems', dataset='pems32'),
            row('full', memory=220., latency=20.),
            row('small', chunk=1, memory=200., latency=30.),
            row('lora', family='LORA', mse=.1, memory=190., latency=10.)]
    assert m.phase_b_decision(rows, **kwargs)['reason'] == 'NO_JENA_B2_TSFM_FRONTIER'


def test_export_recomputes_trial_statistics_and_metric_deltas(tmp_path):
    result = analysis().export_receipts([receipt()], tmp_path, make_figures=False)
    with (tmp_path / 'MEMORY_COST_SUMMARY.csv').open(newline='', encoding='utf-8') as handle:
        values = list(csv.DictReader(handle))
    assert float(values[0]['median_batch_ms']) == 20.
    assert float(values[0]['IQR_batch_ms']) == 10.
    assert float(values[0]['p95_batch_ms']) == pytest.approx(29.)
    assert float(values[0]['origins_per_second']) == 50.
    with (tmp_path / 'ACCURACY_RECHECK.csv').open(newline='', encoding='utf-8') as handle:
        accuracy = list(csv.DictReader(handle))
    assert float(accuracy[0]['delta_MSE']) == pytest.approx(1e-8)
    assert result['receipt_count'] == 1


def test_export_rejects_duplicate_facets_and_mismatched_pid_or_repeats(tmp_path):
    m = analysis()
    with pytest.raises(ValueError, match='duplicate'):
        m.export_receipts([receipt(), receipt()], tmp_path, make_figures=False)
    value = receipt()
    value['trials'][0]['pid'] = 999
    with pytest.raises(ValueError, match='PID'):
        m.export_receipts([value], tmp_path, make_figures=False)
    value = receipt()
    value['summary']['timed_trials'] = 4
    with pytest.raises(ValueError, match='repeat'):
        m.export_receipts([value], tmp_path, make_figures=False)


def test_export_preserves_invalid_failure_without_ranking(tmp_path):
    value = receipt()
    value['status'] = 'INVALID'
    value['failure'] = dict(exception_type='ParityError', message='full V mismatch')
    value['summary']['parity_status'] = 'FAIL'
    value['parity']['status'] = 'FAIL'
    analysis().export_receipts([value], tmp_path, make_figures=False)
    with (tmp_path / 'RUN_MANIFEST.csv').open(newline='', encoding='utf-8') as handle:
        manifest = list(csv.DictReader(handle))
    assert manifest[0]['status'] == 'INVALID'
    assert 'full V mismatch' in manifest[0]['reason']
    assert json.loads((tmp_path / 'PARITY_AUDIT.json').read_text(encoding='utf-8'))['facets'][0]['status'] == 'FAIL'
    with (tmp_path / 'PARETO_TABLE.csv').open(newline='', encoding='utf-8') as handle:
        assert list(csv.DictReader(handle)) == []


def test_export_is_byte_reproducible_and_cpu_context_separate(tmp_path):
    m = analysis()
    cpu = receipt('ridge', family='RAW_RIDGE', device='cpu')
    cpu['summary']['peak_allocated_MiB'] = None
    cpu['summary']['CPU_RSS_sampled_peak_MiB'] = 100.
    first, second = tmp_path / 'a', tmp_path / 'b'
    m.export_receipts([receipt(), cpu], first, make_figures=False)
    m.export_receipts([receipt(), cpu], second, make_figures=False)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    with (first / 'RIDGE_CPU_CONTEXT.csv').open(newline='', encoding='utf-8') as handle:
        assert [r['facet_id'] for r in csv.DictReader(handle)] == ['ridge']


def test_confirm_repeat_protocol_and_figures_are_deterministic(tmp_path):
    value = receipt(stage='confirm', times=tuple(.01 + i * .001 for i in range(20)),
                    memory=(200.,) * 20)
    value['summary']['warmups'] = 10
    value['trials'] = [dict(value['trials'][0], trial=i) for i in range(10)] + [
        trial for trial in value['trials'] if trial['phase'] == 'timed']
    first, second = tmp_path / 'a', tmp_path / 'b'
    analysis().export_receipts([value], first)
    analysis().export_receipts([value], second)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    assert (first / 'FIG1_memory_mse.png').stat().st_size > 1000
    assert not (first / 'FIG3_k_accuracy_memory.png').exists()


def test_phase_b_figure_requires_actual_new_k_confirmation(tmp_path):
    value = receipt(stage='confirm', times=(.01,) * 20, memory=(200.,) * 20)
    value['summary'].update(warmups=10, K=3)
    value['trials'] = [dict(value['trials'][0], trial=i) for i in range(10)] + [
        trial for trial in value['trials'] if trial['phase'] == 'timed']
    analysis().export_receipts([value], tmp_path)
    assert (tmp_path / 'FIG3_k_accuracy_memory.png').exists()


def test_export_rejects_reserved_below_allocated_and_wrong_registered_repeats(tmp_path):
    value = receipt()
    value['trials'][-1]['peak_reserved_MiB'] = 1.
    with pytest.raises(ValueError, match='Reserved'):
        analysis().export_receipts([value], tmp_path, make_figures=False)
    value = receipt(stage='confirm')
    with pytest.raises(ValueError, match='registered'):
        analysis().export_receipts([value], tmp_path, make_figures=False)


def test_same_facet_screen_and_confirm_are_distinct_measurements(tmp_path):
    screened = receipt('same')
    confirmed = receipt('same', stage='confirm', times=(.02,) * 20, memory=(200.,) * 20)
    confirmed['summary']['warmups'] = 10
    confirmed['trials'] = [dict(confirmed['trials'][0], trial=i) for i in range(10)] + [
        trial for trial in confirmed['trials'] if trial['phase'] == 'timed']
    result = analysis().export_receipts([screened, confirmed], tmp_path, make_figures=False)
    assert result['receipt_count'] == 2
    assert result['confirmed_count'] == 1
    audit = json.loads((tmp_path / 'PARITY_AUDIT.json').read_text(encoding='utf-8'))
    assert {value['stage'] for value in audit['facets']} == {'screen', 'confirm'}


def test_channel_macro_recheck_does_not_use_micro_aggregation_or_unobserved_channels(tmp_path):
    value = receipt()
    metrics = dict(MSE=2., MAE=1., channel_sse=[2., 30., 0.],
                   channel_sae=[1., 15., 0.], channel_counts=[2, 10, 0])
    value['parity']['original_metrics'] = metrics
    value['parity']['independent_metrics'] = metrics
    analysis().export_receipts([value], tmp_path, make_figures=False)
    with (tmp_path / 'ACCURACY_RECHECK.csv').open(newline='', encoding='utf-8') as handle:
        output = list(csv.DictReader(handle))[0]
    assert float(output['regenerated_MSE']) == 2.
    assert output['regenerated_metric_source'] == 'INDEPENDENT_FP64_CHANNEL_SUFFICIENT_STATISTICS'
    value['parity']['independent_metrics'] = dict(metrics, MSE=32 / 12)
    with pytest.raises(ValueError, match='contradicts'):
        analysis().export_receipts([value], tmp_path / 'bad', make_figures=False)


def test_export_manifest_records_expensive_row_one_screen_reason(tmp_path):
    original = receipt('original')
    slow = receipt('slow', times=(5., 5., 5.), memory=(190., 190., 190.))
    slow['summary'].update(original=False, row_microbatch=1, effective_row_microbatch=1)
    analysis().export_receipts([original, slow], tmp_path, make_figures=False)
    with (tmp_path / 'RUN_MANIFEST.csv').open(newline='', encoding='utf-8') as handle:
        manifest = {value['facet_id']: value for value in csv.DictReader(handle)}
    assert manifest['slow']['status'] == 'SCREENED'
    assert manifest['slow']['reason'] == 'SCREENED_COST_EXCESS'
    assert manifest['slow']['confirmation_reason'] == 'SCREENED_COST_EXCESS'
    assert manifest['slow']['confirm_selected'] == 'False'
    assert manifest['original']['reason'] == 'ORIGINAL_PATH_REQUIRED'
    assert manifest['original']['confirm_selected'] == 'True'


def test_export_manifest_links_confirm_reason_and_replays_without_external_selection(tmp_path):
    screened = receipt('same')
    confirmed = receipt('same', stage='confirm', times=(.02,) * 20, memory=(200.,) * 20)
    confirmed['summary']['warmups'] = 10
    confirmed['trials'] = [dict(confirmed['trials'][0], trial=i) for i in range(10)] + [
        trial for trial in confirmed['trials'] if trial['phase'] == 'timed']
    orphan = dict(confirmed, facet_id='no_screen', summary=dict(confirmed['summary'], facet_id='no_screen'))
    report = tmp_path / 'published'
    directory = report / 'receipts'
    directory.mkdir(parents=True)
    paths = []
    for value in (screened, confirmed, orphan):
        path = directory / (value['facet_id'] + '_' + value['summary']['stage'] + '.receipt.json')
        path.write_text(json.dumps(value, sort_keys=True) + '\n', encoding='utf-8')
        paths.append(path)
    analysis().export_receipts(paths, report, make_figures=False)
    with (report / 'RUN_MANIFEST.csv').open(newline='', encoding='utf-8') as handle:
        manifest = {(value['facet_id'], value['stage']): value for value in csv.DictReader(handle)}
    for stage in ('screen', 'confirm'):
        assert manifest[('same', stage)]['confirmation_reason'] == 'ORIGINAL_PATH_REQUIRED'
        assert manifest[('same', stage)]['confirm_selected'] == 'True'
    assert manifest[('same', 'confirm')]['reason'] == 'ACTUAL_ISOLATED_PROCESS_MEASUREMENT'
    assert manifest[('same', 'confirm')]['status'] == 'CONFIRMED'
    assert manifest[('no_screen', 'confirm')]['confirm_selected'] == ''
    assert manifest[('no_screen', 'confirm')]['confirmation_reason'] == 'NOT_AVAILABLE_NO_SCREEN_RECEIPT'
    replay = importlib.import_module('research.memory_budget_20261018_v1.replay')
    proof = replay.replay(report, tmp_path / 'fresh_replay', figures=False)
    assert proof['status'] == 'PASS'
    assert not (report / 'CONFIRM_SELECTION_A.json').exists()


def test_export_manifest_keeps_invalid_failure_reason_with_selection_flag(tmp_path):
    value = receipt('invalid')
    value['status'] = 'INVALID'
    value['failure'] = dict(exception_type='ParityError', message='specific mismatch evidence')
    value['summary']['parity_status'] = 'FAIL'
    value['parity']['status'] = 'FAIL'
    analysis().export_receipts([value], tmp_path, make_figures=False)
    with (tmp_path / 'RUN_MANIFEST.csv').open(newline='', encoding='utf-8') as handle:
        manifest = list(csv.DictReader(handle))[0]
    assert manifest['status'] == 'INVALID'
    assert json.loads(manifest['reason']) == value['failure']
    assert manifest['confirmation_reason'] == 'INVALID_PARITY'
    assert manifest['confirm_selected'] == 'False'


def test_renderer_point_keys_identify_adapter_seeds_and_unseeded_baselines():
    m = analysis()
    assert m._point_label(row('a', chunk=1)) == 'B2 s1 K6 r1'
    second = row('b', family='B0', chunk=48)
    second['seed'] = 92602
    assert m._point_label(second) == 'B0 s2 K6 r48'
    assert m._point_label(row('raw', family='RAW', chunk=21)) == 'RAW r21'
    assert m._point_label(row('lora', family='LORA', chunk=6)) == 'LORA r6'
    assert m._point_label(row('g', family='DIRECT_G', chunk=None)) == 'DIRECT_G no F'


def test_renderer_key_table_maps_all_points_without_changing_coordinates():
    import copy
    rows = [row('a', memory=201.234, latency=20.1234),
            row('b', family='RAW', memory=199.876, latency=50.9876, chunk=21)]
    original = copy.deepcopy(rows)
    keys = analysis()._plot_key_rows(rows, 'median_batch_ms')
    assert keys == [['B2 s1 K6 r6', '201.23', '20.1234'], ['RAW r21', '199.88', '50.9876']]
    assert rows == original


def test_svg_clip_ids_are_canonical_without_geometry_or_reference_changes(tmp_path):
    import xml.etree.ElementTree as ET
    template = '<svg xmlns="http://www.w3.org/2000/svg"><defs><clipPath id="{identifier}"><rect x="1.234567" y="2" width="3" height="4"/></clipPath></defs><g clip-path="url(#{identifier})"><path d="M 1 2 L 3 4"/></g></svg>\n'
    first, second = tmp_path / 'one.svg', tmp_path / 'two.svg'
    first.write_text(template.format(identifier='p_random_a'), encoding='utf-8')
    second.write_text(template.format(identifier='p_random_b'), encoding='utf-8')
    analysis()._canonical_svg_ids(first)
    analysis()._canonical_svg_ids(second)
    assert first.read_bytes() == second.read_bytes()
    tree = ET.fromstring(first.read_text(encoding='utf-8'))
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    clip = tree.find('.//svg:clipPath', ns)
    assert clip.find('svg:rect', ns).attrib == dict(x='1.234567', y='2', width='3', height='4')
    assert tree.find('svg:g', ns).attrib['clip-path'] == 'url(#' + clip.attrib['id'] + ')'
    assert tree.find('.//svg:path', ns).attrib['d'] == 'M 1 2 L 3 4'


def test_renderer_replay_in_separate_os_processes(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    value = receipt(stage='confirm', times=(.02,) * 20, memory=(200.,) * 20)
    value['summary']['warmups'] = 10
    value['trials'] = [dict(value['trials'][0], trial=i) for i in range(10)] + [
        trial for trial in value['trials'] if trial['phase'] == 'timed']
    report = tmp_path / 'published'
    directory = report / 'receipts'
    directory.mkdir(parents=True)
    (directory / 'one.receipt.json').write_text(json.dumps(value) + '\n', encoding='utf-8')
    repo = Path(__file__).resolve().parents[1]
    code = ('from pathlib import Path;import sys;'
            'from research.memory_budget_20261018_v1.analysis import export_receipts;'
            'root=Path(sys.argv[1]);export_receipts(sorted((root/"receipts").glob("*.receipt.json")),root)')
    first_env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    first = subprocess.run([sys.executable, '-c', code, str(report)], cwd=repo,
                           env=first_env, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    second_env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
    second = subprocess.run([sys.executable, '-m', 'research.memory_budget_20261018_v1.replay',
        '--report-root', str(report), '--output', str(tmp_path / 'fresh')],
        cwd=repo, env=second_env, capture_output=True, text=True)
    assert second.returncode == 0, second.stdout + second.stderr
    assert json.loads(second.stdout)['status'] == 'PASS'
