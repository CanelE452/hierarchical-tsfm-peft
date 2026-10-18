import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def modules():
    common = importlib.import_module('research.memory_budget_20261018_v1.common')
    runner = importlib.import_module('research.memory_budget_20261018_v1.run')
    return common, runner


def source_rows():
    rows = []
    for dataset, width, latent in [('jena', 21, 6), ('pems32', 32, 8)]:
        for family in (['B2', 'B0'] if dataset == 'jena' else ['B2']):
            for seed in (92601, 92602):
                rows.append(dict(run_id=f'{dataset}_{family}_{seed}', dataset=dataset,
                    family=family, seed=seed, C=width, K=latent, final=True,
                    revin=False, independent_FP64='PASS',
                    bundle_binding={'sha256': 'a' * 64}))
        for family in (['RAW', 'LORA', 'RAW_RIDGE'] if dataset == 'jena' else ['RAW', 'DIRECT_G']):
            rows.append(dict(run_id=f'{dataset}_{family}', dataset=dataset,
                family=family, seed=0, C=width, K=None, final=True,
                independent_FP64='PASS', model_sha='b' * 64))
    return rows


def test_runner_builds_exact_registered_phase_a_facets_with_checkpoint_identity():
    _, runner = modules()
    rows = source_rows()
    facets = runner._facets(rows)
    assert len(facets) == 46
    assert sum(facet['device'] == 'cuda' for facet in facets) == 44
    assert sum(facet['device'] == 'cpu' for facet in facets) == 2
    assert len({facet['facet_id'] for facet in facets}) == 46
    assert all(facet['checkpoint_sha256'] in ('a' * 64, 'b' * 64) for facet in facets)
    assert len([facet for facet in facets if facet['is_original']]) == 22
    raw8 = [facet for facet in facets if facet['accuracy_run_id'] == 'jena_RAW' and facet['B'] == 8]
    assert {facet['row_microbatch'] for facet in raw8} == {128, 48, 1}
    assert next(facet for facet in raw8 if facet['row_microbatch'] == 128)['is_original']
    assert all(facet['row_microbatch'] is None for facet in facets
               if facet['family'] in ('DIRECT_G', 'RAW_RIDGE'))


def test_runner_builds_exact_twenty_new_k_facets_with_fixed_k6_reference():
    _, runner = modules()
    rows = [dict(run_id=f'K{k}_s{seed}', family='B2', dataset='jena',
                 C=21, K=k, seed=seed, bundle_binding={'sha256': 'c' * 64})
            for k in (3, 12) for seed in (92601, 92602)]
    facets = runner._facets(rows)
    assert len(facets) == 20
    assert len({facet['facet_id'] for facet in facets}) == 20
    assert {facet['row_microbatch'] for facet in facets if facet['K'] == 3 and facet['B'] == 8} == {24, 1}
    assert {facet['row_microbatch'] for facet in facets if facet['K'] == 12 and facet['B'] == 8} == {96, 48, 1}


@pytest.fixture
def local_campaign(tmp_path, monkeypatch):
    common, runner = modules()
    root = tmp_path / 'repo'
    cache = root / '.cache' / common.NAME
    report = root / 'reports' / common.NAME
    cache.mkdir(parents=True)
    report.mkdir(parents=True)
    monkeypatch.setattr(common, 'REPO', root)
    paths = dict(repo=root, cache=cache, report=report)
    monkeypatch.setattr(common, 'paths', lambda: paths)
    (report / 'PROTOCOL.md').write_text('registered protocol\n', encoding='utf-8')
    common.sealed_json(report / 'START_STATE.json', dict(protocol_sha256=common.sha(report / 'PROTOCOL.md')))
    common.sealed_json(cache / 'CAMPAIGN.json', dict(asset_root=str(tmp_path / 'assets')))
    return common, runner, paths


def test_prepare_reuses_registered_state_without_opening_assets(local_campaign, monkeypatch):
    common, runner, paths = local_campaign
    monkeypatch.setattr(common, 'source_context', lambda *args: pytest.fail('reuse opened parent assets'))
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs: pytest.fail('reuse launched a command'))
    finished = []
    monkeypatch.setattr(runner, '_finish_prepare', lambda paths: finished.append(paths))
    assert runner.prepare('unused', 'unused', 'unused', []) == common.read(paths['report'] / 'START_STATE.json')
    assert finished == [paths]


@pytest.mark.parametrize('action', ['prepare', 'screen'])
def test_protocol_mutation_is_rejected_before_experiment(local_campaign, monkeypatch, action):
    _, runner, paths = local_campaign
    (paths['report'] / 'PROTOCOL.md').write_text('changed protocol\n', encoding='utf-8')
    monkeypatch.setattr(runner, '_resource_check', lambda *args: pytest.fail('protocol mismatch reached resources'))
    with pytest.raises(ValueError, match='[Pp]rotocol'):
        if action == 'prepare':
            runner.prepare('unused', 'unused', 'unused', [])
        else:
            runner.execute('screen')


def test_finish_prepare_recovers_missing_receipt_without_overwriting_existing_environment(local_campaign, monkeypatch):
    common, runner, paths = local_campaign
    environment = paths['report'] / 'ENVIRONMENT.json'
    common.sealed_json(environment, dict(existing='original snapshot'))
    original = environment.read_bytes()
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs: 'fixture GPU snapshot')
    monkeypatch.setattr(common, 'source_hashes', lambda: {'research/fixture.py': 'c' * 64})
    runner._finish_prepare(paths)
    assert environment.read_bytes() == original
    complete = common.read(paths['cache'] / 'PREPARE_COMPLETE.json')
    assert complete['environment_sha256'] == common.sha(environment)
    assert complete['start_state_sha256'] == common.sha(paths['report'] / 'START_STATE.json')
    assert common.read(paths['report'] / 'EXECUTION_SOURCE_START.json') == {'research/fixture.py': 'c' * 64}
    runner._finish_prepare(paths)
    assert environment.read_bytes() == original


def bound_source_fixture(tmp_path):
    common, _ = modules()
    context = {key: tmp_path / key for key in ('cache', 'report', 'repo', 'asset')}
    for directory in context.values():
        directory.mkdir()
    rows = source_rows()
    for index, row in enumerate(rows):
        prediction = context['asset'] / f'prediction_{index}.npz'
        prediction.write_bytes(f'prediction {index}'.encode())
        row['prediction_binding'] = common.binding(prediction, context)
        if row.get('bundle_binding'):
            bundle = context['asset'] / f'bundle_{index}.pt'
            bundle.write_bytes(f'checkpoint {index}'.encode())
            row['bundle_binding'] = common.binding(bundle, context)
    common.atomic_json(context['cache'] / 'VERIFIED_ROWS.json', rows)
    return common, context, rows


def test_selected_rows_resolves_all_eleven_exact_parent_bindings(tmp_path):
    common, context, rows = bound_source_fixture(tmp_path)
    selected = common.selected_rows(context)
    assert len(selected) == 11
    assert {row['run_id'] for row in selected} == {row['run_id'] for row in rows}


@pytest.mark.parametrize('binding_name', ['prediction_binding', 'bundle_binding'])
def test_parent_prediction_or_checkpoint_mutation_is_rejected(tmp_path, binding_name):
    common, context, rows = bound_source_fixture(tmp_path)
    path = common.resolve_binding(rows[0][binding_name], context)
    path.write_bytes(b'replaced parent artifact')
    with pytest.raises(ValueError, match='SHA mismatch'):
        common.selected_rows(context)


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'fp64', 'on'])
def test_incomplete_or_unverified_parent_rows_are_rejected(tmp_path, mutation):
    common, context, rows = bound_source_fixture(tmp_path)
    if mutation == 'duplicate':
        rows[-1] = dict(rows[0])
    elif mutation == 'missing':
        rows.pop()
    elif mutation == 'fp64':
        rows[0]['independent_FP64'] = 'FAIL'
    else:
        rows[0]['revin'] = True
    common.atomic_json(context['cache'] / 'VERIFIED_ROWS.json', rows)
    with pytest.raises(ValueError):
        common.selected_rows(context)


def test_sealed_artifacts_allow_exact_reuse_and_preserve_original_bytes(tmp_path):
    common, _ = modules()
    path = tmp_path / 'sealed.json'
    common.sealed_json(path, dict(value='original'))
    original = path.read_bytes()
    common.sealed_json(path, dict(value='original'))
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match='Sealed artifact changed'):
        common.sealed_json(path, dict(value='replacement'))
    assert path.read_bytes() == original


def test_resource_check_refuses_live_external_campaign_lease(local_campaign, monkeypatch):
    common, runner, paths = local_campaign
    import psutil
    lease = common.REPO / '.cache' / 'b2_revin_20261018_v1' / 'ACTIVE_GPU_LEASE.json'
    common.atomic_json(lease, dict(pid=777, create_time=123.))
    monkeypatch.setattr(psutil, 'Process', lambda pid=None: SimpleNamespace(create_time=lambda: 123.))
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs: pytest.fail('live lease must block before GPU command'))
    with pytest.raises(RuntimeError, match='BLOCKED_RESOURCE'):
        runner._resource_check(paths['cache'])
    assert lease.exists()


def test_resource_check_refuses_live_authorized_child_when_parent_is_gone(local_campaign, monkeypatch):
    common, runner, paths = local_campaign
    import psutil
    lease = common.REPO / '.cache' / 'adapter_diagnostics_20261017_v2' / 'ACTIVE_GPU_LEASE.json'
    common.atomic_json(lease, dict(pid=777, create_time=123., authorized_children=[dict(pid=778, create_time=124.)]))
    def process(pid=None):
        if pid == 777:
            raise psutil.NoSuchProcess(pid)
        return SimpleNamespace(create_time=lambda: 124.)
    monkeypatch.setattr(psutil, 'Process', process)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs: pytest.fail('live child lease must block before GPU command'))
    with pytest.raises(RuntimeError, match='BLOCKED_RESOURCE'):
        runner._resource_check(paths['cache'])


@pytest.mark.parametrize('name', ['python.exe', 'python', 'python3', 'pythonw.exe'])
def test_resource_check_refuses_unrelated_python_gpu_process(local_campaign, monkeypatch, name):
    _, runner, paths = local_campaign
    import psutil
    monkeypatch.setattr(psutil, 'Process', lambda pid=None: SimpleNamespace(parents=lambda: []))
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda command, **kwargs:
        '2000, 12000, 10' if '--query-gpu=memory.used,memory.total,utilization.gpu' in command
        else f'987654, fixture/{name}, 200 MiB\n')
    with pytest.raises(RuntimeError, match='BLOCKED_RESOURCE'):
        runner._resource_check(paths['cache'])
    assert not (paths['cache'] / 'RESOURCE_CHECK.json').exists()


def test_resource_check_allows_desktop_and_stale_pid_reuse_without_killing(local_campaign, monkeypatch):
    common, runner, paths = local_campaign
    import psutil
    lease = common.REPO / '.cache' / 'b2_revin_20261018_v1' / 'ACTIVE_GPU_LEASE.json'
    common.atomic_json(lease, dict(pid=777, create_time=123.))
    monkeypatch.setattr(psutil, 'Process', lambda pid=None: SimpleNamespace(create_time=lambda: 999., parents=lambda: []))
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda command, **kwargs:
        '2000, 12000, 10' if '--query-gpu=memory.used,memory.total,utilization.gpu' in command
        else '778, fixture/dwm.exe, N/A\n')
    runner._resource_check(paths['cache'])
    assert common.read(paths['cache'] / 'RESOURCE_CHECK.json')['processes'] == [dict(pid=778, process='dwm.exe', memory='N/A')]
    assert lease.exists()


@pytest.mark.parametrize('snapshot', ['2000, 12000, 80', '10800, 12000, 10'])
def test_resource_check_refuses_high_external_device_occupancy(local_campaign, monkeypatch, snapshot):
    _, runner, paths = local_campaign
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda command, **kwargs:
        snapshot if '--query-gpu=memory.used,memory.total,utilization.gpu' in command else '')
    with pytest.raises(RuntimeError, match='high external GPU occupancy'):
        runner._resource_check(paths['cache'])


def test_resource_snapshots_append_history_without_replacing_earlier_samples(local_campaign, monkeypatch):
    common, runner, paths = local_campaign
    import psutil
    monkeypatch.setattr(psutil, 'Process', lambda pid=None: SimpleNamespace(parents=lambda: []))
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda command, **kwargs:
        '2000, 12000, 10' if '--query-gpu=memory.used,memory.total,utilization.gpu' in command else '')
    runner._resource_check(paths['cache'])
    journal = paths['cache'] / 'RESOURCE_CHECKS.jsonl'
    original = journal.read_bytes()
    runner._resource_check(paths['cache'])
    assert journal.read_bytes().startswith(original)
    assert len(journal.read_text().splitlines()) == 2
    assert common.read(paths['cache'] / 'RESOURCE_CHECK.json')['device_values_are_not_PID_memory'] is True


class FakeLease:
    def __init__(self, *args, **kwargs):
        self.checked = 0
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def check(self):
        self.checked += 1
    def child_environment(self):
        return {'TEST_GPU_LEASE': 'fixture'}


@pytest.fixture
def mocked_execution(local_campaign, monkeypatch):
    common, runner, paths = local_campaign
    resources = importlib.import_module('research.informs_broad_validation_20261015_v1.resources')
    analysis = importlib.import_module('research.memory_budget_20261018_v1.analysis')
    facets = runner._facets(source_rows())[:1]
    rows = source_rows()
    common.sealed_json(paths['cache'] / 'FACETS_A.json', facets)
    common.sealed_json(paths['cache'] / 'SOURCE_ROWS.json', rows)
    monkeypatch.setattr(common, 'source_context', lambda *args: {'test': 'context'})
    monkeypatch.setattr(common, 'selected_rows', lambda *args: rows)
    monkeypatch.setattr(resources, 'GPUReservation', FakeLease)
    monkeypatch.setattr(runner, '_resource_check', lambda *args: None)
    exported = []
    monkeypatch.setattr(analysis, 'export_receipts', lambda values, *args, **kwargs: exported.extend(values))
    import psutil
    monkeypatch.setattr(psutil, 'Process', lambda pid=None: SimpleNamespace(create_time=lambda: 123.))
    monkeypatch.setattr(psutil, 'pid_exists', lambda pid: False)
    return common, runner, paths, facets[0], exported


def expected_request(common, paths, facet):
    parent = next(row for row in common.read(paths['cache'] / 'SOURCE_ROWS.json')
                  if row['run_id'] == facet['accuracy_run_id'])
    return dict(asset_root=common.read(paths['cache'] / 'CAMPAIGN.json')['asset_root'],
        resource_cache=str(paths['cache']), accuracy_run_id=facet['accuracy_run_id'],
        batch_size=facet['batch_size'], row_microbatch=facet['row_microbatch'],
        stage='screen', facet_id=facet['facet_id'],
        protocol_sha256=common.sha(paths['report'] / 'PROTOCOL.md'), phase='A',
        expected_candidate_sha256=common.fingerprint(parent))


def test_execute_rejects_changed_parent_checkpoint_before_receipt_reuse(mocked_execution, monkeypatch):
    common, runner, paths, _, _ = mocked_execution
    rows = common.read(paths['cache'] / 'SOURCE_ROWS.json')
    rows[0]['bundle_binding']['sha256'] = 'd' * 64
    monkeypatch.setattr(common, 'selected_rows', lambda *args: rows)
    monkeypatch.setattr(runner, '_resource_check', lambda *args: pytest.fail('changed parent reached resource acquisition'))
    with pytest.raises(ValueError, match='parent accuracy identities'):
        runner.execute('screen')


def test_execute_records_missing_receipt_as_invalid_and_process_identity(mocked_execution, monkeypatch):
    common, runner, paths, facet, exported = mocked_execution
    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        kwargs['stdout'].write('worker crashed before receipt\n')
        return SimpleNamespace(pid=555, wait=lambda: 7)
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    runner.execute('screen')
    stem = facet['facet_id'] + '_screen'
    receipt = common.read(paths['cache'] / 'facets' / (stem + '.receipt.json'))
    process = common.read(paths['cache'] / 'facets' / (stem + '.process.json'))
    assert receipt['status'] == 'INVALID'
    assert receipt['failure']['exception_type'] == 'WORKER_NO_RECEIPT'
    assert process['pid'] == 555 and process['create_time'] == 123.
    assert process['exit_code'] == 7
    assert len(calls) == 1
    assert calls[0][1]['env']['TEST_GPU_LEASE'] == 'fixture'
    assert len(exported) == 1 and exported[0]['status'] == 'INVALID'
    assert 'worker crashed' in (paths['cache'] / 'logs' / (stem + '.log')).read_text()


def test_execute_reuses_complete_receipt_without_spawning_again(mocked_execution, monkeypatch):
    common, runner, paths, facet, _ = mocked_execution
    request = expected_request(common, paths, facet)
    stem = facet['facet_id'] + '_screen'
    receipt_path = paths['cache'] / 'facets' / (stem + '.receipt.json')
    common.sealed_json(receipt_path, dict(status='COMPLETE', facet_id=facet['facet_id'],
        request_identity_sha256=common.fingerprint(request), summary=dict(facet, stage='screen')))
    original = receipt_path.read_bytes()
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('complete receipt reran'))
    runner.execute('screen')
    assert receipt_path.read_bytes() == original


def test_execute_refuses_complete_receipt_with_a_different_request_identity(mocked_execution, monkeypatch):
    common, runner, paths, facet, _ = mocked_execution
    receipt = paths['cache'] / 'facets' / (facet['facet_id'] + '_screen.receipt.json')
    common.sealed_json(receipt, dict(status='COMPLETE', facet_id=facet['facet_id'],
        request_identity_sha256='different-checkpoint-request', summary=dict(facet, stage='screen')))
    original = receipt.read_bytes()
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('identity mismatch silently reran'))
    with pytest.raises(RuntimeError, match='Previous attempt preserved'):
        runner.execute('screen')
    assert receipt.read_bytes() == original


def test_execute_continues_remaining_facets_after_invalid_worker(mocked_execution, monkeypatch):
    common, runner, paths, _, exported = mocked_execution
    facets = runner._facets(source_rows())[:2]
    common.atomic_json(paths['cache'] / 'FACETS_A.json', facets)
    pids = iter([555, 556])
    calls = []
    def popen(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(pid=next(pids), wait=lambda: 7)
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    runner.execute('screen')
    assert len(calls) == 2
    assert len(exported) == 2
    assert all(item['status'] == 'INVALID' for item in exported)
    assert common.read(paths['report'] / 'STATUS.json')['completed'] == 2


def test_confirm_selection_keeps_valid_original_and_excludes_terminal_invalid(mocked_execution, monkeypatch):
    common, runner, paths, _, _ = mocked_execution
    facets = runner._facets(source_rows())[:2]
    common.atomic_json(paths['cache'] / 'FACETS_A.json', facets)
    for index, facet in enumerate(facets):
        status = 'COMPLETE' if index == 0 else 'INVALID'
        common.sealed_json(paths['cache'] / 'facets' / (facet['facet_id'] + '_screen.receipt.json'),
            dict(facet_id=facet['facet_id'], status=status,
                summary=dict(facet, status=status, stage='screen',
                    parity_status='PASS' if index == 0 else 'FAIL',
                    peak_allocated_MiB=200., median_batch_ms=10.)))
    calls = []
    def popen(command, **kwargs):
        calls.append(common.read(command[command.index('--request') + 1]))
        return SimpleNamespace(pid=555, wait=lambda: 7)
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    runner.execute('confirm')
    assert [request['facet_id'] for request in calls] == [facets[0]['facet_id']]
    selection = common.read(paths['report'] / 'CONFIRM_SELECTION_A.json')
    assert {value['facet_id']: (value['confirm_selected'], value['confirmation_reason'])
            for value in selection} == {
        facets[0]['facet_id']: (True, 'ORIGINAL_PATH_REQUIRED'),
        facets[1]['facet_id']: (False, 'INVALID_PARITY')}


def test_confirm_selection_refuses_missing_screen_attempt(mocked_execution, monkeypatch):
    _, runner, _, _, _ = mocked_execution
    monkeypatch.setattr(runner, '_resource_check', lambda *args: pytest.fail('incomplete screens reached resource acquisition'))
    with pytest.raises(ValueError, match='All terminal screen attempts'):
        runner.execute('confirm')


def test_execute_retains_invalid_receipt_and_log_without_retry(mocked_execution, monkeypatch):
    common, runner, paths, facet, _ = mocked_execution
    stem = facet['facet_id'] + '_screen'
    receipt_path = paths['cache'] / 'facets' / (stem + '.receipt.json')
    common.sealed_json(receipt_path, dict(status='INVALID', facet_id=facet['facet_id'],
        summary=dict(facet, stage='screen'), process=dict(pid=555, create_time=123.)))
    logs = paths['cache'] / 'logs'
    logs.mkdir()
    log_path = logs / (stem + '.log')
    log_path.write_text('original failed log', encoding='utf-8')
    before = (receipt_path.read_bytes(), log_path.read_bytes())
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('INVALID receipt reran without retry'))
    runner.execute('screen')
    assert (receipt_path.read_bytes(), log_path.read_bytes()) == before


def test_retry_preserves_receipt_and_log_bytes_before_new_attempt(mocked_execution, monkeypatch):
    common, runner, paths, facet, _ = mocked_execution
    stem = facet['facet_id'] + '_screen'
    receipt_path = paths['cache'] / 'facets' / (stem + '.receipt.json')
    common.sealed_json(receipt_path, dict(status='INVALID', facet_id=facet['facet_id'],
        summary=dict(facet, stage='screen'), process=dict(pid=555, create_time=123.)))
    logs = paths['cache'] / 'logs'
    logs.mkdir()
    log_path = logs / (stem + '.log')
    log_path.write_text('original failed log', encoding='utf-8')
    old_receipt, old_log = receipt_path.read_bytes(), log_path.read_bytes()
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: SimpleNamespace(pid=556, wait=lambda: 8))
    runner.execute('screen', retry_failed=True)
    archived = list((paths['cache'] / 'attempts').iterdir())
    assert any(path.read_bytes() == old_receipt for path in archived if path.suffix == '.json')
    assert any(path.read_bytes() == old_log for path in archived if path.suffix == '.log')
    assert common.read(receipt_path)['process']['pid'] == 556


def test_retry_refuses_same_live_worker_identity_without_archiving(mocked_execution, monkeypatch):
    common, runner, paths, facet, _ = mocked_execution
    import psutil
    receipt = paths['cache'] / 'facets' / (facet['facet_id'] + '_screen.receipt.json')
    common.sealed_json(receipt, dict(status='INVALID', facet_id=facet['facet_id'],
        summary=dict(facet, stage='screen'), process=dict(pid=555, create_time=123.)))
    original = receipt.read_bytes()
    monkeypatch.setattr(psutil, 'pid_exists', lambda pid: True)
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('live worker retried'))
    with pytest.raises(RuntimeError, match='Cannot retry a live worker'):
        runner.execute('screen', retry_failed=True)
    assert receipt.read_bytes() == original
    assert not (paths['cache'] / 'attempts').exists()


def test_orphan_log_refuses_unrequested_retry_and_preserves_bytes(mocked_execution, monkeypatch):
    _, runner, paths, facet, _ = mocked_execution
    logs = paths['cache'] / 'logs'
    logs.mkdir()
    path = logs / (facet['facet_id'] + '_screen.log')
    path.write_text('unresolved worker output', encoding='utf-8')
    original = path.read_bytes()
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('orphan log silently overwritten'))
    with pytest.raises(RuntimeError, match='Orphan log retained'):
        runner.execute('screen')
    assert path.read_bytes() == original


@pytest.fixture
def preservation_fixture(local_campaign, monkeypatch):
    common, runner, paths = local_campaign
    files = {'_docs/history/2026-10-18.md': b'original history\n',
             'reports/old/RESULTS.md': b'original old results\n'}
    for name, content in files.items():
        path = common.REPO / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    presentation = paths['repo'] / 'presentation.md'
    presentation.write_bytes(b'original presentation\n')
    original_status = b' M dirty.py\0?? private.npz\0'
    state = common.read(paths['report'] / 'START_STATE.json')
    state['parent_commit'] = 'parent-commit'
    common.atomic_json(paths['report'] / 'START_STATE.json', state)
    common.sealed_json(paths['cache'] / 'PRESERVATION_BASELINE.json', dict(
        original_root='original-workspace', original_HEAD='original-head',
        original_status_z_sha256=hashlib.sha256(original_status).hexdigest(),
        protected_files={name: hashlib.sha256(value).hexdigest() for name, value in files.items()},
        presentations={'presentation.md': dict(path=str(presentation), sha256=common.sha(presentation))}))
    def check_output(command, **kwargs):
        if command[1] == 'show':
            assert command[2] == 'parent-commit:_docs/history/2026-10-18.md'
            return files['_docs/history/2026-10-18.md']
        if command[1] == 'status':
            return original_status
        pytest.fail('unexpected preservation command')
    monkeypatch.setattr(runner.subprocess, 'check_output', check_output)
    monkeypatch.setattr(runner, '_git', lambda *args, **kwargs: 'original-head')
    return common, runner, paths, presentation


def test_preservation_allows_only_history_append(preservation_fixture):
    common, runner, paths, _ = preservation_fixture
    history = common.REPO / '_docs/history/2026-10-18.md'
    with history.open('ab') as handle:
        handle.write(b'new memory-budget result\n')
    result = runner.preservation()
    assert result['status'] == 'PASS'
    assert result['changed_tracked_files'] == ['_docs/history/2026-10-18.md']
    assert common.read(paths['report'] / 'PRESERVATION.json')['presentations_preserved']


@pytest.mark.parametrize('target', ['history', 'old_result', 'presentation', 'original_HEAD'])
def test_preservation_rejects_old_content_or_original_checkout_changes(preservation_fixture, monkeypatch, target):
    common, runner, paths, presentation = preservation_fixture
    if target == 'history':
        (common.REPO / '_docs/history/2026-10-18.md').write_bytes(b'rewritten history\n')
    elif target == 'old_result':
        (common.REPO / 'reports/old/RESULTS.md').write_bytes(b'rewritten old result\n')
    elif target == 'presentation':
        presentation.write_bytes(b'changed presentation\n')
    else:
        monkeypatch.setattr(runner, '_git', lambda *args, **kwargs: 'different-head')
    with pytest.raises(ValueError, match='Preservation check failed'):
        runner.preservation()
    assert common.read(paths['report'] / 'PRESERVATION.json')['status'] == 'FAIL'


def test_importing_campaign_modules_has_no_experiment_or_filesystem_side_effect(monkeypatch):
    common, runner = modules()
    def forbidden(*args, **kwargs):
        pytest.fail('module import initiated an experiment or write')
    monkeypatch.setattr(Path, 'mkdir', forbidden)
    monkeypatch.setattr(runner.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(runner.subprocess, 'run', forbidden)
    monkeypatch.setattr(runner.subprocess, 'check_output', forbidden)
    importlib.reload(common)
    importlib.reload(runner)
