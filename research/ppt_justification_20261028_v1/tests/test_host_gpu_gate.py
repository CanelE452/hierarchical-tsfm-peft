"""Synthetic Windows host records only; no actual counters, GPU or apps touched."""
from copy import deepcopy
import io
import json
from types import SimpleNamespace

import pytest

from research.ppt_justification_20261028_v1 import host_gpu_gate as host


@pytest.fixture
def scene(tmp_path):
    exe = tmp_path/'desktop.exe'
    exe.write_bytes(b'synthetic reviewed desktop executable')
    identity = dict(pid=20, created=100., exe=host.canonical_exe(exe), exe_sha256=host.sha(exe))
    target = dict(adapter_luid='0x00000000_0x00000001', physical_index=0,
                  engine_keys=['0x00000000_0x00000001/phys0/eng0/3d', '0x00000000_0x00000001/phys0/eng9/copy'])
    policy = dict(device=dict(gpu_uuid='GPU-synthetic'), desktop_processes=[identity], counter_target=target,
                  kernel_engine_provider=dict(kind='windows_system_counter_only', pid=4, name='System', boot_time=50.))
    records = {20: dict(identity, name='desktop.exe', ppid=1, command=['desktop.exe']),
               4: dict(pid=4, created=50., name='System', exe=None, command=[], ppid=0)}
    contexts = [dict(pid=20, gpu_uuid='GPU-synthetic')]
    return policy, records, contexts


def admit(scene, **kwargs):
    return host.validate_identity(*scene, boot_time=50., **kwargs)


def raw_sample(timestamp=0., desktop=1., system=1.):
    return dict(timestamp_unix=timestamp, counters=[
        dict(name='pid_20_luid_0x00000000_0x00000001_phys_0_eng_0_engtype_3D', value=desktop, status=0),
        dict(name='pid_4_luid_0x00000000_0x00000001_phys_0_eng_9_engtype_Copy', value=system, status=0)])


def contract(policy):
    return dict(policy_sha256='a'*64, counter_target=policy['counter_target'], performance=dict(
        sample_interval_seconds=1, preflight_seconds=60, maximum_sample_gap_seconds=2.5,
        background_engine_p95_limit_percent=5., background_engine_max_limit_percent=15.))


def test_reviewed_desktop_and_boot_bound_System_are_explicit(scene):
    snapshot = admit(scene)
    totals = host.background_engines(raw_sample(), scene[0], snapshot)
    assert sorted(totals.values()) == [1., 1.]
    assert snapshot['kernel_engine_provider']['pid'] == 4
    assert snapshot['observed'][0]['exe_sha256'] == scene[0]['desktop_processes'][0]['exe_sha256']


@pytest.mark.parametrize('fault', ['unknown', 'reused_pid', 'spoofed_path', 'changed_image', 'different_gpu',
                                  'System_boot', 'System_name', 'missing_identity', 'duplicate_policy'])
def test_identity_faults_rejected(scene, tmp_path, fault):
    policy, records, contexts = scene
    if fault == 'unknown':
        contexts.append(dict(pid=999, gpu_uuid='GPU-synthetic', type='C+G'))
    elif fault == 'reused_pid':
        records[20]['created'] += 1
    elif fault == 'spoofed_path':
        fake = tmp_path/'other/desktop.exe'; fake.parent.mkdir(); fake.write_bytes(b'same filename')
        records[20]['exe'] = str(fake)
    elif fault == 'changed_image':
        from pathlib import Path
        Path(records[20]['exe']).write_bytes(b'changed executable')
    elif fault == 'different_gpu':
        contexts[0]['gpu_uuid'] = 'GPU-other'
    elif fault == 'System_boot':
        policy['kernel_engine_provider']['boot_time'] = 1.
    elif fault == 'System_name':
        records[4]['name'] = 'Unknown'
    elif fault == 'missing_identity':
        records[20]['exe'] = None
    elif fault == 'duplicate_policy':
        policy['desktop_processes'].append(dict(policy['desktop_processes'][0]))
    with pytest.raises(RuntimeError):
        admit(scene)


@pytest.mark.parametrize('filename', ['python.exe', 'pythonw.exe', 'wsl.exe', 'vmmemWSL.exe', 'ollama.exe'])
def test_compute_process_cannot_be_desktop_even_if_whitelisted(scene, tmp_path, filename):
    exe = tmp_path/filename; exe.write_bytes(b'compute')
    row = scene[0]['desktop_processes'][0]
    row.update(exe=host.canonical_exe(exe), exe_sha256=host.sha(exe))
    scene[1][20]['exe'] = str(exe)
    with pytest.raises(RuntimeError, match='compute executable'):
        admit(scene)


def test_live_experiment_rejected_even_with_empty_gpu_roster(scene):
    scene[1][30] = dict(pid=30, created=120., exe='python.exe', command=['python', '-m', 'research.ppt_justification_20261028_v1.run_one'])
    with pytest.raises(RuntimeError, match='Live experiment'):
        host.validate_identity(scene[0], scene[1], [], boot_time=50.)
    with pytest.raises(RuntimeError, match='Main training can never'):
        admit(scene, allowed_processes={30:120.})


def test_only_exact_lease_worker_can_be_excluded_from_engine_load(scene):
    scene[1][30] = dict(pid=30, created=120., exe='python.exe', command=['python', '-m', 'research.ppt_justification_20261028_v1.cost_profile', '--worker'])
    scene[2].append(dict(pid=30, gpu_uuid='GPU-synthetic'))
    with pytest.raises(RuntimeError, match='Unknown GPU'):
        admit(scene, allowed_processes={30:120.})
    identity = admit(scene, allowed_processes={30:120.}, allowed_gpu_processes={30:120.})
    sample = raw_sample()
    sample['counters'].append(dict(name='pid_30_luid_0x00000000_0x00000001_phys_0_eng_0_engtype_3D', value=97., status=0))
    assert max(host.background_engines(sample, scene[0], identity).values()) == 1.
    with pytest.raises(RuntimeError, match='identity changed'):
        admit(scene, allowed_processes={30:119.}, allowed_gpu_processes={30:119.})


@pytest.mark.parametrize('fault', ['unknown_active', 'missing_engine', 'empty', 'bad_status', 'nan', 'negative', 'duplicate', 'wrong_luid', 'unregistered_system'])
def test_invalid_counter_evidence_never_becomes_zero(scene, fault):
    identity = admit(scene)
    sample = raw_sample()
    if fault == 'unknown_active': sample['counters'][0]['name'] = sample['counters'][0]['name'].replace('pid_20', 'pid_999')
    elif fault == 'missing_engine': sample['counters'].pop()
    elif fault == 'empty': sample['counters'] = []
    elif fault == 'bad_status': sample['counters'][0]['status'] = 0xc0000000
    elif fault == 'nan': sample['counters'][0]['value'] = float('nan')
    elif fault == 'negative': sample['counters'][0]['value'] = -1.
    elif fault == 'duplicate': sample['counters'].append(dict(sample['counters'][0]))
    elif fault == 'wrong_luid':
        for row in sample['counters']: row['name'] = row['name'].replace('0x00000001', '0x00000002')
    else: identity['kernel_engine_provider'] = None
    with pytest.raises(RuntimeError):
        host.background_engines(sample, scene[0], identity)


def samples(scene, n=61):
    identity = admit(scene)
    return [dict(status='PASS', timestamp_unix=float(i), identity=identity, raw=raw_sample(float(i)),
                 background_engines=host.background_engines(raw_sample(float(i)), scene[0], identity)) for i in range(n)]


def test_sixty_seconds_is_not_sixty_samples(scene):
    cfg = contract(scene[0])
    assert host.validate_observation(samples(scene), cfg, required_start=0., required_end=60., preflight=True)['status'] == 'PASS'
    result = host.validate_observation(samples(scene, 60), cfg, required_start=0., required_end=60., preflight=True)
    assert result['status'] == 'CONTAMINATED' and 'preflight_shorter_than_60_seconds' in result['errors']


@pytest.mark.parametrize('fault', ['gap', 'nonmonotonic', 'p95', 'max', 'failed', 'missing_engine', 'late_start'])
def test_contamination_blocks_cost_comparison(scene, fault):
    rows = samples(scene)
    key = next(iter(rows[0]['background_engines']))
    if fault == 'gap':
        for row in rows[10:]: row['timestamp_unix'] += 2.
    elif fault == 'nonmonotonic': rows[10]['timestamp_unix'] = 9.
    elif fault == 'p95':
        for row in rows: row['background_engines'][key] = 5.01
    elif fault == 'max': rows[10]['background_engines'][key] = 15.01
    elif fault == 'failed': rows[10]['status'] = 'FAILED'
    elif fault == 'missing_engine': rows[10]['background_engines'].pop(key)
    elif fault == 'late_start': rows[0]['timestamp_unix'] = .1
    result = host.validate_observation(rows, contract(scene[0]), required_start=0., required_end=60., preflight=True)
    assert result['status'] == 'CONTAMINATED'


def test_recorded_snapshot_must_match_fixed_policy(scene):
    identity = admit(scene)
    host.validate_stored_identity(identity, scene[0])
    identity['observed'][0]['created'] += 1
    with pytest.raises(RuntimeError, match='tuple differs'):
        host.validate_stored_identity(identity, scene[0])


def test_source_or_private_policy_change_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(host, 'REPO', tmp_path)
    monkeypatch.setattr(host, 'PRIVATE', tmp_path/'private')
    path = tmp_path/'contract.json'
    host.write_new(path, dict(schema_version=1, mode='wddm_registered_desktop_concurrency', source_sha256='0'*64))
    with pytest.raises(RuntimeError, match='source differs'):
        host.load_policy(path)


def test_identity_receipt_independently_binds_private_bytes(scene, tmp_path, monkeypatch):
    monkeypatch.setattr(host, 'REPO', tmp_path)
    monkeypatch.setattr(host, 'PRIVATE', tmp_path/'private')
    cfg = contract(scene[0]); cfg_path=tmp_path/'contract.json'; host.write_new(cfg_path, cfg)
    monkeypatch.setattr(host, 'load_policy', lambda p: (cfg, scene[0]))
    snapshot = admit(scene); path=host.PRIVATE/'snapshot.json'; host.write_new(path, snapshot)
    receipt=dict(status='PASS', mode='identity_only', policy_sha256=cfg['policy_sha256'], contract_sha256=host.sha(cfg_path),
        private_snapshot_relative_path='private/snapshot.json', identity_snapshot_sha256=host.sha(path),
        unexpected_gpu_pids=[], desktop_concurrent_load=True, observed_gpu_process_count=1, allowed_desktop_context_count=1)
    assert host.verify_identity_receipt(receipt,cfg_path)['status'] == 'PASS'
    path.write_text('{}')
    with pytest.raises(RuntimeError, match='snapshot hash'):
        host.verify_identity_receipt(receipt,cfg_path)


def test_cost_trace_recomputed_from_raw_counters_and_private_hash(scene, tmp_path, monkeypatch):
    monkeypatch.setattr(host, 'REPO', tmp_path)
    monkeypatch.setattr(host, 'PRIVATE', tmp_path/'private')
    cfg=contract(scene[0]); cfg_path=tmp_path/'contract.json'; host.write_new(cfg_path,cfg)
    monkeypatch.setattr(host, 'load_policy', lambda p: (cfg,scene[0]))
    trace=host.PRIVATE/'trace.jsonl'; trace.parent.mkdir()
    rows=samples(scene)
    trace.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf8')
    receipt=host.validate_observation(rows,cfg,required_start=0.,required_end=60.,preflight=True)
    receipt.update(contract_sha256=host.sha(cfg_path),observer_source_sha256=host.sha(host.__file__),
                   private_trace_relative_path='private/trace.jsonl',private_trace_sha256=host.sha(trace))
    assert host.verify_observation_receipt(receipt,cfg_path)['status']=='PASS'
    receipt['background_engine_statistics'][next(iter(receipt['background_engine_statistics']))]['p95_percent']=0.
    with pytest.raises(RuntimeError,match='summary differs'):
        host.verify_observation_receipt(receipt,cfg_path)


@pytest.fixture
def capture_context(scene, tmp_path, monkeypatch):
    monkeypatch.setattr(host, 'REPO', tmp_path)
    monkeypatch.setattr(host, 'PRIVATE', tmp_path/'private')
    monkeypatch.setattr(host, 'POLICY', host.PRIVATE/'policy.json')
    monkeypatch.setattr(host, 'CONTRACT', tmp_path/'public/contract.json')
    monkeypatch.setattr(host, 'process_records', lambda: scene[1])
    monkeypatch.setattr(host, 'gpu_contexts', lambda: scene[2])
    monkeypatch.setattr(host, 'gpu_device', lambda: scene[0]['device'])
    monkeypatch.setattr(host.psutil, 'boot_time', lambda: 50.)
    return ([20], '0x00000000_0x00000001', 0, True)


def test_cp949_nonzero_capture_preserves_raw_stderr_without_public_leak(capture_context, monkeypatch):
    error='오류: C:\\Users\\비공개\\원본.txt'.encode('cp949')
    def run(*args, **kwargs):
        assert not kwargs.get('text', False) and 'encoding' not in kwargs and kwargs['check'] is False
        return SimpleNamespace(returncode=1, stdout=b'', stderr=error)
    monkeypatch.setattr(host.subprocess, 'run', run)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='counter capture failed') as caught:
            host.capture_policy(*capture_context)
        assert '비공개' not in str(caught.value) and 'C:\\Users' not in str(caught.value)
    files=list(host.PRIVATE.glob('capture_stderr_*.bin'))
    assert len(files)==2 and all(path.read_bytes()==error for path in files)
    assert not host.POLICY.exists() and not host.CONTRACT.exists()


def test_valid_utf8_json_capture_keeps_policy_semantics(capture_context, monkeypatch):
    stdout=json.dumps(raw_sample()).encode('utf8')
    monkeypatch.setattr(host.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout, stderr=b''))
    result=host.capture_policy(*capture_context)
    assert result['status']=='POLICY_CAPTURED_NOT_ADMITTED'
    cfg,policy=host.load_policy(host.CONTRACT)
    assert cfg['performance']['preflight_seconds']==60
    assert cfg['performance']['background_engine_p95_limit_percent']==5.
    assert cfg['performance']['background_engine_max_limit_percent']==15.
    assert cfg['E_scope']=='Identity-only host admission before CUDA at each worker start; controller enforces upstream-process exclusion and shared reservations; no performance claim'
    assert policy['kernel_engine_provider']['pid']==4
    assert [row['pid'] for row in policy['desktop_processes']]==[20]
    assert not list(host.PRIVATE.glob('capture_stderr_*.bin'))


@pytest.mark.parametrize('stdout', [b'\xbc', b'not json', b'{}', b'{"counters":[]}'])
def test_invalid_stdout_or_missing_telemetry_still_refuses_capture(capture_context, monkeypatch, stdout):
    monkeypatch.setattr(host.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout, stderr=b''))
    with pytest.raises(RuntimeError, match='UTF-8 JSON|missing telemetry'):
        host.capture_policy(*capture_context)
    assert not host.POLICY.exists() and not host.CONTRACT.exists()


def test_counter_timeout_keeps_binary_stderr_and_refuses_capture(capture_context, monkeypatch):
    error='시간 초과'.encode('cp949')
    def run(*args, **kwargs):
        raise host.subprocess.TimeoutExpired('powershell.exe',20,stderr=error)
    monkeypatch.setattr(host.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='timed out'):
        host.capture_policy(*capture_context)
    assert next(host.PRIVATE.glob('capture_stderr_*.bin')).read_bytes()==error
    assert not host.POLICY.exists()


def test_observer_stderr_is_a_binary_fd(capture_context, scene, monkeypatch):
    cfg=contract(scene[0]); host.write_new(host.CONTRACT,cfg)
    monkeypatch.setattr(host, 'load_policy', lambda p:(cfg,scene[0]))
    error='원본 오류'.encode('cp949')
    def popen(*args, **kwargs):
        assert 'b' in kwargs['stderr'].mode
        kwargs['stderr'].write(error); kwargs['stderr'].flush()
        return SimpleNamespace(stdout=io.StringIO(json.dumps(raw_sample())+'\n'), poll=lambda:0, wait=lambda **k:0)
    monkeypatch.setattr(host.subprocess, 'Popen', popen)
    observer=host.HostObservation(host.CONTRACT,'synthetic_binary_stderr',{}).start()
    observer.finish(0.,0.)
    assert observer.stderr_path.suffix=='.bin' and observer.stderr_path.read_bytes()==error
