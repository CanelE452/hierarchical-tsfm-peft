"""Prospective WDDM desktop identity admission and bounded cost-host observation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time

import numpy as np
import psutil


REPO = Path(__file__).resolve().parents[2]
PRIVATE = REPO/'.cache/crossdata_component_screen_20261028_v1/host_gpu_gate'
CONTRACT = REPO/'reports/crossdata_component_screen_20261028_v1/HOST_GPU_CONTRACT.json'
POLICY = PRIVATE/'DESKTOP_POLICY.json'
IMAGE_HASHES = {}
MODULES = {'research.ppt_justification_20261028_v1.controller',
           'research.ppt_justification_20261028_v1.run_one',
           'research.ppt_justification_20261028_v1.cost_profile',
           'research.ppt_e_fixed_20261028_v1.controller',
           'research.ppt_e_fixed_20261028_v1.run_one',
           'research.crossdata_component_screen_20261028_v1.controller',
           'research.crossdata_component_screen_20261028_v1.run_one'}
COUNTER = re.compile(r'^pid_(\d+)_luid_(0x[0-9a-f]+_0x[0-9a-f]+)_phys_(\d+)_eng_(\d+)_engtype_(.+)$', re.I)
COUNTER_SCRIPT = r"""
$ErrorActionPreference='Stop'
Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -Continuous | ForEach-Object {
    $stamp = ([DateTimeOffset]$_.Timestamp).ToUnixTimeMilliseconds()/1000.0
    $values = @($_.CounterSamples | ForEach-Object {
        [pscustomobject]@{name=$_.InstanceName; value=$_.CookedValue; status=[long]$_.Status}
    })
    [pscustomobject]@{timestamp_unix=$stamp; counters=$values} | ConvertTo-Json -Depth 4 -Compress
}
"""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def canonical_exe(path):
    require(bool(path) and Path(path).is_absolute(), 'GPU process full executable identity unavailable')
    return os.path.normcase(os.path.realpath(path))


def forbidden_desktop(exe):
    name = Path(exe).name.lower()
    return any(text in name for text in ('python', 'wsl', 'vmmem', 'jupyter', 'ollama', 'llama', 'cuda', 'nvidia-smi'))


def image_sha(path):
    stat = Path(path).stat()
    key = (path, stat.st_size, stat.st_mtime_ns)
    if key not in IMAGE_HASHES:
        IMAGE_HASHES[key] = sha(path)
    return IMAGE_HASHES[key]


def process_records():
    records = {}
    for process in psutil.process_iter(['pid', 'ppid', 'create_time', 'cmdline', 'exe', 'name'], ad_value=None):
        try:
            info = process.info
            records[int(info['pid'])] = dict(pid=int(info['pid']), created=info['create_time'],
                exe=info['exe'], name=info['name'], ppid=info['ppid'], command=info['cmdline'] or [])
        except psutil.NoSuchProcess:
            continue
    return records


def creation_times(pids):
    return {int(pid): psutil.Process(int(pid)).create_time() for pid in pids}


def gpu_contexts():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10,
                            check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(',', 2)]
        require(len(fields) == 3 and fields[0].startswith('GPU-') and fields[1].isdigit(), 'Unparseable GPU context roster')
        rows.append(dict(gpu_uuid=fields[0], pid=int(fields[1])))
    require(len({(r['gpu_uuid'], r['pid']) for r in rows}) == len(rows), 'Duplicate GPU context records')
    return rows


def gpu_device():
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,pci.bus_id,driver_version', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=10, check=True,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    rows = [line.split(',') for line in result.stdout.splitlines() if line.strip()]
    require(len(rows) == 1 and len(rows[0]) == 3, 'Exactly one explicitly bound NVIDIA GPU is required')
    return dict(zip(('gpu_uuid', 'pci_bus_id', 'driver_version'), (value.strip() for value in rows[0])))


def module_name(record):
    command = record['command']
    return command[command.index('-m')+1] if '-m' in command and command.index('-m')+1 < len(command) else None


def validate_kernel_identity(observed, registered):
    if registered is None:
        require(observed is None, 'Unregistered System counter provider')
        return
    fields = ('kind', 'pid', 'name', 'created')
    keys = set(fields) | {'observed_boot_time'}
    for row in (registered, observed):
        require(isinstance(row, dict) and set(row) == keys and
                row['kind'] == 'windows_system_creation_counter_only' and row['pid'] == 4 and row['name'] == 'System',
                'Explicit System creation provider policy required; boot-only policies are unsupported')
        require(all(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] > 0
                    for key in ('created', 'observed_boot_time')), 'System identity/provenance is unavailable')
    require(all(observed[key] == registered[key] for key in fields), 'System creation identity changed')


def validate_identity(policy, records, contexts, allowed_processes=None, allowed_gpu_processes=None, *, boot_time=None):
    allowed_processes, allowed_gpu_processes = allowed_processes or {}, allowed_gpu_processes or {}
    allowed_modules = set()
    for pid, created in {**allowed_processes, **allowed_gpu_processes}.items():
        require(pid in records and records[pid]['created'] == created, 'Allowed worker/parent PID identity changed')
        module = module_name(records[pid])
        require(module in MODULES and module not in {'research.ppt_justification_20261028_v1.controller',
                'research.ppt_justification_20261028_v1.run_one'}, 'Main training can never receive a host-gate exception')
        allowed_modules.add('crossdata' if module.startswith('research.crossdata_component_screen_') else 'E' if module.startswith('research.ppt_e_fixed_') else 'cost')
    require(len(allowed_modules) <= 1, 'Cost and E leases cannot run together')
    conflicts = [pid for pid, row in records.items() if module_name(row) in MODULES and pid not in allowed_processes]
    require(not conflicts, 'Live experiment process prohibits host admission: '+str(conflicts))
    desktops = {row['pid']: row for row in policy['desktop_processes']}
    require(len(desktops) == len(policy['desktop_processes']), 'Duplicate desktop policy PID')
    observed = []
    for context in contexts:
        require(context['gpu_uuid'] == policy['device']['gpu_uuid'], 'Unregistered GPU context device')
        pid = context['pid']
        if pid in allowed_gpu_processes:
            require(pid in allowed_processes, 'GPU worker lacks the validated experiment lease')
            observed.append(dict(pid=pid, role='registered_worker', created=records[pid]['created']))
            continue
        require(pid in desktops and pid in records, 'Unknown GPU context PID: '+str(pid))
        expected, current = desktops[pid], records[pid]
        exe = canonical_exe(current['exe'])
        require(not forbidden_desktop(exe), 'Python/WSL/compute executable cannot be a desktop exception')
        require(current['created'] == expected['created'] and exe == expected['exe'], 'Desktop PID/creation/executable changed: '+str(pid))
        require(image_sha(exe) == expected['exe_sha256'], 'Desktop executable bytes changed: '+str(pid))
        observed.append(dict(pid=pid, role='desktop', created=current['created'], exe=exe, exe_sha256=expected['exe_sha256']))
    system = policy.get('kernel_engine_provider')
    if system is not None:
        require(4 in records and records[4]['name'] == 'System', 'System counter provider identity unavailable')
        observed_system = dict(kind='windows_system_creation_counter_only', pid=4, name=records[4]['name'],
                               created=records[4].get('created'), observed_boot_time=boot_time)
        validate_kernel_identity(observed_system, system)
        system = observed_system
    return dict(timestamp_unix=time.time(), observed=observed, kernel_engine_provider=system,
                allowed_processes={str(pid): created for pid, created in allowed_processes.items()})


def load_policy(contract_path=CONTRACT):
    contract = read(contract_path)
    require(contract['schema_version'] == 1 and contract['mode'] == 'wddm_registered_desktop_concurrency', 'Unknown host contract')
    require(contract['source_sha256'] == sha(__file__), 'Host gate source differs from the public contract')
    policy_path = (REPO/contract['private_policy_relative_path']).resolve()
    require(policy_path.is_relative_to(PRIVATE.resolve()), 'Host policy is outside its private campaign directory')
    require(sha(policy_path) == contract['policy_sha256'], 'Immutable private desktop policy changed')
    policy = read(policy_path)
    validate_kernel_identity(policy.get('kernel_engine_provider'), policy.get('kernel_engine_provider'))
    require(policy['device'] == contract['device'] and policy['counter_target'] == contract['counter_target'], 'Host device/counter binding differs')
    require(contract['performance'] == dict(sample_interval_seconds=1, preflight_seconds=60,
                maximum_sample_gap_seconds=2.5, background_engine_p95_limit_percent=5.,
                background_engine_max_limit_percent=15.), 'Unregistered performance host limits')
    return contract, policy


def admit_identity(contract_path=CONTRACT, *, allowed_processes=None):
    contract, policy = load_policy(contract_path)
    snapshot = validate_identity(policy, process_records(), gpu_contexts(), allowed_processes, boot_time=psutil.boot_time())
    PRIVATE.mkdir(parents=True, exist_ok=True)
    path = PRIVATE/f'identity_{time.time_ns()}_{os.getpid()}.json'
    write_new(path, snapshot)
    return dict(status='PASS', mode='identity_only', policy_sha256=contract['policy_sha256'],
                contract_sha256=sha(contract_path), identity_snapshot_sha256=sha(path),
                private_snapshot_relative_path=path.relative_to(REPO).as_posix(),
                observed_gpu_process_count=len(snapshot['observed']),
                allowed_desktop_context_count=sum(row['role']=='desktop' for row in snapshot['observed']),
                unexpected_gpu_pids=[], desktop_concurrent_load=True,
                scope='Exact registered desktop identities; no claim of an idle or exclusive GPU; no performance-load test in this mode')


def validate_stored_identity(snapshot, policy, *, gpu_worker_allowed=False):
    desktops = {row['pid']: row for row in policy['desktop_processes']}
    seen = set()
    for row in snapshot['observed']:
        require(row['pid'] not in seen, 'Duplicate observed process identity')
        seen.add(row['pid'])
        if row['role'] == 'registered_worker':
            require(gpu_worker_allowed and snapshot['allowed_processes'].get(str(row['pid'])) == row['created'], 'Unbound recorded GPU worker identity')
        else:
            require(row['role'] == 'desktop' and row['pid'] in desktops, 'Unregistered recorded desktop identity')
            require(all(row[key] == desktops[row['pid']][key] for key in ('pid', 'created', 'exe', 'exe_sha256')), 'Recorded desktop tuple differs from immutable policy')
    validate_kernel_identity(snapshot['kernel_engine_provider'], policy.get('kernel_engine_provider'))


def verify_identity_receipt(receipt, contract_path=CONTRACT):
    contract, policy = load_policy(contract_path)
    path = (REPO/receipt['private_snapshot_relative_path']).resolve()
    require(path.is_relative_to(PRIVATE.resolve()) and sha(path) == receipt['identity_snapshot_sha256'], 'Private identity snapshot hash differs')
    snapshot = read(path)
    validate_stored_identity(snapshot, policy)
    require(receipt['status'] == 'PASS' and receipt['mode'] == 'identity_only' and
            receipt['policy_sha256'] == contract['policy_sha256'] and receipt['contract_sha256'] == sha(contract_path) and
            receipt['unexpected_gpu_pids'] == [] and receipt['desktop_concurrent_load'] is True and
            receipt['observed_gpu_process_count'] == receipt['allowed_desktop_context_count'] == len(snapshot['observed']), 'Identity receipt summary differs')
    return dict(status='PASS', identity_snapshot_sha256=receipt['identity_snapshot_sha256'], policy_sha256=contract['policy_sha256'])


def background_engines(sample, policy, identity):
    require(type(sample['timestamp_unix']) in (int, float) and math.isfinite(sample['timestamp_unix']), 'Invalid counter timestamp')
    require(sample['counters'], 'GPU engine telemetry missing')
    known = {row['pid'] for row in identity['observed']}
    desktops = {row['pid'] for row in policy['desktop_processes']}
    workers = {row['pid'] for row in identity['observed'] if row['role'] == 'registered_worker'}
    target = policy['counter_target']
    totals, seen = {}, set()
    for row in sample['counters']:
        match = COUNTER.fullmatch(row['name'])
        require(match is not None and row['status'] in (0, 1), 'Invalid/missing GPU engine counter status')
        pid, luid, physical, engine, kind = match.groups()
        pid, physical, engine = int(pid), int(physical), int(engine)
        value = row['value']
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid GPU engine utilization')
        require(row['name'].lower() not in seen, 'Duplicate GPU engine counter')
        seen.add(row['name'].lower())
        if luid.lower() != target['adapter_luid'] or physical != target['physical_index']:
            continue
        key = f"{luid.lower()}/phys{physical}/eng{engine}/{kind.lower()}"
        totals.setdefault(key, 0.)
        if value > 0:
            system = pid == 4 and identity['kernel_engine_provider'] is not None
            require(pid in known or system, 'Active GPU engine PID lacks a verified context identity: '+str(pid))
            require(pid in desktops or pid in workers or system, 'Unregistered GPU engine process')
        if pid not in workers:
            totals[key] += float(value)
    require(bool(totals), 'No telemetry from the explicitly bound target GPU/LUID')
    require(set(totals) == set(target['engine_keys']), 'Target GPU engine telemetry incomplete or changed')
    return totals


def validate_observation(samples, contract, *, required_start, required_end, preflight=False):
    errors = []
    if len(samples) < 2:
        errors.append('fewer_than_two_counter_samples')
    valid = [row for row in samples if row.get('status') == 'PASS']
    if len(valid) != len(samples):
        errors.append('identity_or_counter_failure')
    times = [row['timestamp_unix'] for row in valid]
    if not times or times[0] > required_start or times[-1] < required_end:
        errors.append('measurement_interval_not_fully_covered')
    if times and any(not 0 < later-earlier <= contract['performance']['maximum_sample_gap_seconds'] for earlier, later in zip(times, times[1:])):
        errors.append('counter_gap_or_nonmonotonic_time')
    if preflight and (len(times) < 61 or times[-1]-times[0] < 60):
        errors.append('preflight_shorter_than_60_seconds')
    engines = sorted({key for row in valid for key in row['background_engines']})
    if any(set(row['background_engines']) != set(contract['counter_target']['engine_keys']) for row in valid):
        errors.append('physical_engine_telemetry_incomplete')
    statistics = {}
    for key in engines:
        values = [row['background_engines'][key] for row in valid if key in row['background_engines']]
        p95, maximum = float(np.quantile(values, .95)), max(values)
        statistics[key] = dict(p95_percent=p95, max_percent=maximum)
        if p95 > 5. or maximum > 15.:
            errors.append('background_load_limit_exceeded:'+key)
    return dict(status='CONTAMINATED' if errors else 'PASS', mode='performance_preflight' if preflight else 'performance_observation',
                errors=errors, sample_count=len(samples), first_timestamp_unix=times[0] if times else None,
                last_timestamp_unix=times[-1] if times else None, required_start_unix=required_start,
                required_end_unix=required_end, background_engine_statistics=statistics,
                policy_sha256=contract['policy_sha256'], desktop_concurrent_load=True,
                scope='Registered desktop concurrency; all target GPU engines observed at 1Hz; subsecond bursts and absolute exclusivity are not certified')


class HostObservation:
    def __init__(self, contract_path, label, allowed_processes, *, gpu_worker=False):
        require(re.fullmatch(r'[A-Za-z0-9_-]+', label) is not None, 'Unsafe host observation label')
        self.contract_path = Path(contract_path)
        self.contract, self.policy = load_policy(contract_path)
        self.allowed = allowed_processes
        self.gpu_allowed = {os.getpid(): allowed_processes[os.getpid()]} if gpu_worker else {}
        self.path = PRIVATE/(label+'.jsonl')
        self.stderr_path = PRIVATE/(label+'_sampler_stderr.bin')
        self.samples, self.failure = [], None
        self.ready = threading.Event()
        self.process = None

    def start(self):
        PRIVATE.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('x', encoding='utf8', newline='\n')
        self.stderr = self.stderr_path.open('xb')
        try:
            self.process = subprocess.Popen(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', COUNTER_SCRIPT],
                stdout=subprocess.PIPE, stderr=self.stderr, text=True, encoding='utf8',
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.thread = threading.Thread(target=self._read, name='registered-host-counter-observer', daemon=True)
            self.thread.start()
            require(self.ready.wait(15) and self.samples and self.samples[0]['status'] == 'PASS', 'Host observer has no valid first sample')
        except BaseException:
            self.finish(time.time(), time.time())
            raise
        return self

    def _read(self):
        try:
            for line in self.process.stdout:
                raw = json.loads(line)
                entry = dict(timestamp_unix=raw['timestamp_unix'], raw=raw)
                try:
                    identity = validate_identity(self.policy, process_records(), gpu_contexts(), self.allowed,
                        self.gpu_allowed, boot_time=psutil.boot_time())
                    entry.update(status='PASS', identity=identity, background_engines=background_engines(raw, self.policy, identity))
                except BaseException as exc:
                    entry.update(status='FAILED', error=type(exc).__name__+': '+str(exc))
                self.stream.write(json.dumps(entry, allow_nan=False)+'\n')
                self.stream.flush()
                self.samples.append(entry)
                self.ready.set()
        except BaseException as exc:
            self.failure = type(exc).__name__+': '+str(exc)
            self.ready.set()

    def wait_preflight(self):
        start = self.samples[0]['timestamp_unix']
        deadline = time.monotonic()+75
        while time.monotonic() < deadline:
            if self.failure or any(row['status'] != 'PASS' for row in self.samples):
                break
            if len(self.samples) >= 61 and self.samples[-1]['timestamp_unix']-start >= 60:
                break
            time.sleep(.1)
        return self.finish(start, start+60, preflight=True)

    def finish(self, required_start, required_end, *, preflight=False):
        deadline = time.monotonic()+6
        while self.process is not None and self.process.poll() is None and time.monotonic() < deadline:
            if self.samples and self.samples[-1]['timestamp_unix'] >= required_end:
                break
            time.sleep(.1)
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()  # Only this observer's own PowerShell child.
            self.process.wait(timeout=10)
            if hasattr(self, 'thread'):
                self.thread.join(timeout=10)
                require(not self.thread.is_alive(), 'Host sampler reader did not finish')
        self.stream.close()
        self.stderr.close()
        result = validate_observation(self.samples, self.contract, required_start=required_start,
                                      required_end=required_end, preflight=preflight)
        if self.failure:
            result['status'] = 'CONTAMINATED'
            result['errors'].append('sampler_reader_failure')
        result.update(contract_sha256=sha(self.contract_path), private_trace_relative_path=self.path.relative_to(REPO).as_posix(),
                      private_trace_sha256=sha(self.path), observer_source_sha256=sha(__file__))
        return result


def verify_observation_receipt(receipt, contract_path=CONTRACT):
    contract, policy = load_policy(contract_path)
    require(receipt['contract_sha256'] == sha(contract_path) and receipt['observer_source_sha256'] == sha(__file__), 'Host observation source/contract changed')
    path = (REPO/receipt['private_trace_relative_path']).resolve()
    require(path.is_relative_to(PRIVATE.resolve()) and sha(path) == receipt['private_trace_sha256'], 'Host private trace changed')
    samples = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
    for entry in samples:
        if entry.get('status') == 'PASS':
            require(entry['timestamp_unix']==entry['raw']['timestamp_unix'], 'Host sample timestamp differs from raw counter')
            validate_stored_identity(entry['identity'], policy, gpu_worker_allowed=receipt['mode']=='performance_observation')
            require(entry['background_engines'] == background_engines(entry['raw'], policy, entry['identity']), 'Host engine totals differ from raw counters')
    recomputed = validate_observation(samples, contract, required_start=receipt['required_start_unix'],
        required_end=receipt['required_end_unix'], preflight=receipt['mode']=='performance_preflight')
    require(all(receipt[key] == value for key, value in recomputed.items()), 'Host observation summary differs from trace')
    require(receipt['status'] == 'PASS', 'Cost host condition contaminated; no comparison PASS')
    return dict(status='PASS', private_trace_sha256=receipt['private_trace_sha256'], policy_sha256=contract['policy_sha256'])


def preserve_capture_stderr(value):
    if value:
        PRIVATE.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='wb', prefix='capture_stderr_', suffix='.bin', dir=PRIVATE, delete=False) as stream:
            stream.write(value)


def capture_policy(pids, adapter_luid, physical_index, include_system=False):
    require(os.name == 'nt', 'WDDM desktop policy capture is Windows-only')
    require(re.fullmatch(r'0x[0-9a-f]+_0x[0-9a-f]+', adapter_luid, re.I) is not None, 'Explicit counter adapter LUID required')
    require(pids and len(pids) == len(set(pids)), 'Explicit distinct reviewed desktop PIDs required')
    records, contexts, device = process_records(), gpu_contexts(), gpu_device()
    roster = {row['pid'] for row in contexts if row['gpu_uuid'] == device['gpu_uuid']}
    entries = []
    for pid in pids:
        require(pid in roster and pid in records, 'Requested desktop PID is absent from the current target GPU roster')
        record = records[pid]
        exe = canonical_exe(record['exe'])
        require(not forbidden_desktop(exe) and module_name(record) not in MODULES and record['created'] is not None,
                'Experiment/compute/unknown process cannot enter desktop policy')
        entries.append(dict(pid=pid, created=record['created'], exe=exe, exe_sha256=image_sha(exe)))
    kernel = None
    if include_system:
        require(4 in records and records[4]['name'] == 'System', 'Windows System provider could not be identified')
        kernel = dict(kind='windows_system_creation_counter_only', pid=4, name='System',
                      created=records[4]['created'], observed_boot_time=psutil.boot_time())
        validate_kernel_identity(kernel, kernel)
    try:
        counter_result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
            COUNTER_SCRIPT.replace('-Continuous', '-MaxSamples 1')], capture_output=True,
            timeout=20, check=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired as exc:
        preserve_capture_stderr(exc.stderr)
        raise RuntimeError('PowerShell counter capture timed out; no desktop policy created') from None
    preserve_capture_stderr(counter_result.stderr)
    require(counter_result.returncode == 0, 'PowerShell counter capture failed; raw stderr retained privately; no desktop policy created')
    try:
        counter_sample = json.loads(counter_result.stdout.decode('utf8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError('PowerShell counter stdout is not valid UTF-8 JSON; no desktop policy created') from None
    require(isinstance(counter_sample, dict) and isinstance(counter_sample.get('counters'), list)
            and counter_sample['counters'], 'Counter capture has missing telemetry; no desktop policy created')
    keys = set()
    for row in counter_sample['counters']:
        match = COUNTER.fullmatch(row['name'])
        require(match is not None and row['status'] in (0, 1), 'Counter capture contains invalid telemetry')
        _, luid, physical, engine, kind = match.groups()
        if luid.lower() == adapter_luid.lower() and int(physical) == physical_index:
            keys.add(f'{luid.lower()}/phys{int(physical)}/eng{int(engine)}/{kind.lower()}')
    require(bool(keys), 'Root-specified counter GPU/LUID is absent')
    policy = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), device=device,
                  counter_target=dict(adapter_luid=adapter_luid.lower(), physical_index=physical_index, engine_keys=sorted(keys)),
                  desktop_processes=sorted(entries, key=lambda row: row['pid']), kernel_engine_provider=kernel,
                  capture_is_not_admission=True, scope='Root-selected desktop identities only; other live experiment PIDs are never admitted')
    write_new(POLICY, policy)
    contract = dict(schema_version=1, mode='wddm_registered_desktop_concurrency', created_utc=policy['created_utc'],
                    private_policy_relative_path=POLICY.relative_to(REPO).as_posix(), policy_sha256=sha(POLICY),
                    device=device, counter_target=policy['counter_target'], source_sha256=sha(__file__),
                    desktop_identity_count=len(entries), kernel_provider='Windows System PID4 + exact creation time; boot estimate is provenance only; counter attribution only' if kernel else None,
                    performance=dict(sample_interval_seconds=1, preflight_seconds=60, maximum_sample_gap_seconds=2.5,
                        background_engine_p95_limit_percent=5., background_engine_max_limit_percent=15.),
                    limits=['Engineering admission limits fixed before measurement; not a universal safety/performance guarantee.',
                            'Known desktop concurrency is retained, including video encode and System Copy engines.',
                            '1Hz samples cannot exclude subsecond bursts or certify an absolutely idle/exclusive GPU.',
                            'New/unknown GPU identities or missing telemetry fail closed; no result-based trial deletion or automatic repeat.'],
                    E_scope='Identity-only host admission before CUDA at each worker start; controller enforces upstream-process exclusion and shared reservations; no performance claim',
                    cost_scope='60-second all-engine preflight and continuous identical observer covering every measured facet')
    write_new(CONTRACT, contract)
    return dict(status='POLICY_CAPTURED_NOT_ADMITTED', policy_sha256=contract['policy_sha256'],
                contract_path=str(CONTRACT), desktop_identity_count=len(entries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-policy', action='store_true', required=True)
    parser.add_argument('--desktop-pids', required=True)
    parser.add_argument('--adapter-luid', required=True)
    parser.add_argument('--physical-index', type=int, required=True)
    parser.add_argument('--include-system-provider', action='store_true')
    args = parser.parse_args()
    result = capture_policy([int(value) for value in args.desktop_pids.split(',')], args.adapter_luid,
                            args.physical_index, args.include_system_provider)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
