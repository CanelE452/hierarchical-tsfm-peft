"""V12 accounting core adapted from the verified v11 runtime; old ledgers are never used."""
import contextlib
import copy
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache/tsfm_peft_a_confirmation_v12_20261001'
OLD = ROOT / 'research/tsfm_peft_internal_vs_subspace_v11_20260930'
DATASETS = {'robin': (17, 5, 24), 'jena': (21, 6, 144), 'hog': (32, 8, 24)}
SEEDS = (92601, 92602)
LIMITS = {'real_fit': 60, 'neural_basic': 36, 'coefficient_basic': 0,
          'technical_reserve': 4, 'gpu_seconds': 28800, 'cpu_analysis_seconds': 7200,
          'cpu_check_seconds': 1800, 'synthetic_sessions': 6, 'synthetic_steps': 10,
          'storage_bytes': 10 * 1024**3, 'download_bytes': 0,
          'conflict_wait_per_incident_seconds': 1800, 'conflict_wait_total_seconds': 3600}
FIT_CATEGORIES = ('neural_fit', 'coefficient_fit')
TIME_CATEGORIES = ('gpu', 'cpu_analysis', 'cpu_check')


def read_json(path):
    path = Path(path)
    for retry in range(6):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except PermissionError:
            if retry == 5:
                raise
            time.sleep(.05 * 2**retry)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    for retry in range(6):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if retry == 5:
                raise
            time.sleep(.05 * 2**retry)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def artifact(path, expected=None):
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    sha = digest(path)
    if expected is not None and sha != expected:
        raise RuntimeError(f'Artifact hash mismatch: {path}')
    return {'path': str(path), 'sha256': sha, 'bytes': path.stat().st_size}


def import_file(path, name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def source_receipt():
    paths = list(HERE.glob('*.py')) + list(HERE.glob('data_contract*.json'))
    paths += [HERE / name for name in ('PLAN.md', 'APPROVAL.txt', 'REQUEST_PLAN.txt', 'protocol.json', 'reuse_manifest.json')]
    paths += [HERE / name for name in ('AUTHORIZATION_SUPPLEMENT.txt', 'PLAN_SUPPLEMENT.md',
                                      'confirmation_protocol.json', 'confirmation_binding.json')]
    paths += [OLD / name for name in ('runtime_v11.py', 'model_v11.py')]
    paths += [ROOT / 'src/hier_peft/lora.py']
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths if p.is_file()}


def snapshot_sources(receipt):
    snapshots = {}
    for relative, expected in receipt.items():
        source = ROOT / relative
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError('Source changed during reservation; retry after edits finish: ' + relative)
        destination = CACHE / 'source_snapshots' / expected / source.name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        elif digest(destination) != expected:
            raise RuntimeError('Stored source snapshot changed')
        snapshots[relative] = str(destination)
    return snapshots


def environment_receipt():
    names = ('torch', 'numpy', 'scipy', 'peft', 'chronos-forecasting', 'transformers', 'pandas')
    return {'python': sys.version, 'executable': sys.executable, 'command': sys.argv,
            'cwd': str(Path.cwd()), 'versions': {n: importlib.metadata.version(n) for n in names}}


def storage_bytes():
    total = 0
    for base in (HERE, CACHE):
        if not base.exists():
            continue
        for path in base.rglob('*'):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                # Atomic JSON replacement can remove an enumerated temporary file.
                continue
    return total


def require_storage(reserve=0):
    used = storage_bytes()
    if used + reserve > LIMITS['storage_bytes']:
        raise RuntimeError('V12_STORAGE_LIMIT')
    return used


def pid_alive(pid):
    pid = int(pid)
    if pid <= 0:
        return False
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextlib.contextmanager
def ledger_transaction():
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = CACHE / 'ledger.lock'
    token = {'pid': os.getpid(), 'created_utc': time.time(), 'token': uuid.uuid4().hex}
    deadline = time.monotonic() + 15
    while True:
        try:
            with lock.open('x', encoding='utf-8') as handle:
                json.dump(token, handle)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                owner = read_json(lock)
                raise RuntimeError(f'Ledger lock occupied; inspect PID before recovery: {owner}, alive={pid_alive(owner["pid"])}')
            time.sleep(.1)
    try:
        ledger = read_json(HERE / 'ledger.json')
        yield ledger
        save_json(HERE / 'ledger.json', ledger)
    finally:
        if lock.exists() and read_json(lock).get('token') == token['token']:
            lock.unlink()


def elapsed(record):
    if record['status'] == 'running':
        if pid_alive(record['pid']):
            return max(record.get('elapsed_s', 0), time.time() - record['started_utc'])
        return record.get('elapsed_s', 0)
    return record.get('elapsed_s', 0)


def budget_snapshot():
    jobs = read_json(HERE / 'ledger.json')['jobs']
    fits = [j for j in jobs if j['category'] in FIT_CATEGORIES]
    return {'real_fit_attempts': len(fits),
            'neural_fit_attempts': sum(j['category'] == 'neural_fit' for j in fits),
            'coefficient_fit_attempts': sum(j['category'] == 'coefficient_fit' for j in fits),
            'technical_reserve_attempts': sum(bool(j['metadata'].get('technical_retry')) for j in fits),
            'synthetic_sessions': sum(bool(j.get('synthetic')) for j in jobs),
            **{c + '_seconds': sum(elapsed(j) for j in jobs if j['category'] == c) for c in TIME_CATEGORIES},
            'active_jobs': [{**{k: j.get(k) for k in ('id', 'label', 'pid', 'category', 'progress', 'heartbeat_utc')},
                            'pid_alive': pid_alive(j['pid'])} for j in jobs if j['status'] == 'running'],
            'storage_bytes': storage_bytes(), 'limits': LIMITS}


def verify_provenance():
    provenance = read_json(HERE / 'provenance.json')
    for name, key in (('PLAN.md', 'plan_sha256'), ('APPROVAL.txt', 'approval_sha256'),
                      ('REQUEST_PLAN.txt', 'request_sha256'), ('protocol.json', 'protocol_sha256')):
        if digest(HERE / name) != provenance[key]:
            raise RuntimeError(f'Immutable approval/protocol changed: {name}')
    if provenance.get('reuse_manifest_sha256') and digest(HERE / 'reuse_manifest.json') != provenance['reuse_manifest_sha256']:
        raise RuntimeError('Pinned reuse manifest changed')


def confirmation_protocol():
    binding = read_json(HERE / 'confirmation_binding.json')
    for name, expected in binding['files'].items():
        if digest(HERE / name) != expected:
            raise RuntimeError('Confirmation contract changed: ' + name)
    return read_json(HERE / 'confirmation_protocol.json')


class Job:
    def __init__(self, category, label, reserve_s=60, metadata=None, synthetic=False):
        if category not in TIME_CATEGORIES + FIT_CATEGORIES:
            raise ValueError(category)
        if synthetic and category != 'cpu_check':
            raise ValueError('Synthetic optimizer sessions belong to CPU checks')
        self.label, self.category, self.reserve = label, category, float(reserve_s)
        self.metadata, self.synthetic = copy.deepcopy(metadata or {}), synthetic
        if 'fit_id' not in self.metadata and 'id' in self.metadata:
            self.metadata['fit_id'] = self.metadata['id']
        self.updates = 0
        self._stop = threading.Event()
        self._heartbeat_error = None

    def __enter__(self):
        verify_provenance()
        require_storage()
        if self.category in FIT_CATEGORIES:
            if (HERE / 'test_exposure.json').exists() or (HERE / 'confirmation_test_exposure.json').exists():
                raise RuntimeError('No fitting after new confirmation exposure')
            fit_protocol = confirmation_protocol()
            if self.metadata.get('dataset') not in fit_protocol['units'] or not self.metadata.get('fit_id'):
                raise ValueError('Fit reservation requires approved dataset and fit_id')
        if self.category == 'gpu':
            output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                              '--format=csv,noheader'], text=True)
            others = [row for row in output.splitlines() if 'python' in row.lower()
                      and row.split(',')[0].strip() != str(os.getpid())]
            if others:
                raise RuntimeError(f'Other GPU Python process; do not terminate: {others}')
        sources = source_receipt()
        source_snapshots = snapshot_sources(sources)
        with ledger_transaction() as ledger:
            jobs = ledger['jobs']
            if any(j['label'] == self.label for j in jobs):
                raise RuntimeError('Every retry requires a fresh job label')
            if any(j['category'] == self.category and j['status'] == 'running' for j in jobs):
                raise RuntimeError('Unclosed same-category job; verify PID before continuing')
            if self.category in FIT_CATEGORIES:
                fits = [j for j in jobs if j['category'] in FIT_CATEGORIES]
                if len(fits) >= LIMITS['real_fit']:
                    raise RuntimeError('REAL_FIT_LIMIT')
                if any(j['status'] == 'running' for j in fits):
                    raise RuntimeError('Only one real-data fit can run at a time')
                protocol = fit_protocol
                fit_ids = {row['id'] for row in protocol['neural_fits' if self.category == 'neural_fit' else 'coefficient_fits']}
                previous = [j for j in fits if j['metadata'].get('fit_id') == self.metadata['fit_id']]
                retry = bool(self.metadata.get('technical_retry'))
                if self.metadata['fit_id'] not in fit_ids:
                    raise RuntimeError('Fit is outside the frozen confirmation matrix')
                if previous and not retry:
                    raise RuntimeError('Restart/retry consumes technical reserve')
                if retry and (not self.metadata.get('reason') or
                              sum(bool(j['metadata'].get('technical_retry')) for j in fits) >= LIMITS['technical_reserve']):
                    raise RuntimeError('Technical reserve exhausted or missing recovery reason')
                parent_category = 'gpu' if self.category == 'neural_fit' else 'cpu_analysis'
                parents = [j for j in jobs if j['category'] == parent_category and j['status'] == 'running' and j['pid'] == os.getpid()]
                if len(parents) != 1:
                    raise RuntimeError(f'Fit must be nested inside one same-process {parent_category} job')
                self.metadata['parent_job_id'] = parents[0]['id']
            else:
                cap = LIMITS[self.category + '_seconds']
                used = sum(elapsed(j) for j in jobs if j['category'] == self.category)
                if used + self.reserve > cap:
                    raise RuntimeError(f'Insufficient {self.category} budget: {used}+{self.reserve}>{cap}')
            if self.synthetic and sum(bool(j.get('synthetic')) for j in jobs) >= LIMITS['synthetic_sessions']:
                raise RuntimeError('SYNTHETIC_SESSION_LIMIT')
            self.id, self.started = len(jobs), time.perf_counter()
            jobs.append({'id': self.id, 'label': self.label, 'category': self.category, 'synthetic': self.synthetic,
                         'status': 'running', 'pid': os.getpid(), 'started_utc': time.time(),
                         'heartbeat_utc': time.time(), 'reserved_seconds': self.reserve,
                         'elapsed_s': 0, 'updates': 0, 'metadata': self.metadata, 'source': sources,
                         'source_snapshots': source_snapshots})
        self._thread = threading.Thread(target=self._auto_heartbeat, daemon=True)
        self._thread.start()
        return self

    def _auto_heartbeat(self):
        while not self._stop.wait(10):
            try:
                self.heartbeat(None)
            except Exception as error:
                self._heartbeat_error = traceback.format_exc()
                return

    def check_limits(self):
        if self._heartbeat_error:
            raise RuntimeError('Job heartbeat failed: ' + self._heartbeat_error)
        snapshot = budget_snapshot()
        for category in TIME_CATEGORIES:
            if snapshot[category + '_seconds'] >= LIMITS[category + '_seconds']:
                raise RuntimeError(category.upper() + '_TOTAL_LIMIT')
        require_storage(1024 * 1024)

    def heartbeat(self, progress, updates=None):
        if updates is not None:
            if updates < self.updates or (self.synthetic and updates > LIMITS['synthetic_steps']):
                raise RuntimeError('Invalid/decreased optimizer update accounting')
            self.updates = int(updates)
        with ledger_transaction() as ledger:
            record = ledger['jobs'][self.id]
            record.update(elapsed_s=time.perf_counter() - self.started, heartbeat_utc=time.time(), updates=self.updates)
            if progress is not None:
                record['progress'] = progress
        self.check_limits()

    def __exit__(self, kind, exc, tb):
        self._stop.set()
        self._thread.join(timeout=20)
        with ledger_transaction() as ledger:
            record = ledger['jobs'][self.id]
            record.update(status='failed' if kind or self._heartbeat_error else 'complete',
                          elapsed_s=time.perf_counter() - self.started, ended_utc=time.time(), updates=self.updates)
            if exc:
                record['error'] = ''.join(traceback.format_exception(kind, exc, tb))
            if self._heartbeat_error:
                record['heartbeat_error'] = self._heartbeat_error
        if not kind and self._heartbeat_error:
            raise RuntimeError('Background heartbeat/limit failure: ' + self._heartbeat_error)
