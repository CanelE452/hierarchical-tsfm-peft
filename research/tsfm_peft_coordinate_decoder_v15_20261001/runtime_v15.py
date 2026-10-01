"""V15-only accounting and data contracts; old generations are read-only."""
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
CACHE = ROOT / '.cache' / HERE.name
DATASETS = {'robin': (17, 5, 24), 'jena': (21, 6, 144), 'hog': (32, 8, 24),
            'peacock_education': (13, 4, 24)}
SEEDS = (92601, 92602)
FAMILIES = ('raw_free', 'pca_free')
LIMITS = {'real_fit': 52, 'neural_basic': 0, 'coefficient_basic': 48, 'technical_reserve': 4,
          'gpu_seconds': 3600, 'cpu_analysis_seconds': 1800, 'cpu_check_seconds': 600,
          'storage_bytes': 2 * 1024**3, 'synthetic_sessions': 0, 'synthetic_steps': 0,
          'download_bytes': 0, 'conflict_wait_per_incident_seconds': 1800, 'conflict_wait_total_seconds': 3600}
TIME_CATEGORIES = ('gpu', 'cpu_analysis', 'cpu_check')


def read_json(path):
    for retry in range(6):
        try:
            return json.loads(Path(path).read_text(encoding='utf-8'))
        except PermissionError:
            if retry == 5:
                raise
            time.sleep(.05 * 2**retry)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    for retry in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if retry == 5:
                raise
            time.sleep(.05 * 2**retry)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest(path):
    value = hashlib.sha256()
    with resolve(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def artifact(path, expected=None):
    path = resolve(path)
    sha = digest(path)
    if expected is not None and sha != expected:
        raise RuntimeError('Artifact hash mismatch: ' + str(path))
    return {'path': str(path), 'sha256': sha, 'bytes': path.stat().st_size}


def import_file(path, name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def verify_provenance():
    value = read_json(HERE / 'provenance.json')
    for name, sha in value['files'].items():
        if digest(HERE / name) != sha:
            raise RuntimeError('Immutable v15 provenance changed: ' + name)


def protocol():
    verify_provenance()
    return read_json(HERE / 'protocol.json')


def source_receipt():
    old = ROOT / 'research/tsfm_peft_internal_vs_subspace_v11_20260930'
    paths = list(HERE.glob('*.py')) + [old / name for name in ('runtime_v11.py', 'model_v11.py')]
    paths += [ROOT / 'src/hier_peft/lora.py']
    for folder, names in (
        ('tsfm_peft_group_basis_v13_20261001', ('cost_v13.py', 'evaluate_v13.py', 'runtime_v13.py', 'model_v13.py', 'train_v13.py')),
        ('tsfm_peft_a_confirmation_v12_20261001', ('cost_confirmation_v12.py', 'evaluate_confirmation_v12.py',
         'runtime_confirmation_v12.py', 'model_confirmation_v12.py', 'train_confirmation_v12.py'))):
        paths += [ROOT / 'research' / folder / name for name in names]
    paths += [HERE / name for name in ('PLAN.md', 'AUTHORIZATION.txt', 'protocol.json', 'provenance.json',
                                      'reuse_manifest.json', 'data_manifest.json')]
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in paths if path.is_file()}


def snapshot_sources(sources):
    result = {}
    for relative, sha in sources.items():
        payload = (ROOT / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha:
            raise RuntimeError('Source changed during reservation: ' + relative)
        path = CACHE / 'source_snapshots' / sha / Path(relative).name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        elif digest(path) != sha:
            raise RuntimeError('Saved source snapshot changed')
        result[relative] = str(path)
    return result


def environment_receipt():
    names = ('torch', 'numpy', 'scipy', 'peft', 'chronos-forecasting', 'transformers', 'pandas')
    return {'python': sys.version, 'executable': sys.executable, 'command': sys.argv,
            'cwd': str(Path.cwd()), 'versions': {name: importlib.metadata.version(name) for name in names}}


def storage_bytes():
    total = 0
    for base in (HERE, CACHE):
        for path in base.rglob('*') if base.exists() else ():
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def require_storage(reserve=0):
    used = storage_bytes()
    if used + reserve > LIMITS['storage_bytes']:
        raise RuntimeError('V15_STORAGE_LIMIT')
    return used


def pid_alive(pid):
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = ctypes.c_ulong()
            return not kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextlib.contextmanager
def ledger_transaction():
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / 'ledger.lock'
    owner = {'pid': os.getpid(), 'token': uuid.uuid4().hex, 'created_utc': time.time()}
    deadline = time.monotonic() + 15
    while True:
        try:
            with path.open('x', encoding='utf-8') as stream:
                json.dump(owner, stream)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError('Ledger lock occupied; preserve and inspect owner: ' + repr(read_json(path)))
            time.sleep(.1)
    try:
        ledger = read_json(HERE / 'ledger.json')
        yield ledger
        save_json(HERE / 'ledger.json', ledger)
    finally:
        if path.exists() and read_json(path).get('token') == owner['token']:
            path.unlink()


def elapsed(row):
    if row['status'] == 'running' and pid_alive(row['pid']):
        return max(row['elapsed_s'], time.time() - row['started_utc'])
    return row['elapsed_s']


def budget_snapshot():
    jobs = read_json(HERE / 'ledger.json')['jobs']
    fits = [row for row in jobs if row['category'] == 'coefficient_fit']
    return {'real_fit_attempts': len(fits), 'neural_fit_attempts': 0, 'coefficient_fit_attempts': len(fits),
            'technical_reserve_attempts': sum(bool(row['metadata'].get('technical_retry')) for row in fits),
            'synthetic_sessions': sum(bool(row.get('synthetic')) for row in jobs),
            **{kind + '_seconds': sum(elapsed(row) for row in jobs if row['category'] == kind) for kind in TIME_CATEGORIES},
            'active_jobs': [{**{key: row.get(key) for key in ('id', 'label', 'pid', 'category', 'progress', 'heartbeat_utc')},
                             'pid_alive': pid_alive(row['pid'])} for row in jobs if row['status'] == 'running'],
            'storage_bytes': storage_bytes(), 'limits': LIMITS}


class Job:
    def __init__(self, category, label, reserve_s=60, metadata=None, synthetic=False):
        if category not in (*TIME_CATEGORIES, 'coefficient_fit') or synthetic and category != 'cpu_check':
            raise ValueError('Invalid v15 job category')
        self.category, self.label, self.reserve = category, label, float(reserve_s)
        self.metadata, self.synthetic = copy.deepcopy(metadata or {}), synthetic
        self.updates, self._heartbeat_error = 0, None
        self._stop = threading.Event()

    def __enter__(self):
        p = protocol()
        require_storage(1024 * 1024)
        if self.category == 'coefficient_fit' and any((HERE / name).exists() for name in ('selection_seal.json', 'test_exposure.json')):
            raise RuntimeError('No fitting after joint selection seal or evaluation exposure')
        if self.category == 'gpu':
            output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
            others = [row for row in output.splitlines() if 'python' in row.lower() and row.split(',')[0].strip() != str(os.getpid())]
            if others:
                raise RuntimeError('Other GPU Python process; do not terminate: ' + repr(others))
        sources = source_receipt()
        snapshots = snapshot_sources(sources)
        with ledger_transaction() as ledger:
            jobs = ledger['jobs']
            if any(row['label'] == self.label for row in jobs):
                raise RuntimeError('Every attempt/job requires a new label')
            if any(row['category'] == self.category and row['status'] == 'running' for row in jobs):
                raise RuntimeError('Unclosed same-category job; verify PID before recovery')
            if self.category == 'coefficient_fit':
                fits = [row for row in jobs if row['category'] == 'coefficient_fit']
                spec = next((row for row in p['fits'] if row['id'] == self.metadata.get('fit_id')), None)
                if spec is None or spec['dataset'] != self.metadata.get('dataset'):
                    raise ValueError('Fit is outside the prescribed matrix')
                prior = [row for row in fits if row['metadata']['fit_id'] == spec['id']]
                retry = self.metadata.get('technical_retry', False)
                if len(fits) >= LIMITS['real_fit'] or prior and (not retry or any(row['status'] != 'failed' for row in prior)):
                    raise RuntimeError('Fit limit, duplicate valid fit, or invalid recovery')
                if retry and (not prior or not self.metadata.get('reason') or sum(bool(row['metadata'].get('technical_retry')) for row in fits) >= LIMITS['technical_reserve']):
                    raise RuntimeError('Invalid technical reserve use')
                parents = [row for row in jobs if row['category'] == 'cpu_analysis' and row['status'] == 'running' and row['pid'] == os.getpid()]
                if len(parents) != 1:
                    raise RuntimeError('Coefficient fitting requires one same-process CPU-analysis parent')
                self.metadata['parent_job_id'] = parents[0]['id']
            elif sum(elapsed(row) for row in jobs if row['category'] == self.category) + self.reserve > LIMITS[self.category + '_seconds']:
                raise RuntimeError('Insufficient reserved category budget')
            if self.synthetic and sum(bool(row.get('synthetic')) for row in jobs) >= LIMITS['synthetic_sessions']:
                raise RuntimeError('Synthetic session limit')
            self.id, self.started = len(jobs), time.perf_counter()
            jobs.append({'id': self.id, 'category': self.category, 'label': self.label, 'status': 'running',
                         'pid': os.getpid(), 'synthetic': self.synthetic, 'started_utc': time.time(),
                         'heartbeat_utc': time.time(), 'elapsed_s': 0, 'reserved_seconds': self.reserve,
                         'updates': 0, 'metadata': self.metadata, 'source': sources, 'source_snapshots': snapshots})
        self._thread = threading.Thread(target=self._auto_heartbeat, daemon=True)
        self._thread.start()
        return self

    def _auto_heartbeat(self):
        while not self._stop.wait(10):
            try:
                self.heartbeat(None)
            except Exception:
                self._heartbeat_error = traceback.format_exc()
                return

    def check_limits(self):
        if self._heartbeat_error:
            raise RuntimeError('Background heartbeat failed: ' + self._heartbeat_error)
        budget = budget_snapshot()
        for category in TIME_CATEGORIES:
            if budget[category + '_seconds'] >= LIMITS[category + '_seconds']:
                raise RuntimeError(category + ' total time limit')
        require_storage(1024 * 1024)

    def heartbeat(self, progress, updates=None):
        if updates is not None:
            if updates < self.updates or self.synthetic and updates > LIMITS['synthetic_steps']:
                raise RuntimeError('Invalid optimizer-update accounting')
            self.updates = int(updates)
        with ledger_transaction() as ledger:
            row = ledger['jobs'][self.id]
            row.update(elapsed_s=time.perf_counter() - self.started, heartbeat_utc=time.time(), updates=self.updates)
            if progress is not None:
                row['progress'] = progress
        self.check_limits()

    def __exit__(self, kind, exc, tb):
        self._stop.set()
        self._thread.join(timeout=20)
        with ledger_transaction() as ledger:
            row = ledger['jobs'][self.id]
            row.update(status='failed' if kind or self._heartbeat_error else 'complete', ended_utc=time.time(),
                       elapsed_s=time.perf_counter() - self.started, updates=self.updates)
            if exc:
                row['error'] = ''.join(traceback.format_exception(kind, exc, tb))
            if self._heartbeat_error:
                row['heartbeat_error'] = self._heartbeat_error
        if not kind and self._heartbeat_error:
            raise RuntimeError(self._heartbeat_error)


def configure():
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def load_data(dataset, include_test=False):
    import numpy as np
    protocol()
    manifest = read_json(HERE / 'data_manifest.json')
    if include_test:
        seal = read_json(HERE / 'selection_seal.json')
        exposure = read_json(HERE / 'test_exposure.json')
        if exposure['seal_sha256'] != digest(HERE / 'selection_seal.json'):
            raise RuntimeError('Joint evaluation exposure seal changed')
        for name in ('selected.json', 'protocol.json', 'data_manifest.json'):
            if seal['files'][name] != digest(HERE / name):
                raise RuntimeError('Joint selection binding changed: ' + name)
        if set(seal['datasets']) != set(DATASETS):
            raise RuntimeError('Every exposed development dataset must be selected jointly')
    entry = manifest['datasets'][dataset]['test' if include_test else 'trainval']
    path = artifact(entry['path'], entry['sha256'])['path']
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    c, k, period = DATASETS[dataset]
    if (data['x'].shape[1] != c or data['pca_basis'].shape != (c, k)
            or data['finite'].shape != data['x'].shape):
        raise ValueError('V15 arrays differ from the fixed original-PCA shape contract')
    data.update(_path=Path(path), _dataset=dataset, _C=c, _K=k, _phase_period=period)
    return data


_numeric = import_file(ROOT / 'research/tsfm_peft_internal_vs_subspace_v11_20260930/runtime_v11.py', 'v15_pinned_numeric_only')
tensors, batches = _numeric.tensors, _numeric.batches
phase_sample, masked_macro_loss, score_arrays = _numeric.phase_sample, _numeric.masked_macro_loss, _numeric.score_arrays
