"""Isolated zero-fit Chronos-2 control campaign; old records are read-only."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache/level_chronos2_controls_v1'
DATASETS = ('robin', 'peacock_education', 'jena')
SEEDS = (92601, 92602)
SCOPE = 'Post-exposure zero-shot controls; no new fitting or independent confirmation'
LIMITS = {'gpu_seconds': 10800, 'cpu_seconds': 3600, 'download_bytes': 4 * 1024**3,
          'storage_bytes': 8 * 1024**3, 'fits': 0}
_lock = threading.RLock()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path).resolve()
    if not (path.is_relative_to(HERE) or path.is_relative_to(CACHE)):
        raise PermissionError('Writes are limited to the new campaign directories')
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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def artifact(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    actual = digest(path)
    if expected is not None and actual != expected:
        raise RuntimeError('Artifact bytes differ: ' + str(path))
    return {'path': str(path.resolve()), 'sha256': actual, 'bytes': path.stat().st_size}


def import_file(path, name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def configure():
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def environment():
    import torch
    return {'python': sys.version, 'executable': sys.executable,
       'versions': {n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'scipy', 'chronos-forecasting', 'transformers', 'peft', 'huggingface-hub')},
       'device': torch.cuda.get_device_name(), 'total_gpu_bytes': torch.cuda.get_device_properties(0).total_memory,
       'dtype': 'float32', 'tf32': False, 'cpu_threads': 4}


def budget_snapshot():
    record = read_json(HERE / 'ledger.json')
    now = time.time()
    elapsed = lambda j: now - j['started_utc'] if j['status'] == 'running' else j['elapsed_s']
    value = {kind + '_seconds': sum(elapsed(j) for j in record['jobs'] if j['category'] == kind)
             for kind in ('gpu', 'cpu')}
    value.update(download_bytes=record['download_bytes'], fits=0, limits=LIMITS,
                 active=[j for j in record['jobs'] if j['status'] == 'running'])
    return value


def storage_bytes():
    return sum(p.stat().st_size for base in (HERE, CACHE) for p in base.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))


class Job:
    def __init__(self, category, label, reserve_s=0, metadata=None):
        if category not in ('gpu', 'cpu'):
            raise ValueError('Fitting/optimizer jobs are not authorized')
        self.category, self.label, self.reserve, self.metadata = category, label, reserve_s, metadata or {}
        self._stop = threading.Event()
        self._error = None

    def __enter__(self):
        with _lock:
            ledger = read_json(HERE / 'ledger.json')
            budget = budget_snapshot()
            if budget['active']:
                raise RuntimeError('A campaign job is still running; inspect its actual PID before resuming')
            if budget[self.category + '_seconds'] + self.reserve > LIMITS[self.category + '_seconds']:
                raise RuntimeError('Insufficient remaining ' + self.category + ' budget')
            self.id, self.started = len(ledger['jobs']), time.perf_counter()
            ledger['jobs'].append({'id': self.id, 'category': self.category, 'label': self.label,
                'pid': os.getpid(), 'status': 'running', 'started_utc': time.time(),
                'elapsed_s': 0, 'metadata': self.metadata, 'fit_count': 0})
            save_json(HERE / 'ledger.json', ledger)
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def _monitor(self):
        while not self._stop.wait(10):
            try:
                self.heartbeat(None)
            except BaseException:
                self._error = traceback.format_exc()
                return

    def check_limits(self):
        if self._error:
            raise RuntimeError(self._error)
        budget = budget_snapshot()
        if any(budget[k + '_seconds'] >= LIMITS[k + '_seconds'] - 10 for k in ('gpu', 'cpu')):
            raise RuntimeError('Campaign time cap reached')
        if budget['download_bytes'] > LIMITS['download_bytes']:
            raise RuntimeError('Download cap exceeded')

    def heartbeat(self, progress=None, **extra):
        with _lock:
            ledger = read_json(HERE / 'ledger.json')
            row = ledger['jobs'][self.id]
            row.update(elapsed_s=time.perf_counter() - self.started, heartbeat_utc=time.time(), **extra)
            if progress is not None:
                row['progress'] = progress
            save_json(HERE / 'ledger.json', ledger)
        self.check_limits()

    def __exit__(self, kind, exc, tb):
        self._stop.set()
        self._thread.join(timeout=12)
        with _lock:
            ledger = read_json(HERE / 'ledger.json')
            row = ledger['jobs'][self.id]
            row.update(status='failed' if kind or self._error else 'complete',
                       elapsed_s=time.perf_counter() - self.started, ended_utc=time.time())
            if exc:
                row['error'] = ''.join(traceback.format_exception(kind, exc, tb))
            if self._error:
                row['heartbeat_error'] = self._error
            save_json(HERE / 'ledger.json', ledger)
        if not kind and self._error:
            raise RuntimeError(self._error)


def check_seal():
    seal = read_json(HERE / 'evaluation_seal.json')
    for path, expected in seal['files'].items():
        artifact(HERE / path, expected)
    return seal
