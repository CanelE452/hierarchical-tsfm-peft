"""Separate v10 accounting and pinned read-only access to earlier artifacts."""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache/tsfm_peft_ridge_correction_v10_20260929'
V9 = ROOT / 'research/tsfm_peft_staged_p_v9_20260929'
LIMITS = {'fits': 18, 'gpu': 1800, 'cpu_analysis': 1800, 'cpu_check': 300,
          'cost_gpu': 600, 'storage_bytes': 1024**3, 'downloads': 0, 'optimizer_sessions': 0}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    for retry in range(6):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if retry == 5:
                raise
            time.sleep(.05 * 2**retry)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def import_file(path, name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def old_runtime():
    return import_file(V9 / 'runtime_v9.py', '_ridge_readonly_v9_runtime')


def parent_receipt_for(dataset, seed):
    module = import_file(V9 / 'model_v9.py', '_ridge_readonly_v9_model')
    return module.parent_receipt_for(dataset, seed)


def load_parent(dataset, seed, device='cpu'):
    module = import_file(V9 / 'model_v9.py', '_ridge_readonly_v9_model')
    parent = module._load_parent({'dataset': dataset, 'seed': seed}, module.parent_receipt_for(dataset, seed), device)
    return parent.requires_grad_(False).eval()


def load_data(dataset, include_test=False):
    if include_test:
        seal = read_json(HERE / 'selection_seal.json')
        exposure = read_json(HERE / 'test_exposure.json')
        assert seal['selected_sha256'] == digest(HERE / 'selected.json')
        assert seal['reuse_manifest_sha256'] == digest(HERE / 'reuse_manifest.json')
        assert exposure['seal_sha256'] == digest(HERE / 'selection_seal.json')
    return old_runtime().load_data(dataset, include_test=include_test)


def score_arrays(pred, target, mask):
    return old_runtime().score_arrays(pred, target, mask)


def tensors(data, origins, device='cuda'):
    return old_runtime().tensors(data, origins, device=device)


def storage_bytes():
    return sum(p.stat().st_size for base in (HERE, CACHE) if base.exists() for p in base.rglob('*') if p.is_file())


@contextlib.contextmanager
def transaction():
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = CACHE / 'ledger.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(str(os.getpid()))
    try:
        value = read_json(HERE / 'ledger.json')
        yield value
        save_json(HERE / 'ledger.json', value)
    finally:
        lock.unlink()


def elapsed(record):
    return time.time() - record['started_utc'] if record['status'] == 'running' else record.get('elapsed_s', 0)


def budget_snapshot():
    jobs = read_json(HERE / 'ledger.json')['jobs']
    return {'fit_attempts': sum(j['category'] == 'fit' for j in jobs),
            **{key + '_seconds': sum(elapsed(j) for j in jobs if j['category'] == key)
               for key in ('gpu', 'cpu_analysis', 'cpu_check')},
            'cost_gpu_seconds': sum(elapsed(j) for j in jobs if j['category'] == 'gpu' and j['metadata'].get('purpose') == 'cost'),
            'active': [j['label'] for j in jobs if j['status'] == 'running'],
            'storage_bytes': storage_bytes(), 'limits': LIMITS}


class Job:
    def __init__(self, label, category='gpu', reserve_seconds=60, metadata=None):
        assert category in ('gpu', 'cpu_analysis', 'cpu_check', 'fit')
        self.label, self.category, self.reserve = label, category, reserve_seconds
        self.metadata = metadata or {}

    def __enter__(self):
        provenance = read_json(HERE / 'provenance.json')
        assert digest(HERE / 'PLAN.md') == provenance['plan_sha256']
        assert digest(HERE / 'APPROVAL.txt') == provenance['approval_sha256']
        if self.category == 'fit' and (HERE / 'test_exposure.json').exists():
            raise RuntimeError('No fitting after v10 evaluation exposure')
        if self.category == 'gpu':
            output = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                                    capture_output=True, text=True, check=True).stdout
            if any('python' in line.lower() and line.split(',')[0].strip() != str(os.getpid()) for line in output.splitlines()):
                raise RuntimeError('Another GPU Python job exists; do not terminate it')
        with transaction() as ledger:
            jobs = ledger['jobs']
            assert not any(j['label'] == self.label for j in jobs), 'Use a fresh retry label and preserve outputs'
            assert not any(j['category'] == self.category and j['status'] == 'running' for j in jobs), 'Inspect active job/PID'
            if self.category == 'fit':
                assert sum(j['category'] == 'fit' for j in jobs) < LIMITS['fits']
                assert self.metadata.get('dataset') in ('robin', 'jena')
            else:
                used = sum(elapsed(j) for j in jobs if j['category'] == self.category)
                assert used + self.reserve <= LIMITS[self.category], 'Insufficient reserved job budget'
            if self.metadata.get('purpose') == 'cost':
                assert self.category == 'gpu'
                used = sum(elapsed(j) for j in jobs if j['metadata'].get('purpose') == 'cost')
                assert used + self.reserve <= LIMITS['cost_gpu']
            assert storage_bytes() < LIMITS['storage_bytes']
            self.id, self.started = len(jobs), time.perf_counter()
            jobs.append({'id': self.id, 'label': self.label, 'category': self.category, 'metadata': self.metadata,
                         'status': 'running', 'started_utc': time.time(), 'pid': os.getpid(),
                         'reserved_seconds': self.reserve, 'elapsed_s': 0,
                         'sources': {p.name: digest(p) for p in HERE.glob('*.py')}})
        return self

    def check_limits(self):
        budget = budget_snapshot()
        for category in ('gpu', 'cpu_analysis', 'cpu_check'):
            if budget[category + '_seconds'] >= LIMITS[category]:
                raise RuntimeError(category + ' budget exhausted')
        if budget['cost_gpu_seconds'] >= LIMITS['cost_gpu'] or budget['storage_bytes'] >= LIMITS['storage_bytes']:
            raise RuntimeError('Cost/storage budget exhausted')

    def heartbeat(self, progress, updates=None):
        self.check_limits()
        with transaction() as ledger:
            ledger['jobs'][self.id].update(progress=progress, elapsed_s=time.perf_counter() - self.started)
        print(progress, flush=True)

    def __exit__(self, kind, exc, tb):
        with transaction() as ledger:
            row = ledger['jobs'][self.id]
            row.update(status='failed' if kind else 'complete', elapsed_s=time.perf_counter() - self.started, ended_utc=time.time())
            if exc:
                row['error'] = ''.join(traceback.format_exception(kind, exc, tb))


if __name__ == '__main__':
    if (HERE / 'ledger.json').exists():
        raise RuntimeError('Already initialized')
    save_json(HERE / 'provenance.json', {'plan_sha256': digest(HERE / 'PLAN.md'),
              'approval_sha256': digest(HERE / 'APPROVAL.txt'), 'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    save_json(HERE / 'ledger.json', {'limits': LIMITS, 'jobs': [], 'created_utc': time.time()})
    print('v10 ledger initialized; no computation started')
