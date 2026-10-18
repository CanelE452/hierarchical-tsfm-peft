"""New-campaign artifacts, immutable evidence and explicit resource limits."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
REPORT = ROOT / 'reports' / CODE.name
CACHE = ROOT / '.cache' / CODE.name


class BudgetExceeded(RuntimeError):
    pass


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def state_sha(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False, default=_json_default) + '\n').encode('utf-8')


def _destination(path, allowed_root=None):
    path = Path(path).resolve()
    roots = (Path(allowed_root).resolve(),) if allowed_root is not None else (CACHE.resolve(), REPORT.resolve())
    if not any(path.is_relative_to(root) and path != root for root in roots):
        raise ValueError('Artifact path escapes this campaign')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def seal_json(path, value, allowed_root=None):
    data = json_bytes(value)
    path = _destination(path, allowed_root)
    with path.open('xb') as stream:
        stream.write(data)


def atomic_json(path, value):
    data = json_bytes(value)
    path = _destination(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    with temp.open('xb') as stream:
        stream.write(data)
    temp.replace(path)


def seal_npz(path, allowed_root=None, **arrays):
    path = _destination(path, allowed_root)
    if allowed_root is None and not path.is_relative_to(CACHE.resolve()):
        raise ValueError('Prediction/feature arrays must remain private')
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def budget_status(started, now, gpu_committed, gpu_active, cache_bytes):
    elapsed = max(0., now - started)
    used = gpu_committed + gpu_active
    if elapsed > 10800 or used > 5400 or cache_bytes > 6 * 1024**3:
        raise BudgetExceeded('Registered wall/GPU/storage budget exhausted')
    return {'elapsed_seconds': elapsed, 'GPU_used_seconds': used,
            'new_cache_bytes': cache_bytes, 'wall_remaining_seconds': 10800 - elapsed,
            'GPU_remaining_seconds': 5400 - used}


def choose_updates(seconds_per_update, gpu_used, elapsed, overhead_seconds):
    if seconds_per_update <= 0 or min(gpu_used, elapsed, overhead_seconds) < 0:
        raise ValueError('Invalid TRAIN timing estimate')
    available = min(5400 * .8 - gpu_used, 10800 * .8 - elapsed)
    selected = None
    for updates in (300, 100):
        if 8 * updates * seconds_per_update + overhead_seconds <= available:
            selected = updates
            break
    return {'updates': selected,
            'status': 'FIXED_300_UPDATES' if selected == 300 else 'BOUNDED_100_UPDATE_PILOT' if selected == 100 else 'NOT_RUN_BUDGET',
            'eight_arm_estimate_seconds': 8 * selected * seconds_per_update + overhead_seconds if selected else None,
            'training_available_seconds': available, 'seconds_per_update': seconds_per_update,
            'overhead_seconds': overhead_seconds, 'selection_uses_VAL': False}


def current_budget():
    state = read_json(REPORT / 'START_STATE.json')
    ledger = read_json(CACHE / 'GPU_LEDGER.json')
    active = max(0., time.time() - ledger['active_started_unix']) if ledger.get('status') == 'RUNNING' else 0.
    size = sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())
    return budget_status(state['started_unix'], time.time(), ledger['GPU_seconds'], active, size)


@contextmanager
def gpu_lease(stage):
    current_budget()
    lock = CACHE / 'GPU.lock'
    with lock.open('x', encoding='ascii') as stream:
        stream.write(str(os.getpid()))
    ledger = read_json(CACHE / 'GPU_LEDGER.json')
    if ledger['status'] != 'IDLE':
        lock.unlink()
        raise RuntimeError('A prior campaign GPU lease is still active; inspect its actual process')
    started = time.time()
    ledger.update(status='RUNNING', active_started_unix=started, active_stage=stage, owner_pid=os.getpid())
    atomic_json(CACHE / 'GPU_LEDGER.json', ledger)
    outcome = 'FAILED'
    try:
        yield
        outcome = 'COMPLETE'
    finally:
        ended = time.time()
        ledger['GPU_seconds'] += ended - started
        ledger['stages'].append({'stage': stage, 'started_unix': started, 'ended_unix': ended,
                                 'wall_seconds': ended - started, 'status': outcome})
        ledger.update(status='IDLE')
        for key in ('active_started_unix', 'active_stage', 'owner_pid'):
            ledger.pop(key, None)
        atomic_json(CACHE / 'GPU_LEDGER.json', ledger)
        lock.unlink()


def progress(stage, **values):
    atomic_json(CACHE / 'PROGRESS.json', {'stage': stage, 'pid': os.getpid(),
                                       'updated_unix': time.time(), **current_budget(), **values})
