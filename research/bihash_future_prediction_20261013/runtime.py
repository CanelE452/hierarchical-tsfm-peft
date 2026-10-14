"""Fresh artifacts and bounded, terminal-only resource receipts for this study."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import time

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name
GOAL_STARTED_UNIX = 1791902100
GPU_SECONDS = 14400
CPU_SECONDS = 7200
CAMPAIGN_SECONDS = 28800
STORAGE_BYTES = 12 * 1024**3
PROCESS_BYTES = 8 * 1024**3
PILOT_MEASURED_GPU_SECONDS = 475.5248562004417
PILOT_UNMETERED_RESERVE_SECONDS = 60.


def jsonable(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [jsonable(v) for v in value]
    return value


def json_line(value):
    return json.dumps(jsonable(value), ensure_ascii=True, allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''): digest.update(chunk)
    return digest.hexdigest()


def _destination(path):
    path = Path(path).resolve()
    if not (path.is_relative_to(HERE) or path.is_relative_to(CACHE)):
        raise ValueError('WRITE_OUTSIDE_NEW_STUDY')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write(path, value):
    payload = json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with _destination(path).open('x', encoding='utf-8') as handle: handle.write(payload)


def _artifact(path):
    return {'path': Path(path).resolve().relative_to(ROOT).as_posix(),
            'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def save(path, state):
    path = _destination(path)
    with path.open('xb') as handle: torch.save(state, handle)
    return _artifact(path)


def save_npz(path, **arrays):
    path = _destination(path)
    with path.open('xb') as handle: np.savez_compressed(handle, **arrays)
    return {**_artifact(path), 'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
            for key, value in arrays.items()}}


def tensor(value, device='cuda'):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def clone_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def state_hash(module):
    digest = hashlib.sha256()
    state = module.state_dict() if hasattr(module, 'state_dict') else module
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def source_snapshot(tag):
    folder = CACHE / 'source_snapshots' / tag
    folder.mkdir(parents=True, exist_ok=False)
    paths = [HERE / name for name in ('model.py', 'data.py', 'train.py', 'run.py', 'probes.py',
             'benchmark.py', 'runtime.py', 'PLAN_KO.md', 'SOURCE_NOTES_KO.md', 'FRESH_SOURCE_PLAN.json',
             'download_new.py') if (HERE / name).is_file()]
    paths += [ROOT / name for name in (
        'research/smallk_predictive_alignment_20261013/engine.py',
        'research/forecast_component_study_20261013/component_experiment.py',
        'research/forecast_generalization_study_20261013/experiment.py',
        'research/forecast_generalization_study_20261013/diagnose_constant_gradient.py',
        'research/past_only_repr_diagnosis_20261012/common.py')]
    result = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = folder / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with path.open('rb') as source, destination.open('xb') as target: shutil.copyfileobj(source, target)
        result[relative.as_posix()] = sha(path)
    write(folder / 'MANIFEST.json', {'tag': tag, 'created_unix': time.time(), 'sha256': result})
    return result


class BudgetHeld(RuntimeError):
    pass


class Limit:
    def __init__(self, tag, gpu=True):
        self.tag, self.gpu = tag, bool(gpu)
        self.process = psutil.Process()
        self.started_unix = self.process.create_time()
        self.start = time.perf_counter()
        self.initial_elapsed = max(0., time.time() - self.started_unix)
        self.last_storage = -60.
        self.rss = self.vram = self.reserved = 0
        self.closed = False
        self.path = HERE / 'resources' / f'{os.getpid()}_{time.time_ns()}_{tag}.json'
        self.previous = []
        for path in (HERE / 'resources').glob('*.json'):
            receipt = read(path)
            if receipt.get('status') not in ('COMPLETE', 'FAILED_PRESERVED', 'BUDGET_HELD_PRESERVED'):
                raise RuntimeError('NONTERMINAL_RESOURCE_FILE: ' + str(path))
            self.previous.append(receipt)
        self.storage = 0
        self.check()

    def _elapsed(self):
        return self.initial_elapsed + time.perf_counter() - self.start

    def check(self):
        self.rss = max(self.rss, self.process.memory_info().rss)
        if self.gpu and torch.cuda.is_initialized():
            self.vram = max(self.vram, torch.cuda.max_memory_allocated())
            self.reserved = max(self.reserved, torch.cuda.max_memory_reserved())
        elapsed = self._elapsed()
        measured = sum(r['wall_seconds'] for r in self.previous if r['gpu'] == self.gpu) + elapsed
        budget = GPU_SECONDS if self.gpu else CPU_SECONDS
        if self.gpu: measured += PILOT_MEASURED_GPU_SECONDS + PILOT_UNMETERED_RESERVE_SECONDS
        if elapsed - self.last_storage >= 30:
            roots = (HERE, CACHE, ROOT / 'research/bihash_future_alignment_20261013',
                     ROOT / '.cache/bihash_future_alignment_20261013')
            self.storage = sum(p.stat().st_size for folder in roots if folder.exists()
                               for p in folder.rglob('*') if p.is_file())
            self.last_storage = elapsed
            if self.storage > STORAGE_BYTES or shutil.disk_usage(ROOT).free < 2 * 1024**3:
                raise BudgetHeld('NEW_STUDY_STORAGE_OR_DISK_GUARD')
        if measured > budget:
            raise BudgetHeld(f'PROCESS_WALL_BUDGET: {measured:.3f} > {budget}')
        if time.time() - GOAL_STARTED_UNIX > CAMPAIGN_SECONDS:
            raise BudgetHeld('CAMPAIGN_WALL_BUDGET')
        if self.rss > PROCESS_BYTES or self.vram > PROCESS_BYTES:
            raise BudgetHeld('PROCESS_RSS_OR_ALLOCATED_VRAM_GUARD')
        return {'elapsed_seconds': elapsed, 'budget_counted_seconds': measured, 'storage_bytes': self.storage}

    def close(self, status='COMPLETE', error=None):
        if self.closed: return _artifact(self.path)
        if self.gpu and torch.cuda.is_initialized(): torch.cuda.synchronize()
        self.rss = max(self.rss, self.process.memory_info().rss)
        if self.gpu and torch.cuda.is_initialized():
            self.vram = max(self.vram, torch.cuda.max_memory_allocated())
            self.reserved = max(self.reserved, torch.cuda.max_memory_reserved())
        receipt = {'tag': self.tag, 'gpu': self.gpu, 'pid': os.getpid(), 'status': status,
                   'started_unix': self.started_unix, 'finished_unix': time.time(),
                   'wall_seconds': self._elapsed(), 'sampled_peak_rss': self.rss,
                   'peak_allocated_vram': self.vram, 'peak_reserved_vram': self.reserved,
                   'new_study_storage_bytes': self.storage, 'failure_reason': None if error is None else str(error),
                   'wall_scope': 'process creation to terminal receipt, includes Python startup and CPU work',
                   'pilot_measured_seconds_counted': PILOT_MEASURED_GPU_SECONDS if self.gpu else 0.,
                   'pilot_unmetered_fixture_reserve_not_measurement': PILOT_UNMETERED_RESERVE_SECONDS if self.gpu else 0.,
                   'previous_receipts': len(self.previous)}
        write(self.path, receipt)
        self.closed = True
        print(json_line({'resource_terminal': _artifact(self.path), 'status': status,
                         'wall_seconds': receipt['wall_seconds']}), flush=True)
        return _artifact(self.path)

