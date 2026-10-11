"""Local I/O, observed metrics and cumulative limits for direct professor experiments."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name


def serializable(value):
    if isinstance(value, Path):
        return value.relative_to(ROOT).as_posix() if value.is_relative_to(ROOT) else str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path).resolve()
    if not (path.is_relative_to(HERE) or path.is_relative_to(CACHE)):
        raise ValueError('Writes must remain in the new experiment tree')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    temporary.write_text(json.dumps(serializable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensor_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_digest(model):
    value = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value.update(key.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def observed_metrics(prediction, target, mask, std=None):
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError('Prediction, target and original observation mask must align')
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError('Non-finite prediction or observed target')
    axes = tuple(range(prediction.ndim - 1))
    count = mask.sum(axes)
    valid = count > 0
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    cmse = np.divide(np.square(error).sum(axes), count, out=np.full(count.shape, np.nan), where=valid)
    cmae = np.divide(np.abs(error).sum(axes), count, out=np.full(count.shape, np.nan), where=valid)
    result = {'mse': float(cmse[valid].mean()) if valid.any() else None,
              'mae': float(cmae[valid].mean()) if valid.any() else None,
              'channel_mse': [float(v) if np.isfinite(v) else None for v in cmse],
              'channel_mae': [float(v) if np.isfinite(v) else None for v in cmae],
              'observed_counts': count, 'unobserved_channels': np.flatnonzero(~valid)}
    if std is not None:
        result['physical_channel_mae'] = [float(a * b) if np.isfinite(a) else None
                                           for a, b in zip(cmae, std)]
    if prediction.ndim == 3:
        result['origin_count'] = len(prediction)
        result['horizon_mse'] = []
        result['horizon_mae'] = []
        for step in range(prediction.shape[1]):
            part = observed_metrics(prediction[:, step], target[:, step], mask[:, step])
            result['horizon_mse'].append(part['mse'])
            result['horizon_mae'].append(part['mae'])
    return result


def masked_loss(prediction, target, mask):
    error = torch.where(mask, prediction - target, torch.zeros_like(prediction))
    return error.square().sum() / mask.sum().clamp_min(1)


class BudgetExceeded(RuntimeError):
    pass


class Resources:
    def __init__(self):
        self.proc = psutil.Process()
        self.path = HERE / 'resources' / f'{os.getpid()}_{time.time_ns()}.json'
        self.limits = read_json(HERE / 'RESOURCE_BUDGET.json')
        self.value = {'pid': os.getpid(), 'created': self.proc.create_time(),
                      'gpu_seconds': 0., 'cpu_seconds': time.process_time(), 'stages': [],
                      'peak_rss_bytes': 0, 'peak_cuda_allocated_bytes': 0,
                      'peak_cuda_reserved_bytes': 0, 'F_forward_calls': 0,
                      'F_backward_calls': 0, 'storage_bytes': 0}
        self.active_start = None
        self.active_gpu = False
        self.last_storage = 0.

    def check(self, **progress):
        now = time.time()
        self.value['cpu_seconds'] = time.process_time()
        self.value['peak_rss_bytes'] = max(self.value['peak_rss_bytes'], self.proc.memory_info().rss)
        if torch.cuda.is_initialized():
            self.value['peak_cuda_allocated_bytes'] = max(self.value['peak_cuda_allocated_bytes'], torch.cuda.max_memory_allocated())
            self.value['peak_cuda_reserved_bytes'] = max(self.value['peak_cuda_reserved_bytes'], torch.cuda.max_memory_reserved())
        gpu = self.limits['previous_gpu_seconds'] + self.value['gpu_seconds']
        cpu = self.limits['previous_cpu_seconds'] + self.value['cpu_seconds']
        for path in (HERE / 'resources').glob('*.json'):
            if path == self.path:
                continue
            other = read_json(path)
            gpu += other['gpu_seconds']
            cpu += other['cpu_seconds']
        if self.active_gpu:
            gpu += time.perf_counter() - self.active_start
        if now - self.limits['campaign_start_unix'] > self.limits['wall_seconds']:
            raise BudgetExceeded('cumulative wall-time cap')
        if gpu > self.limits['gpu_seconds'] or cpu > self.limits['cpu_seconds']:
            raise BudgetExceeded('cumulative GPU/CPU cap')
        if self.value['peak_rss_bytes'] > self.limits['ram_bytes'] or self.value['peak_cuda_allocated_bytes'] > self.limits['cuda_allocated_bytes']:
            raise BudgetExceeded('process RAM/allocated VRAM cap')
        if now - self.last_storage > 60:
            self.value['storage_bytes'] = sum(p.stat().st_size for folder in (HERE, CACHE) if folder.exists() for p in folder.rglob('*') if p.is_file())
            self.last_storage = now
            if self.value['storage_bytes'] + self.limits['previous_storage_bytes'] > self.limits['storage_bytes']:
                raise BudgetExceeded('cumulative new artifact storage cap')
        if progress:
            write_json(HERE / 'progress.json', {'updated_unix': now, 'pid': os.getpid(),
                                                'cumulative_gpu_seconds': gpu,
                                                'cumulative_cpu_seconds': cpu, **progress})

    @contextmanager
    def stage(self, name, gpu=False, **fields):
        if self.active_start is not None:
            raise RuntimeError('Resource stages must not nest')
        self.active_start, self.active_gpu = time.perf_counter(), gpu
        started = time.time()
        try:
            self.check(stage=name, **fields)
            yield
        finally:
            if gpu and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - self.active_start
            self.value['gpu_seconds'] += elapsed if gpu else 0
            self.value['stages'].append({'name': name, 'gpu': gpu, 'wall_seconds': elapsed,
                                         'started_unix': started, 'ended_unix': time.time(), **fields})
            self.active_start = None
            self.active_gpu = False
            self.flush()

    def flush(self):
        self.value['cpu_seconds'] = time.process_time()
        write_json(self.path, self.value)
