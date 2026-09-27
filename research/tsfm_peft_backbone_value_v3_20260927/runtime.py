"""Shared accounting and unchanged v1 data/metric contract for approved v3 work."""
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache/tsfm_peft_backbone_value_v3_20260927'
V1 = ROOT / 'research/tsfm_peft_development_20260926'
V2 = ROOT / 'research/tsfm_peft_followup_v2_20260926'
LIMITS = {'real_fit': 16, 'synthetic_sessions': 6, 'synthetic_steps': 10,
          'gpu_seconds': 10800, 'cpu_check_seconds': 900, 'storage_bytes': 5 * 1024**3}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    for retry in range(6):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if retry == 5:
                raise
            time.sleep(0.05 * 2**retry)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_receipt():
    files = list(HERE.glob('*.py')) + [HERE / 'PLAN.md', HERE / 'protocol.json', V1 / 'data_contract.json']
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files if p.exists()}


def environment_receipt():
    return {'python': sys.version, 'executable': sys.executable, 'command': sys.argv,
            'cwd': str(Path.cwd()), 'versions': {n: importlib.metadata.version(n) for n in
             ('torch', 'numpy', 'peft', 'chronos-forecasting', 'transformers')}}


def storage_bytes():
    return sum(p.stat().st_size for base in (HERE, CACHE) if base.exists()
               for p in base.rglob('*') if p.is_file())


def require_storage(reserve=0):
    used = storage_bytes()
    if used + reserve > LIMITS['storage_bytes']:
        raise RuntimeError('V3_STORAGE_LIMIT')
    return used


@contextlib.contextmanager
def ledger_transaction():
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = CACHE / 'ledger.lock'
    with lock.open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()))
    try:
        path = HERE / 'ledger.json'
        ledger = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
            'limits': LIMITS, 'jobs': [], 'v2_violation_preserved': '22/20; not retrospectively approved'}
        yield ledger
        save_json(path, ledger)
    finally:
        lock.unlink()


def elapsed(record):
    return time.time() - record['started_utc'] if record['status'] == 'running' else record.get('elapsed_s', 0)


class Job:
    def __init__(self, label, category='gpu', reserve_seconds=60, synthetic=False, metadata=None):
        if category not in ('gpu', 'cpu_check', 'cpu_analysis', 'real_fit'):
            raise ValueError(category)
        self.label, self.category, self.reserve = label, category, reserve_seconds
        self.synthetic, self.metadata = synthetic, metadata or {}

    def __enter__(self):
        require_storage()
        if self.category == 'gpu':
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                                    capture_output=True, text=True, check=True)
            others = [line for line in result.stdout.splitlines() if 'python' in line.lower()
                      and line.split(',')[0].strip() != str(os.getpid())]
            if others:
                raise RuntimeError(f'Other GPU Python process: {others}; never terminate it')
        with ledger_transaction() as ledger:
            jobs = ledger['jobs']
            if any(r['label'] == self.label for r in jobs):
                raise RuntimeError('Use a new id for every started retry; never overwrite a ledger job')
            if self.category == 'gpu' and any(r['category'] == 'gpu' and r['status'] == 'running' for r in jobs):
                raise RuntimeError('An unclosed GPU job exists; inspect its actual process before recovery')
            if self.category == 'real_fit' and sum(r['category'] == 'real_fit' for r in jobs) >= LIMITS['real_fit']:
                raise RuntimeError('REAL_FIT_LIMIT')
            if self.synthetic and sum(r.get('synthetic', False) for r in jobs) >= LIMITS['synthetic_sessions']:
                raise RuntimeError('SYNTHETIC_SESSION_LIMIT')
            cap = {'gpu': LIMITS['gpu_seconds'], 'cpu_check': LIMITS['cpu_check_seconds']}.get(self.category)
            used = sum(elapsed(r) for r in jobs if r['category'] == self.category)
            if cap and used + self.reserve > cap:
                raise RuntimeError(f'Insufficient {self.category} time for reservation: {used}+{self.reserve}>{cap}')
            self.started = time.perf_counter()
            self.id = len(jobs)
            jobs.append({'id': self.id, 'label': self.label, 'category': self.category, 'synthetic': self.synthetic,
                         'status': 'running', 'reserved_utc': time.time(), 'started_utc': time.time(),
                         'reserved_seconds': self.reserve, 'elapsed_s': 0, 'pid': os.getpid(),
                         'updates': 0, 'metadata': self.metadata, 'source': source_receipt()})
        return self

    def check_limits(self):
        ledger = json.loads((HERE / 'ledger.json').read_text(encoding='utf-8'))
        for category, key in (('gpu', 'gpu_seconds'), ('cpu_check', 'cpu_check_seconds')):
            if sum(elapsed(r) for r in ledger['jobs'] if r['category'] == category) >= LIMITS[key]:
                raise RuntimeError(f'{category.upper()}_TOTAL_LIMIT')
        require_storage(1024 * 1024)

    def heartbeat(self, progress, updates=None):
        with ledger_transaction() as ledger:
            r = ledger['jobs'][self.id]
            r.update(elapsed_s=time.perf_counter() - self.started, heartbeat_utc=time.time(), progress=progress)
            if updates is not None:
                if self.synthetic and updates > LIMITS['synthetic_steps']:
                    raise RuntimeError('SYNTHETIC_STEP_LIMIT')
                r['updates'] = updates
        self.check_limits()

    def __exit__(self, kind, exc, tb):
        with ledger_transaction() as ledger:
            r = ledger['jobs'][self.id]
            r.update(status='failed' if kind else 'complete', elapsed_s=time.perf_counter() - self.started,
                     ended_utc=time.time())
            if exc:
                r['error'] = ''.join(traceback.format_exception(kind, exc, tb))


def load_data(dataset):
    import numpy as np
    contract = json.loads((V1 / 'data_contract.json').read_text(encoding='utf-8'))
    key = {'electricity': 'electricity_first32', 'bull': 'bdg2_bull_office'}[dataset]
    path = ROOT / contract['datasets'][key]['npz']
    protocol = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
    if digest(path) != protocol['data_sha256'][dataset]:
        raise ValueError('Approved data hash changed')
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    data['_path'], data['finite'] = path, data['targetmask']
    if dataset == 'electricity':
        data['train_start'], data['train_end'] = data['split_bounds'][:2]
    else:
        periods = contract['datasets'][key]['split_periods']['train']
        data['train_start'], data['train_end'] = [np.searchsorted(data['times'], np.datetime64(t, 'ns').astype(np.int64)) for t in periods]
    return data


def tensors(data, origins, device='cuda'):
    import numpy as np
    import torch
    x = np.stack([data['x'][o - 512:o] for o in origins])
    y = np.stack([data['x'][o:o + 48] for o in origins])
    mask = np.stack([data['finite'][o:o + 48] for o in origins])
    return [torch.as_tensor(a, device=device) for a in (x, y, mask)]


def phase_sample(origins, seed, epoch, n=512):
    import numpy as np
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    groups = [origins[origins % 24 == phase] for phase in range(24)]
    selected = []
    for phase, group in enumerate(groups):
        count = n // 24 + int(phase < n % 24)
        if not len(group):
            raise ValueError('Empty TRAIN phase')
        selected.extend(rng.choice(group, count, replace=len(group) < count))
    return rng.permutation(selected)


def score_arrays(pred, target, mask):
    import numpy as np
    n = mask.sum(axis=(0, 1))
    if not np.all(n > 0) or not np.isfinite(pred).all():
        raise ValueError('Invalid evaluation channel/forecast')
    squared = (pred.astype(np.float64) - target) ** 2 * mask
    absolute = np.abs(pred.astype(np.float64) - target) * mask
    return {'mse': float(np.mean(squared.sum(axis=(0, 1)) / n)),
            'mae': float(np.mean(absolute.sum(axis=(0, 1)) / n)),
            'channel_mse': (squared.sum(axis=(0, 1)) / n).tolist(), 'target_counts': n.tolist(), 'origins': len(pred)}
