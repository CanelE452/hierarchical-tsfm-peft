"""V7-only paths, reservations and guarded data access; no model import here."""
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
CACHE = ROOT / '.cache/tsfm_peft_practical_controls_v7_20260928'
V1 = ROOT / 'research/tsfm_peft_development_20260926'
V2 = ROOT / 'research/tsfm_peft_followup_v2_20260926'
V3 = ROOT / 'research/tsfm_peft_backbone_value_v3_20260927'
V6 = ROOT / 'research/tsfm_peft_level_confirmation_v6_20260928'
V6_CACHE = ROOT / '.cache/tsfm_peft_level_confirmation_v6_20260928'
LIMITS = {'real_fit': 20, 'synthetic_sessions': 6, 'synthetic_steps': 10,
          'gpu_seconds': 14400, 'cpu_check_seconds': 900, 'storage_bytes': 5 * 1024**3,
          'download_bytes': 0, 'conflict_wait_per_incident_seconds': 1800,
          'conflict_wait_total_seconds': 3600}


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
    paths = list(HERE.glob('*.py')) + list(HERE.glob('data_contract*.json')) + [HERE / 'PLAN.md', HERE / 'protocol.json']
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths if p.exists()}


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
        raise RuntimeError('V7_STORAGE_LIMIT')
    return used


@contextlib.contextmanager
def ledger_transaction():
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = CACHE / 'ledger.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(str(os.getpid()))
    try:
        path = HERE / 'ledger.json'
        ledger = json.loads(path.read_text(encoding='utf-8'))
        yield ledger
        save_json(path, ledger)
    finally:
        lock.unlink()


def elapsed(record):
    return time.time() - record['started_utc'] if record['status'] == 'running' else record.get('elapsed_s', 0)


def budget_snapshot():
    ledger = json.loads((HERE / 'ledger.json').read_text(encoding='utf-8'))
    jobs = ledger['jobs']
    return {'real_fit_attempts': sum(r['category'] == 'real_fit' for r in jobs),
            'synthetic_sessions': sum(bool(r.get('synthetic')) for r in jobs),
            'gpu_seconds': sum(elapsed(r) for r in jobs if r['category'] == 'gpu'),
            'cpu_check_seconds': sum(elapsed(r) for r in jobs if r['category'] == 'cpu_check'),
            'data_prepare_seconds': sum(elapsed(r) for r in jobs if r['category'] == 'data_prepare'),
            'cpu_analysis_seconds': sum(elapsed(r) for r in jobs if r['category'] == 'cpu_analysis'),
            'active_jobs': [{k: r.get(k) for k in ('id', 'label', 'pid', 'category', 'progress')}
                            for r in jobs if r['status'] == 'running'],
            'storage_bytes': storage_bytes(), 'limits': LIMITS}


class Job:
    def __init__(self, label, category='gpu', reserve_seconds=60, synthetic=False, metadata=None):
        if category not in ('gpu', 'cpu_check', 'cpu_analysis', 'real_fit', 'data_prepare'):
            raise ValueError(category)
        if synthetic and category != 'cpu_check':
            raise ValueError('Synthetic optimizer sessions use CPU check category')
        self.label, self.category, self.reserve = label, category, reserve_seconds
        self.synthetic, self.metadata = synthetic, metadata or {}

    def __enter__(self):
        require_storage()
        if self.category == 'real_fit' and self.metadata.get('dataset') not in ('robin', 'jena'):
            raise ValueError('Every real fit must identify its approved dataset before reservation')
        provenance_path = HERE / 'provenance.json'
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        if digest(HERE / 'PLAN.md') != provenance['plan_sha256']:
            raise RuntimeError('Approved PLAN changed')
        if digest(HERE / 'protocol.json') != provenance['protocol_sha256']:
            raise RuntimeError('Precommitted protocol changed')
        if digest(HERE / 'APPROVAL.txt') != provenance['approval_sha256']:
            raise RuntimeError('Approval text changed')
        if self.category == 'real_fit' and (HERE / f'test_exposure_{self.metadata.get("dataset", "unknown")}.json').exists():
            raise RuntimeError('No fitting after TEST exposure')
        if self.category == 'gpu':
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                                    capture_output=True, text=True, check=True)
            others = [line for line in result.stdout.splitlines() if 'python' in line.lower()
                      and line.split(',')[0].strip() != str(os.getpid())]
            if others:
                raise RuntimeError(f'Other GPU Python process; do not terminate it: {others}')
        with ledger_transaction() as ledger:
            jobs = ledger['jobs']
            if any(r['label'] == self.label for r in jobs):
                raise RuntimeError('Every started retry requires a new ledger label')
            if self.category in ('gpu', 'real_fit') and any(
                    r['category'] == self.category and r['status'] == 'running' for r in jobs):
                raise RuntimeError('Unclosed active job: inspect actual PID, never duplicate execution')
            if self.category == 'real_fit' and sum(r['category'] == 'real_fit' for r in jobs) >= LIMITS['real_fit']:
                raise RuntimeError('REAL_FIT_LIMIT')
            if self.synthetic and sum(bool(r.get('synthetic')) for r in jobs) >= LIMITS['synthetic_sessions']:
                raise RuntimeError('SYNTHETIC_SESSION_LIMIT')
            cap = {'gpu': LIMITS['gpu_seconds'], 'cpu_check': LIMITS['cpu_check_seconds']}.get(self.category)
            used = sum(elapsed(r) for r in jobs if r['category'] == self.category)
            if cap and used + self.reserve > cap:
                raise RuntimeError(f'Insufficient {self.category} reservation: {used}+{self.reserve}>{cap}')
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
            record = ledger['jobs'][self.id]
            record.update(elapsed_s=time.perf_counter() - self.started, heartbeat_utc=time.time(), progress=progress)
            if updates is not None:
                if self.synthetic and updates > LIMITS['synthetic_steps']:
                    raise RuntimeError('SYNTHETIC_STEP_LIMIT')
                record['updates'] = updates
        self.check_limits()

    def __exit__(self, kind, exc, tb):
        with ledger_transaction() as ledger:
            record = ledger['jobs'][self.id]
            record.update(status='failed' if kind else 'complete', elapsed_s=time.perf_counter() - self.started,
                          ended_utc=time.time())
            if exc:
                record['error'] = ''.join(traceback.format_exception(kind, exc, tb))


def load_data(dataset, include_test=False):
    import numpy as np
    if dataset not in ('robin', 'jena'):
        raise ValueError('Only the two approved datasets are allowed')
    contract = json.loads((HERE / f'data_contract_{dataset}.json').read_text(encoding='utf-8'))
    if include_test:
        seal_path = HERE / f'evaluation_seal_{dataset}.json'
        exposure_path = HERE / f'test_exposure_{dataset}.json'
        if not seal_path.exists() or not exposure_path.exists():
            raise RuntimeError('TEST requires dataset-specific selection seal and exposure record')
        exposure = json.loads(exposure_path.read_text(encoding='utf-8'))
        if exposure.get('seal_sha256') != digest(seal_path):
            raise RuntimeError('TEST exposure/selection seal mismatch')
    entry = contract['test' if include_test else 'trainval']
    path = Path(entry['path'])
    if not path.is_absolute():
        path = ROOT / path
    if digest(path) != entry['sha256']:
        raise RuntimeError('Sealed data content changed')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data['_path'] = path
    data['_dataset'] = dataset
    data['_phase_period'] = 24 if dataset == 'robin' else 144
    for name in ('train_start', 'train_end'):
        if name in data:
            data[name] = int(data[name])
    expected = 17 if dataset == 'robin' else 21
    if data['x'].shape[1] != expected:
        raise ValueError('Approved channel count changed')
    return data


def tensors(data, origins, device='cuda'):
    import numpy as np
    import torch
    x = np.stack([data['x'][o - 512:o] for o in origins])
    y = np.stack([data['x'][o:o + 48] for o in origins])
    mask = np.stack([data['finite'][o:o + 48] for o in origins])
    return [torch.as_tensor(a, device=device) for a in (x, y, mask)]


def phase_sample(origins, seed, epoch, n=512, period=24):
    import numpy as np
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    selected = []
    for phase in range(period):
        group = origins[origins % period == phase]
        count = n // period + int(phase < n % period)
        if len(group) < count:
            raise ValueError('TRAIN phase insufficient for approved sampling without replacement')
        selected.extend(rng.choice(group, count, replace=False))
    return rng.permutation(selected)


def score_arrays(pred, target, mask):
    import numpy as np
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError('Prediction/target/mask axes mismatch')
    count = mask.sum(axis=(0, 1))
    if not np.all(count > 0) or not np.isfinite(pred).all():
        raise ValueError('Undefined channel score or nonfinite forecast; do not silently exclude channels')
    error = pred.astype(np.float64) - target
    squared = np.where(mask, error**2, 0.0)
    absolute = np.where(mask, np.abs(error), 0.0)
    sums, abs_sums = squared.sum(axis=(0, 1)), absolute.sum(axis=(0, 1))
    return {'mse': float(np.mean(sums / count)), 'mae': float(np.mean(abs_sums / count)),
            'channel_mse': (sums / count).tolist(), 'channel_mae': (abs_sums / count).tolist(),
            'squared_error_sums': sums.tolist(), 'absolute_error_sums': abs_sums.tolist(),
            'target_counts': count.tolist(), 'origins': len(pred)}
