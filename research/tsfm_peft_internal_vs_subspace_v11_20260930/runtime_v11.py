"""V11 reservations, immutable provenance and guarded data access."""
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
CACHE = ROOT / '.cache/tsfm_peft_internal_vs_subspace_v11_20260930'
DATASETS = {'robin': (17, 5, 24), 'jena': (21, 6, 144), 'hog': (32, 8, 24)}
SEEDS = (92601, 92602)
LIMITS = {'real_fit': 60, 'neural_basic': 40, 'coefficient_basic': 12,
          'technical_reserve': 8, 'gpu_seconds': 28800, 'cpu_analysis_seconds': 7200,
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
    return sum(p.stat().st_size for base in (HERE, CACHE) if base.exists()
               for p in base.rglob('*') if p.is_file())


def require_storage(reserve=0):
    used = storage_bytes()
    if used + reserve > LIMITS['storage_bytes']:
        raise RuntimeError('V11_STORAGE_LIMIT')
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


class Job:
    def __init__(self, label, category='gpu', reserve_seconds=60, metadata=None, synthetic=False):
        if category not in TIME_CATEGORIES + FIT_CATEGORIES:
            raise ValueError(category)
        if synthetic and category != 'cpu_check':
            raise ValueError('Synthetic optimizer sessions belong to CPU checks')
        self.label, self.category, self.reserve = label, category, float(reserve_seconds)
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
            if (HERE / 'test_exposure.json').exists():
                raise RuntimeError('No fitting after v11 evaluation exposure')
            if self.metadata.get('dataset') not in DATASETS or not self.metadata.get('fit_id'):
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
                protocol = read_json(HERE / 'protocol.json')
                fit_ids = {row['id'] for row in protocol['neural_fits' if self.category == 'neural_fit' else 'coefficient_fits']}
                previous = [j for j in fits if j['metadata'].get('fit_id') == self.metadata['fit_id']]
                retry = bool(self.metadata.get('technical_retry'))
                if self.metadata['fit_id'] not in fit_ids and not retry:
                    raise RuntimeError('Unplanned fit requires an identified technical recovery')
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
                self._heartbeat_error = repr(error)
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


def sources_for(dataset, seed, require_direct=False):
    sources = dict(read_json(HERE / 'reuse_manifest.json')['sources'][dataset][str(seed)])
    selected_path = HERE / 'selected_direct.json'
    if selected_path.exists():
        selected = read_json(selected_path).get('selected', {}).get(dataset, {}).get('direct_nlinear')
        if selected:
            index = SEEDS.index(int(seed))
            sources['direct_checkpoint'] = selected['checkpoints'][index]
            sources['direct_checkpoint_sha256'] = selected['checkpoint_sha256'][index]
    direct_path = HERE / 'direct_sources.json'
    if direct_path.exists():
        extra = read_json(direct_path).get(dataset, {}).get(str(seed), {})
        sources.update(extra)
    if require_direct and not sources.get('direct_checkpoint'):
        raise RuntimeError(f'Direct NLinear has not been selected for {dataset}/{seed}')
    return sources


def load_data(dataset, include_test=False):
    import numpy as np
    if dataset not in DATASETS:
        raise ValueError('Only Robin, Jena, Hog are approved')
    if include_test:
        seal = read_json(HERE / 'selection_seal.json')
        exposure = read_json(HERE / 'test_exposure.json')
        for name in ('selected', 'gamma_selection', 'reuse_manifest', 'protocol'):
            if seal[name + '_sha256'] != digest(HERE / (name + '.json')):
                raise RuntimeError('Selection seal content changed: ' + name)
        if exposure['seal_sha256'] != digest(HERE / 'selection_seal.json'):
            raise RuntimeError('TEST exposure seal mismatch')
        if set(seal['datasets']) != set(DATASETS):
            raise RuntimeError('All three datasets must be selected before any evaluation')
    contract = read_json(HERE / f'data_contract_{dataset}.json')
    reuse = read_json(HERE / 'reuse_manifest.json')['datasets'][dataset]
    if digest(HERE / f'data_contract_{dataset}.json') != reuse['contract']['sha256']:
        raise RuntimeError('Pinned data contract changed')
    entry = contract['test' if include_test else 'trainval']
    path = resolve_path(entry['path'])
    if digest(path) != entry['sha256']:
        raise RuntimeError('Sealed data bytes changed')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    channels, latent, period = DATASETS[dataset]
    if data['x'].shape[1] != channels or data['basis'].shape != (channels, latent):
        raise ValueError('Approved channel/PCA dimensions changed')
    if data['finite'].shape != data['x'].shape:
        raise ValueError('Target mask shape mismatch')
    data.update(_path=path, _dataset=dataset, _phase_period=period)
    for name in ('train_start', 'train_end'):
        if name in data:
            data[name] = int(data[name])
    return data


def tensors(data, origins, device='cuda'):
    import numpy as np
    import torch
    x = np.stack([data['x'][int(o) - 512:int(o)] for o in origins])
    y = np.stack([data['x'][int(o):int(o) + 48] for o in origins])
    mask = np.stack([data['finite'][int(o):int(o) + 48] for o in origins])
    if x.shape[1] != 512 or y.shape[1] != 48:
        raise ValueError('Origin outside approved context/target bounds')
    return [torch.as_tensor(a, device=device) for a in (x, y, mask)]


def batches(data, origins, batch_size=4, device='cuda'):
    for start in range(0, len(origins), batch_size):
        group = origins[start:start + batch_size]
        yield (group, *tensors(data, group, device=device))


def phase_sample(origins, seed, epoch, n=512, period=24):
    import numpy as np
    origins = np.asarray(origins)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    selected = []
    for phase in range(period):
        group = origins[origins % period == phase]
        count = n // period + int(phase < n % period)
        if len(group) < count:
            raise ValueError('TRAIN phase insufficient for sampling without replacement')
        selected.extend(rng.choice(group, count, replace=False))
    return rng.permutation(selected)


def masked_macro_loss(prediction, target, mask):
    import torch
    mask = mask.bool()
    count = mask.sum((0, 1))
    valid = count > 0
    if not bool(valid.any()):
        raise ValueError('Batch has no observed target')
    error = torch.where(mask, prediction - target, torch.zeros_like(prediction))
    return (error.square().sum((0, 1))[valid] / count[valid]).mean()


def score_arrays(pred, target, mask):
    import numpy as np
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError('Prediction/target/mask axes mismatch')
    mask = mask.astype(bool)
    count = mask.sum(axis=(0, 1))
    if not np.all(count > 0) or not np.isfinite(pred).all() or not np.isfinite(target[mask]).all():
        raise ValueError('Undefined channel score or nonfinite observed output')
    error = np.where(mask, pred.astype(np.float64) - target, 0.)
    squared, absolute, signed = (error**2).sum((0, 1)), np.abs(error).sum((0, 1)), error.sum((0, 1))
    return {'mse': float(np.mean(squared / count)), 'mae': float(np.mean(absolute / count)),
            'signed_mean_error': float(np.mean(signed / count)),
            'channel_mse': (squared / count).tolist(), 'channel_mae': (absolute / count).tolist(),
            'channel_bias': (signed / count).tolist(), 'squared_error_sums': squared.tolist(),
            'absolute_error_sums': absolute.tolist(), 'signed_error_sums': signed.tolist(),
            'target_counts': count.tolist(), 'origins': len(pred)}
