"""Bounded matched learned E/D control; original research artifacts are read-only."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name
DATASETS = ('robin', 'peacock_education', 'jena')
SEEDS = (92601, 92602)
LIMITS = {'gpu_seconds': 10800, 'new_fits': 12, 'new_test_predictions': 6}
_lock = threading.RLock()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def public_paths(value):
    if isinstance(value, str):
        for prefix in (str(ROOT), ROOT.as_posix()):
            value = value.replace(prefix + '/', '').replace(prefix + chr(92), '')
        return value
    if isinstance(value, dict):
        return {k: public_paths(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_paths(v) for v in value]
    return value


def read_json(path):
    return json.loads(resolve(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = resolve(path).resolve()
    if not (path.is_relative_to(HERE) or path.is_relative_to(CACHE)):
        raise PermissionError('Writes limited to the new campaign directories')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(public_paths(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * 2**attempt)


def digest(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def artifact(path, expected=None):
    path = resolve(path).resolve()
    actual = digest(path)
    if expected is not None and actual != expected:
        raise RuntimeError('Artifact changed: ' + str(path))
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': actual, 'bytes': path.stat().st_size}


def array_hash(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def state_hash(module):
    import numpy as np
    h = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(np.asarray(value).tobytes())
    return h.hexdigest()


def import_file(path, name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, resolve(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def dataset_contract(dataset):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    return read_json(HERE / ('data_contract_' + dataset + '.json'))


def load_data(dataset, include_test=False):
    import numpy as np
    contract = dataset_contract(dataset)
    entry = contract['test' if include_test else 'trainval']
    artifact(entry['path'], entry['sha256'])
    with np.load(resolve(entry['path']), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if data['columns'].tolist() != contract['columns'] or data['basis'].shape != (contract['c'], contract['k']):
        raise RuntimeError('Original data channel/basis contract changed')
    data.update(_path=entry['path'], _C=contract['c'], _K=contract['k'], _dataset=dataset)
    return data


def inputs(data, origins):
    import numpy as np
    import torch
    return torch.from_numpy(np.stack([data['x'][int(o)-512:int(o)] for o in origins]).astype(np.float32))


def targets(data, origins):
    import numpy as np
    return (np.stack([data['x'][int(o):int(o)+48] for o in origins]),
            np.stack([data['finite'][int(o):int(o)+48] for o in origins]))


def score_arrays(prediction, target, mask):
    import numpy as np
    mask = mask.astype(bool)
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('Prediction/target/mask shape differs')
    error = np.where(mask, prediction.astype(np.float64)-target.astype(np.float64), 0.)
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError('Nonfinite forecast/observed target')
    count = mask.sum(axis=(0, 1)).astype(np.float64)
    if np.any(count<=0):
        raise ValueError('Every fixed channel needs observed targets')
    sums = {'squared': np.square(error).sum(axis=1).sum(axis=0),
            'absolute': np.abs(error).sum(axis=1).sum(axis=0), 'signed': error.sum(axis=1).sum(axis=0)}
    result = {metric: float(np.mean(sums[key]/count)) for metric, key in
              (('mse','squared'), ('mae','absolute'), ('signed_mean_error','signed'))}
    result.update(channel_mse=(sums['squared']/count).tolist(), channel_mae=(sums['absolute']/count).tolist(),
                  squared_error_sums=sums['squared'].tolist(), absolute_error_sums=sums['absolute'].tolist(),
                  signed_error_sums=sums['signed'].tolist(), target_counts=count.astype(np.int64).tolist(),
                  target_count=int(count.sum()), origins=len(prediction))
    return result


def original_levels(dataset):
    return read_json(HERE / 'parent_reuse_manifest.json')['units'][dataset]['levels']


def load_original(row, device='cuda', deploy=True):
    parent = ROOT / 'research/level_bolt_backbone_scaling_v1_20261006'
    # The reused baseline loader binds its own runtime and source hashes literally.
    import_file(parent / 'runtime.py', 'runtime')
    loader = import_file(parent / 'baselines.py', 'matched_v18_original_baselines')
    return loader.load_baseline(row, device=device, deploy=deploy).module


def configure():
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def environment():
    import torch
    return {'python': sys.version, 'executable': Path(sys.executable).name,
            'versions': {key: importlib.metadata.version(key) for key in ('torch', 'numpy', 'chronos-forecasting', 'transformers', 'huggingface-hub')},
            'gpu': torch.cuda.get_device_name(0), 'gpu_bytes': torch.cuda.get_device_properties(0).total_memory,
            'dtype': 'float32', 'tf32': False, 'cpu_threads': 4}


def budget_snapshot():
    ledger = read_json(HERE / 'ledger.json')
    now = time.time()
    return {'gpu_seconds': sum(now-j['started_utc'] if j['status']=='running' else j['elapsed_s'] for j in ledger['jobs'] if j['category']=='gpu'),
            'new_fits': len(ledger['fits']), 'new_test_predictions': len(ledger['test_predictions']),
            'active': [j for j in ledger['jobs'] if j['status']=='running'], 'limits': LIMITS}


def begin_fit(run_id, metadata=None):
    with _lock:
        ledger = read_json(HERE / 'ledger.json')
        if any(row['id']==run_id for row in ledger['fits']):
            raise RuntimeError('Fit already attempted; preserve its result and inspect before resuming: ' + run_id)
        if len(ledger['fits']) >= LIMITS['new_fits']:
            raise RuntimeError('STOP_RESOURCE: twelve-fit cap')
        ledger['fits'].append({'id': run_id, 'status': 'running', 'started_utc': time.time(), **(metadata or {})})
        save_json(HERE / 'ledger.json', ledger)


def finish_fit(run_id, metadata=None, status='complete'):
    with _lock:
        ledger = read_json(HERE / 'ledger.json')
        row = next(row for row in ledger['fits'] if row['id']==run_id)
        row.update(status=status, ended_utc=time.time(), **(metadata or {}))
        save_json(HERE / 'ledger.json', ledger)


def register_test(run_id, metadata=None):
    check_selection_seal()
    with _lock:
        ledger = read_json(HERE / 'ledger.json')
        if any(row['id']==run_id for row in ledger['test_predictions']):
            raise RuntimeError('TEST already attempted; reuse completed predictions: ' + run_id)
        if len(ledger['test_predictions']) >= LIMITS['new_test_predictions']:
            raise RuntimeError('STOP_RESOURCE: six-TEST cap')
        ledger['test_predictions'].append({'id': run_id, 'status': 'running', 'started_utc': time.time(), **(metadata or {})})
        save_json(HERE / 'ledger.json', ledger)


def finish_test(run_id, metadata=None, status='complete'):
    with _lock:
        ledger = read_json(HERE / 'ledger.json')
        row = next(row for row in ledger['test_predictions'] if row['id']==run_id)
        row.update(status=status, ended_utc=time.time(), **(metadata or {}))
        save_json(HERE / 'ledger.json', ledger)


def check_source_seal():
    seal = read_json(HERE / 'source_seal.json')
    for row in seal['artifacts']:
        artifact(row['path'], row['sha256'])
    return seal


def check_selection_seal():
    check_source_seal()
    seal = read_json(HERE / 'selection_seal.json')
    for row in seal['artifacts']:
        artifact(row['path'], row['sha256'])
    if not seal['all_trainval_selection_complete_before_test']:
        raise RuntimeError('Joint TRAIN/VAL selection is not sealed')
    return seal


class Job:
    def __init__(self, category, label, reserve_s=0, metadata=None):
        if category not in ('gpu', 'cpu'):
            raise ValueError(category)
        self.category, self.label, self.reserve, self.metadata = category, label, reserve_s, metadata or {}
        self.stop = threading.Event()

    def __enter__(self):
        with _lock:
            ledger = read_json(HERE / 'ledger.json')
            budget = budget_snapshot()
            if budget['active']:
                raise RuntimeError('Campaign job is running; inspect PID and do not duplicate')
            if self.category=='gpu' and budget['gpu_seconds']+self.reserve >= LIMITS['gpu_seconds']:
                raise RuntimeError('STOP_RESOURCE: insufficient cumulative GPU budget')
            self.id, self.start = len(ledger['jobs']), time.perf_counter()
            ledger['jobs'].append({'id': self.id, 'category': self.category, 'label': self.label, 'pid': os.getpid(),
                                  'status': 'running', 'started_utc': time.time(), 'elapsed_s': 0, 'metadata': self.metadata})
            save_json(HERE / 'ledger.json', ledger)
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()
        return self

    def monitor(self):
        while not self.stop.wait(2):
            with _lock:
                if budget_snapshot()['gpu_seconds'] >= LIMITS['gpu_seconds']:
                    ledger = read_json(HERE / 'ledger.json')
                    ledger['jobs'][self.id].update(status='STOP_RESOURCE', elapsed_s=time.perf_counter()-self.start,
                                                 error='GPU cumulative hard cap reached', ended_utc=time.time())
                    save_json(HERE / 'ledger.json', ledger)
                    os._exit(75)

    def check_limits(self):
        if budget_snapshot()['gpu_seconds'] >= LIMITS['gpu_seconds']-5:
            raise RuntimeError('STOP_RESOURCE: GPU cumulative hard cap imminent')

    def heartbeat(self, progress=None, **extra):
        with _lock:
            ledger = read_json(HERE / 'ledger.json')
            ledger['jobs'][self.id].update(elapsed_s=time.perf_counter()-self.start, heartbeat_utc=time.time(), **extra)
            if progress is not None:
                ledger['jobs'][self.id]['progress'] = progress
            save_json(HERE / 'ledger.json', ledger)
        self.check_limits()

    def __exit__(self, kind, exc, tb):
        self.stop.set()
        self.thread.join(timeout=3)
        with _lock:
            ledger = read_json(HERE / 'ledger.json')
            row = ledger['jobs'][self.id]
            row.update(status='failed' if kind else 'complete', elapsed_s=time.perf_counter()-self.start, ended_utc=time.time())
            if exc:
                row['error'] = ''.join(traceback.format_exception(kind, exc, tb))
            save_json(HERE / 'ledger.json', ledger)
