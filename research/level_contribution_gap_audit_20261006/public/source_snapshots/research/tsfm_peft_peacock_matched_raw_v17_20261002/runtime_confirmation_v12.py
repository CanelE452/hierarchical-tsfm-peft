"""TEST 노출 후 보충 대조 (v12 확인 실험 아님).

Administrative isolation for an unchanged copy of the v12 fitting algorithm.
All writes are constrained to this new directory; prior ledgers are not used.
"""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'research/tsfm_peft_a_confirmation_v12_20261001'
CACHE = HERE / 'local_artifacts'
SEEDS = (92601, 92602)
SCOPE = 'TEST 노출 후 보충 대조 (v12 확인 실험 아님)'
LIMITS = {'attempts': 4, 'training_seconds': 2400, 'evaluation_seconds': 1800}


def read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(value, dict):
        value.pop('_v17_scope', None)
    return value


def save_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE):
        raise PermissionError('Output outside the approved v17 folder')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value, _v17_scope=SCOPE) if isinstance(value, dict) else value
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def artifact(path, expected=None):
    path = Path(path).resolve()
    actual = digest(path)
    if expected is not None and actual != expected:
        raise RuntimeError('PREMISE FAILED: hash mismatch: ' + str(path))
    return {'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size}


def import_file(path, name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


numeric = import_file(ROOT / 'research/tsfm_peft_internal_vs_subspace_v11_20260930/runtime_v11.py', 'v17_original_numerics')
tensors, batches = numeric.tensors, numeric.batches
phase_sample, masked_macro_loss, score_arrays = numeric.phase_sample, numeric.masked_macro_loss, numeric.score_arrays


def protocol():
    return read_json(HERE / 'protocol.json')


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def load_data(dataset, include_test=False):
    if dataset != 'peacock_education':
        raise ValueError('Only Peacock is authorized')
    contract_path = OLD / f'data_contract_confirmation_{dataset}.json'
    contract = read_json(contract_path)
    manifest = read_json(OLD / 'confirmation_data_manifest.json')
    artifact(contract_path, manifest[dataset]['contract_sha256'])
    if include_test:
        artifact(OLD / 'confirmation_selection_seal.json', read_json(OLD / 'confirmation_test_exposure.json')['seal_sha256'])
    unit = protocol()['units'][dataset]
    entry = contract['test' if include_test else 'trainval']
    path = Path(entry['path'])
    artifact(path, entry['sha256'])
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if data['x'].shape[1] != 13 or data['basis'].shape != (13, 4):
        raise ValueError('Peacock dimensions changed')
    if data['finite'].shape != data['x'].shape or data['columns'].tolist() != unit['columns']:
        raise ValueError('Mask or channel order mismatch')
    data.update(_path=path, _dataset=dataset, _phase_period=24, _C=13, _K=4)
    return data


def sources_for(*args):
    raise RuntimeError('No A or parent adaptation is authorized in v17')


def environment_receipt():
    return {'python': sys.version, 'executable': sys.executable, 'pid': os.getpid(),
            'versions': {n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'peft', 'chronos-forecasting', 'transformers')},
            'scope': SCOPE}


def budget_snapshot():
    ledger = read_json(HERE / 'ledger.json')
    now = time.time()
    elapsed = lambda j: now - j['started_utc'] if j['status'] == 'running' else j['elapsed_s']
    return {'attempts': sum(j['category'] == 'neural_fit' for j in ledger['jobs']),
            'training_seconds': sum(elapsed(j) for j in ledger['jobs'] if j['category'] == 'gpu'),
            'evaluation_seconds': ledger['preflight_evaluation_seconds'] + sum(elapsed(j) for j in ledger['jobs'] if j['category'] == 'evaluation'),
            'limits': LIMITS}


class Job:
    def __init__(self, category, label, reserve_s=0, metadata=None):
        if category not in ('gpu', 'neural_fit', 'evaluation'):
            raise ValueError(category)
        self.category, self.label, self.metadata = category, label, metadata or {}

    def __enter__(self):
        ledger = read_json(HERE / 'ledger.json')
        if self.category in ('gpu', 'neural_fit'):
            if (HERE / 'selection_seal.json').exists() or (HERE / 'confirmation_test_exposure.json').exists():
                raise RuntimeError('No fitting after the v17 seal')
            if not read_json(HERE / 'implementation_checks.json')['pass']:
                raise RuntimeError('Implementation premises have not passed')
        if self.category == 'neural_fit':
            if budget_snapshot()['attempts'] >= 4:
                raise RuntimeError('Four-attempt hard cap')
            if any(j['status'] == 'running' and j['category'] == 'neural_fit' for j in ledger['jobs']):
                raise RuntimeError('A fit is already running')
            assert self.metadata['dataset'] == 'peacock_education' and self.metadata['family'] == 'level'
            assert self.metadata['role'] == 'matched_raw' and self.metadata['lr'] == .001
            assert self.metadata['seed'] in SEEDS
        elif any(j['status'] == 'running' for j in ledger['jobs']):
            raise RuntimeError('Another job is recorded as running; inspect its actual PID')
        self.id, self.started = len(ledger['jobs']), time.perf_counter()
        ledger['jobs'].append({'id': self.id, 'category': self.category, 'label': self.label,
             'metadata': self.metadata, 'status': 'running', 'pid': os.getpid(),
             'started_utc': time.time(), 'elapsed_s': 0, 'updates': 0})
        save_json(HERE / 'ledger.json', ledger)
        self.check_limits()
        return self

    def check_limits(self):
        budget = budget_snapshot()
        # Leave a small margin for the next batch/JSON write; caps never increase.
        if budget['training_seconds'] >= 2395 or budget['evaluation_seconds'] >= 1795:
            raise RuntimeError('Approved elapsed-time cap reached')

    def heartbeat(self, progress, updates=None):
        ledger = read_json(HERE / 'ledger.json')
        row = ledger['jobs'][self.id]
        row.update(elapsed_s=time.perf_counter() - self.started, heartbeat_utc=time.time(), progress=progress)
        if updates is not None:
            row['updates'] = int(updates)
        save_json(HERE / 'ledger.json', ledger)
        self.check_limits()

    def __exit__(self, kind, exc, tb):
        ledger = read_json(HERE / 'ledger.json')
        row = ledger['jobs'][self.id]
        row.update(status='failed' if kind else 'complete', elapsed_s=time.perf_counter() - self.started,
                   ended_utc=time.time())
        if exc:
            row['error'] = str(exc)
        save_json(HERE / 'ledger.json', ledger)
