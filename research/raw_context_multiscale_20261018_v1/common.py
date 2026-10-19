"""Local experiment roots and the inherited, pre-T data contract."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from research.b2_revin_20261018_v1.contracts import (
    atomic_json, binding, fingerprint, read, resolve_binding, sealed_json, sha,
)


NAME = 'raw_context_multiscale_20261018_v1'
REPO = Path(__file__).resolve().parents[2]
SEEDS = (92601, 92602)
ATOL, RTOL = 1e-5, 1e-4
INITIAL_EPOCHS = (.25, 1., 3.)
EXTENDED_EPOCHS = (.25, 1., 3., 4., 5., 6.)
PEMS_EXTENDED_EPOCHS = (*EXTENDED_EPOCHS, 9., 12.)
RECIPE = dict(lr=.001, effective_batch=16, microbatch=4,
              F_row_microbatch=128, checkpoint_inputs=False)
EXPECTED = dict(jena=dict(C=21, K=12, L=512, H=48, N=20965, V_origins=43),
                pems32=dict(C=32, K=8, L=512, H=12, N=17856, V_origins=148))


def paths(repo=REPO):
    repo = Path(repo).resolve()
    return dict(repo=repo, cache=repo/'.cache'/NAME, report=repo/'reports'/NAME)


def context(asset_root, *, repo=REPO, device='cuda', deadline=None):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    ctx = dict(paths(repo), asset=Path(asset_root).resolve(), device=device,
               deadline=1e12 if deadline is None else float(deadline))
    from research.adapter_diagnostics_20261017_v2.existing_jobs import _modules
    ctx['modules'] = _modules(ctx)
    return ctx


def load_task(ctx, dataset):
    if dataset not in EXPECTED:
        raise ValueError('Only the registered Jena/PeMS32 cohorts are admitted')
    from .data import bounded_task
    task, access = bounded_task(ctx, dataset)
    ctx.setdefault('data_access', {})[dataset] = access
    expected = read(ctx['repo']/'reports/b2_revin_20261018_v1/DATA_CONTRACT.json')['tasks'][dataset]
    shape = (expected['materialized_prefix_rows'], expected['C'])
    if task['physical'].shape != shape or task['observed'].shape != shape:
        raise ValueError('The admitted physical prefix or original mask changed')
    if task['observed'].dtype != np.bool_ or list(task['columns']) != expected['columns']:
        raise ValueError('Original mask dtype or channel order changed')
    if task['data_sha256'] != expected['data_sha256']:
        raise ValueError('The inherited data identity changed')
    task = dict(task, dataset=dataset, K=EXPECTED[dataset]['K'],
                split_ranges={key: tuple(value) for key, value in expected['split_ranges'].items()})
    for key in ('L', 'H'):
        if int(task[key]) != EXPECTED[dataset][key]:
            raise ValueError('Original temporal contract changed: '+key)
    origins = ctx['modules']['data'].origins(task['split_ranges']['V_select'], task['L'], task['H'], task['H'])
    if len(origins) != EXPECTED[dataset]['V_origins']:
        raise ValueError('The original V origin cohort changed')
    return task


def model_sources(repo=REPO):
    repo = Path(repo)
    names = ('common.py', 'data.py', 'models.py', 'training.py')
    result = {(Path('research')/NAME/name).as_posix(): sha(repo/'research'/NAME/name) for name in names}
    for relative in ('priority_b2_validation_20261016_v1/safe_jena_prefix.py',
                     'priority_core_followup_20261016_v1/pems_fixed_head_runner.py',
                     'priority_pems_subset_20261016_v1/subset_minimum_runner.py'):
        path = Path('research/informs_broad_validation_20261015_v1')/relative
        result[path.as_posix()] = sha(repo/path)
    return result


def array_identity(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def event(ctx, stage, status, **values):
    directory = Path(ctx['cache'])
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(stage=stage, status=status, time_unix=time.time(), pid=os.getpid(), **values)
    with (directory/'EVENTS.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
    return record


def statistics(prediction, target, observed):
    """Independent FP64 sufficient statistics at every horizon and channel."""
    p, y, mask = np.asarray(prediction), np.asarray(target), np.asarray(observed)
    if p.ndim != 3 or p.shape != y.shape or mask.shape != y.shape or mask.dtype != np.bool_:
        raise ValueError('Aligned original-channel arrays and bool masks required')
    if not np.isfinite(p).all() or not np.isfinite(y[mask]).all():
        raise ValueError('Observed predictions and targets must be finite')
    delta = np.where(mask, p.astype(np.float64)-y.astype(np.float64), 0.)
    sse = np.square(delta).sum(axis=0, dtype=np.float64)
    sae = np.abs(delta).sum(axis=0, dtype=np.float64)
    counts = mask.sum(axis=0, dtype=np.int64)
    ccounts = counts.sum(axis=0)
    active = ccounts > 0
    if not active.any():
        raise ValueError('No observed target for channel macro metrics')
    mse = math.fsum((sse.sum(axis=0)[active]/ccounts[active]).tolist())/int(active.sum())
    mae = math.fsum((sae.sum(axis=0)[active]/ccounts[active]).tolist())/int(active.sum())
    return dict(mse=mse, mae=mae, n_observed=int(ccounts.sum()), n_origins=len(p),
                horizon=p.shape[1], channels=p.shape[2], channel_counts=ccounts.tolist(),
                horizon_channel_sse=sse.tolist(), horizon_channel_sae=sae.tolist(),
                horizon_channel_count=counts.tolist(), accumulation='float64',
                aggregation='equal-channel observed macro')


def save_prediction(path, *, prediction, target, observed, origins, task, metadata):
    path = Path(path)
    arrays = dict(pred_normalized=np.asarray(prediction, np.float32),
                  target_normalized=np.asarray(target, np.float32),
                  observed=np.asarray(observed), origins=np.asarray(origins, np.int64),
                  channel_ids=np.asarray(task['columns']))
    if arrays['observed'].dtype != np.bool_:
        raise ValueError('Do not coerce an altered target mask')
    score = statistics(arrays['pred_normalized'], arrays['target_normalized'], arrays['observed'])
    identities = {key: array_identity(value) for key, value in arrays.items()}
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (any(not np.array_equal(saved[key], value) for key, value in arrays.items())
                    or json.loads(str(saved['metadata_json'].item())) != metadata):
                raise ValueError('Preserve changed prediction attempts; never overwrite a sealed result')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    return dict(score=score, arrays=identities, prediction_sha256=sha(path), metadata=metadata)
