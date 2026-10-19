"""New campaign journal, immutable registration and explicit resource clock."""
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from research.raw_context_multiscale_20261018_v1 import common as inherited

NAME = 'incremental_latent_adaptation_20261019_v1'
REPO = Path(__file__).resolve().parents[2]
REPORT = REPO/'reports'/NAME
CACHE = REPO/'.cache'/NAME
SEEDS = (92601, 92602)
DATASETS = ('jena', 'pems32')
METHODS = ('RESIDUAL_F', 'RESIDUAL_Q', 'RESIDUAL_RAW_MLP', 'READOUT_ONLY')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    os.replace(temporary, path)


def seal(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError('Preserve prior sealed evidence: '+str(path))
    if not path.exists():
        atomic(path, value)
    return value


def csv_write(path, rows):
    rows = list(rows)
    path = Path(path)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def event(stage, **values):
    row = dict(stage=stage, time_unix=time.time(), pid=os.getpid(), **values)
    with (CACHE/'EVENTS.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    return row


def start():
    return read(REPORT/'EXECUTION_START.json')


def ledger():
    rows = []
    for p in (CACHE/'process_receipts').glob('*.json'):
        rows.append(read(p))
    active = CACHE/'ACTIVE_PROCESS.json'
    if active.exists():
        row = read(active)
        if not any(r['session_id'] == row['session_id'] for r in rows):
            rows.append(dict(row, end_unix=time.time(), active=True))
    gpu = sum(r['end_unix']-r['start_unix'] for r in rows if r.get('gpu_owned'))
    return dict(start_unix=start()['start_unix'], elapsed_seconds=time.time()-start()['start_unix'],
                GPU_owned_process_seconds=gpu, sessions=rows,
                GPU_accounting='Owned process spawn-to-exit wall; includes initialization, cache, evaluation and idle within workers',
                training_deadline_unix=start()['training_deadline_unix'], final_deadline_unix=start()['final_deadline_unix'])


def deadline(training=True):
    clock = start()
    limit = clock['training_deadline_unix'] if training else clock['final_deadline_unix']
    remaining_gpu = clock['gpu_owned_limit_seconds']-ledger()['GPU_owned_process_seconds']
    return min(limit, time.time()+max(0, remaining_gpu))


def score(p, y, mask):
    stats = inherited.statistics(p, y, mask)
    return {k: stats[k] for k in ('mse', 'mae', 'n_observed', 'n_origins')}


def identity(array):
    return inherited.array_identity(array)


def register():
    core = ('models.py', 'engine.py', 'data_contract.py', 'probes.py')
    paths = [REPO/'research'/NAME/name for name in core]
    paths += [REPO/'research/informs_broad_validation_20261015_v1'/name
              for name in ('networks.py','training.py','gpu_runtime.py')]
    sources = {p.relative_to(REPO).as_posix(): sha(p) for p in paths}
    if (REPORT/'EXPERIMENT_PLAN.json').exists():
        existing=read(REPORT/'EXPERIMENT_PLAN.json')
        if existing['sources']!=sources or existing['document_sha256']!=start()['document_sha256']:
            raise ValueError('Registered scientific source or document changed')
        return existing
    plan = dict(campaign=NAME, version=1, document_sha256=start()['document_sha256'],
        initial_main=start()['initial_remote_main'], registration_unix=time.time(),
        methods=list(METHODS), datasets=list(DATASETS), seeds=list(SEEDS),
        axes=inherited.EXPECTED, correction='G + phi(z,F(z)) @ zero_initialized_D; no additional x_last',
        parents='exact A60 G single seed92601 and original S Jena K12 cap3 / PeMS K8 cap6',
        optimizer=dict(name='AdamW', lr=.001, weight_decay=0, clip_norm=1,
                       effective_batch=16, microbatch=4, F_row_microbatch=128, dtype='FP32'),
        endpoints=[0,.25,1,3], extension_endpoints=[4,5,6],
        extension_rule='if any last checkpoint strictly best, all four methods and both seeds for that dataset extend together if complete-group budget fits',
        selection='M_select minimum MSE; exact tie earlier epoch; lambda exact tie larger',
        residual_probes=['ZERO','MEAN','OWN','ALL','S_DIFF'], lambdas=[.01,.1,1,10],
        ridge_objective='for channel c: sum_h sum_i observed*(r-XW-b)^2/N_c + lambda*sum_h ||W_h||^2; intercept unpenalized; normal penalty lambda*N_c',
        feature_scaling='fit features on M_fit only; fresh full-Mcal scaling for refit',
        C_gate='both F seeds beat G; mean F beats min(mean alpha,Q,RAW_MLP,READOUT_ONLY), within numerical-error ties reject; complete-group budget',
        C_priority=['jena','pems32'], C_methods=['SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX'],
        C_blocks=dict(jena=[6,6],pems32=[4,4]), C_shared_P='fixed original A60 S',
        V_policy='no new V numeric reads/forecasts until all B/C decisions sealed; evaluate fixed full V once for final models',
        T_policy='no new protected numeric access or full-source numeric hash',
        refit='fresh reset to same A60 parent initialization on full Mcal using selected epoch',
        resource_clock=start(), sources=sources,
        limitations=['V repeatedly used development data','M_select repeated selection','upstream preprocessing/pretraining overlap unresolved','two seeds do not give confidence intervals'])
    return seal(REPORT/'EXPERIMENT_PLAN.json', plan)
