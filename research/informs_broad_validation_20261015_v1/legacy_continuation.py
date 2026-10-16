"""E0: explicitly queued legacy L512/H48 continuation, never a new-data fit.

Import/inventory/plan do not load source arrays or a foundation model. Only the
explicit ``run --execute-training`` entry point does so. Historical weights and
Adam moments are reused; missing historical RNG makes the initial transition
RESTARTED_NOT_EXACT_RESUME. No legacy file is written or patched.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback

import numpy as np

from resources import BudgetExceeded, ResourceBusy, validate_gpu_child

HERE = Path(__file__).resolve().parent
OLD_NAME = 'temporal_frontend_campaign_20261015_v1'
ENDPOINTS = (4800, 9600, 19200)
MODEL_RELATIVE = '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8'
FIDELITY = 'RESTARTED_NOT_EXACT_RESUME'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ahash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def extended_schedule(n, updates, seed):
    """Preserve old2400 exactly; a fixed independent permutation stream follows."""
    if n < 1 or updates < 2400 or updates > ENDPOINTS[-1]:
        raise ValueError('schedule requires n>0 and 2400<=updates<=19200')
    rng = np.random.default_rng(seed); count = 2400*4
    if count <= n:
        prefix = np.floor((np.arange(count)+.5)*n/count).astype(np.int64)
        rng.shuffle(prefix)
    else:
        prefix = np.concatenate([rng.permutation(n) for _ in range((count+n-1)//n)])[:count]
    extension_count = (updates-2400)*4
    extension_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2400, 0xE0]))
    extension = (np.concatenate([extension_rng.permutation(n)
                 for _ in range((extension_count+n-1)//n)])[:extension_count]
                 if extension_count else np.empty(0, dtype=np.int64))
    return np.concatenate((prefix, extension)).reshape(updates, 4)


def optimizer_steps(state):
    return {int(item['step']) for item in state.get('state', {}).values() if 'step' in item}


def validate_carry(carry, expected_updates):
    steps = optimizer_steps(carry.get('optimizer', {}))
    if carry.get('updates') != expected_updates or len(carry.get('curve', [])) != expected_updates:
        raise ValueError('carry update/learning-curve mismatch')
    if not steps or steps != {expected_updates}:
        raise ValueError(f'optimizer steps do not equal {expected_updates}: {steps}')
    return {'optimizer_step':expected_updates, 'optimizer_state_count':len(carry['optimizer']['state'])}


def restore_optimizer(model, state, lr, expected_step):
    import torch
    if optimizer_steps(state) != {expected_step}:
        raise ValueError('Adam moments must be reused at the exact carry update')
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.)
    optimizer.load_state_dict(state)
    if any(group['lr'] != lr or group['weight_decay'] != 0. for group in optimizer.param_groups):
        raise ValueError('legacy optimizer hyperparameters changed')
    return optimizer


def capture_rng(include_cuda):
    import torch
    nstate = np.random.get_state()
    value = {'python':random.getstate(), 'numpy':(nstate[0], nstate[1].tolist(),
             int(nstate[2]), int(nstate[3]), float(nstate[4])), 'torch_cpu':torch.get_rng_state()}
    if include_cuda:
        value['torch_cuda'] = torch.cuda.get_rng_state_all()
    return value


def restore_rng(value, include_cuda):
    import torch
    random.setstate(value['python'])
    n = value['numpy']; np.random.set_state((n[0], np.asarray(n[1], np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(value['torch_cpu'])
    if include_cuda:
        torch.cuda.set_rng_state_all(value['torch_cuda'])


def check_work_boundary(lease, reserve_seconds=5.):
    """Validate the live parent's lease before CUDA/work; reserve save time."""
    if lease is None:
        raise ResourceBusy('E0 training requires the central parent GPU lease')
    lease.check()
    if lease.remaining <= reserve_seconds:
        raise BudgetExceeded('E0 stopped at a work boundary to preserve its full carry')


class LeaseBoundFoundation:
    def __init__(self, foundation, lease):
        self.foundation, self.lease = foundation, lease

    def __getattr__(self, name):
        return getattr(self.foundation, name)

    def __call__(self, *args, **kwargs):
        check_work_boundary(self.lease)
        return self.foundation(*args, **kwargs)


def pending_checkpoint_update(update, checkpoints):
    return update if update % 400 == 0 and not any(row['update'] == update for row in checkpoints) else None


def save_budget_checkpoint(path, model, optimizer, curve, checkpoints, sampler, contract, *, include_cuda):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    update = len(curve)
    carry = dict(updates=update, optimizer=optimizer.state_dict(), curve=curve,
                 checkpoints=checkpoints, rng=capture_rng(include_cuda), sampler=sampler,
                 continuation_fidelity=FIDELITY, budget_resume_fidelity='EXACT_WITHIN_NEW_RUN')
    validate_carry(carry, update)
    if sampler.get('next_update') != update:
        raise ValueError('Budget carry next batch/update differs')
    state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    with path.open('xb') as stream:
        torch.save(dict(state=state, carry=carry, budget_contract=contract), stream)
    write_json(path.with_suffix(path.suffix+'.json'), dict(artifact_sha256=sha(path), budget_contract=contract,
                                                        update=update, status='BUDGET_LIMIT'))
    return dict(checkpoint_sha256=sha(path), update=update)


def load_budget_checkpoint(path, expected_contract):
    import torch
    path = Path(path)
    sidecar = read_json(path.with_suffix(path.suffix+'.json'))
    if sidecar['budget_contract'] != expected_contract or sidecar['artifact_sha256'] != sha(path):
        raise ValueError('Budget checkpoint source/identity/file hash differs')
    saved = torch.load(path, weights_only=True, map_location='cpu')
    if saved['budget_contract'] != expected_contract:
        raise ValueError('Budget checkpoint internal contract differs')
    update = saved['carry']['updates']
    validate_carry(saved['carry'], update)
    if saved['carry']['sampler']['next_update'] != update or sidecar['update'] != update:
        raise ValueError('Budget checkpoint next batch/update differs')
    return saved


def best_checkpoint(checkpoints):
    return min(checkpoints, key=lambda row:(row['B']['mse'], row['update']))


def model_spec(receipt):
    keys = ('id', 'type', 'dataset', 'fold', 'kind', 'r', 'seed', 'front_kwargs')
    result = {key:copy.deepcopy(receipt[key]) for key in keys if key in receipt}
    result.setdefault('type', 'ADAPTER'); result.setdefault('fold', 'primary')
    if 'base' in receipt:
        result['base'] = model_spec(receipt['base'])
    return result


def bind_legacy_paths(rt, dio, asset_root, campaign_root, cache_root, model_relative=MODEL_RELATIVE):
    asset_root = Path(asset_root).resolve(); campaign_root = Path(campaign_root).resolve()
    cache_root = Path(cache_root).resolve()
    if campaign_root.name == OLD_NAME or campaign_root == asset_root/'research'/OLD_NAME:
        raise ValueError('legacy output directory is forbidden')
    guard_cache = asset_root/'.cache'/campaign_root.name
    if not cache_root.is_relative_to(guard_cache):
        raise ValueError('cache must remain inside unchanged legacy data-guard new-campaign allowance')
    model = (asset_root/model_relative).resolve()
    if not model.is_relative_to(asset_root):
        raise ValueError('foundation model must be an explicit local asset-root path')
    if getattr(dio, '_GUARD_INSTALLED', False):
        raise RuntimeError('use a fresh subprocess; data guard was already installed')
    rt.ROOT = dio.ROOT = asset_root
    # The unchanged guard permits HERE and ROOT/.cache/HERE.name. Keep these roots aligned.
    dio.HERE = campaign_root
    rt.HERE = campaign_root/'legacy_e0'; rt.CACHE = cache_root/'legacy_e0'; rt.MODEL = model
    return {'asset_root':str(asset_root), 'source_root':str(asset_root/'research'/OLD_NAME),
            'receipt_root':str(rt.HERE), 'cache_root':str(rt.CACHE),
            'data_guard_here':str(dio.HERE), 'data_guard_cache_root':str(guard_cache),
            'model_path':str(model)}


def legacy_ids(asset_root):
    report = (Path(asset_root)/'research'/OLD_NAME/'REPORT_KO_v3.md').read_text(encoding='utf-8-sig')
    line = next(line for line in report.splitlines() if line.startswith('미수렴·미해결 ID:'))
    ids = re.findall(r'`([^`]+)`', line)
    if len(ids) != 12 or len(set(ids)) != 12:
        raise ValueError('expected the frozen report exact twelve unresolved IDs')
    return ids


def receipt_for(asset_root, legacy_id):
    if not re.fullmatch(r'[A-Za-z0-9_]+', legacy_id):
        raise ValueError('invalid legacy ID')
    path = Path(asset_root)/'research'/OLD_NAME/'fits'/f'{legacy_id}_2400.json'
    result = read_json(path)
    if result['id'] != legacy_id or result['updates'] != 2400 or result.get('fold', 'primary') != 'primary':
        raise ValueError('legacy receipt identity/contract mismatch')
    return result, path


def checkpoint_path(asset_root, legacy_id, update=2400):
    return Path(asset_root)/'.cache'/OLD_NAME/'training'/legacy_id/f'last_{update}.pt'


def validate_sources(asset_root):
    old = Path(asset_root)/'research'/OLD_NAME
    lock = read_json(old/'SELECTION_LOCK.json')
    names = ('runtime.py', 'campaign.py', 'models.py', 'data.py')
    hashes = {name:sha(old/name) for name in names}
    if any(hashes[name] != lock['source_hashes'][name] for name in names):
        raise ValueError('frozen legacy scientific source hash mismatch')
    if sha(old/'FOLD_STATISTICS.json') != lock['statistics_hash']:
        raise ValueError('frozen legacy statistics metadata hash mismatch')
    return lock, hashes


def inventory(asset_root):
    """Metadata and small adapter PT only. Never load original dataset arrays."""
    import torch
    asset_root = Path(asset_root).resolve(); old = asset_root/'research'/OLD_NAME
    _, source_hashes = validate_sources(asset_root)
    ids = legacy_ids(asset_root)
    with (HERE/'UNRESOLVED_RESOLUTION.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if {row['legacy_id'] for row in rows} != set(ids):
        raise ValueError('ledger/report exact IDs differ')
    result = []
    for row in rows:
        rec, path = receipt_for(asset_root, row['legacy_id'])
        last = checkpoint_path(asset_root, row['legacy_id'])
        state = torch.load(last, weights_only=True, map_location='cpu')
        evidence = validate_carry(state['carry'], 2400)
        if state['carry']['curve'] != rec['learning_curve'] or state['carry']['checkpoints'] != rec['checkpoints']:
            raise ValueError('legacy checkpoint/receipt carry differs')
        final_cp = next(item for item in rec['checkpoints'] if item['update'] == 2400)
        if tensor_state_hash(state['state']) != final_cp['state_sha256']:
            raise ValueError('last2400 tensor state/CP2400 hash differs')
        prefix = extended_schedule(rec['legal_A_origins'], 2400, rec['seed'])
        if ahash(prefix) != rec['origin_schedule_sha256']:
            raise ValueError('original sampler reconstruction failed')
        if sha(last) != row['last_state_sha256']:
            raise ValueError('ledger/checkpoint SHA differs')
        result.append({'id':rec['id'], **evidence, 'last_sha256':sha(last),
                       'rng_present':'rng' in state['carry'], 'sampler_present':'sampler' in state['carry'],
                       'original_schedule_reconstructed':True, 'status':row['status']})
    return {'status':'INVENTORY_VALID', 'count':len(result), 'source_hashes':source_hashes,
            'rows':result, 'source_arrays_opened':False, 'foundation_loaded':False}


def queue_plan(asset_root, ids, include_controls):
    with (HERE/'UNRESOLVED_RESOLUTION.csv').open(encoding='utf-8-sig', newline='') as stream:
        ledger = {row['legacy_id']:row for row in csv.DictReader(stream)}
    control_groups = {root:[root]+[name for name in row['matched_control_ids'].split(';') if name]
                      for root, row in ledger.items()}
    allowed = set(ledger).union(*(set(group) for group in control_groups.values()))
    roots = list(dict.fromkeys(ids or ledger))
    if any(name not in allowed for name in roots):
        raise ValueError('requested ID is neither an original unresolved route nor its registered matched control')
    selected = list(roots)
    if include_controls:
        selected += [name for root in roots for group in control_groups.values() if root in group for name in group]
    selected = list(dict.fromkeys(selected))
    rows = []
    for name in selected:
        rec, _ = receipt_for(asset_root, name)
        if not checkpoint_path(asset_root, name).is_file():
            raise FileNotFoundError(f'matched last2400 missing: {name}')
        rows.append({'id':name, 'role':'UNRESOLVED' if name in ledger else 'MATCHED_CONTROL',
                     'dataset':rec['dataset'], 'seed':rec['seed'], 'status':'PENDING_QUEUE',
                     'continuation_fidelity':FIDELITY, 'endpoints':list(ENDPOINTS),
                     'incremental_updates_per_endpoint':[2400,4800,9600],
                     'same_legacy_question':True})
    groups = {key:value for key,value in control_groups.items() if any(name in value for name in roots)}
    return {'status':'READY_RESTARTED_NOT_EXACT_RESUME', 'routes':rows, 'route_count':len(rows),
            'max_incremental_updates':len(rows)*(19200-2400),
            'include_controls':include_controls, 'matched_comparison_groups':groups,
            'unexecuted_controls_status':'PENDING_QUEUE',
            'fair_final_comparison_ready':False, 'old_length':512, 'old_horizon':48,
            'new_INFORMS_data_contract_used':False, 'research_success_count':0}


def load_legacy(asset_root, cache_root):
    old = Path(asset_root)/'research'/OLD_NAME
    lock, hashes = validate_sources(asset_root)
    modules = {}
    for name in ('models', 'runtime', 'data', 'campaign'):
        if name in sys.modules:
            raise RuntimeError(f'fresh subprocess required: module name already in use: {name}')
        spec = importlib.util.spec_from_file_location(name, old/f'{name}.py')
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module); modules[name] = module
    paths = bind_legacy_paths(modules['runtime'], modules['data'], asset_root, HERE, cache_root)
    return modules, lock, hashes, paths


def build_legacy_model(spec, state, modules):
    """Construct original classes; full last-state loading retains a G parent."""
    import torch
    rt, cp, models = modules['runtime'], modules['campaign'], modules['models']
    kind = spec['type']
    if kind == 'ADAPTER':
        front = models.TemporalFront(spec['kind'], channels=state['E'].shape[0], **spec.get('front_kwargs', {}))
        model = rt.Adapter(front, state['center'].numpy(), state['E'].numpy())
    elif kind == 'RAW': model = cp.Raw()
    elif kind == 'CURRENT': model = cp.Current()
    elif kind == 'G':
        base_state = {name[5:]:tensor for name, tensor in state.items() if name.startswith('base.')}
        base = build_legacy_model(spec['base'], base_state, modules)
        model = cp.WithG(base)
    else:
        raise ValueError(f'unsupported legacy continuation type: {kind}')
    model.load_state_dict(state, strict=True)
    return model


def update_ledger(legacy_id, status, reason):
    path = HERE/'UNRESOLVED_RESOLUTION.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); fieldnames = reader.fieldnames; rows = list(reader)
    found = False
    for row in rows:
        if row['legacy_id'] == legacy_id:
            row['status'] = status; row['reason'] = reason; found = True
    if found:
        temporary = path.with_suffix('.csv.tmp')
        with temporary.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
        temporary.replace(path)


def run_one(asset_root, cache_root, legacy_id, endpoint, plan, *, child_lease=None, resume_budget_checkpoint=None):
    """Actual GPU work: invoked by root queue only, never by inventory/tests."""
    import torch
    rec, rec_path = receipt_for(asset_root, legacy_id)
    if endpoint not in ENDPOINTS:
        raise ValueError('endpoint must be 4800, 9600, or 19200')
    lease = child_lease if child_lease is not None else validate_gpu_child(cache_root)
    check_work_boundary(lease)
    modules, old_lock, sources, paths = load_legacy(asset_root, cache_root)
    rt, cp = modules['runtime'], modules['campaign']
    rt.HERE.mkdir(parents=True, exist_ok=True); rt.CACHE.mkdir(parents=True, exist_ok=True)
    attempt_path = rt.HERE/'fits'/f'{legacy_id}_{endpoint}.json'
    if attempt_path.exists() or (rt.CACHE/'training'/legacy_id/f'last_{endpoint}.pt').exists():
        raise FileExistsError('endpoint artifacts already exist; never overwrite/retrain silently')
    started = time.time(); token = rt.CACHE/'EXECUTION.lock'
    with token.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps({'pid':os.getpid(), 'id':legacy_id, 'endpoint':endpoint}))
    initialized = False
    try:
        initial_begin = 2400 if endpoint == 4800 else ENDPOINTS[ENDPOINTS.index(endpoint)-1]
        begin = initial_begin
        old_last = checkpoint_path(asset_root, legacy_id)
        last_path = old_last if begin == 2400 else rt.CACHE/'training'/legacy_id/f'last_{begin}.pt'
        contract = dict(legacy_id=legacy_id, endpoint=endpoint, initial_begin_update=initial_begin,
                        parent_checkpoint_sha256=sha(last_path), legacy_receipt_sha256=sha(rec_path),
                        legacy_last2400_sha256=sha(old_last), source_hashes=sources,
                        runner_source_sha256=sha(Path(__file__)), statistics_sha256=old_lock['statistics_hash'],
                        foundation_sha256=old_lock['F_model_hash'],
                        resource_guard_source_sha256=sha(Path(__file__).with_name('resources.py')))
        if resume_budget_checkpoint is not None:
            resume_path = (Path(cache_root)/Path(resume_budget_checkpoint)).resolve()
            if not resume_path.is_relative_to(rt.CACHE.resolve()) or not resume_path.is_file():
                raise ValueError('Budget resume checkpoint must stay inside this new E0 cache')
            last = load_budget_checkpoint(resume_path, contract)
            begin = int(last['carry']['updates'])
            if not initial_begin <= begin <= endpoint:
                raise ValueError('Budget resume update outside requested endpoint segment')
            if begin == endpoint and last['carry']['sampler'].get('pending_checkpoint_update') != endpoint:
                raise ValueError('Endpoint-state budget resume must have its final assessment pending')
            last_path = resume_path
        else:
            last = torch.load(last_path, weights_only=True, map_location='cpu')
        carry = last['carry']; evidence = validate_carry(carry, begin)
        if resume_budget_checkpoint is None and begin == 2400 and (carry['curve'] != rec['learning_curve'] or carry['checkpoints'] != rec['checkpoints']):
            raise ValueError('legacy carry/receipt mismatch')
        if resume_budget_checkpoint is None and begin == 2400:
            expected = next(item for item in rec['checkpoints'] if item['update'] == 2400)['state_sha256']
            if tensor_state_hash(last['state']) != expected:
                raise ValueError('legacy last2400 tensor hash/receipt mismatch')
        elif resume_budget_checkpoint is None:
            prior = read_json(rt.HERE/'fits'/f'{legacy_id}_{begin}.json')
            if (prior['id'] != legacy_id or prior['endpoint'] != begin or
                prior['last_file_sha256'] != sha(last_path) or prior['source_hashes'] != sources or
                prior['new_runner_source_sha256'] != sha(Path(__file__)) or
                prior['statistics_sha256'] != old_lock['statistics_hash']):
                raise ValueError('prior endpoint identity/artifact/source/statistics mismatch')
        check_work_boundary(lease)
        win = cp.dataset_windows(rec['dataset'], rec.get('fold', 'primary'))
        if len(win.origins['A']) != rec['legal_A_origins']:
            raise ValueError('legacy A-origin pool changed')
        statistics = read_json(Path(asset_root)/'research'/OLD_NAME/'FOLD_STATISTICS.json')[rec['dataset']]['primary']
        np.testing.assert_array_equal(win.mean, np.asarray(statistics['A_mean'], np.float32))
        np.testing.assert_array_equal(win.std, np.asarray(statistics['A_std'], np.float32))
        order = extended_schedule(len(win.origins['A']), endpoint, rec['seed'])
        if ahash(order[:2400]) != rec['origin_schedule_sha256']:
            raise ValueError('legacy sampler prefix mismatch')
        if begin > 2400 and carry['sampler']['schedule_sha256'] != ahash(order[:begin]):
            raise ValueError('new segment sampler prefix mismatch')
        spec = model_spec(rec)
        check_work_boundary(lease)
        model = build_legacy_model(spec, last['state'], modules).cuda().eval()
        if rt.state_hash(model) != tensor_state_hash(last['state']):
            raise ValueError('CPU/GPU loaded last-state mismatch')
        optimizer = restore_optimizer(model, carry['optimizer'], rec['learning_rate'], begin)
        parameters = [p for p in model.parameters() if p.requires_grad]
        check_work_boundary(lease)
        bolt = rt.Bolt()
        guarded_bolt = LeaseBoundFoundation(bolt, lease)
        if bolt.initial_hash != old_lock['F_model_hash']:
            raise ValueError('legacy frozen foundation hash changed')
        if any(p.requires_grad or p.grad is not None for p in bolt.model.parameters()):
            raise ValueError('foundation must remain frozen')
        if {id(p) for p in parameters}.intersection(id(p) for p in bolt.model.parameters()):
            raise ValueError('foundation parameters included in optimizer')
        new_seed = int(np.random.SeedSequence([rec['seed'],2400,0xE0,1]).generate_state(1)[0])
        if begin == 2400 and resume_budget_checkpoint is None:
            random.seed(new_seed); np.random.seed(new_seed); torch.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
        else:
            restore_rng(carry['rng'], include_cuda=True)
        curve = list(carry['curve']); checkpoints = list(carry['checkpoints'])
        best = best_checkpoint(checkpoints)
        best_path = (Path(asset_root)/'.cache'/OLD_NAME/'checkpoints'/legacy_id/f"cp{best['update']}.pt"
                     if best['update'] <= 2400 else rt.CACHE/'checkpoints'/legacy_id/f"cp{best['update']}.pt")
        best_state = torch.load(best_path, weights_only=True, map_location='cpu')
        initialized = True
        current_update = begin
        def assess_checkpoint(update):
            nonlocal best, best_state
            check_work_boundary(lease)
            a, _ = cp.assess(model, guarded_bolt, win, 'A', win.eval_A)
            check_work_boundary(lease)
            b, _ = cp.assess(model, guarded_bolt, win, 'B')
            check_work_boundary(lease)
            row = {'update':update, 'A':a, 'B':b, 'state_sha256':rt.state_hash(model)}
            state = rt.clone(model)
            path = rt.CACHE/'checkpoints'/legacy_id/f'cp{update}.pt'
            if path.exists():
                raise FileExistsError('Preserve an existing unreferenced E0 CP; audit before retry')
            rt.save(path, state)
            checkpoints.append(row)
            if (b['mse'], update) < (best['B']['mse'], best['update']):
                best, best_state = row, state
            rt.event(stage='E0_CHECKPOINT', id=legacy_id, update=update, B_mse=b['mse'])
        if resume_budget_checkpoint is not None and carry['sampler'].get('pending_checkpoint_update') is not None:
            if carry['sampler']['pending_checkpoint_update'] != begin:
                raise ValueError('Pending assessment state/update differs')
            assess_checkpoint(begin)
        for step in range(begin, endpoint):
            check_work_boundary(lease)
            x, y, mask = win.get(win.origins['A'][order[step]])
            optimizer.zero_grad(set_to_none=True)
            tick = time.perf_counter(); prediction = model(x, guarded_bolt)
            loss = rt.loss(prediction, y, mask, win.counts, len(win.origins['A']))
            if not torch.isfinite(loss): raise FloatingPointError('NONFINITE_LOSS')
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm): raise FloatingPointError('NONFINITE_GRADIENT')
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in parameters):
                raise FloatingPointError('NONFINITE_PARAMETER')
            torch.cuda.synchronize()
            curve.append({'update':step+1, 'loss':float(loss.detach()), 'gradient_norm':float(norm),
                          'clipped':bool(norm>1), 'gradient_by_parameter':{}, 'seconds':time.perf_counter()-tick})
            current_update = step+1
            if (step+1)%100 == 0:
                rt.event(stage='E0_UPDATE', id=legacy_id, update=step+1, loss=curve[-1]['loss'])
            if (step+1)%400 == 0 or step+1 == endpoint:
                assess_checkpoint(step+1)
        if optimizer_steps(optimizer.state_dict()) != {endpoint}:
            raise ValueError('optimizer carry counter did not advance to endpoint')
        frozen_after = rt.state_hash(bolt.model)
        if frozen_after != bolt.initial_hash:
            raise ValueError('frozen foundation changed')
        last_state = rt.clone(model)
        sampler = {'algorithm':'old2400_then_independent_full_permutations_v1',
                   'seed_sequence':[rec['seed'],2400,0xE0], 'schedule_sha256':ahash(order),
                   'legacy_prefix_sha256':ahash(order[:2400]),
                   'post2400_sha256':ahash(order[2400:]), 'legal_A_origins':len(win.origins['A']),
                   'batch':4, 'next_update':endpoint}
        new_carry = {'updates':endpoint, 'optimizer':optimizer.state_dict(), 'curve':curve,
                     'checkpoints':checkpoints, 'rng':capture_rng(include_cuda=True), 'sampler':sampler,
                     'continuation_fidelity':FIDELITY}
        rt.save(rt.CACHE/'training'/legacy_id/f'last_{endpoint}.pt', {'state':last_state, 'carry':new_carry})
        model.load_state_dict(best_state)
        selected_path = rt.CACHE/'selected'/f'{legacy_id}_{endpoint}.pt'
        rt.save(selected_path, rt.clone(model))
        improving = best['update'] == endpoint or cp.still_decreasing(checkpoints)
        touched = np.unique(order.reshape(-1)); used_origins = win.origins['A'][touched]
        input_times = np.unique((used_origins[:,None]+np.arange(-512,0)[None]).reshape(-1))
        target_times = np.unique((used_origins[:,None]+np.arange(48)[None]).reshape(-1))
        receipt = {'id':legacy_id, 'legacy_spec':spec, 'endpoint':endpoint, 'begin_update':begin,
                   'incremental_updates':endpoint-begin, 'total_updates_including_old':endpoint,
                   'status':'CONTINUED_ENDPOINT_COMPLETE', 'continuation_fidelity':FIDELITY,
                   'historical_rng_missing':True, 'historical_sampler_state_missing':True,
                   'new_seed_at_2400':new_seed, 'new_runner_source_sha256':sha(Path(__file__)),
                   'new_segment_rng_restored':begin>2400, 'optimizer_reused':True,
                   'resumed_budget_checkpoint':resume_budget_checkpoint is not None,
                   'budget_resume_fidelity':'EXACT_WITHIN_NEW_RUN' if resume_budget_checkpoint is not None else None,
                   'optimizer_before':evidence, 'optimizer_after_step':endpoint,
                   'sampler':sampler, 'selected_update':best['update'], 'selected_B':best['B'],
                   'selected_state_sha256':rt.state_hash(model), 'selected_file_sha256':sha(selected_path),
                   'last_file_sha256':sha(rt.CACHE/'training'/legacy_id/f'last_{endpoint}.pt'),
                   'parent_last_sha256':sha(last_path), 'legacy_receipt_sha256':sha(rec_path),
                   'legacy_last2400_sha256':sha(old_last), 'source_hashes':sources,
                   'statistics_sha256':old_lock['statistics_hash'], 'paths':paths,
                   'length':512, 'horizon':48, 'fold':'primary', 'checkpoint_every_updates':400,
                   'batch':4, 'optimizer':'AdamW', 'learning_rate':rec['learning_rate'],
                   'weight_decay':0., 'gradient_clipping_norm':1., 'scheduler':None,
                   'effective_origin_epochs':endpoint*4/len(win.origins['A']),
                   'legal_A_origins':len(win.origins['A']), 'distinct_A_origins':len(touched),
                   'unique_input_times':len(input_times), 'unique_target_times':len(target_times),
                   'unique_true_observed_target_entries':int(win.data['observed'][target_times].sum()),
                   'repeated_origin_exposures':int(endpoint*4-len(touched)),
                   'clipping_fraction':float(np.mean([row['clipped'] for row in curve])),
                   'A_monitor_origins':win.eval_A.tolist(), 'B_origins':win.origins['B'].tolist(),
                   'checkpoints':checkpoints, 'learning_curve':curve,
                   'resolution_status':'STILL_IMPROVING_UNRESOLVED' if improving else 'CONVERGENCE_UNPROVEN',
                   'research_success':False, 'historically_unseen':False,
                   'selection_uses':['legacy_A_monitor','legacy_B'], 'C_scored':False,
                   'protected_TEST_accessed':False, 'F_unchanged':frozen_after==bolt.initial_hash,
                   'F_model_hash':frozen_after, 'queue_registry':plan,
                   'started_unix':started, 'finished_unix':time.time()}
        rt.write(attempt_path, receipt)
        update_ledger(legacy_id, receipt['resolution_status'],
                      f'Completed endpoint {endpoint}; weights/Adam reused; historical RNG unavailable. Receipt: legacy_e0/fits/{attempt_path.name}')
        return {'id':legacy_id, 'endpoint':endpoint, 'selected_update':best['update'],
                'status':receipt['status'], 'resolution_status':receipt['resolution_status']}
    except (BudgetExceeded, ResourceBusy) as error:
        stamp = time.time_ns()
        budget = dict(id=legacy_id, endpoint=endpoint, status='BUDGET_LIMIT' if isinstance(error, BudgetExceeded) else 'RESOURCE_GUARD_STOP',
                      reason=str(error), endpoint_complete=False, research_success=False,
                      continuation_fidelity=FIDELITY, initialized=initialized,
                      started_unix=started, finished_unix=time.time(), new_runner_source_sha256=sha(Path(__file__)))
        if initialized:
            sampler = dict(algorithm='old2400_then_independent_full_permutations_v1',
                           seed_sequence=[rec['seed'],2400,0xE0], schedule_sha256=ahash(order[:current_update]),
                           legacy_prefix_sha256=ahash(order[:2400]), post2400_sha256=ahash(order[2400:current_update]),
                           legal_A_origins=len(win.origins['A']), batch=4, next_update=current_update)
            sampler['pending_checkpoint_update'] = pending_checkpoint_update(current_update, checkpoints)
            budget_path = rt.CACHE/'training'/legacy_id/f'budget_u{current_update}_target{endpoint}_{stamp}.pt'
            evidence = save_budget_checkpoint(budget_path, model, optimizer, curve, checkpoints, sampler, contract, include_cuda=True)
            relative = budget_path.relative_to(Path(cache_root).resolve()).as_posix()
            budget.update(current_update=current_update, next_update=current_update, optimizer_after_step=current_update,
                          budget_checkpoint_cache_relative_path=relative, budget_checkpoint_sha256=evidence['checkpoint_sha256'],
                          budget_resume_fidelity='EXACT_WITHIN_NEW_RUN', no_partial_endpoint_score_published=True,
                          resume_command=f'python legacy_queue.py run --asset-root <asset-root> --cache-root <cache-root> --id {legacy_id} --endpoint {endpoint} --resume-budget-checkpoint {relative} --execute-training')
        else:
            budget.update(no_optimizer_update=True, resume_budget_checkpoint=resume_budget_checkpoint,
                          resume_command=f'python legacy_queue.py run --asset-root <asset-root> --cache-root <cache-root> --id {legacy_id} --endpoint {endpoint} --execute-training' +
                                         (f' --resume-budget-checkpoint {resume_budget_checkpoint}' if resume_budget_checkpoint else ''))
        write_json(rt.HERE/'budgets'/f'{legacy_id}_{endpoint}_{stamp}.json', budget)
        update_ledger(legacy_id, budget['status'], f'Endpoint {endpoint} incomplete; explicit full-carry resume required')
        return budget
    except Exception as error:
        write_json(rt.HERE/f'FAILURE_{legacy_id}_{endpoint}_{time.time_ns()}.json',
                   {'id':legacy_id, 'endpoint':endpoint, 'status':'FAILED_ATTEMPT_PRESERVED',
                    'error':repr(error), 'traceback':traceback.format_exc(), 'paths':paths,
                    'started_unix':started, 'finished_unix':time.time(), 'research_success':False})
        raise
    finally:
        token.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inventory','plan','run'))
    parser.add_argument('--asset-root', default=os.environ.get('INFORMS_ASSET_ROOT'))
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--id', action='append', dest='ids')
    parser.add_argument('--endpoint', type=int, choices=ENDPOINTS)
    parser.add_argument('--include-controls', action='store_true')
    parser.add_argument('--execute-training', action='store_true')
    parser.add_argument('--resume-budget-checkpoint', help='Cache-relative immutable E0 budget carry; same ID/target/source only')
    args = parser.parse_args()
    if not args.asset_root: parser.error('--asset-root or INFORMS_ASSET_ROOT is required')
    asset_root = Path(args.asset_root).resolve()
    if args.command == 'inventory':
        result = inventory(asset_root)
    else:
        plan = queue_plan(asset_root, args.ids, args.include_controls)
        if args.command == 'plan': result = plan
        else:
            if not args.execute_training or args.endpoint is None:
                parser.error('run requires --execute-training and an explicit --endpoint')
            if len(plan['routes']) != 1:
                parser.error('run accepts one ID per fresh subprocess; use plan for the control queue')
            cache_root = args.cache_root or asset_root/'.cache'/HERE.name
            result = run_one(asset_root, cache_root, plan['routes'][0]['id'], args.endpoint, plan,
                             resume_budget_checkpoint=args.resume_budget_checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
