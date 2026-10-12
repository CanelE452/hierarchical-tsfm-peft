"""Gated, balanced E5 objective comparison on a small common encoder."""
from argparse import ArgumentParser
from pathlib import Path
import copy
import hashlib
import json
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import (HERE, CACHE, L, H, SEEDS, LAMBDAS, Budget, BudgetHeld, access_guard,
                    ahash, sha, read, write, load_data, windows, masks, score,
                    physical_score, ridge_grouped)
from model_neural import TinyEncoder, ReconstructionProbe

OBJECTIVES = ('FULL_REC', 'CAUSAL_REC', 'CAUSAL_CONTRAST', 'CAUSAL_FORECAST')
UPDATES, BATCH, READOUT_LAGS = 600, 8, 32


def training_origins(bounds):
    return np.arange(bounds[0] + L, bounds[1] - H + 1, L + H, dtype=np.int64)


def make_batches(origins, batch, updates, seed):
    if len(origins) < batch or batch < 2:
        raise ValueError('BLOCKED_PAIR_CONSTRUCTION: at least two nonoverlapping input windows required')
    rng = np.random.default_rng(seed)
    result, remaining = [], np.array([], dtype=np.int64)
    for _ in range(updates):
        if len(remaining) < batch:
            remaining = rng.permutation(origins)
        result.append(remaining[:batch])
        remaining = remaining[batch:]
    return np.asarray(result, dtype=np.int64)


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def masked_mse(prediction, target, observed):
    if not bool(observed.any()):
        raise ValueError('No observed supervised target')
    return ((prediction-target).square() * observed).sum() / observed.sum()


def contrastive_loss(p, q, timestamps, receptive_field=63, temperature=.2):
    if p.shape != q.shape or p.ndim != 2 or len(p) != len(timestamps):
        raise ValueError('Contrastive pair shape mismatch')
    n = len(p)
    embeddings = torch.cat([p, q], 0)
    times = torch.cat([timestamps, timestamps], 0)
    delta = (times[:, None]-times[None, :]).abs()
    negative = delta >= receptive_field
    positive_index = (torch.arange(2*n, device=p.device) + n) % (2*n)
    positive = torch.zeros_like(negative)
    positive[torch.arange(2*n, device=p.device), positive_index] = True
    if not bool((negative.sum(1) > 0).all()):
        raise ValueError('BLOCKED_PAIR_CONSTRUCTION: some anchors have no legal negative')
    logits = embeddings @ embeddings.T / temperature
    allowed = negative | positive
    logits = logits.masked_fill(~allowed, -torch.inf)
    loss = F.cross_entropy(logits, positive_index)
    return loss, {'anchors': 2*n, 'positive_pairs': 2*n,
                  'negative_pairs': int(negative.sum()),
                  'same_timestamp_negative_pairs': int((negative & (delta == 0)).sum()),
                  'near_timestamp_negative_pairs': int((negative & (delta < receptive_field)).sum()),
                  'minimum_negative_time_separation': receptive_field}


def objective_head(condition, channels):
    if condition.endswith('REC'):
        return ReconstructionProbe(channels)
    if condition.endswith('FORECAST'):
        return nn.Linear(16, H*channels)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))


def source_pins():
    return {str(p): sha(p) for p in [Path(__file__), HERE/'model_neural.py', HERE/'common.py',
                                    HERE/'APPROVED_INSTRUCTIONS.md', HERE/'PLAN.md']}


def create_plan(dataset, tag, device='cuda'):
    data = load_data(dataset)
    origins = training_origins(data['fit_bounds'])
    actual_batch = min(BATCH, len(origins))
    if actual_batch < 2:
        raise ValueError('BLOCKED_PAIR_CONSTRUCTION: fewer than two nonoverlapping FIT windows')
    model = TinyEncoder(data['x'].shape[1])
    plan = {
        'experiment': 'E5', 'status': 'PLANNED_NOT_EXECUTED', 'dataset': dataset, 'device': device,
        'purpose': 'Separate temporal information direction and encoder learning objectives using frozen common readouts',
        'simple_alternative': 'Causal reconstruction and FIT-only lagged direct ridge',
        'varied_factors': ['FULL versus CAUSAL padding within REC', 'REC/CONTRAST/FORECAST objectives within CAUSAL'],
        'fixed': {'L': L, 'H': H, 'hidden': 32, 'output': 16, 'kernel': 3,
                  'dilations': list(model.dilations), 'receptive_field': model.receptive_field,
                  'encoder_parameters': model.parameter_count, 'seeds': SEEDS, 'updates': UPDATES,
                  'batch': actual_batch, 'microbatch': actual_batch, 'accumulation': 1,
                  'optimizer': 'Adam', 'learning_rate': .001, 'readout_lags': list(range(33)),
                  'ridge_lambdas': LAMBDAS, 'normalization': 'FIT_FIXED_INPUT + TIMESTAMP_CHANNEL_LAYERNORM',
                  'readout_selection': 'TUNE physical channel macro MSE; declared lambda order breaks ties',
                  'probe': 'timestamp-only d16->128->ReLU->C; Adam1e-3;600 updates; TUNE observed reconstruction MSE chooses grid0:100:600'},
        'objectives': list(OBJECTIVES), 'training_origins': origins.tolist(),
        'training_origin_hash': ahash(origins), 'fit_bounds': data['fit_bounds'],
        'tune_bounds': data['tune_bounds'], 'val_bounds': data['val_bounds'],
        'readout_origins': {k: v.tolist() for k,v in data['origins'].items()},
        'input_source': data['source'], 'input_source_sha256': data['source_hash'],
        'contrast': {'family': 'symmetric NT-Xent', 'temperature': .2, 'mask_probability_observed_inputs': .1,
                     'projection': '16->32->ReLU->16; L2 normalized',
                     'positive': 'same raw timestamp in two independently masked views',
                     'anchors': list(range(model.receptive_field-1, L, model.receptive_field)),
                     'negative_exclusion': '|raw_timestamp_difference| < receptive_field, including duplicate/overlap timestamps',
                     'cross_instance_and_temporal_negatives': True,
                     'independent_window_definition': 'nonoverlapping input+future intervals; statistical independence not assumed'},
        'decisions': {'improved': 'retain objective as a candidate within this structure/budget',
                      'unchanged': 'simpler causal reconstruction may suffice',
                      'worse': 'reject this configuration, not the objective family',
                      'stop': ['unsafe information path', 'nonfinite values', 'budget/memory/asset failure']},
        'estimated_latency_not_measured': True,
        'output_array_upper_bytes': int(sum(len(v) for v in data['origins'].values()) *
                                        (L*16 + H*data['x'].shape[1])*4),
        'source_pins': source_pins(),
    }
    path = HERE/'plans'/f'E5_{dataset}_{tag}.json'
    write(path, plan)
    return path, plan


def check_plan(path, digest, gate_path, gate_digest):
    if sha(path) != digest or sha(gate_path) != gate_digest:
        raise ValueError('Plan/gate byte binding changed')
    plan, gate = read(path), read(gate_path)
    if gate.get('experiment') != 'E5' or gate.get('safety_pass') is not True or gate.get('execute') is not True:
        raise ValueError('E1 safety and E3/E4 scientific execution gate not approved')
    for p, digest in plan['source_pins'].items():
        if sha(p) != digest:
            raise ValueError(f'Frozen implementation changed: {p}')
    return plan, gate


def take_history(data, origins, device):
    x = windows(dict(data,origins={'e5':np.asarray(origins,dtype=np.int64)}),'e5')
    m = np.stack([data['mask'][o-L:o] for o in origins])
    return torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)


def take_future(data, origins, device):
    y = np.stack([data['x'][o:o+H] for o in origins])
    m = np.stack([data['mask'][o:o+H] for o in origins])
    return torch.from_numpy(y).to(device), torch.from_numpy(m).to(device)


def run_objective(model, head, condition, data, origins, device, random_seed, target_floor=None):
    x, observed = take_history(data, origins, device)
    exposure = {'input_samples': len(origins), 'input_timesteps': int(len(origins)*L),
                'observed_input_values': int(observed.sum()), 'future_target_values': 0,
                'future_target_timesteps': 0}
    if condition.endswith('CONTRAST'):
        generator = torch.Generator(device=device).manual_seed(random_seed)
        masked_a = (torch.rand(x.shape, device=device, generator=generator) < .1) & observed
        masked_b = (torch.rand(x.shape, device=device, generator=generator) < .1) & observed
        anchors = list(range(model.receptive_field-1, L, model.receptive_field))
        pa = F.normalize(head(model(x.masked_fill(masked_a, 0))[:, anchors]).reshape(-1,16), dim=-1)
        pb = F.normalize(head(model(x.masked_fill(masked_b, 0))[:, anchors]).reshape(-1,16), dim=-1)
        timestamps = torch.tensor(origins-L, device=device)[:, None] + torch.tensor(anchors, device=device)[None]
        loss, audit = contrastive_loss(pa, pb, timestamps.flatten(), model.receptive_field)
        exposure.update(audit)
        exposure['encoder_input_timesteps'] = 2*exposure['input_timesteps']
        exposure['masked_observed_values'] = int(masked_a.sum()+masked_b.sum())
        exposure['positive_timestamps'] = timestamps.detach().cpu().flatten().tolist()
        exposure['positive_information_cutoffs'] = exposure['positive_timestamps']
        return loss, exposure
    latent = model(x)
    exposure['encoder_input_timesteps'] = exposure['input_timesteps']
    if condition.endswith('FORECAST'):
        future, future_mask = take_future(data, origins, device)
        output = head(latent[:, -1]).reshape(len(origins), H, data['x'].shape[1])
        exposure['future_target_values'] = int(future_mask.sum())
        exposure['future_target_timesteps'] = len(origins)*H
        return masked_mse(output, future, future_mask), exposure
    if target_floor is not None:
        ts = torch.tensor(origins-L, device=device)[:,None] + torch.arange(L, device=device)[None]
        observed = observed & (ts[:,:,None] >= target_floor)
    return masked_mse(head(latent), x, observed), exposure


def save_torch(path, value):
    path = Path(path)
    if not path.is_relative_to(CACHE):
        raise ValueError('Checkpoint outside new cache')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        torch.save(value, f)
    return sha(path)


def save_npz(path, **values):
    path = Path(path)
    if not path.is_relative_to(CACHE):
        raise ValueError('Array outside new cache')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **values)
    return sha(path)


def train_encoder(condition, seed, initial, data, batches, output, cache, budget, device):
    torch.manual_seed(seed)
    model = TinyEncoder(data['x'].shape[1], causal=condition != 'FULL_REC').to(device)
    model.load_state_dict(initial)
    initial_hash = state_hash(model.state_dict())
    torch.manual_seed(seed+17)
    head = objective_head(condition, data['x'].shape[1]).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=.001)
    curves, exposures, checkpoints = [], [], []
    started = time.perf_counter()
    for step in range(UPDATES+1):
        if step % 100 == 0:
            model.eval(); head.eval()
            tune_origins = training_origins(data['tune_bounds'])[:len(batches[0])]
            if len(tune_origins) < 2:
                raise ValueError('BLOCKED_PAIR_CONSTRUCTION: TUNE curve needs two nonoverlapping windows')
            with torch.no_grad():
                fit_loss, _ = run_objective(model, head, condition, data, batches[0], device, seed+200000,
                                           target_floor=data['fit_bounds'][0])
                tune_loss, _ = run_objective(model, head, condition, data, tune_origins, device, seed+200001,
                                            target_floor=data['tune_bounds'][0])
            curve = {'update': step, 'fit_fixed_batch_objective_loss': float(fit_loss),
                     'tune_fixed_batch_objective_loss': float(tune_loss),
                     'cross_objective_loss_comparison_valid': False,
                     'wall_seconds': time.perf_counter()-started}
            if not np.isfinite([curve['fit_fixed_batch_objective_loss'], curve['tune_fixed_batch_objective_loss']]).all():
                raise ValueError('Nonfinite objective curve')
            curves.append(curve)
            cp = cache/f'E_u{step:04}.pt'
            digest = save_torch(cp, {'encoder': model.state_dict(), 'head': head.state_dict(),
                                    'optimizer': optimizer.state_dict(), 'step': step, 'condition': condition,
                                    'seed': seed, 'encoder_initial_hash': initial_hash,
                                    'batch_order_hash': ahash(batches)})
            checkpoints.append({'update': step, 'path': str(cp), 'sha256': digest})
            write(output/f'E_curve_u{step:04}.json', curve)
            budget.check()
        if step == UPDATES:
            break
        model.train(); head.train()
        optimizer.zero_grad(set_to_none=True)
        loss, exposure = run_objective(model, head, condition, data, batches[step], device, seed+step+1)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
        exposure.update({'update': step+1, 'origins': batches[step].tolist(),
                         'loss': float(loss.detach()), 'history_end_inclusive': (batches[step]-1).tolist(),
                         'target_end_exclusive': (batches[step]+H).tolist() if condition.endswith('FORECAST') else batches[step].tolist()})
        exposures.append(exposure)
        if (step+1) % 10 == 0:
            budget.check()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    write(output/'E_training.json', {'condition': condition, 'seed': seed, 'actual_updates': UPDATES,
          'encoder_parameters': model.parameter_count, 'head_parameters': sum(p.numel() for p in head.parameters()),
          'encoder_initial_hash': initial_hash, 'encoder_final_hash': state_hash(model.state_dict()),
          'batch_order_hash': ahash(batches), 'curves': curves, 'checkpoints': checkpoints,
          'actual_sample_exposures': sum(v['input_samples'] for v in exposures),
          'actual_encoder_timestep_exposures': sum(v['encoder_input_timesteps'] for v in exposures),
          'actual_future_target_values': sum(v['future_target_values'] for v in exposures),
          'actual_future_target_timesteps': sum(v['future_target_timesteps'] for v in exposures),
          'wall_seconds': time.perf_counter()-started,
          'training_exposures': exposures})
    return model


def export_latents(model, data, budget, device):
    exports = {}
    for split, origins in data['origins'].items():
        values = np.empty((len(origins), L, 16), dtype=np.float32)
        for start in range(0, len(origins), BATCH):
            x, _ = take_history(data, origins[start:start+BATCH], device)
            with torch.no_grad():
                values[start:start+BATCH] = model(x).cpu().numpy()
            budget.check()
        exports[split] = values
    return exports


def probe_evaluation(probe, latent, data, split, budget, device, return_array=False):
    origins = data['origins'][split]
    output = np.empty((len(origins), L, data['x'].shape[1]), dtype=np.float32)
    for start in range(0, len(origins), BATCH):
        with torch.no_grad():
            z = torch.from_numpy(latent[start:start+BATCH]).to(device)
            output[start:start+BATCH] = probe(z).cpu().numpy()
        budget.check()
    target = windows(data, split)
    observed = masks(data, split, target=False)
    floor = {'fit': data['fit_bounds'][0], 'tune': data['tune_bounds'][0], 'val': data['val_bounds'][0]}[split]
    ts = origins[:,None]-L + np.arange(L)[None]
    observed &= ts[:,:,None] >= floor
    metrics = score(output*data['std']+data['mean'], target*data['std']+data['mean'], observed)
    metrics['normalized'] = score(output, target, observed)
    return (metrics, output, observed) if return_array else metrics


def train_probe(seed, latents, data, output, cache, budget, device):
    torch.manual_seed(seed+23)
    probe = ReconstructionProbe(data['x'].shape[1]).to(device)
    initial_hash = state_hash(probe.state_dict())
    order = make_batches(np.arange(len(latents['fit'])), min(BATCH,len(latents['fit'])), UPDATES, seed+29)
    optimizer = torch.optim.Adam(probe.parameters(), lr=.001)
    targets, observed = windows(data,'fit'), masks(data,'fit',target=False)
    curves, checkpoints, best = [], [], None
    started = time.perf_counter()
    for step in range(UPDATES+1):
        if step % 100 == 0:
            probe.eval()
            metric = probe_evaluation(probe, latents['tune'], data, 'tune', budget, device)
            curves.append({'update': step, 'tune_observed_reconstruction': metric})
            cp = cache/f'D_u{step:04}.pt'
            digest = save_torch(cp, {'decoder':probe.state_dict(), 'optimizer':optimizer.state_dict(), 'step':step})
            checkpoints.append({'update':step,'path':str(cp),'sha256':digest})
            if best is None or metric['mse'] < best[0]:
                best = (metric['mse'], step, copy.deepcopy(probe.state_dict()))
            budget.check()
        if step == UPDATES:
            break
        js = order[step]
        z = torch.from_numpy(latents['fit'][js]).to(device)
        y = torch.from_numpy(targets[js]).to(device)
        m = torch.from_numpy(observed[js]).to(device)
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        loss = masked_mse(probe(z), y, m)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite reconstruction probe loss')
        loss.backward(); optimizer.step()
        if (step+1)%10 == 0:
            budget.check()
    probe.load_state_dict(best[2]); probe.eval()
    metric, prediction, mask = probe_evaluation(probe,latents['val'],data,'val',budget,device,True)
    path = cache/'RECONSTRUCTION_PROBE_VAL.npz'
    digest = save_npz(path,prediction=prediction,mask=mask,origins=data['origins']['val'])
    receipt = {'actual_updates':UPDATES,'selected_update':best[1],'selection_scope':'TUNE_OBSERVED_RECONSTRUCTION_MSE',
               'initial_hash':initial_hash,'selected_hash':state_hash(probe.state_dict()),
               'batch_order_hash':ahash(order),'parameter_count':sum(p.numel() for p in probe.parameters()),
               'wall_seconds':time.perf_counter()-started,'curves':curves,'checkpoints':checkpoints,
               'val':metric,'prediction_path':str(path),'prediction_sha256':digest,
               'target_timestamp_mask': 'only timestamps inside the relevant FIT/TUNE/VAL split are scored'}
    write(output/'RECONSTRUCTION_PROBE.json',receipt)
    return receipt


def latent_descriptives(z):
    flat = z.reshape(-1, z.shape[-1]).astype(np.float64)
    centered = flat-flat.mean(0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T@centered/len(flat)),0.)
    total = eigenvalues.sum()
    if total == 0:
        effective_rank = 0.
    else:
        weights = eigenvalues[eigenvalues>0]/total
        effective_rank = float(np.exp(-(weights*np.log(weights)).sum()))
    return {'per_coordinate_variance':flat.var(0).tolist(),'effective_rank_entropy':effective_rank,
            'exact_constant_all_coordinates': bool((flat.var(0)==0).all()),
            'rank_is_descriptive_not_a_gate':True}


def readout_rows(data):
    rows = {}
    for split, oo in data['origins'].items():
        target_times = (oo[:,None] + np.arange(H)[None]).flatten()
        uniq, counts = np.unique(target_times,return_counts=True)
        rows[split] = {'origins':oo.tolist(),'origin_count':len(oo),'target_rows':len(target_times),
                       'unique_target_timestamps':len(uniq),'target_overlap_histogram':
                           {str(int(c)):int((counts==c).sum()) for c in np.unique(counts)},
                       'feature_timestamps':(oo[:,None]-1-np.arange(33)[None]).tolist(),
                       'feature_export':'last33 timestamps of fixed-origin E(history512); FULL may use later within-history timestamps',
                       'target_start':oo.tolist(),'target_end_exclusive':(oo+H).tolist()}
    return rows


def save_ridge(path, fits):
    arrays = {}
    for i, (js, model) in enumerate(fits):
        prefix = f'group{i}_'
        arrays[prefix+'outputs'] = np.asarray(js,dtype=np.int64)
        for name in ('mean','std','ym','train','coef','dual'):
            value = getattr(model,name)
            if value is not None:
                arrays[prefix+name] = value
        arrays[prefix+'lambda'] = np.asarray(model.lam)
    return save_npz(path,**arrays)


def remaining_seconds(budget):
    used = 0.
    for p in (HERE/'resources').glob('*.json'):
        r = read(p)
        if r['gpu'] == budget.gpu and (not budget.gpu or r['phase'] == 2):
            used += r['wall_seconds']
    return (5400 if budget.gpu else 3600)-used


def cost_preflight(data,plan,budget,device,output,local_limit=None,enforce_forecast=True):
    timings = {}
    started = time.perf_counter()
    def check():
        budget.check()
        if local_limit is not None and time.perf_counter()-started >= local_limit:
            raise BudgetHeld('COST_ONLY local wall limit reached')
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    origins = np.asarray(plan['training_origins'])[:plan['fixed']['batch']]
    for condition in OBJECTIVES:
        torch.manual_seed(SEEDS[0])
        model = TinyEncoder(data['x'].shape[1],condition!='FULL_REC').to(device)
        head = objective_head(condition,data['x'].shape[1]).to(device)
        durations = []
        for rep in range(6):
            model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
            sync(); start = time.perf_counter()
            loss,_ = run_objective(model,head,condition,data,origins,device,SEEDS[0]+rep)
            loss.backward()
            sync()
            durations.append(time.perf_counter()-start)
            check()
        timings[condition] = {'median_forward_backward_seconds':float(np.median(durations[1:])),
                              'samples':durations,'optimizer_updates':0}
        del model,head
    probe = ReconstructionProbe(data['x'].shape[1]).to(device)
    z = torch.zeros((len(origins),L,16),device=device)
    y,m = take_history(data,origins,device)
    durations = []
    for rep in range(6):
        probe.zero_grad(set_to_none=True)
        sync(); start = time.perf_counter()
        masked_mse(probe(z),y,m).backward()
        sync(); durations.append(time.perf_counter()-start)
        check()
    per_probe = float(np.median(durations[1:]))
    # The margin includes readout/export/curve/checkpoint overhead; it is a forecast, not a guarantee.
    forecast = 2*(2*UPDATES*sum(t['median_forward_backward_seconds'] for t in timings.values()) +
                  8*UPDATES*per_probe) + 180
    check()
    receipt = {'objective_timings':timings,'probe_forward_backward_seconds':per_probe,
               'estimated_balanced_job_wall_seconds':forecast,'estimated_margin_multiplier':2,
               'remaining_shared_budget_wall_seconds':remaining_seconds(budget),
               'charged_budget':'PHASE2_GPU90MIN' if budget.gpu else 'GLOBAL_CPU60MIN',
               'device':device.type,'actual_profile_wall_seconds':time.perf_counter()-started,
               'actual_optimizer_updates':0,'performance_scores_used_for_scope_selection':False,
               'forecast_is_not_completion_guarantee':True}
    write(output/'COST_PREFLIGHT.json',receipt)
    if enforce_forecast and forecast >= receipt['remaining_shared_budget_wall_seconds']:
        raise BudgetHeld('Balanced E5 cost forecast exceeds remaining shared wall budget')
    return receipt


def run_balanced(plan_path, plan_sha, gate_path, gate_sha, tag, requested_device='cuda'):
    plan, gate = check_plan(plan_path,plan_sha,gate_path,gate_sha)
    dataset = plan['dataset']
    data = load_data(dataset)
    if data['source_hash'] != plan['input_source_sha256']:
        raise ValueError('Input bytes differ from frozen plan')
    for split in data['origins']:
        if data['origins'][split].tolist() != plan['readout_origins'][split]:
            raise ValueError('Readout origins differ from frozen plan')
    if plan.get('device','cuda') != requested_device:
        raise ValueError('Execution device differs from frozen plan')
    device = torch.device(requested_device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('BLOCKED_MISSING_GPU')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = HERE/'results'/f'E5_{dataset}_{tag}'
    cache = CACHE/f'E5_{dataset}_{tag}'
    output.mkdir(parents=True,exist_ok=False)
    cache.mkdir(parents=True,exist_ok=False)
    result = {'experiment':'E5','dataset':dataset,'status':'RUNNING_CONFIRMED',
              'plan_path':str(plan_path),'plan_sha256':plan_sha,'gate_path':str(gate_path),'gate_sha256':gate_sha,
              'source_pins':source_pins(),'conditions':[],'protocol':'NEW_FIT',
              'scope':'FULL_VAL_DEVELOPMENT_DIAGNOSTIC','gate_reason':gate.get('reason')}
    try:
        with Budget(2,gpu=device.type=='cuda',label=f'E5 balanced {dataset} objectives {device.type}') as budget:
            result['cost_preflight'] = cost_preflight(data,plan,budget,device,output)
            for seed in SEEDS:
                torch.manual_seed(seed)
                initial = TinyEncoder(data['x'].shape[1],causal=True).state_dict()
                initial = {k:v.clone() for k,v in initial.items()}
                batches = make_batches(np.asarray(plan['training_origins']),plan['fixed']['batch'],UPDATES,seed+31)
                save_npz(cache/f'batch_order_s{seed}.npz',origins=batches)
                for condition in OBJECTIVES:
                    out, cp = output/f'{condition}_s{seed}', cache/f'{condition}_s{seed}'
                    out.mkdir(); cp.mkdir()
                    model = train_encoder(condition,seed,initial,data,batches,out,cp,budget,device)
                    latent = export_latents(model,data,budget,device)
                    features = {s:z[:,-1-np.arange(33),:].reshape(len(z),-1) for s,z in latent.items()}
                    # CPU ridge is charged within this GPU-using process, not to a second overlapping CPU ledger.
                    predicted, ridge, fits = ridge_grouped(features['fit'],windows(data,'fit',True),masks(data,'fit'),
                                    features['tune'],windows(data,'tune',True),masks(data,'tune'),features['val'],
                                    score_scale=data['std'],budget=budget)
                    y_tune = np.zeros_like(windows(data,'tune',True),dtype=float).reshape(len(features['tune']),-1)
                    for js, fitted in fits:
                        y_tune[:,js] = fitted.predict(features['tune'])
                    y_tune = y_tune.reshape(len(features['tune']),H,data['x'].shape[1])
                    metric, tune_metric = physical_score(data,predicted,'val'), physical_score(data,y_tune,'tune')
                    pred_path = cp/'DIRECT_RIDGE_VAL.npz'
                    pred_hash = save_npz(pred_path,prediction=predicted,mask=masks(data,'val'),origins=data['origins']['val'],
                        target=windows(data,'val',True),tune_prediction=y_tune,tune_origins=data['origins']['tune'])
                    ridge_path = cp/'DIRECT_RIDGE_COEFFICIENTS.npz'
                    ridge_hash = save_ridge(ridge_path,fits)
                    probe = train_probe(seed,latent,data,out,cp,budget,device)
                    record = {'condition':condition,'seed':seed,'encoder_updates':UPDATES,'probe_updates':UPDATES,
                              'encoder_final_path':str(cp/'E_u0600.pt'),'encoder_final_sha256':sha(cp/'E_u0600.pt'),
                              'encoder_initial_hash':state_hash(initial),'batch_order_hash':ahash(batches),
                              'direct_ridge':ridge,'tune':tune_metric,'val':metric,
                              'prediction_path':str(pred_path),'prediction_sha256':pred_hash,
                              'ridge_coefficients_path':str(ridge_path),'ridge_coefficients_sha256':ridge_hash,
                              'reconstruction_probe':probe,'representation_statistics':latent_descriptives(latent['fit']),
                              'row_metadata':readout_rows(data),'fit_normalization_mean':data['mean'].tolist(),
                              'fit_normalization_std':data['std'].tolist(),
                              'raw_recovery_note':data['raw_recovery_note'],
                              'head_not_used_in_frozen_direct_readout':True,
                              'physical_scale_note':'Channel macro mixes heterogeneous units within this dataset; normalized and per-channel errors accompany it'}
                    write(out/'RESULT.json',record)
                    result['conditions'].append({'condition':condition,'seed':seed,'path':str(out/'RESULT.json'),
                                                  'sha256':sha(out/'RESULT.json'),'val':metric,'tune':tune_metric})
                    write(output/'PROGRESS.json',result,replace=True)
                    del model,latent,features
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    budget.check()
            result['budget_ledger'] = str(budget.path)
        result['status'] = 'COMPLETE_BALANCED'
        result['budget_ledger_sha256'] = sha(result['budget_ledger'])
        result['interpretation'] = 'Fixed600 training budget; not convergence proof or independent generalization. Different objectives have different heads/FLOPs/target exposure.'
        write(output/'RESULT.json',result)
        return output/'RESULT.json'
    except BaseException as error:
        result['status'] = 'BUDGET_HELD_PARTIAL' if isinstance(error,BudgetHeld) else 'FAILED_PRESERVED'
        result['reason'] = str(error)
        write(output/'FAILURE.json',result)
        raise


def profile_only(plan_path,digest,tag,requested_device):
    if sha(plan_path) != digest:
        raise ValueError('Plan bytes changed')
    plan = read(plan_path)
    if plan.get('device','cuda') != requested_device:
        raise ValueError('Profile device differs from frozen plan')
    for p,d in plan['source_pins'].items():
        if sha(p) != d:
            raise ValueError(f'Frozen source changed: {p}')
    device = torch.device(requested_device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('BLOCKED_MISSING_GPU')
    torch.set_num_threads(4)
    output = HERE/'results'/f'E5_COST_ONLY_{plan["dataset"]}_{tag}'
    output.mkdir(parents=True,exist_ok=False)
    with Budget(2,gpu=device.type=='cuda',label=f'E5 COST_ONLY {device.type} preflight') as budget:
        data = load_data(plan['dataset'])
        receipt = cost_preflight(data,plan,budget,device,output,local_limit=29,enforce_forecast=False)
        receipt.update({'status':'COST_ONLY_COMPLETE_NOT_SCIENCE','science_updates':0,
                        'source_pins':source_pins(),'plan_path':str(plan_path),'plan_sha256':digest,
                        'budget_ledger':str(budget.path)})
    receipt['budget_ledger_sha256'] = sha(budget.path)
    write(output/'RESULT.json',receipt)
    return output/'RESULT.json'


def main():
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--plan',action='store_true')
    group.add_argument('--run',action='store_true')
    group.add_argument('--profile',action='store_true')
    parser.add_argument('--dataset',choices=['jena','robin'],default='jena')
    parser.add_argument('--tag',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--frozen-plan',type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--gate',type=Path)
    parser.add_argument('--gate-sha256')
    args = parser.parse_args()
    if not args.tag.replace('_','').replace('-','').isalnum():
        raise ValueError('Use a simple fresh attempt tag')
    access_guard()
    if args.plan:
        with Budget(2,gpu=False,label='E5 fixed objective plan'):
            path, plan = create_plan(args.dataset,args.tag,args.device)
        print(json.dumps({'plan':str(path),'sha256':sha(path),'parameters':plan['fixed']['encoder_parameters'],
                          'actual_science_updates':0},ensure_ascii=False))
    elif args.profile:
        if any(v is None for v in [args.frozen_plan,args.plan_sha256]):
            parser.error('--profile requires a frozen plan and SHA256')
        path = profile_only(args.frozen_plan,args.plan_sha256,args.tag,args.device)
        print(json.dumps({'profile':str(path),'sha256':sha(path),'actual_science_updates':0},ensure_ascii=False))
    else:
        if any(v is None for v in [args.frozen_plan,args.plan_sha256,args.gate,args.gate_sha256]):
            parser.error('--run requires frozen plan and gate paths with their SHA256')
        path = run_balanced(args.frozen_plan,args.plan_sha256,args.gate,args.gate_sha256,args.tag,args.device)
        print(json.dumps({'result':str(path),'sha256':sha(path)},ensure_ascii=False))


if __name__ == '__main__':
    main()
