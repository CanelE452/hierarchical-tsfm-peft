"""Past-prefix lag-graph prototype, bounded balanced graph-condition comparison."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
import psutil
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from common import (HERE, CACHE, L, H, SEEDS, LAMBDAS, Ridge, Budget, BudgetHeld,
                    access_guard, ahash, sha, read, write, load_data, score, ridge_grouped,
                    select_origins)
from model_neural import TinyEncoder, ReconstructionProbe
from synthetic import SYNTHETIC_SEEDS, make_record

MODES = ('SELF_ONLY', 'FIXED_LAG_GRAPH', 'PAST_UPDATED_LAG_GRAPH', 'WRONG_EDGE_GRAPH')
MAX_LAG, UPDATE_INTERVAL, TOP_K, UPDATES, BATCH = 32, 32, 2, 600, 8


def pins():
    return {str(p): sha(p) for p in [Path(__file__), HERE/'model_neural.py', HERE/'common.py',
            HERE/'synthetic.py', HERE/'APPROVED_INSTRUCTIONS.md', HERE/'PLAN.md']}


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def lagged_features(x, destinations):
    destinations = np.asarray(destinations, dtype=np.int64)
    if len(destinations) and (destinations.min() < MAX_LAG or destinations.max() >= len(x)):
        raise ValueError('Graph feature timestamp outside the supplied observed prefix')
    return np.asarray(x)[destinations[:, None]-np.arange(1, MAX_LAG+1), :].reshape(len(destinations), -1)


def empty_graph(channels):
    return {'source': np.full((channels, TOP_K), -1, np.int16),
            'lag': np.zeros((channels, TOP_K), np.int16),
            'weight': np.zeros((channels, TOP_K), np.float32),
            'fit_samples_by_destination': np.zeros(channels, np.int32), 'fit_support_length': 0}


def select_edges(beta):
    channels = beta.shape[1]
    if beta.shape != (MAX_LAG*channels, channels):
        raise ValueError('Lag coefficient tensor shape differs')
    g = empty_graph(channels)
    for destination in range(channels):
        candidates = [(abs(float(beta[lag*channels+source, destination])), source, lag+1,
                       float(beta[lag*channels+source, destination]))
                      for lag in range(MAX_LAG) for source in range(channels)
                      if source != destination and beta[lag*channels+source, destination] != 0]
        candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
        chosen = candidates[:TOP_K]
        scale = max(1., sum(v[0] for v in chosen))
        for edge, (_, source, lag, weight) in enumerate(chosen):
            g['source'][destination, edge] = source
            g['lag'][destination, edge] = lag
            g['weight'][destination, edge] = weight/scale
    return g


def fit_graph(prefix, observed, lam, return_models=False):
    prefix, observed = np.asarray(prefix), np.asarray(observed, bool)
    if prefix.shape != observed.shape or prefix.ndim != 2 or len(prefix) > L:
        raise ValueError('Graph must receive only this origin prefix, at most512 observed timestamps')
    channels = prefix.shape[1]
    graph = empty_graph(channels)
    graph['fit_support_length'] = len(prefix)
    destinations = np.arange(MAX_LAG, len(prefix))
    fits = []
    if len(destinations) < 2:
        return (graph, fits) if return_models else graph
    features = lagged_features(prefix, destinations)
    beta = np.zeros((MAX_LAG*channels, channels), dtype=float)
    groups = {}
    for j in range(channels):
        valid = observed[destinations, j]
        graph['fit_samples_by_destination'][j] = valid.sum()
        groups.setdefault(valid.tobytes(), []).append(j)
    for js in groups.values():
        valid = observed[destinations, js[0]]
        if valid.sum() < 2:
            continue
        model = Ridge(lam).fit(features[valid], prefix[destinations[valid]][:, js])
        fits.append((js, model))
        standardized = model.coef if model.coef is not None else model.train.T@model.dual
        beta[:, js] = standardized/model.std[:, None]
    chosen = select_edges(beta)
    chosen['fit_samples_by_destination'] = graph['fit_samples_by_destination']
    chosen['fit_support_length'] = len(prefix)
    return (chosen, fits) if return_models else chosen


def wrong_graph(graph):
    result = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in graph.items()}
    channels = len(graph['source'])
    if channels < 3 and np.any(graph['source'] >= 0):
        raise ValueError('Cannot build nonself wrong-source permutation with fewer than3 channels')
    for destination in range(channels):
        foreign = [c for c in range(channels) if c != destination]
        permutation = dict(zip(foreign, foreign[1:]+foreign[:1]))
        for edge, source in enumerate(graph['source'][destination]):
            if source >= 0:
                result['source'][destination, edge] = permutation[int(source)]
    if np.any((result['source'] >= 0) & (result['source'] == np.arange(channels)[:, None])):
        raise ValueError('Wrong-source permutation introduced self edges')
    if not np.array_equal(result['lag'], graph['lag']) or not np.array_equal(result['weight'], graph['weight']):
        raise ValueError('Wrong graph changed lag or signed weight')
    return result


def graph_messages(history, observed, mode, fixed, lam):
    history, observed = np.asarray(history), np.asarray(observed, bool)
    if mode not in MODES or len(history) > L or history.shape != observed.shape:
        raise ValueError('Graph message input outside declared history contract')
    channels = history.shape[1]
    graph = empty_graph(channels) if mode in ('SELF_ONLY', 'PAST_UPDATED_LAG_GRAPH') else fixed
    if mode == 'WRONG_EDGE_GRAPH':
        graph = wrong_graph(fixed)
    message = np.zeros_like(history, dtype=np.float32)
    changes, cutoffs, samples, sources, lags, weights = [], [], [], [], [], []
    cutoff = -1
    events = [0]+list(range(UPDATE_INTERVAL-1,len(history),UPDATE_INTERVAL))
    for event, s in enumerate(events):
        if mode == 'PAST_UPDATED_LAG_GRAPH' and (s+1) % UPDATE_INTERVAL == 0:
            graph = fit_graph(history[:s+1], observed[:s+1], lam)
            cutoff = s if np.any(graph['source'] >= 0) else -1
        changes.append(s); cutoffs.append(cutoff)
        samples.append(graph['fit_samples_by_destination'].copy())
        sources.append(graph['source'].copy()); lags.append(graph['lag'].copy()); weights.append(graph['weight'].copy())
        end = events[event+1] if event+1 < len(events) else len(history)
        times = np.arange(s,end)[:,None,None]
        valid = (graph['source'][None] >= 0) & (times >= graph['lag'][None])
        values = history[np.maximum(times-graph['lag'][None],0),np.maximum(graph['source'][None],0)]
        weighted = np.where(valid, values*graph['weight'][None], 0).astype(np.float32)
        message[s:end] = weighted[:,:,0]+weighted[:,:,1]
    # The initial trace is omitted for UPDATED so each row is one declared32-tick event.
    first = 1 if mode == 'PAST_UPDATED_LAG_GRAPH' and changes and changes[0] == 0 else 0
    trace = {'update_indices': np.asarray(changes[first:], np.int16),
             'effective_cutoff_inclusive': np.asarray(cutoffs[first:], np.int16),
             'observed_fit_samples': np.asarray(samples[first:], np.int32),
             'source': np.asarray(sources[first:], np.int16), 'lag': np.asarray(lags[first:], np.int16),
             'weight': np.asarray(weights[first:], np.float32)}
    return message, trace


def pooled_pairs(records, split, maximum=None):
    pairs = np.asarray([(i, int(o)) for i, d in enumerate(records) for o in d['origins'][split]], np.int64)
    return select_origins(pairs, maximum) if maximum is not None else pairs


def remaining(gpu=False):
    used = sum(r['wall_seconds'] for p in (HERE/'resources').glob('*.json') for r in [read(p)]
               if r['gpu'] == gpu and (not gpu or r['phase'] == 2))
    return max(0., (5400 if gpu else 3600)-used)


class AllocationBudget(Budget):
    def __init__(self, phase, gpu, label, allocation):
        self.allocation = float(allocation)
        super().__init__(phase,gpu,label)

    def check(self):
        super().check()
        if self.record['wall_seconds'] >= self.allocation:
            raise BudgetHeld('Root allocated wall seconds exhausted within this E6 job')


def scope_records(dataset, summary_path):
    summary = read(summary_path)
    if summary.get('status') not in ('COMPLETE','COMPLETE_VERIFIED') or summary.get('synthetic_alignment') != 'PASS':
        raise ValueError('E2 completed noiseless alignment gate required')
    records, evidence = [], []
    synthetic = dataset.startswith('synthetic_')
    for entry in summary['records']:
        path = Path(entry['path'])
        if sha(path) != entry['sha256']:
            raise ValueError('E2 result bytes differ from summary')
        result = read(path)
        if result.get('dataset') != dataset:
            continue
        if result.get('status') != 'COMPLETE':
            raise ValueError('E2 source record incomplete')
        if not synthetic and result['LAG_SIGNAL'] not in ('LAG_SIGNAL', 'LAG_SIGNAL_MSE_ONLY'):
            raise ValueError('NOT_APPLICABLE_WITH_REASON: no prespecified real-data linear-lag signal')
        if synthetic:
            data, _ = make_record(dataset.removeprefix('synthetic_'), result['synthetic']['seed'])
        else:
            data = load_data(dataset, frozen=False)
        if data['source_hash'] != result['source_hash']:
            raise ValueError('E2 reused record/source mismatch')
        for split, origins in data['origins'].items():
            if ahash(origins) != result['origin_hashes'][split]:
                raise ValueError('E2 origin order mismatch')
        shared = result['shared_arrays']
        if not Path(shared['path']).is_file() or sha(shared['path']) != shared['sha256']:
            raise ValueError('E2 actual source arrays unavailable')
        records.append(data)
        evidence.append({'record_id': len(records)-1, 'path': str(path), 'sha256': sha(path),
                         'source_hash': data['source_hash'], 'seed': data.get('metadata', {}).get('seed'),
                         'e2_gate': result['LAG_SIGNAL'], 'e2_models': result['models']})
    if len(records) != (len(SYNTHETIC_SEEDS) if synthetic else 1):
        raise ValueError('BLOCKED_MISSING_ASSET: complete declared E2 record set unavailable')
    return records, evidence


def objective_choice(e5_path):
    if e5_path is None:
        return 'FORECAST', {'rule': 'INCOMPLETE_E5_DEFAULT_FORECAST', 'e5_complete': False}
    result = read(e5_path)
    if result.get('dataset') != 'jena' or result.get('status') != 'COMPLETE_BALANCED':
        return 'FORECAST', {'rule': 'INCOMPLETE_E5_DEFAULT_FORECAST', 'e5_complete': False,
                            'path': str(e5_path), 'sha256': sha(e5_path)}
    means = {}
    for objective in ('CAUSAL_REC', 'CAUSAL_FORECAST'):
        rows = [r for r in result['conditions'] if r['condition'] == objective]
        if sorted(r['seed'] for r in rows) != sorted(SEEDS):
            raise ValueError('E5 objective choice requires both declared seeds')
        means[objective] = float(np.mean([r['tune']['mse'] for r in rows]))
    chosen = 'REC' if means['CAUSAL_REC'] <= means['CAUSAL_FORECAST'] else 'FORECAST'
    return chosen, {'rule': 'JENA_TWO_SEED_TUNE_DIRECT_RIDGE_MSE_REC_VS_FORECAST_TIE_REC',
                    'e5_complete': True, 'mean_tune_mse': means, 'path': str(e5_path), 'sha256': sha(e5_path),
                    'selection_is_not_VAL_confirmation': True}


def graph_cost_preflight(records, pairs, budget):
    data = records[0]
    o = int(pairs['fit'][0, 1]); x, m = prepare_history(data,o), data['mask'][o-L:o]
    measured = []
    for length in (64, 128, 256, 512):
        start = time.perf_counter(); fit_graph(x[:length], m[:length], 1.)
        measured.append({'prefix_length': length, 'fit_seconds': time.perf_counter()-start})
        budget.check()
    start = time.perf_counter()
    graph_messages(x, m, 'PAST_UPDATED_LAG_GRAPH', empty_graph(x.shape[1]), 1.)
    full_history_seconds = time.perf_counter()-start
    fixed = fit_graph(x,m,1.)
    static_seconds = []
    for mode in ('SELF_ONLY','FIXED_LAG_GRAPH','WRONG_EDGE_GRAPH'):
        start = time.perf_counter(); graph_messages(x,m,mode,fixed,1.)
        static_seconds.append(time.perf_counter()-start); budget.check()
    torch.set_num_threads(2); torch.manual_seed(92799)
    model = TinyEncoder(2*x.shape[1],causal=True).eval()
    tensor = torch.from_numpy(np.repeat(np.concatenate((x,np.zeros_like(x)),axis=1)[None],BATCH,0))
    forward = []
    with torch.no_grad():
        model(tensor)
        for _ in range(3):
            start = time.perf_counter(); model(tensor); forward.append(time.perf_counter()-start)
            budget.check()
    origins = sum(len(v) for v in pairs.values())
    # Only cost is inspected;1.5 is declared allowance, not a proven upper bound.
    estimate = 1.5*(full_history_seconds*origins + 5*len(records)*measured[-1]['fit_seconds'])
    per_forward = float(np.median(forward))
    neural_estimate = per_forward*(len(MODES)*len(SEEDS)*UPDATES*4 +
        len(MODES)*len(SEEDS)*sum(int(np.ceil(len(v)/BATCH)) for v in pairs.values()) +
        len(MODES)*len(SEEDS)*25*(1+32/BATCH))
    bench_graph_estimate = 25*(1+32)*len(SEEDS)*(full_history_seconds+3*max(static_seconds))
    return {'measured_prefix_fits': measured, 'uncached_updated_history_seconds': full_history_seconds,
            'static_history_message_seconds':dict(zip(('SELF_ONLY','FIXED_LAG_GRAPH','WRONG_EDGE_GRAPH'),static_seconds)),
            'CPU_B8_forward_seconds':forward,'CPU_threads':2,
            'CPU_balanced_neural_export_benchmark_estimated_seconds':1.5*(neural_estimate+bench_graph_estimate),
            'CPU_complete_scope_estimate_seconds':estimate+1.5*(neural_estimate+bench_graph_estimate),
            'training_backprop_cost_multiplier_assumed':4,
            'unprofiled_costs':['ridge readout fitting/prediction','serialization/compression','full scoring'],
            'history_count': origins, 'fit_calls_per_updated_history': L//UPDATE_INTERVAL-1,
            'graph_cpu_estimated_seconds_with_allowance': estimate, 'allowance_factor': 1.5,
            'estimate_is_not_rigorous_upper_bound': True, 'lambda_used_for_cost_only': 1.,
            'actual_neural_updates': 0, 'TUNE_or_VAL_performance_inspected': False}


def create_plan(dataset, summary_path, e1_path, e5_path, tag, cpu_allocation, budget, device='cuda'):
    e1 = read(e1_path)
    if not (e1.get('e2_allowed') is True or e1.get('status') in ('PASS', 'VERIFIED') or (
            e1.get('experiment') == 'E1' and e1.get('safety_pass') is True and e1.get('status') == 'COMPLETE_VERIFIED')):
        raise ValueError('E1 safe execution path gate required')
    records, evidence = scope_records(dataset, summary_path)
    objective, decision = objective_choice(e5_path)
    pairs = {s: pooled_pairs(records, s, {'fit': 256, 'tune': 64, 'val': None}[s]) for s in ('fit','tune','val')}
    cost = graph_cost_preflight(records, pairs, budget)
    model = TinyEncoder(2*records[0]['x'].shape[1], causal=True)
    available = min(remaining(False), cpu_allocation if cpu_allocation is not None else remaining(False))
    estimate = cost['CPU_complete_scope_estimate_seconds'] if device == 'cpu' else cost['graph_cpu_estimated_seconds_with_allowance']
    status = 'READY' if estimate <= available else 'BUDGET_HELD'
    plan = {'experiment': 'E6', 'dataset': dataset, 'status': status,'compute_device':device,
            'purpose': 'Whether correctly directed fixed/updated past lag summaries add prediction value',
            'simple_alternative': 'SELF_ONLY, wrong-source graph, and actual E2 matched direct lag ridge',
            'varied_factor': 'SELF/FIXED/PAST_UPDATED/WRONG graph message used during separate training',
            'fixed': {'L': L, 'H': H, 'latent_dimension': 16, 'hidden': 32, 'kernel': 3,
                      'dilations': list(model.dilations), 'receptive_field': model.receptive_field,
                      'encoder_parameters': model.parameter_count, 'updates': UPDATES, 'batch': BATCH,
                      'microbatch': BATCH, 'accumulation': 1, 'optimizer': 'Adam', 'learning_rate': .001,
                      'CPU_threads':2,
                      'seeds': SEEDS, 'primary_readout': 'frozen E last33 timestamps -> common direct ridge',
                      'primary_readout_lambdas': LAMBDAS, 'readout_selection': 'TUNE physical channel macro MSE',
                      'encoder_checkpoint_selection': 'fixed600; grid is learning curve only',
                      'matched_raw_baseline': 'same pooled256/64/fullVAL pair IDs, raw lag0..32 ALL_CHANNEL_LAG ridge and persistence'},
            'objective': objective, 'objective_decision': decision,
            'graph': {'lags': list(range(1,33)), 'lambda_grid': LAMBDAS,
                      'lambda_selection': 'FIT-last<=512 one-step ridge -> TUNE first future point physical macro MSE; once per balanced scope',
                      'edge_selection': 'top2 foreign source-lag by |raw-FIT-normalized-feature coefficient|; ties source then lag',
                      'weight': 'signed beta; divide by max(1,sum(abs(chosen))) per destination',
                      'self_branch': 'all raw channels, concatenated with per-destination signed neighbor message',
                      'self_branch_limitation': 'TinyEncoder mixes raw channels: SELF_ONLY is no graph message, not channel-independent model',
                      'fixed': 'each record FIT last<=512 timestamps; targets and features inside that segment',
                      'updated': 'only origin history[a:s+1]; update at31,63,...511; short/insufficient observed prefixes have no foreign message',
                      'future': 'last observed graph held; direct48 readout does not consume future graph or observations',
                      'wrong': 'destination-specific cyclic permutation of foreign source IDs; same count, lag, signed weight',
                      'physical_causal_discovery_claim': False},
            'pairs': {s: v.tolist() for s,v in pairs.items()}, 'pair_hashes': {s: ahash(v) for s,v in pairs.items()},
            'records': [{'record_id': i, 'source': d['source'], 'source_hash': d['source_hash'],
                         'fit_bounds': d['fit_bounds'], 'tune_bounds': d['tune_bounds'], 'val_bounds': d['val_bounds'],
                         'normalization_fit_scope': d['normalization_fit_scope'], 'mean': d['mean'], 'std': d['std'],
                         'synthetic': d.get('metadata')} for i,d in enumerate(records)],
            'reused_E2_evidence': evidence, 'E2_summary': {'path': str(summary_path), 'sha256': sha(summary_path)},
            'E1_gate': {'path': str(e1_path), 'sha256': sha(e1_path)},
            'cost_only_preflight': cost, 'cpu_available_at_plan_seconds': available,
            'array_estimated_bytes': int(sum(len(v) for v in pairs.values())*L*records[0]['x'].shape[1]*4*4),
            'normalization': 'per-record FIT fixed input statistics; timestamp channel LayerNorm; no time pooling normalization',
            'independence': 'pooled synthetic record training is one model per mode/seed; ten evaluation records are not ten independent models',
            'decisions': {'fixed_better_self_wrong': 'candidate value of directed relationship summary',
                          'updated_better_only_change': 'conditional value of adaptation',
                          'similar': 'fixed graph may suffice', 'worse_than_E2': 'neural graph extra value not secured',
                          'stop': 'unsafe/mask/split/nonfinite/asset/budget/memory/concurrent GPU failure'},
            'failure_reason': None if status == 'READY' else 'Complete FULL_VAL balanced declared device scope cost forecast exceeds allocated remaining CPU; no origin/data/objective change',
            'source_pins': pins(), 'actual_science_updates': 0}
    path = HERE/'plans'/f'E6_{dataset}_{tag}.json'; write(path, plan)
    return path, plan


def rows(records, pairs, target=False, observed=False):
    array = []
    for record, origin in pairs:
        d = records[record]; x = d['mask'] if observed else d['x']
        array.append(x[origin:origin+H] if target else (
            x[origin-L:origin] if observed else prepare_history(d,origin)))
    return np.asarray(array, dtype=bool if observed else np.float32)


def prepare_segment(data, start, end):
    raw, observed = data['physical'][start:end],data['mask'][start:end]
    last = np.asarray(data['mean'],dtype=raw.dtype).copy()
    filled = np.empty_like(raw)
    for t in range(len(raw)):
        last[observed[t]] = raw[t,observed[t]]
        filled[t] = last
    return ((filled-data['mean'])/data['std']).astype(np.float32)


def prepare_history(data, origin):
    if origin-L < 0:
        raise ValueError('History before source record')
    return prepare_segment(data,origin-L,origin)


def selected_graph_lambda(records, pairs, budget):
    options, fixed_states = [], []
    for lam in LAMBDAS:
        models, graphs = [], []
        for d in records:
            end = d['fit_bounds'][1]; start = max(d['fit_bounds'][0], end-L)
            x, m = prepare_segment(d,start,end), d['mask'][start:end]
            destinations = np.arange(MAX_LAG, len(x))
            graph, by_mask_group = fit_graph(x,m,lam,return_models=True)
            if any(n < 2 for n in graph['fit_samples_by_destination']):
                raise ValueError('Insufficient observed FIXED graph regression targets')
            models.append(by_mask_group); graphs.append(graph); budget.check()
        p, y, m, scale = [], [], [], []
        for rid, o in pairs:
            d = records[rid]
            history = prepare_history(d,int(o))
            feature = history[-np.arange(1,MAX_LAG+1)].reshape(1,-1)
            prediction = np.zeros(d['x'].shape[1])
            for js,model in models[rid]:
                prediction[js] = model.predict(feature)[0]
            p.append(prediction)
            y.append(d['x'][o]); m.append(d['mask'][o]); scale.append(d['std'])
        metric = score(np.asarray(p)[:,None,:]*np.asarray(scale)[:,None,:],
                       np.asarray(y)[:,None,:]*np.asarray(scale)[:,None,:], np.asarray(m)[:,None,:])
        options.append({'lambda': lam, 'TUNE_one_step_physical_mse': metric['mse'],
                        'TUNE_one_step_physical_mae': metric['mae']})
        fixed_states.append(graphs)
    selected = min(range(len(options)), key=lambda i: options[i]['TUNE_one_step_physical_mse'])
    return LAMBDAS[selected], fixed_states[selected], options


def save_npz(path, **arrays):
    path = Path(path).resolve()
    if not path.is_relative_to(CACHE):
        raise ValueError('E6 arrays outside new campaign cache')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)
    return {'path': str(path), 'sha256': sha(path), 'array_hashes': {k: ahash(v) for k,v in arrays.items()}}


def save_torch(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(CACHE):
        raise ValueError('E6 checkpoint outside new campaign cache')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        torch.save(value, f)
    return {'path': str(path), 'sha256': sha(path)}


def precompute(records, pairs, fixed, lam, cache, output, budget):
    blocks, trace_refs, diagnostics = {}, {}, {}
    for split, chosen in pairs.items():
        history = rows(records, chosen)
        observed = rows(records, chosen, observed=True)
        messages = {mode: np.zeros_like(history) for mode in MODES if mode != 'SELF_ONLY'}
        graph_trace = {mode: [] for mode in messages}
        for i, (rid, origin) in enumerate(chosen):
            for mode in messages:
                message, trace = graph_messages(history[i], observed[i], mode, fixed[rid], lam)
                cutoff = trace['effective_cutoff_inclusive'].astype(np.int64)
                if mode == 'PAST_UPDATED_LAG_GRAPH':
                    cutoff = np.where(cutoff >= 0,cutoff+origin-L,-1)
                else:
                    cutoff[:] = records[rid]['fit_bounds'][1]-1
                trace['effective_cutoff_inclusive'] = cutoff
                messages[mode][i] = message
                graph_trace[mode].append(trace)
            if i % 16 == 0:
                budget.check()
                print(f'E6 graph preprocessing {split} {i+1}/{len(chosen)}', flush=True)
        blocks[split] = {'history': history, 'messages': messages}
        arrays = {'record_origin_pairs': chosen, 'input_history': history, 'input_mask': observed}
        for mode, value in messages.items():
            arrays[f'{mode}_neighbor_message'] = value
            for k in graph_trace[mode][0]:
                arrays[f'{mode}_trace_{k}'] = np.asarray([t[k] for t in graph_trace[mode]])
        diagnostics[split] = {}
        for mode, traces in graph_trace.items():
            source = np.asarray([t['source'] for t in traces])
            lag = np.asarray([t['lag'] for t in traces])
            weight = np.asarray([t['weight'] for t in traces])
            change = ((source[:,1:] != source[:,:-1]) | (lag[:,1:] != lag[:,:-1]))
            diag = {'mean_foreign_edge_id_change_per_recorded_event': float(change.mean()),
                    'mean_signed_weight_l1_change_per_destination_event': float(np.abs(np.diff(weight,axis=1)).sum(-1).mean()),
                    'events_include_short_prefix_start': True,
                    'change_is_not_success_or_causal_discovery': True}
            if records[0].get('metadata'):
                hits, totals = [], []
                for i,(rid,o) in enumerate(chosen):
                    meta = records[rid]['metadata']
                    changed = meta['change_timestamp_effective'] is not None and o-1 >= meta['change_timestamp_effective']
                    true_sources = meta['changed_sources'] if changed else meta['initial_sources']
                    true_lags = meta['changed_lags'] if changed else meta['initial_lags']
                    hits.append(sum(bool(np.any((source[i,-1,3+j] == s) & (lag[i,-1,3+j] == ell)))
                                    for j,(s,ell) in enumerate(zip(true_sources,true_lags))))
                    totals.append(len(true_sources))
                diag.update({'known_synthetic_receiver_edge_lag_hits_at_origin': int(sum(hits)),
                             'known_synthetic_receiver_edge_lag_possible': int(sum(totals)),
                             'known_edge_lag_recovery_fraction': float(sum(hits)/sum(totals)),
                             'known_relation_scope': 'synthetic generative directed lag, not inferred physical causality'})
            diagnostics[split][mode] = diag
        trace_refs[split] = save_npz(cache/f'GRAPH_INPUT_{split}.npz', **arrays)
        write(output/f'GRAPH_PREPROCESS_{split}.json', {'split': split, 'arrays': trace_refs[split],
              'graph_support_start': (chosen[:,1]-L).tolist(), 'input_end_inclusive': (chosen[:,1]-1).tolist(),
              'trace_cutoff_rule': 'absolute raw record timestamp; UPDATED short/no foreign=-1; FIXED/WRONG use frozen FIT artifact cutoff',
              'UPDATED_cutoff_rule': 'each row cutoff=origin-512+local observed index; never origin graph retroactively',
              'fixed_cutoffs_exclusive': [d['fit_bounds'][1] for d in records],
              'future_graph': 'held at history511; future observations never passed to graph_messages',
              'graph_diagnostics':diagnostics[split]})
    return blocks, trace_refs, diagnostics


def model_input(block, mode, indices=None):
    x = block['history'] if indices is None else block['history'][indices]
    msg = np.zeros_like(x) if mode == 'SELF_ONLY' else (
        block['messages'][mode] if indices is None else block['messages'][mode][indices])
    return np.concatenate((x, msg), axis=2)


def physical_metrics(records, pairs, pred, target, mask):
    mean = np.asarray([records[r]['mean'] for r,o in pairs])[:,None,:]
    std = np.asarray([records[r]['std'] for r,o in pairs])[:,None,:]
    metric = score(pred*std+mean, target*std+mean, mask)
    metric['normalized'] = score(pred,target,mask)
    metric['per_record'] = []
    for rid in np.unique(pairs[:,0]):
        idx = np.flatnonzero(pairs[:,0] == rid)
        d = records[rid]
        base = score(pred[idx]*d['std']+d['mean'], target[idx]*d['std']+d['mean'], mask[idx])
        base['record_id'] = int(rid)
        base['blocks'] = [score(pred[b]*d['std']+d['mean'], target[b]*d['std']+d['mean'],mask[b])
                          for b in np.array_split(idx, 4) if len(b)]
        change = d.get('metadata', {}).get('change_timestamp_effective')
        if change is not None:
            ts = pairs[idx,1,None]+np.arange(H)
            for name, valid in [('before_change',ts < change),('after_change',ts >= change)]:
                base[name] = score(pred[idx]*d['std']+d['mean'],target[idx]*d['std']+d['mean'], mask[idx] & valid[:,:,None])
        metric['per_record'].append(base)
    return metric


def readout_features(latent):
    return latent[:, -1-np.arange(33), :].reshape(len(latent), -1)


def matched_baselines(records,pairs,blocks,cache,output,budget):
    target = {s:rows(records,pairs[s],target=True) for s in pairs}
    observed = {s:rows(records,pairs[s],target=True,observed=True) for s in pairs}
    feat = {s:block['history'][:,-1-np.arange(33),:].reshape(len(block['history']),-1) for s,block in blocks.items()}
    scale = np.asarray([records[r]['std'] for r,o in pairs['tune']])[:,None,:]
    predicted,info,fits = ridge_grouped(feat['fit'],target['fit'],observed['fit'],feat['tune'],target['tune'],
                                      observed['tune'],feat['val'],score_scale=scale,budget=budget)
    tune_pred = predict_ridge(fits,feat['tune'],target['fit'].shape[-1])
    results = {}
    for method,p,pt in [('ALL_CHANNEL_LAG',predicted,tune_pred),('PERSISTENCE',
                         np.repeat(blocks['val']['history'][:,-1:,:],H,1),
                         np.repeat(blocks['tune']['history'][:,-1:,:],H,1))]:
        ref = save_npz(cache/f'MATCHED_BASELINE_{method}.npz',prediction=p,target=target['val'],mask=observed['val'],
                      pairs=pairs['val'],mean=np.asarray([records[r]['mean'] for r,o in pairs['val']]),
                      std=np.asarray([records[r]['std'] for r,o in pairs['val']]),tune_prediction=pt,tune_pairs=pairs['tune'])
        results[method] = {'val':physical_metrics(records,pairs['val'],p,target['val'],observed['val']),
                           'tune':physical_metrics(records,pairs['tune'],pt,target['tune'],observed['tune']),
                           'arrays':ref,'fit':info if method=='ALL_CHANNEL_LAG' else {'fitted_coefficients':0}}
    write(output/'MATCHED_BASELINES.json',{'models':results,'pair_hashes':{s:ahash(p) for s,p in pairs.items()},
          'synthetic_reference_difference':'Existing E2 separately fits each record; this new baseline pools exactly the E6 neural FIT256/TUNE64 rows',
          'optimizer_steps':0,'source_pins':pins()})
    return results


def evaluate_objective(e, head, inputs, target, mask, objective, device):
    x = torch.from_numpy(inputs).to(device)
    y = torch.from_numpy(target).to(device); m = torch.from_numpy(mask).to(device)
    z = e(x)
    p = head(z[:, -1]).reshape(len(x),H,-1) if objective == 'FORECAST' else head(z)
    if not bool(m.any()):
        raise ValueError('No observed objective targets')
    loss = ((p-y).square()*m).sum()/m.sum()
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite graph objective loss')
    return loss


def train_condition(mode, seed, initial, head_initial, batches, blocks, targets, masks_, plan, output, cache, budget, device):
    channels = targets['fit'].shape[-1]
    model = TinyEncoder(2*channels, causal=True).to(device); model.load_state_dict(initial)
    head = nn.Linear(16,H*channels) if plan['objective'] == 'FORECAST' else ReconstructionProbe(channels)
    head = head.to(device); head.load_state_dict(head_initial)
    opt = torch.optim.Adam(list(model.parameters())+list(head.parameters()), lr=.001)
    curves, checkpoints, start = [], [], time.perf_counter()
    for step in range(UPDATES+1):
        if step % 100 == 0:
            model.eval(); head.eval()
            with torch.no_grad():
                fit_loss = evaluate_objective(model,head,model_input(blocks['fit'],mode,batches[0]),
                    targets['fit'][batches[0]],masks_['fit'][batches[0]],plan['objective'],device)
                ti = np.arange(min(BATCH,len(targets['tune'])))
                tune_loss = evaluate_objective(model,head,model_input(blocks['tune'],mode,ti),
                    targets['tune'][ti],masks_['tune'][ti],plan['objective'],device)
            curve = {'update':step,'FIT_fixed_batch_objective_loss':float(fit_loss),
                     'TUNE_fixed_batch_objective_loss':float(tune_loss),'wall_seconds':time.perf_counter()-start}
            curves.append(curve); write(output/f'CURVE_u{step:04}.json',curve)
            checkpoints.append(save_torch(cache/f'E_u{step:04}.pt', {'encoder':model.state_dict(),
                'objective_head':head.state_dict(),'optimizer':opt.state_dict(),'step':step,'mode':mode,'seed':seed,
                'initial_hash':state_hash(initial),'batch_order_hash':ahash(batches)}))
            budget.check()
        if step == UPDATES:
            break
        model.train(); head.train(); opt.zero_grad(set_to_none=True)
        i = batches[step]
        loss = evaluate_objective(model,head,model_input(blocks['fit'],mode,i),targets['fit'][i],masks_['fit'][i],plan['objective'],device)
        loss.backward(); opt.step()
        if (step+1)%10 == 0:
            budget.check()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    training = {'actual_updates':UPDATES,'encoder_initial_hash':state_hash(initial),'head_initial_hash':state_hash(head_initial),
                'encoder_final_hash':state_hash(model.state_dict()),'batch_order_hash':ahash(batches),
                'actual_batch':BATCH,'samples_seen':UPDATES*BATCH,'encoder_timestamps_seen':UPDATES*BATCH*L,
                'future_target_timesteps_seen':UPDATES*BATCH*H if plan['objective'] == 'FORECAST' else 0,
                'encoder_parameters':model.parameter_count,'head_parameters':sum(p.numel() for p in head.parameters()),
                'curves':curves,'checkpoints':checkpoints,'wall_seconds':time.perf_counter()-start,
                'selection':'fixed600; no TUNE/VAL encoder checkpoint selection'}
    write(output/'TRAINING.json',training)
    return model, training


def exports(model, blocks, mode, device, budget):
    features = {}
    for split, block in blocks.items():
        z = np.empty((len(block['history']), 33*16), np.float32)
        for start in range(0,len(z),BATCH):
            idx = np.arange(start,min(start+BATCH,len(z)))
            with torch.no_grad():
                latent = model(torch.from_numpy(model_input(block,mode,idx)).to(device)).cpu().numpy()
            z[idx] = readout_features(latent); budget.check()
        features[split] = z
    return features


def predict_ridge(fits, features, channels):
    p = np.zeros((len(features),H*channels))
    for js, model in fits:
        p[:,js] = model.predict(features)
    return p.reshape(len(features),H,channels)


def benchmark(model, fits, records, pairs, fixed, lam, mode, device, budget):
    result = []
    for batch in (1,32):
        selected = pairs[np.arange(batch)%len(pairs)]
        samples, graph_seconds = [], []
        for iteration in range(25):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            started = time.perf_counter(); graph_start = time.perf_counter()
            xs = []
            for rid,o in selected:
                d = records[rid]
                history, observed = prepare_history(d,int(o)), d['mask'][o-L:o]
                msg,_ = graph_messages(history,observed,mode,fixed[rid],lam)
                xs.append(np.concatenate((history,msg),1))
            graph_elapsed = time.perf_counter()-graph_start
            with torch.no_grad():
                latent = model(torch.from_numpy(np.asarray(xs,np.float32)).to(device)).cpu().numpy()
            predict_ridge(fits,readout_features(latent),records[0]['x'].shape[1])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            if iteration >= 5:
                samples.append(elapsed); graph_seconds.append(graph_elapsed)
            budget.check()
        result.append({'batch':batch,'warmup':5,'timed':20,'input_repetitions':max(0,batch-len(pairs)),
                       'seconds':samples,'p50_seconds':float(np.median(samples)),'p95_seconds':float(np.percentile(samples,95)),
                       'uncached_graph_seconds':graph_seconds,'graph_included':True,
                       'scope':'observed physical history+mask -> window-local causal fill/FIT normalization+uncached graph+E+ridge+CPU output; file/model loading excluded',
                       'preprocessing_excluded':'one-time source TRAIN/VAL FP32 inverse and FIT statistic fitting are shared fitting/source preparation, outside the inference timer',
                       'cache':False})
    # Separate full-prefix graph update and a held-edge message event.
    d = records[int(pairs[0,0])]; o = int(pairs[0,1]); x, m = prepare_history(d,o),d['mask'][o-L:o]
    start = time.perf_counter(); g = fit_graph(x,m,lam); updated = time.perf_counter()-start
    start = time.perf_counter()
    for j in range(x.shape[1]):
        sum(weight*x[-1-lag,source] for source,lag,weight in zip(g['source'][j],g['lag'][j],g['weight'][j]) if source >= 0)
    held = time.perf_counter()-start
    return {'B1_B32':result,'single_last_prefix_graph_update_seconds':updated,
            'single_nonupdate_held_message_seconds':held,'update_event_scope':'one graph fit on already prepared history512, no neural operation'}


def check_execution(plan_path, digest, gate_path, gate_digest):
    if sha(plan_path) != digest or sha(gate_path) != gate_digest:
        raise ValueError('Frozen plan/execution gate changed')
    plan, gate = read(plan_path), read(gate_path)
    gpu = plan.get('compute_device','cuda') == 'cuda'
    required = ('execute','safety_pass','legacy_gpu_clear','single_gpu_dispatch_authorized') if gpu else (
        'execute','safety_pass','cpu_dispatch_authorized')
    if plan['status'] != 'READY' or gate.get('experiment') != 'E6' or not all(gate.get(k) is True for k in required):
        raise ValueError('E6 safety/cost/device dispatch gate not authorized')
    if gate.get('plan_sha256') != digest:
        raise ValueError('E6 gate bound to another plan')
    allocation = float(gate.get('gpu_allocation_seconds' if gpu else 'cpu_allocation_seconds',0))
    if not 0 < allocation <= remaining(gpu):
        raise ValueError('E6 device wall allocation missing or larger than remaining campaign cap')
    for path, expected in plan['source_pins'].items():
        if sha(path) != expected:
            raise ValueError(f'E6 frozen source changed: {path}')
    for reference in (plan['E1_gate'],plan['E2_summary']):
        if sha(reference['path']) != reference['sha256']:
            raise ValueError('E6 predecessor evidence changed')
    for p in (HERE/'resources').glob('*.json'):
        r = read(p)
        if gpu and r['gpu'] and r['status'] == 'RUNNING_CONFIRMED' and r['pid'] != os.getpid() and psutil.pid_exists(r['pid']):
            raise ValueError('Another new-campaign GPU job is live')
    return plan, gate


def run_balanced(plan_path, digest, gate_path, gate_digest, tag):
    plan, gate = check_execution(plan_path,digest,gate_path,gate_digest)
    records, _ = scope_records(plan['dataset'],plan['E2_summary']['path'])
    pairs = {s: np.asarray(v,np.int64) for s,v in plan['pairs'].items()}
    for s,v in pairs.items():
        if ahash(v) != plan['pair_hashes'][s]:
            raise ValueError('E6 origin/record order changed')
    output = HERE/'results'/f'E6_{plan["dataset"]}_{tag}'
    cache = CACHE/f'E6_{plan["dataset"]}_{tag}'
    output.mkdir(parents=True,exist_ok=False); cache.mkdir(parents=True,exist_ok=False)
    result = {'experiment':'E6','dataset':plan['dataset'],'status':'RUNNING_CONFIRMED','plan_path':str(plan_path),
              'plan_sha256':digest,'gate_path':str(gate_path),'gate_sha256':gate_digest,'conditions':[],
              'source_pins':pins(),'objective':plan['objective'],'pair_hashes':plan['pair_hashes'],
              'scope':'NEW_FIT FULL_VAL DEVELOPMENT_DIAGNOSTIC','TEST_access':0}
    try:
        cpu_allocation = min(float(gate.get('cpu_allocation_seconds',remaining(False))),remaining(False))
        with threadpool_limits(limits=2), AllocationBudget(2,gpu=False,label=f'E6 graph preprocessing {tag}',allocation=cpu_allocation) as cpu:
            if plan['cost_only_preflight']['graph_cpu_estimated_seconds_with_allowance'] > remaining(False):
                raise BudgetHeld('Actual remaining CPU no longer covers complete balanced precomputation')
            lam, fixed, options = selected_graph_lambda(records,pairs['tune'],cpu)
            fixed_ref = save_npz(cache/'FIXED_GRAPHS.npz',**{f'record{i}_{k}':np.asarray(v) for i,g in enumerate(fixed) for k,v in g.items()})
            write(output/'GRAPH_LAMBDA.json',{'selected_lambda':lam,'TUNE_grid':options,'fixed_graph_arrays':fixed_ref,
                  'normalization':'record-specific FIT input stats; local prefix feature scaler FIT only',
                  'future48_graph':'held at final observed timestamp; no actual future observations'})
            blocks, traces, graph_diagnostics = precompute(records,pairs,fixed,lam,cache,output,cpu)
            baseline = matched_baselines(records,pairs,blocks,cache,output,cpu)
            targets = {s:rows(records,pairs[s],target=plan['objective']=='FORECAST') for s in pairs}
            objective_masks = {s:rows(records,pairs[s],target=plan['objective']=='FORECAST',observed=True) for s in pairs}
            if plan['objective'] == 'REC':
                for split in ('tune','val'):
                    for i,(rid,o) in enumerate(pairs[split]):
                        floor = records[rid][f'{split}_bounds'][0]
                        objective_masks[split][i] &= (np.arange(o-L,o)[:,None] >= floor)
            future = {s:rows(records,pairs[s],target=True) for s in pairs}
            future_masks = {s:rows(records,pairs[s],target=True,observed=True) for s in pairs}
            result['graph_preprocessing_ledger'] = str(cpu.path)
        result['graph_preprocessing_ledger_sha256'] = sha(result['graph_preprocessing_ledger'])
        result['graph_arrays'] = traces
        result['graph_diagnostics'] = graph_diagnostics
        result['matched_baselines'] = baseline
        device = torch.device(plan.get('compute_device','cuda'))
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('BLOCKED_MISSING_GPU')
        torch.set_num_threads(plan['fixed']['CPU_threads']); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        neural_allocation = gate['gpu_allocation_seconds'] if device.type == 'cuda' else (
            cpu_allocation-cpu.record['wall_seconds'])
        with threadpool_limits(limits=2), AllocationBudget(2,gpu=device.type=='cuda',label=f'E6 balanced {tag}',allocation=neural_allocation) as gpu:
            for seed in SEEDS:
                torch.manual_seed(seed); initial = {k:v.clone() for k,v in TinyEncoder(2*records[0]['x'].shape[1]).state_dict().items()}
                torch.manual_seed(seed+17)
                head = nn.Linear(16,H*records[0]['x'].shape[1]) if plan['objective'] == 'FORECAST' else ReconstructionProbe(records[0]['x'].shape[1])
                head_initial = {k:v.clone() for k,v in head.state_dict().items()}
                rng = np.random.default_rng(seed+31)
                batches = np.stack([rng.choice(len(pairs['fit']),BATCH,replace=False) for _ in range(UPDATES)])
                save_npz(cache/f'BATCH_ORDER_s{seed}.npz',indices=batches,record_origin_pairs=pairs['fit'][batches])
                for mode in MODES:
                    if time.perf_counter()-gpu.start >= neural_allocation:
                        raise BudgetHeld('Root allocated neural wall reached before completing balanced scope')
                    out, cp = output/f'{mode}_s{seed}',cache/f'{mode}_s{seed}'; out.mkdir(); cp.mkdir()
                    model, training = train_condition(mode,seed,initial,head_initial,batches,blocks,targets,objective_masks,
                                                      plan,out,cp,gpu,device)
                    feat = exports(model,blocks,mode,device,gpu)
                    tune_scale = np.asarray([records[r]['std'] for r,o in pairs['tune']])[:,None,:]
                    predicted,ridge,fits = ridge_grouped(feat['fit'],future['fit'],future_masks['fit'],feat['tune'],future['tune'],
                        future_masks['tune'],feat['val'],score_scale=tune_scale,budget=gpu)
                    p_tune = predict_ridge(fits,feat['tune'],future['fit'].shape[-1])
                    ridge_arrays = {}
                    for i,(js,fitted) in enumerate(fits):
                        ridge_arrays.update({f'group{i}_output_indices':np.asarray(js,np.int64),
                            f'group{i}_feature_mean':fitted.mean,f'group{i}_feature_std':fitted.std,
                            f'group{i}_target_mean':fitted.ym,f'group{i}_FIT_features':fitted.train,
                            f'group{i}_coefficient' if fitted.coef is not None else f'group{i}_dual':
                                fitted.coef if fitted.coef is not None else fitted.dual})
                    ridge_ref = save_npz(cp/'RIDGE_SELECTED_STATE.npz',**ridge_arrays)
                    metrics = physical_metrics(records,pairs['val'],predicted,future['val'],future_masks['val'])
                    tune_metric = physical_metrics(records,pairs['tune'],p_tune,future['tune'],future_masks['tune'])
                    pred_ref = save_npz(cp/'PREDICTIONS.npz',prediction=predicted,target=future['val'],mask=future_masks['val'],
                        pairs=pairs['val'],mean=np.asarray([records[r]['mean'] for r,o in pairs['val']]),
                        std=np.asarray([records[r]['std'] for r,o in pairs['val']]),tune_prediction=p_tune,
                        tune_target=future['tune'],tune_mask=future_masks['tune'],tune_pairs=pairs['tune'])
                    cost = benchmark(model,fits,records,pairs['val'],fixed,lam,mode,device,gpu)
                    record = {'experiment_id':'E6','dataset':plan['dataset'],'seed':seed,'mode':mode,'status':'COMPLETE',
                        'objective':plan['objective'],'encoder_family':'TinyCausalGraphTemporalPrototype','latent_shape':[L,16],
                        'compute_device':str(device),
                        'predictor':'FIT_ONLY_DIRECT_RIDGE_LAST33','decoder':'not used in common primary future readout',
                        'trainable_parts':'E + common objective head; E frozen for new common ridge',
                        'training':training,'ridge':ridge,'tune':tune_metric,'val':metrics,'arrays':pred_ref,'cost':cost,
                        'ridge_selected_state':ridge_ref,'source_revision':read(HERE/'PROTOCOL.json')['HEAD'],
                        'fit_bounds':[{str(i):d['fit_bounds']} for i,d in enumerate(records)],
                        'tune_bounds':[{str(i):d['tune_bounds']} for i,d in enumerate(records)],
                        'eval_bounds':[{str(i):d['val_bounds']} for i,d in enumerate(records)],
                        'input_support':'origin-512 through origin-1; dynamic graphs use only prefixes with same start',
                        'latent_timestamp_rule':'z_s at observed timestamp s; causalRF63 plus causal graph estimation',
                        'target_mask_hash':ahash(future_masks['val']),'forecast_origins_hash':ahash(pairs['val']),
                        'graph_fit_cutoff':'per-record FIT end for FIXED/WRONG; local-prefix last observed update tick for UPDATED',
                        'graph_arrays':traces,'normalization_fit_scope':'FIT_ONLY_PER_RECORD',
                        'graph_diagnostics':graph_diagnostics['val'].get(mode,{'no_foreign_graph':True}),
                        'no_history_retroactive_graph':True,'future_graph_held':True,'source_pins':pins(),
                        'remaining_uncertainties':['repeated development VAL','only2 neural seeds','fixed600 not convergence proof',
                            'raw self branch is multichannel; graph delta is extra relational message',
                            'pooled records are not independent model repetitions']}
                    write(out/'RESULT.json',record)
                    result['conditions'].append({'mode':mode,'seed':seed,'path':str(out/'RESULT.json'),
                                                'sha256':sha(out/'RESULT.json'),'val':metrics,'tune':tune_metric})
                    write(output/'PROGRESS.json',result,replace=True)
                    print(f'E6 {plan["dataset"]} {mode} seed={seed} MSE={metrics["mse"]:.9g} MAE={metrics["mae"]:.9g}',flush=True)
                    del model,feat
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gpu.check()
            result['neural_budget_ledger'] = str(gpu.path)
        result['neural_budget_ledger_sha256'] = sha(result['neural_budget_ledger'])
        result['compute_device'] = str(device)
        result['status'] = 'COMPLETE_BALANCED'
        result['interpretation'] = 'Saved raw predictions require independent verification; gate/cost safety does not establish predictive benefit'
        write(output/'RESULT.json',result)
        return output/'RESULT.json'
    except BaseException as error:
        result['status'] = 'BUDGET_HELD_PARTIAL' if isinstance(error,BudgetHeld) else 'FAILED_PRESERVED'
        result['failure_reason'] = str(error); write(output/'FAILURE.json',result)
        raise


def main():
    parser = ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--plan',action='store_true'); group.add_argument('--run',action='store_true')
    parser.add_argument('--dataset',choices=['synthetic_fixed','synthetic_changed','jena','robin'],default='synthetic_fixed')
    parser.add_argument('--e2-summary',type=Path); parser.add_argument('--e1-gate',type=Path)
    parser.add_argument('--e5-result',type=Path); parser.add_argument('--cpu-allocation-seconds',type=float)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--tag',required=True); parser.add_argument('--frozen-plan',type=Path)
    parser.add_argument('--plan-sha256'); parser.add_argument('--gate',type=Path); parser.add_argument('--gate-sha256')
    args = parser.parse_args()
    if not args.tag.replace('_','').replace('-','').isalnum():
        raise ValueError('Use a fresh simple attempt tag')
    access_guard()
    if args.plan:
        if args.e2_summary is None or args.e1_gate is None:
            parser.error('--plan requires --e2-summary and --e1-gate')
        with threadpool_limits(limits=2), Budget(2,gpu=False,label=f'E6 cost-only plan {args.tag}') as budget:
            path,plan = create_plan(args.dataset,args.e2_summary,args.e1_gate,args.e5_result,args.tag,args.cpu_allocation_seconds,budget,args.device)
        print(json.dumps({'plan':str(path),'sha256':sha(path),'status':plan['status'],'cost':plan['cost_only_preflight']},ensure_ascii=False))
    else:
        if any(v is None for v in [args.frozen_plan,args.plan_sha256,args.gate,args.gate_sha256]):
            parser.error('--run requires frozen plan and gate paths with their SHA256')
        path = run_balanced(args.frozen_plan,args.plan_sha256,args.gate,args.gate_sha256,args.tag)
        print(json.dumps({'result':str(path),'sha256':sha(path)},ensure_ascii=False))


if __name__ == '__main__':
    main()
