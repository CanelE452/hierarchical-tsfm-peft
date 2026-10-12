"""Saved-state runtime closure only: no fitting, selection, loss, or TEST reads."""
import argparse
import gc
import re
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from common import (Budget, BudgetHeld, CACHE, H, HERE, L, ROOT, access_guard,
                    ahash, jsonable, load_data, read, sha, windows, write)

METHODS = ('LATENT_PERSISTENCE', 'LATENT_SELF_RIDGE', 'LATENT_JOINT_RIDGE', 'DIRECT_RAW_RIDGE')
PARTS = ('e2_cpu', 'e4_gpu', 'e5_gpu')
SOURCES = ('cost_closure_v1.py', 'common.py', 'models_io.py', 'model_neural.py', 'e2.py', 'e5.py')


def pin(plan, path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f'Bound saved asset changed: {path}')
    plan['pins'][str(path)] = digest
    return str(path)


def build_plan(part, e5_results=()):
    plan = {'experiment': 'COST_CLOSURE', 'part': part, 'status': 'PLANNED_COST_ONLY',
            'phase': 1 if part == 'e4_gpu' else 2, 'gpu': part != 'e2_cpu',
            'L': L, 'H': H, 'batches': [1, 32], 'warmup': 5, 'timed': 20,
            'pins': {}, 'cases': [], 'science_updates': 0, 'scores_computed': False,
            'input_scope': 'Preloaded physical TRAIN/VAL history only; model/statistics/disk loading excluded',
            'parity_tolerance_normalized_atol': 1e-4,
            'physical_postprocessing_included_new_timings': True,
            'device_runtime_warning': 'CPU and GPU measurements are not interchangeable',
            'selection': 'Already selected saved coefficients and E600/D checkpoints; no new selection'}
    for name in SOURCES:
        pin(plan, HERE/name)
    for ds in ('jena', 'robin'):
        contract_path = ROOT/f'research/level_matched_learned_ed_v18_20261006/data_contract_{ds}.json'
        contract = read(contract_path)
        pin(plan, contract_path)
        pin(plan, ROOT/contract['trainval']['path'], contract['trainval']['sha256'])
    if part == 'e2_cpu':
        for ds in ('jena', 'robin'):
            path = HERE/'e2'/f'run01_{ds}.json'
            r = read(path)
            if r['status'] != 'COMPLETE':
                raise ValueError('E2 saved run is incomplete')
            pin(plan, path)
            shared = r['shared_arrays']
            pin(plan, shared['path'], shared['sha256'])
            for method, row in r['models'].items():
                pin(plan, row['arrays']['path'], row['arrays']['sha256'])
            plan['cases'].append({'dataset': ds, 'result': str(path),
                                  'state': shared['path'], 'methods': list(r['models'])})
        plan['cost_forecast_seconds'] = 50
    elif part == 'e4_gpu':
        files = sorted(p for p in (HERE/'e34').glob('*.json') if not p.name.startswith('SUMMARY_'))
        for path in files:
            r = read(path)
            if r.get('experiment_id') != 'E3_E4' or not r['status'].startswith('COMPLETE'):
                continue
            pin(plan, path)
            folder = Path(r['folder'])
            prefix = next(v for v in r['E3'] if v['method'] == 'PREFIX_BOLT_REC_D')
            decoder = folder/'PREFIX'/f'D_{prefix["decoder"]["selected_update"]}.pt'
            pin(plan, decoder)
            pin(plan, folder/'PREFIX'/'EXPORTS.npz')
            cp = r['checkpoint_record']
            pin(plan, cp['checkpoint'], cp['checkpoint_sha256'])
            pin(plan, cp['result'], cp['result_sha256'])
            for method in METHODS:
                row = next(v for v in r['E4'] if v['method'] == method)
                pin(plan, row['prediction_path'], row['prediction_hash'])
                if method != 'LATENT_PERSISTENCE':
                    pin(plan, folder/'E4'/f'{method}_FIT_STATE.npz')
            plan['cases'].append({'dataset': r['dataset'], 'dim': r['dim'], 'seed': r['seed'],
                                  'result': str(path), 'folder': str(folder), 'decoder': str(decoder),
                                  'selected_D_hash': prefix['decoder']['selected_D_hash']})
        if len(plan['cases']) != 8:
            raise ValueError('All eight completed E3/E4 cases are required')
        rep = next(v for v in plan['cases'] if (v['dataset'], v['dim'], v['seed']) == ('jena', 320, 92601))
        r = read(rep['result'])
        bench = next(v for v in r['E3'] if v['method'] == 'PREFIX_BOLT_REC_D')['end_to_end_cost']
        estimate = 4*25*sum(v['p50_seconds'] for v in bench) + 60
        plan['representative_end_to_end_case'] = rep
        plan['cost_forecast_seconds'] = estimate
        plan['forecast_basis'] = 'Four native paths times 25 repetitions times existing actual PREFIX+BOLT+D medians B1+B32, plus60s; forecast only'
        plan['reused_BOLT_cost'] = {'source_result': rep['result'], 'timings': bench,
            'timer_boundary': 'Original causal ffill+normalization+PREFIX+Bolt+D+normalized CPUoutput; physical inverse excluded',
            'comparable_timer_to_new_measurements': False,
            'd320_note': '320 exceeds raw channel count; this is expansion, not compression'}
    else:
        if not e5_results:
            raise ValueError('Supply completed Jena and Robin E5 RESULT.json paths')
        for path in e5_results:
            path = Path(path).resolve()
            r = read(path)
            if r.get('experiment') != 'E5' or r['status'] != 'COMPLETE_BALANCED':
                raise ValueError('Only completed balanced E5 states can be benchmarked')
            pin(plan, path)
            pin(plan, r['plan_path'], r['plan_sha256'])
            for child in r['conditions']:
                pin(plan, child['path'], child['sha256'])
                c = read(child['path'])
                for key, hk in [('encoder_final_path', 'encoder_final_sha256'),
                                ('ridge_coefficients_path', 'ridge_coefficients_sha256'),
                                ('prediction_path', 'prediction_sha256')]:
                    pin(plan, c[key], c[hk])
                plan['cases'].append({'dataset': r['dataset'], 'condition': c['condition'],
                                      'seed': c['seed'], 'result': child['path'],
                                      'parent_result': str(path)})
        if len(plan['cases']) != 16 or {v['dataset'] for v in plan['cases']} != {'jena', 'robin'}:
            raise ValueError('Both datasets, all four objectives, and two seeds are required')
        plan['cost_forecast_seconds'] = 180
    return plan


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def ridge_groups(state, style):
    pattern = r'^group(\d+)_outputs$' if style == 'group' else r'^g(\d+)_outputs$'
    ids = sorted(int(m.group(1)) for k in state if (m := re.match(pattern, k)))
    if not ids:
        raise ValueError('BLOCKED_SAVED_STATE: no grouped coefficients')
    groups = []
    for i in ids:
        prefix = f'group{i}_' if style == 'group' else f'g{i}_'
        groups.append({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    return groups


def predict_groups(features, groups, output_count):
    prediction = np.zeros((len(features), output_count), dtype=np.float64)
    for g in groups:
        v = (features-g['mean'])/g['std']
        y = v@g['coef']+g['ym'] if 'coef' in g else v@g['train'].T@g['dual']+g['ym']
        prediction[:, g['outputs'].astype(int)] = y
    return prediction


def take_raw_history(data, origins):
    raw = np.stack([data['physical'][int(o)-L:int(o)] for o in origins])
    observed = np.stack([data['mask'][int(o)-L:int(o)] for o in origins])
    if data['normalization_fit_scope'] == 'FIT_ONLY':
        initial = np.broadcast_to(data['mean'], (len(origins), len(data['mean']))).copy()
    else:
        initial = np.stack([data['filled'][int(o)-L-1] if o-L > 0 else data['mean'] for o in origins])
    filled = np.empty_like(raw)
    last = initial
    for t in range(L):
        last[observed[:, t]] = raw[:, t][observed[:, t]]
        filled[:, t] = last
    return ((filled-data['mean'])/data['std']).astype(np.float32)


def physical_output(normalized, data):
    return (np.asarray(normalized)*data['std']+data['mean']).copy()


def parity(normalized, expected, data):
    delta = np.asarray(normalized, dtype=float)-np.asarray(expected, dtype=float)
    maximum = float(np.max(np.abs(delta)))
    return {'normalized_max_abs': maximum, 'physical_max_abs': float(np.max(np.abs(delta*data['std']))),
            'normalized_atol': 1e-4, 'status': 'PASS' if maximum <= 1e-4 else 'FLAG_DEVICE_OR_NUMERICAL_DIFFERENCE',
            'new_performance_score_computed': False}


def measure(fn, batch, budget, local_start, allocation, gpu):
    import torch
    def check():
        budget.check()
        if time.perf_counter()-local_start >= allocation:
            raise BudgetHeld('Local saved-state cost allocation exhausted')
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated() if gpu else 0
    samples = []
    for rep in range(25):
        check()
        if gpu:
            torch.cuda.synchronize()
        started = time.perf_counter()
        out = fn()
        if gpu:
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        if not isinstance(out, np.ndarray) or not np.isfinite(out).all():
            raise ValueError('Cost output is not finite CPU numpy')
        if rep >= 5:
            samples.append(elapsed)
        check()
    return {'batch': batch, 'warmup': 5, 'timed': 20, 'samples_seconds': samples,
            'p50_seconds': float(np.median(samples)), 'p95_seconds': float(np.quantile(samples, .95)),
            'seconds_per_window_p50': float(np.median(samples))/batch,
            'device': 'cuda' if gpu else 'cpu', 'physical_inverse_and_CPU_output_included': True,
            'baseline_allocated_vram': baseline_allocated,
            'peak_allocated_vram': torch.cuda.max_memory_allocated() if gpu else 0,
            'peak_reserved_vram': torch.cuda.max_memory_reserved() if gpu else 0,
            'vram_peak_scope': 'Measurement-reset peak includes persistent saved model footprint; reserved allocator may retain earlier cases',
            'model_loading_disk_and_guards_excluded': True, 'output_shape': list(out.shape)}


def data_binding(data, expected_mean=None, expected_std=None):
    if expected_mean is not None and not np.array_equal(data['mean'], np.asarray(expected_mean, dtype=np.float32)):
        raise ValueError('Frozen normalization mean differs')
    if expected_std is not None and not np.array_equal(data['std'], np.asarray(expected_std, dtype=np.float32)):
        raise ValueError('Frozen normalization std differs')
    return {'source': data['source'], 'source_sha256': data['source_hash'],
            'normalization_fit_scope': data['normalization_fit_scope'],
            'normalization_mean_hash': ahash(data['mean']), 'normalization_std_hash': ahash(data['std']),
            'history': '[o-512:o]', 'first_future': 'o', 'L': L, 'H': H}


def e2_cost(plan, budget, started, allocation, rows):
    from e2 import feature_block
    for case in plan['cases']:
        r, state = read(case['result']), load_arrays(case['state'])
        data = load_data(case['dataset'])
        binding = data_binding(data, state['mean'], state['std'])
        origins = state['val_origins']
        if not np.array_equal(origins, data['origins']['val']):
            raise ValueError('E2 origin binding differs')
        for method in case['methods']:
            groups = []
            for key in state:
                match = re.match(r'^'+method+r'_(\d+)_outputs$', key)
                if match:
                    prefix = f'{method}_{match.group(1)}_'
                    groups.append({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
            def head(x):
                if method == 'PERSISTENCE':
                    return np.repeat(x[:, -1:], H, 1)
                pred = np.zeros((len(x), H*len(data['mean'])))
                for g in groups:
                    c = None if method == 'ALL_CHANNEL_LAG' else int(g['outputs'][0]) % len(data['mean'])
                    f = (feature_block(x, method, c)-g['feature_mean'])/g['feature_std']
                    y = f@g['coefficient'] if 'coefficient' in g else f@g['train_standardized'].T@g['dual']
                    pred[:, g['outputs'].astype(int)] = y+g['target_mean']
                return pred.reshape(len(x), H, -1)
            saved = load_arrays(r['models'][method]['arrays']['path'])['prediction_val']
            for n in plan['batches']:
                index = np.arange(n) % len(origins)
                oo = origins[index]
                x = take_raw_history(data, oo)
                check = parity(head(x), saved[index], data)
                row = {'experiment': 'E2', 'dataset': case['dataset'], 'method': method,
                       'binding': binding, 'origins': oo.tolist(), 'origin_hash': ahash(oo), 'parity': check,
                       'unique_inputs': min(n, len(origins)), 'repeated_inputs': max(0, n-len(origins)),
                       'end_to_end': measure(lambda: physical_output(head(take_raw_history(data, oo)), data), n, budget, started, allocation, False),
                       'cached_head_only': measure(lambda: physical_output(head(x), data), n, budget, started, allocation, False),
                       'input_imputation': 'Per-window FIT mean initialization and causal observed-value forward fill'}
                rows.append(row)
                print(f'COST E2 {case["dataset"]} {method} B{n} SAVED', flush=True)
    return rows


def load_e4_native(case):
    import torch
    from models_io import ReconstructionDecoder, state_digest
    r = read(case['result'])
    folder = Path(case['folder'])
    with np.load(folder/'E4'/'LATENT_PERSISTENCE.npz', allow_pickle=False) as z:
        channels = z['prediction'].shape[-1]
    decoder = ReconstructionDecoder(case['dim'], channels).to('cuda')
    decoder.load_state_dict(torch.load(case['decoder'], map_location='cpu', weights_only=True)['state_dict'])
    decoder.eval().requires_grad_(False)
    if state_digest(decoder) != case['selected_D_hash']:
        raise ValueError('Saved selected PREFIX decoder state differs')
    states = {m: ridge_groups(load_arrays(folder/'E4'/f'{m}_FIT_STATE.npz'), 'g' if m == 'DIRECT_RAW_RIDGE' else 'group')
              for m in METHODS if m != 'LATENT_PERSISTENCE'}
    @torch.no_grad()
    def head(z, method):
        if method == 'LATENT_PERSISTENCE':
            p = np.repeat(z[:, -1:], H, 1)
        elif method == 'LATENT_SELF_RIDGE':
            p = np.zeros((len(z), H*case['dim']))
            lag = z[:, -1-np.arange(33)]
            for g in states[method]:
                c = int(g['outputs'][0]) % case['dim']
                f = (lag[:, :, c]-g['mean'])/g['std']
                y = f@g['coef']+g['ym'] if 'coef' in g else f@g['train'].T@g['dual']+g['ym']
                p[:, g['outputs'].astype(int)] = y
            p = p.reshape(len(z), H, case['dim'])
        else:
            f = z[:, -1-np.arange(33)].reshape(len(z), -1)
            count = H*(channels if method == 'DIRECT_RAW_RIDGE' else case['dim'])
            p = predict_groups(f, states[method], count).reshape(len(z), H, -1)
        if method == 'DIRECT_RAW_RIDGE':
            return p
        return decoder(torch.as_tensor(p, device='cuda', dtype=torch.float32)).cpu().numpy()
    return head, decoder


def e4_cost(plan, budget, started, allocation, rows):
    from models_io import encode_prefix, load_ts2, state_digest
    for case in plan['cases']:
        data = load_data(case['dataset'], frozen=True, sizes=(64, 32, 32))
        folder = Path(case['folder'])
        with np.load(folder/'PREFIX'/'EXPORTS.npz', allow_pickle=False) as zz:
            latent = zz['val']
        head, decoder = load_e4_native(case)
        for method in METHODS:
            row = next(v for v in read(case['result'])['E4'] if v['method'] == method)
            with np.load(row['prediction_path'], allow_pickle=False) as z:
                saved, origins = z['prediction'], z['origins']
                binding = data_binding(data, z['mean'], z['std'])
            if not np.array_equal(origins, data['origins']['val']):
                raise ValueError('E4 saved origin contract differs')
            for n in plan['batches']:
                index = np.arange(n) % len(origins)
                z = latent[index]
                cost = measure(lambda: physical_output(head(z, method), data), n, budget, started, allocation, True)
                rows.append({'experiment': 'E4', 'dataset': case['dataset'], 'dim': case['dim'], 'seed': case['seed'],
                    'method': method, 'measurement': 'CACHED_PREFIX_HEAD_ONLY', 'cached_head_only': cost,
                    'parity': parity(head(z, method), saved[index], data), 'binding': binding,
                    'origins': origins[index].tolist(), 'origin_hash': ahash(origins[index]),
                    'encoder_excluded': True, 'dim320_is_expansion': case['dim'] == 320})
        del head, decoder, latent
        gc.collect()
        print(f'COST E4 cached {case["dataset"]} d{case["dim"]} s{case["seed"]} SAVED', flush=True)
    case = plan['representative_end_to_end_case']
    data = load_data(case['dataset'], frozen=True, sizes=(64, 32, 32))
    encoder, record = load_ts2(case['dataset'], case['dim'], case['seed'], 'cuda')
    if record['checkpoint_sha256'] != plan['pins'][str(Path(record['checkpoint']).resolve())]:
        raise ValueError('Representative encoder checkpoint binding differs')
    before = state_digest(encoder)
    head, decoder = load_e4_native(case)
    for method in METHODS:
        row = next(v for v in read(case['result'])['E4'] if v['method'] == method)
        with np.load(row['prediction_path'], allow_pickle=False) as z:
            saved, origins = z['prediction'], z['origins']
        for n in plan['batches']:
            index = np.arange(n) % len(origins)
            oo = origins[index]
            def normalized():
                return head(encode_prefix(encoder, take_raw_history(data, oo), 'cuda', batch=32), method)
            check = parity(normalized(), saved[index], data)
            cost = measure(lambda: physical_output(normalized(), data), n, budget, started, allocation, True)
            rows.append({'experiment': 'E4', 'dataset': case['dataset'], 'dim': case['dim'], 'seed': case['seed'],
                'method': method, 'measurement': 'REPRESENTATIVE_FULL_PREFIX_NATIVE', 'end_to_end': cost,
                'parity': check, 'binding': data_binding(data), 'origins': oo.tolist(), 'origin_hash': ahash(oo),
                'input_imputation': 'Carried observed physical state at a-1; TRAIN_ALL frozen normalizer',
                'dim320_is_expansion': True, 'prefix': 'Same-a exact-length prefix at every one of512timestamps'})
            print(f'COST E4 full PREFIX {method} B{n} SAVED', flush=True)
    if state_digest(encoder) != before:
        raise ValueError('Frozen representative encoder changed')
    return rows


def e5_cost(plan, budget, started, allocation, rows):
    import torch
    from model_neural import TinyEncoder
    from e5 import state_hash
    data_by_dataset = {}
    for case in plan['cases']:
        r = read(case['result'])
        data = data_by_dataset.setdefault(case['dataset'], None)
        if data is None:
            data = load_data(case['dataset'])
            data_by_dataset[case['dataset']] = data
        binding = data_binding(data, r['fit_normalization_mean'], r['fit_normalization_std'])
        model = TinyEncoder(len(data['mean']), causal=case['condition'] != 'FULL_REC').to('cuda')
        checkpoint = torch.load(r['encoder_final_path'], map_location='cpu', weights_only=True)
        if checkpoint['step'] != 600 or checkpoint['condition'] != case['condition'] or checkpoint['seed'] != case['seed']:
            raise ValueError('E5 E600 identity differs')
        model.load_state_dict(checkpoint['encoder'])
        model.eval().requires_grad_(False)
        encoder_state_before = state_hash(model.state_dict())
        groups = ridge_groups(load_arrays(r['ridge_coefficients_path']), 'group')
        with np.load(r['prediction_path'], allow_pickle=False) as z:
            saved, origins = z['prediction'], z['origins']
        if not np.array_equal(origins, data['origins']['val']):
            raise ValueError('E5 saved origin contract differs')
        @torch.no_grad()
        def encode(x):
            return model(torch.as_tensor(x, device='cuda', dtype=torch.float32)).cpu().numpy()
        def head(z):
            f = z[:, -1-np.arange(33)].reshape(len(z), -1)
            return predict_groups(f, groups, H*len(data['mean'])).reshape(len(z), H, -1)
        raw_x = take_raw_history(data, origins[:1])
        perturbed = raw_x.copy()
        observed = data['mask'][int(origins[0])-L:int(origins[0])]
        perturb_mask = observed & (np.arange(L)[:, None] > 256)
        perturbed[0][perturb_mask] += np.float32(3)
        original_latent, changed_latent = encode(raw_x), encode(perturbed)
        anchor_delta = float(np.max(np.abs(original_latent[:, 256]-changed_latent[:, 256])))
        causal = case['condition'] != 'FULL_REC'
        causality_check = {'test': 'E600 local timestamp256; observed inputs after256 perturbed by+3 normalized units',
            'anchor_local_timestamp': 256, 'changed_after_local_timestamp': 256,
            'maximum_anchor_latent_delta': anchor_delta, 'causal_expected': causal, 'atol': 1e-5,
            'status': ('PASS' if anchor_delta <= 1e-5 else 'FAIL_CAUSAL_HISTORY_REVISION') if causal else 'FULL_CONTEXT_REVISION_DIAGNOSTIC',
            'interpretation': 'Within-history revision test B; not actual forecast-origin future leakage A',
            'actual_future_o_cold_global_loader_test': 'NOT_PERFORMED_IN_THIS_COST_CLOSURE',
            'forecast_input_end_inclusive': int(origins[0])-1, 'origin': int(origins[0]),
            'encoder_state_before': encoder_state_before}
        budget.check()
        for n in plan['batches']:
            index = np.arange(n) % len(origins)
            oo = origins[index]
            latent = encode(take_raw_history(data, oo))
            check = parity(head(latent), saved[index], data)
            full = measure(lambda: physical_output(head(encode(take_raw_history(data, oo))), data), n, budget, started, allocation, True)
            cached = measure(lambda: physical_output(head(latent), data), n, budget, started, allocation, True)
            rows.append({'experiment': 'E5', 'dataset': case['dataset'], 'condition': case['condition'],
                'seed': case['seed'], 'end_to_end': full, 'cached_head_only': cached, 'parity': check,
                'binding': binding, 'origins': oo.tolist(), 'origin_hash': ahash(oo),
                'input_imputation': 'Per-window FIT mean initialization and causal forward fill',
                'encoder_padding': 'symmetric FULL' if case['condition'] == 'FULL_REC' else 'left-only CAUSAL',
                'trained_encoder_history_causality': causality_check,
                'decoder_used': False, 'unique_inputs': min(n, len(origins)), 'repeated_inputs': max(0, n-len(origins))})
        encoder_state_after = state_hash(model.state_dict())
        if encoder_state_after != encoder_state_before:
            raise ValueError('Frozen E600 encoder changed during runtime/causality checks')
        causality_check['encoder_state_after'] = encoder_state_after
        causality_check['encoder_unchanged'] = True
        del model, checkpoint, latent, groups
        gc.collect()
        print(f'COST E5 {case["dataset"]} {case["condition"]} s{case["seed"]} SAVED', flush=True)
    return rows


def main():
    import json
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--part', choices=PARTS)
    parser.add_argument('--e5-results', type=Path, nargs='*', default=[])
    parser.add_argument('--frozen-plan', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--gate-sha256')
    parser.add_argument('--tag')
    args = parser.parse_args()
    access_guard()
    if args.dry_run:
        if args.part is None:
            parser.error('--dry-run requires --part')
        print(json.dumps(build_plan(args.part, args.e5_results), ensure_ascii=False, default=jsonable))
        return
    if any(v is None for v in (args.frozen_plan, args.plan_sha256, args.gate, args.gate_sha256, args.tag)):
        parser.error('--run requires frozen plan+SHA, gate+SHA, and fresh tag')
    if not args.tag.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Fresh tag must be a filename-safe identifier')
    if sha(args.frozen_plan) != args.plan_sha256 or sha(args.gate) != args.gate_sha256:
        raise ValueError('Cost plan/gate bytes changed')
    plan, gate = read(args.frozen_plan), read(args.gate)
    if plan['part'] not in PARTS or (args.part and args.part != plan['part']):
        raise ValueError('Cost part mismatch')
    if gate.get('experiment') != 'COST_CLOSURE' or gate.get('execute') is not True or gate.get('safety_pass') is not True:
        raise ValueError('Explicit Root safety/execute gate required')
    if gate.get('plan_sha256') != args.plan_sha256 or gate.get('part') != plan['part']:
        raise ValueError('Gate must bind this exact cost plan and part')
    if plan['gpu'] and (gate.get('gpu_exclusive') is not True or gate.get('legacy_gpu_clear') is not True):
        raise ValueError('Root-exclusive GPU dispatch and legacy terminal evidence required')
    allocation = float(gate['allocation_seconds'])
    if allocation <= 0 or allocation > {'e2_cpu': 70, 'e4_gpu': 1000, 'e5_gpu': 200}[plan['part']]:
        raise ValueError('Local cost allocation exceeds approved representative limit')
    output = HERE/'results'/f'COST_CLOSURE_{plan["part"]}_{args.tag}'
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'experiment': 'COST_CLOSURE', 'part': plan['part'], 'status': 'RUNNING_CONFIRMED',
        'plan_path': str(args.frozen_plan), 'plan_sha256': args.plan_sha256,
        'gate_path': str(args.gate), 'gate_sha256': args.gate_sha256, 'rows': [],
        'science_updates': 0, 'scores_computed': False, 'allocation_seconds': allocation,
        'pins': plan['pins'], 'timer_contract': 'Preloaded physicalrawhistory extraction+causalffill+declarednormalization+savedE/P/D+physicalinverse+CPUoutput; loading/disk/guards excluded',
        'cached_head_contract': 'Already computed latent/normalizedhistory→savedhead→physicalinverse→CPUoutput; E/raw preprocessing excluded',
        'runtime_comparison_limit': 'CPU/GPU and different batch/timer boundaries must be shown separately'}
    try:
        import torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with threadpool_limits(2), Budget(plan['phase'], plan['gpu'], f'Saved runtime closure {plan["part"]}') as budget:
            started = time.perf_counter()
            for path, digest in plan['pins'].items():
                if sha(path) != digest:
                    raise ValueError(f'Pinned saved asset differs: {path}')
            functions = {'e2_cpu': e2_cost, 'e4_gpu': e4_cost, 'e5_gpu': e5_cost}
            functions[plan['part']](plan, budget, started, allocation, receipt['rows'])
            if time.perf_counter()-started >= allocation:
                raise BudgetHeld('Saved cost closure exceeded local allocation')
            for path, digest in plan['pins'].items():
                if sha(path) != digest:
                    raise ValueError(f'Pinned asset changed during cost measurement: {path}')
            receipt['budget_ledger'] = str(budget.path)
            receipt['reused_BOLT_cost'] = plan.get('reused_BOLT_cost')
        receipt['status'] = 'COMPLETE_COST_ONLY'
        receipt['budget_ledger_sha256'] = sha(receipt['budget_ledger'])
        receipt['parity_flags'] = sum(v['parity']['status'] != 'PASS' for v in receipt['rows'])
        write(output/'RESULT.json', receipt)
        print(json.dumps({'result': str(output/'RESULT.json'), 'sha256': sha(output/'RESULT.json'),
                          'status': receipt['status'], 'rows': len(receipt['rows']), 'parity_flags': receipt['parity_flags']}))
    except BaseException as error:
        receipt['status'] = 'BUDGET_HELD_COST_ONLY' if isinstance(error, BudgetHeld) else 'FAILED_PRESERVED_COST_ONLY'
        receipt['reason'] = str(error)
        write(output/'FAILURE.json', receipt)
        raise


if __name__ == '__main__':
    main()
