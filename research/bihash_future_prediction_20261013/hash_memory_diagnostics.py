"""CPU-only, independent future-memory partition diagnostics from sealed checkpoints.

This script does not import the training model, use a GPU or open any EVAL data.
Hard-code collisions describe a partition; they do not establish its usefulness.
"""
from pathlib import Path
import argparse
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
import runtime as rt


def statistics(phi, logits, valid):
    phi = np.asarray(phi, dtype=np.float64)[valid]
    logits = np.asarray(logits, dtype=np.float64)[valid]
    n, bits = logits.shape
    if n == 0:
        return {'valid_memory_entries': 0, 'status': 'NO_VALID_FUTURE_MEMORY'}
    codes = np.where(logits >= 0, 1, -1).astype(np.int8)
    positive = (codes > 0).mean(0)
    entropy = np.zeros(bits, np.float64)
    interior = (positive > 0) & (positive < 1)
    entropy[interior] = -positive[interior] * np.log2(positive[interior]) - (1 - positive[interior]) * np.log2(1 - positive[interior])
    _, counts = np.unique(codes, axis=0, return_counts=True)
    frequencies = counts / n
    collision_pairs = int(np.sum(counts * (counts - 1) // 2))
    total_pairs = n * (n - 1) // 2
    centered = phi - phi.mean(0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    energy_sum = float(energy.sum())
    relative_rank = int((singular > (singular[0] * 1e-6 if len(singular) else 0.)).sum())
    if energy_sum > 0:
        probability = energy[energy > 0] / energy_sum
        effective_rank = float(np.exp(-np.sum(probability * np.log(probability))))
        energy99 = int(np.searchsorted(np.cumsum(energy) / energy_sum, .99) + 1)
    else:
        effective_rank, energy99 = 0., 0
    margin = np.abs(logits).reshape(-1)
    return {'valid_memory_entries': n, 'hash_bits': bits, 'phi_dimensions': phi.shape[1],
        'phi_centered_max_rank': min(max(n - 1, 0), phi.shape[1]), 'phi_centered_numpy_rank': int(np.linalg.matrix_rank(centered)),
        'phi_centered_rank_relative_tol_1e_minus_6': relative_rank, 'phi_effective_spectral_rank': effective_rank,
        'phi_99_percent_energy_dimensions': energy99, 'phi_variance': phi.var(0).tolist(),
        'phi_mean_variance': float(phi.var(0).mean()), 'phi_singular_values': singular.tolist(),
        'future_key_positive_fraction': positive.tolist(), 'future_key_bit_entropy_bits': entropy.tolist(),
        'future_key_mean_bit_entropy_bits': float(entropy.mean()), 'zero_entropy_bits': int((entropy == 0).sum()),
        'unique_hard_codes': len(counts), 'largest_code_bucket': int(counts.max()),
        'code_bucket_sizes_descending': sorted(counts.tolist(), reverse=True),
        'hard_code_entropy_bits': float(-np.sum(frequencies * np.log2(frequencies))),
        'all_same_hard_code': bool(len(counts) == 1), 'collision_pairs': collision_pairs,
        'total_memory_pairs': total_pairs, 'collision_pair_fraction': collision_pairs / total_pairs if total_pairs else None,
        'sign_margin_abs_mean': float(margin.mean()), 'sign_margin_abs_p05': float(np.quantile(margin, .05)),
        'sign_margin_abs_p50': float(np.quantile(margin, .5)), 'sign_margin_abs_min': float(margin.min()),
        'sign_margin_abs_max': float(margin.max()),
        'sign_margin_abs_below': {str(threshold): int((margin < threshold).sum()) for threshold in (1e-6, 1e-4, .01)},
        'exact_zero_logits': int((logits == 0).sum()), 'sample_scope': 'FIT historical future-memory entries only'}


def _load(artifact):
    path = rt.ROOT / artifact['path']
    if rt.sha(path) != artifact['sha256']: raise ValueError('DIAGNOSTIC_ARTIFACT_SHA: ' + str(path))
    return torch.load(path, map_location='cpu', weights_only=True)


def _phi(state):
    delta, mask = state['bank_delta'], state['bank_observed']
    features = torch.cat((torch.where(mask, delta, torch.zeros_like(delta)).flatten(1), mask.float().flatten(1)), dim=1)
    phi = F.linear(F.gelu(F.linear(features, state['future_autoencoder.encoder.0.weight'],
        state['future_autoencoder.encoder.0.bias'])), state['future_autoencoder.encoder.2.weight'], state['future_autoencoder.encoder.2.bias'])
    logits = F.linear(phi, state['future_hash.weight'], state['future_hash.bias'])
    return phi, logits


def _head_hash(state, prefix):
    return rt.state_hash({name: value for name, value in state.items() if name.startswith(prefix)})


def _delta(initial, selected, prefix):
    pieces = [((selected[k] - value).double().flatten()) for k, value in initial.items() if k.startswith(prefix)]
    if not pieces: return {'max_abs': 0., 'l2': 0.}
    vector = torch.cat(pieces)
    return {'max_abs': float(vector.abs().max()), 'l2': float(vector.norm())}


def state_diagnostic(state):
    with torch.no_grad(): cpu_phi, cpu_logits = _phi(state)
    valid = state['bank_valid'].numpy().astype(bool)
    if not bool(state['memory_is_frozen']): raise ValueError('CHECKPOINT_MEMORY_NOT_FROZEN')
    saved_phi, saved_logits = state['memory_embedding'], state['memory_hash_logits']
    cpu_codes = torch.where(cpu_logits >= 0, 1, -1)
    saved_codes = torch.where(saved_logits >= 0, 1, -1)
    differences = torch.nonzero((cpu_codes != saved_codes) & state['bank_valid'][:, None], as_tuple=False)
    disagreement = [{'origin': int(state['bank_origins'][row]), 'bit': int(bit),
        'cpu_logit': float(cpu_logits[row, bit]), 'saved_gpu_logit': float(saved_logits[row, bit])} for row, bit in differences[:64]]
    return {'saved_GPU_deployment_memory': statistics(saved_phi.numpy(), saved_logits.numpy(), valid),
        'independent_CPU_recalculation': statistics(cpu_phi.numpy(), cpu_logits.numpy(), valid),
        'cache_recalculation': {'phi_max_abs': float((cpu_phi - saved_phi).abs().max()),
            'logits_max_abs': float((cpu_logits - saved_logits).abs().max()),
            'hard_sign_disagreement_count': len(differences), 'first_64_disagreements': disagreement,
            'note': 'FP32 CPU/GPU rounding may affect signs near zero; original saved GPU cache describes actual deployment keys'},
        'bank_origins': state['bank_origins'].tolist(), 'bank_end_exclusive': state['bank_end_exclusive'].tolist()}


def inspect_case(case, source_record):
    first = next(cp for cp in case['training']['checkpoints'] if cp['update'] == 0)
    selected = case['training']['selected_checkpoint']
    initial = _load(first)['model']; current = _load(selected)['model']
    common = _load(case['future_pretrain'])['future_autoencoder']
    for key, value in common.items():
        if not torch.equal(initial['future_autoencoder.' + key], value):
            raise ValueError('COMMON_PRETRAIN_NOT_EXACT_CLONE: ' + key)
    for key in ('bank_delta', 'bank_observed', 'bank_valid', 'bank_origins', 'bank_end_exclusive'):
        if not torch.equal(initial[key], current[key]): raise ValueError('BANK_MUTATED_AFTER_INITIALIZATION: ' + key)
    valid = initial['bank_valid']
    a, b = state_diagnostic(initial), state_diagnostic(current)
    first_codes = torch.where(initial['memory_hash_logits'] >= 0, 1, -1)[valid]
    selected_codes = torch.where(current['memory_hash_logits'] >= 0, 1, -1)[valid]
    changed = first_codes != selected_codes
    method = case['method']
    code_rule = 'hard +1/-1 from I>=0' if method.startswith('B_') and method not in ('B_COSINE', 'B_UNIFORM') else (
        'continuous projected cosine logits; hard keys are diagnostic counterfactual' if method == 'B_COSINE' else (
            'uniform eligible retrieval; hard keys are unused diagnostic counterfactual' if method == 'B_UNIFORM' else
            'future branch and hash heads removed in prediction; hard keys are teacher mechanism diagnostics'))
    result = {'dataset': case['dataset'], 'seed': case['seed'], 'method': method, 'weight': case['weight'],
        'label': case['label'], 'K_TSFM_input_channels': case['K'], 'R_future_representation': current['memory_embedding'].shape[1],
        'hash_bits': current['memory_hash_logits'].shape[1], 'trained_updates': case['training']['updates'],
        'selected_update': case['training']['selected_update'], 'initial_checkpoint': first, 'selected_checkpoint': selected,
        'case_source': source_record, 'future_pretrain': case['future_pretrain'], 'common_pretrained_AE_exact_clone': True,
        'bank_unchanged_from_u0': True, 'deployment_code_rule': code_rule,
        'initial_map_hashes': {'Ws': _head_hash(initial, 'student_hash.'), 'Wf': _head_hash(initial, 'future_hash.')},
        'selected_map_hashes': {'Ws': _head_hash(current, 'student_hash.'), 'Wf': _head_hash(current, 'future_hash.')},
        'parameter_changes': {name: _delta(initial, current, prefix) for name, prefix in (
            ('Ws', 'student_hash.'), ('Wf', 'future_hash.'), ('phi_encoder', 'future_autoencoder.encoder.'))},
        'u0': a, 'selected': b,
        'partition_change': {'valid_memory_entries': int(valid.sum()), 'changed_code_rows': int(changed.any(-1).sum()),
            'bit_flip_fraction': float(changed.float().mean()) if changed.numel() else None,
            'phi_rms_change': float((current['memory_embedding'][valid] - initial['memory_embedding'][valid]).square().mean().sqrt()) if valid.any() else None},
        'limitations': ['Gradients or parameter changes do not demonstrate forecasting usefulness.',
            'Collision frequency can reflect either shared useful patterns or collapse; forecasting comparisons decide.',
            'The bank is a finite FIT-memory sample, not an independently labelled future-pattern population.']}
    if 'FIXED' in method:
        result['fixed_map_caveat'] = 'Ws/Wf are frozen; E/phi features can move and therefore the resulting partition is not globally fixed.'
        if result['parameter_changes']['Ws']['max_abs'] != 0 or result['parameter_changes']['Wf']['max_abs'] != 0:
            raise ValueError('DECLARED_FIXED_MAP_CHANGED')
    if method == 'NO_ALIGN': result['baseline_label'] = 'Hash maps receive no forecasting/alignment gradient; phi receives common future reconstruction supervision.'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, action='append', default=[])
    parser.add_argument('--case', type=Path, action='append', default=[])
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-cases', type=int, default=174)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    folder = rt.HERE / 'diagnostics_v1' / args.tag; folder.mkdir(parents=True, exist_ok=False)
    limit = rt.Limit(args.tag, gpu=False)
    output = {'status': 'PARTIAL_INPUT_COVERAGE', 'created_unix': time.time(), 'script_sha256': rt.sha(__file__),
        'expected_cases': args.expected_cases, 'source_artifacts': [], 'case_diagnostics': [], 'failed_cases': [],
        'data_access': 'Verified FIT-memory tensors in completed checkpoint; no dataset loader, EVAL target or GPU access'}
    sources = []
    for path in args.result:
        value = rt.read(path)
        if value['status'] != 'COMPLETE': raise ValueError('RESULT_NOT_TERMINAL_COMPLETE: ' + str(path))
        record = {'path': str(path.resolve().relative_to(rt.ROOT)), 'sha256': rt.sha(path), 'type': 'completed_result'}
        output['source_artifacts'].append(record)
        sources.extend((case, record) for case in value['cases'])
    for path in args.case:
        value = rt.read(path)
        if value['training']['updates'] != 600 or len(value['training']['curve']) != 600:
            raise ValueError('CASE_NOT_COMPLETED_600: ' + str(path))
        record = {'path': str(path.resolve().relative_to(rt.ROOT)), 'sha256': rt.sha(path), 'type': 'completed_individual_case'}
        output['source_artifacts'].append(record); sources.append((value, record))
    identities = [(c['dataset'], c['seed'], c['method'], c['weight']) for c, _ in sources]
    if len(identities) != len(set(identities)): raise ValueError('DUPLICATE_DIAGNOSTIC_INPUT_IDENTITY')
    status = 'COMPLETE'
    try:
        for case, record in sources:
            try:
                value = inspect_case(case, record)
                output['case_diagnostics'].append(value)
                rt.write(folder / (case['dataset'] + '_' + case['label'] + '.json'), value)
                print(rt.json_line({'future_key_diagnostic': case['dataset'] + '_' + case['label'],
                    'selected_unique_codes': value['selected']['saved_GPU_deployment_memory']['unique_hard_codes'],
                    'selected_zero_entropy_bits': value['selected']['saved_GPU_deployment_memory']['zero_entropy_bits']}), flush=True)
            except Exception as exc:
                if isinstance(exc, rt.BudgetHeld): raise
                output['failed_cases'].append({'identity': [case['dataset'], case['seed'], case['method'], case['weight']],
                    'error_type': type(exc).__name__, 'error': str(exc)})
            limit.check()
        if len(output['case_diagnostics']) == args.expected_cases and not output['failed_cases']:
            output['status'] = 'COMPLETE_INPUT_COVERAGE'
        if output['failed_cases']: status = 'FAILED_PRESERVED'
    except Exception as exc:
        status = 'BUDGET_HELD_PRESERVED' if isinstance(exc, rt.BudgetHeld) else 'FAILED_PRESERVED'
        output['terminal_error'] = str(exc)
    output['actual_completed_case_count'] = len(output['case_diagnostics'])
    output['uncovered_case_count'] = max(args.expected_cases - len(output['case_diagnostics']), 0)
    output['resource_receipt'] = limit.close(status, output.get('terminal_error'))
    rt.write(folder / 'RESULT.json', output)
    print(rt.json_line({'future_key_diagnostics_terminal': str(folder / 'RESULT.json'), 'status': output['status'],
        'actual_completed_case_count': output['actual_completed_case_count']}), flush=True)


if __name__ == '__main__': main()
