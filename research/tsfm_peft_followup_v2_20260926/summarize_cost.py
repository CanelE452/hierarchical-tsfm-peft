"""Summarize completed timing artifacts without loading a model or tensor library."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TIMING_KEYS = ('milliseconds_per_origin', 'origins_per_second')
CAPACITY_KEYS = (
    'task_channels', 'backbone_series_per_origin',
    'registered_trainable_parameters_before_merge', 'total_parameters_before_merge',
    'total_deployed_parameters', 'deployed_parameter_bytes',
    'fitted_coefficient_count', 'fixed_basis_elements',
    'deployed_tensor_elements', 'deployed_tensor_bytes', 'torch_requires_grad_parameters',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.resolve().relative_to(ROOT)).replace('\\', '/')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def quantile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def timing_summary(values):
    if not values or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError('timing samples must be finite and positive')
    return dict(median=median(values), q25=quantile(values, .25),
                q75=quantile(values, .75), min=min(values), max=max(values))


def memory_values(row):
    return {
        'cuda_peak_allocated_bytes': row.get('peak_allocated_bytes'),
        'cuda_peak_reserved_bytes': row.get('peak_reserved_bytes'),
        'cuda_resident_allocated_bytes': row.get('resident_allocated_bytes'),
        'process_rss_bytes': row.get('process_cpu_rss_bytes',
                                     row.get('after_process_cpu_memory_bytes', {}).get('rss')),
        'process_baseline_rss_bytes': row.get('baseline_process_cpu_memory_bytes', {}).get('rss'),
        'process_peak_working_set_bytes': row.get('after_process_cpu_memory_bytes', {}).get('peak_wset'),
    }


def verify_row(row):
    durations = row['seconds_per_24_origins']
    if len(durations) != 20:
        raise ValueError(f'{row["id"]}: expected 20 complete passes per order block')
    values = {'milliseconds_per_origin': [v * 1000 / 24 for v in durations],
              'origins_per_second': [24 / v for v in durations]}
    for key, samples in values.items():
        computed = timing_summary(samples)
        for field, value in computed.items():
            if not math.isclose(value, row[key][field], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'{row["id"]}: saved {key}.{field} differs from raw timing passes')


def summarize_fit(identifier, rows, spec, kind, source, selection, channels):
    if len(rows) != 6 or {(r['block'], r['batch_origins']) for r in rows} != {
            (block, batch) for block in range(3) for batch in (1, 4)}:
        raise ValueError(f'{identifier}: require exactly 3 blocks x 2 batch sizes')
    first = rows[0]
    dataset = first['dataset']
    arm = spec['arm'] if spec is not None else first['arm']
    if any(r['dataset'] != dataset or r.get('seed') != first.get('seed') for r in rows):
        raise ValueError(f'{identifier}: inconsistent dataset or seed')
    for row in rows:
        verify_row(row)
    capacity = {key: first.get(key) for key in CAPACITY_KEYS}
    if any(row.get(key) != capacity[key] for row in rows for key in CAPACITY_KEYS):
        raise ValueError(f'{identifier}: capacity metadata changed between blocks')
    capacity['task_channels'] = channels[dataset]
    if kind == 'linear':
        capacity['backbone_series_per_origin'] = 0
    capacity['total_deployed_bytes'] = (capacity['deployed_parameter_bytes'] if kind == 'neural'
                                        else capacity['deployed_tensor_bytes'])
    compression_arm = arm in ('compress', 'residual', 'raw_bypass')
    latent = spec['latent'] if compression_arm else None
    if arm == 'factor_linear':
        basis_elements = capacity['fixed_basis_elements']
        if basis_elements % channels[dataset]:
            raise ValueError(f'{identifier}: nonintegral factor basis rank')
        latent = basis_elements // channels[dataset]
    mode = spec.get('ed_mode', 'current') if compression_arm else None
    selected = None
    roles = list(spec.get('roles', [])) if spec is not None else ['linear_baseline']
    if arm in ('residual', 'raw_bypass'):
        chosen = selection['selection'][dataset][arm]
        selected = identifier in chosen['fits']
        if selected and (mode != chosen['mode'] or latent != chosen['latent']):
            raise ValueError(f'{identifier}: selected fit has wrong mode or K')
        if spec.get('selected_for_own_arm_by_final_VAL') != selected:
            raise ValueError(f'{identifier}: cost-spec selection role is stale')
    if arm == 'lora' and not all(r.get('merge_parity', {}).get('pass') is True for r in rows):
        raise ValueError(f'{identifier}: merged LoRA parity is not verified')
    batches = {}
    for batch in (1, 4):
        members = sorted((r for r in rows if r['batch_origins'] == batch), key=lambda r: r['block'])
        blocks = [dict(block=r['block'], order_position=r['order_position'],
                       **{key: r[key] for key in TIMING_KEYS}, memory=memory_values(r)) for r in members]
        memory_max = {key: max(r['memory'][key] for r in blocks) if all(
            r['memory'][key] is not None for r in blocks) else None for key in blocks[0]['memory']}
        batches[str(batch)] = {
            'blocks': blocks,
            'across_3_block_medians': {
                key: timing_summary([r[key]['median'] for r in members]) for key in TIMING_KEYS},
            'memory_max_over_3_blocks': memory_max,
        }
    return dict(id=identifier, dataset=dataset, arm=arm, ed_mode=mode, latent=latent,
                seed=first.get('seed'), kind=kind, source=source, roles=roles,
                final_val_selected_for_own_arm=selected, capacity=capacity, batches=batches,
                merge_parity=[r['merge_parity'] for r in rows] if arm == 'lora' else None)


def aggregate_group(identifier, fits):
    first = fits[0]
    if first['arm'] in ('f0', 'shared_linear', 'factor_linear'):
        if len(fits) != 1 or first['seed'] is not None:
            raise ValueError(f'{identifier}: deterministic baseline should have one measured fit')
    elif sorted(f['seed'] for f in fits) != [92601, 92602]:
        raise ValueError(f'{identifier}: both trained seeds must be measured')
    if len({f['final_val_selected_for_own_arm'] for f in fits}) != 1:
        raise ValueError(f'{identifier}: partial selection within a configuration')
    capacity = {}
    for key in first['capacity']:
        values = [fit['capacity'][key] for fit in fits]
        if values != [values[0]] * len(values):
            raise ValueError(f'{identifier}: parameter capacity differs by seed')
        capacity[key] = values[0]
    batches = {}
    for batch in ('1', '4'):
        per_fit = {fit['id']: {key: fit['batches'][batch]['across_3_block_medians'][key]['median']
                              for key in TIMING_KEYS} for fit in fits}
        memory = {}
        for key in first['batches'][batch]['memory_max_over_3_blocks']:
            values = [fit['batches'][batch]['memory_max_over_3_blocks'][key] for fit in fits]
            memory[key] = None if any(v is None for v in values) else {
                'mean_of_fit_maxima': mean(values), 'maximum_across_fits': max(values)}
        batches[batch] = dict(
            mean_of_fit_block_medians={key: mean(v[key] for v in per_fit.values()) for key in TIMING_KEYS},
            fit_block_medians=per_fit, memory=memory)
    return dict(id=identifier, dataset=first['dataset'], arm=first['arm'],
                ed_mode=first['ed_mode'], latent=first['latent'], kind=first['kind'],
                fits=[fit['id'] for fit in fits], seeds=[fit['seed'] for fit in fits],
                roles=sorted({role for fit in fits for role in fit['roles']}),
                final_val_selected_for_own_arm=first['final_val_selected_for_own_arm'],
                capacity=capacity, batches=batches)


def relative_cost(candidate, reference):
    def center(group, batch, metric):
        return group['batches'][batch]['mean_of_fit_block_medians'][metric]
    candidate_latency = center(candidate, '1', 'milliseconds_per_origin')
    reference_latency = center(reference, '1', 'milliseconds_per_origin')
    candidate_throughput = center(candidate, '4', 'origins_per_second')
    reference_throughput = center(reference, '4', 'origins_per_second')
    return dict(candidate=candidate['id'], reference=reference['id'],
                batch1_latency_candidate_over_reference=candidate_latency / reference_latency,
                batch1_speedup_reference_over_candidate=reference_latency / candidate_latency,
                batch4_throughput_candidate_over_reference=candidate_throughput / reference_throughput,
                hardware_scope='GPU neural versus CPU linear deployment' if reference['kind'] == 'linear'
                               else 'same GPU neural deployment')


def build_summary(neural_path, linear_path, specs_path, selection_path):
    neural, linear = read_json(neural_path), read_json(linear_path)
    specs, selection = read_json(specs_path), read_json(selection_path)
    sources = {}
    for name, path in [('neural', neural_path), ('linear', linear_path),
                       ('cost_specs', specs_path), ('final_selection', selection_path)]:
        sources[name] = dict(path=relative(path), sha256=digest(path))
    if neural.get('specs_sha256') != sources['cost_specs']['sha256']:
        raise ValueError('neural cost was not measured from the supplied cost specs')
    spec_by_id = {s['id']: s for s in specs}
    if len(spec_by_id) != len(specs):
        raise ValueError('duplicate cost spec ID')
    if any(s.get('source_selection_sha256') != sources['final_selection']['sha256'] for s in specs):
        raise ValueError('cost specs do not refer to the supplied final VAL selection')
    for report in (neural, linear):
        if (report.get('status'), report.get('order_blocks'), report.get('passes_per_order_block')) != ('complete', 3, 20):
            raise ValueError('only complete 3-block, 20-pass timing reports are accepted')
    if neural['val_origins'] != linear['val_origins']:
        raise ValueError('neural and linear timing inputs use different VAL origins')
    channels = {}
    for row in neural['rows']:
        if row['dataset'] in channels and channels[row['dataset']] != row['task_channels']:
            raise ValueError('inconsistent task channel count')
        channels[row['dataset']] = row['task_channels']
    if {r['id'] for r in neural['rows']} != set(spec_by_id):
        raise ValueError('measured neural fit set differs from requested cost cohort')
    fits = {}
    for kind, report in [('neural', neural), ('linear', linear)]:
        grouped = defaultdict(list)
        for row in report['rows']:
            grouped[row['id']].append(row)
        for identifier, rows in sorted(grouped.items()):
            if identifier in fits:
                raise ValueError('duplicate measured fit across reports')
            if kind == 'linear' and not report['parity'][identifier]['pass']:
                raise ValueError('linear deployment parity is not verified')
            fits[identifier] = summarize_fit(identifier, rows, spec_by_id.get(identifier), kind,
                                             sources[kind]['path'], selection, channels)
    grouped = defaultdict(list)
    for fit in fits.values():
        key = f'{fit["dataset"]}/{fit["arm"]}/{fit["ed_mode"] or "none"}/k{fit["latent"] or "none"}'
        grouped[key].append(fit)
    groups = {key: aggregate_group(key, members) for key, members in sorted(grouped.items())}
    comparisons = {}
    for dataset in sorted(channels):
        dataset_groups = [g for g in groups.values() if g['dataset'] == dataset]
        selected = [g for g in dataset_groups if g['arm'] == 'residual' and g['final_val_selected_for_own_arm']]
        if len(selected) != 1:
            raise ValueError('exactly one final-selected residual group is required per dataset')
        references = {}
        for arm in ('f0', 'lora', 'raw_bypass', 'shared_linear', 'factor_linear'):
            candidates = [g for g in dataset_groups if g['arm'] == arm and (
                arm != 'raw_bypass' or g['final_val_selected_for_own_arm'])]
            if len(candidates) != 1:
                raise ValueError(f'{dataset}: unique {arm} reference is unavailable')
            references[arm] = relative_cost(selected[0], candidates[0])
        comparisons[dataset] = references
    return dict(
        status='complete', schema_version=1, sources=sources,
        summarizer=dict(path=relative(Path(__file__)), sha256=digest(Path(__file__))),
        definitions={
            'timing_unit': 'One pass processes 24 VAL origins; milliseconds/origin = seconds*1000/24; throughput = 24/seconds.',
            'within_block': 'median/q25/q75/min/max of the 20 raw passes in one order block; preserved separately for each block. Raw durations remain in the source JSON.',
            'across_blocks': 'Per-fit median/q25/q75/min/max of exactly 3 block medians, not pooled 60-pass quantiles. Quantiles use linear interpolation at (n-1)*p, matching numpy.quantile default.',
            'group_center': 'Arithmetic mean of per-fit medians of 3 block medians across the two trained seeds; single deterministic baseline fit remains unchanged. Matches plot_results.py mean-marker x values.',
            'uncertainty': 'Within-block IQR, between-block quartiles and seed variation are distinct. These repeated timings do not give independent data evidence or confidence intervals; no combined group IQR is claimed.',
            'memory': 'Per-fit memory summary is maximum over 3 blocks, separately by batch. Group reports mean of those fit maxima and maximum across fits. CUDA allocated, CUDA reserved and whole-process CPU RSS are distinct measurements, not summed or interchangeable; unavailable CUDA metrics for CPU baselines are null.',
            'capacity': 'Registered trainable count is before LoRA merge; deployed bytes are all neural parameter tensors or all linear buffers. Linear fitted coefficient count is separate from its zero requires-grad deployment count. Full F0/LoRA have latent=null because cost-spec latent does not compress their input.',
            'ratios': 'Latency candidate/reference: below 1 is lower latency. Latency speedup reference/candidate: above 1 is faster. Throughput candidate/reference: above 1 is higher throughput. Ratios use group centers, not averages of seed ratios.',
            'scope': 'Neural GPU CPU-input-to-CPU-output timing includes H2D/D2H; linear timing uses CPU FP32. Neither includes load/disk/scoring. CPU-linear versus GPU-neural ratios describe these deployment routes, not matched-device kernel speedups.',
        },
        measurement_conditions={kind: {key: report.get(key) for key in (
            'precision', 'warmup_calls', 'passes_per_order_block', 'order_blocks', 'torch_threads',
            'timed_scope', 'memory_scope', 'device', 'torch_version', 'cuda_version',
            'float32_matmul_precision', 'allow_tf32_matmul')} for kind, report in [('neural', neural), ('linear', linear)]},
        checks=dict(neural_fit_count=len(specs), linear_fit_count=len(fits) - len(specs),
                    raw_pass_summaries_recomputed=True, all_fit_batches_have_3_distinct_blocks=True,
                    final_selection_hash_and_roles_verified=True, same_24_val_origins_verified=True),
        fits=fits, groups=groups, selected_residual_relative_cost=comparisons)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neural', type=Path, default=HERE / 'neural_cost_corrected.json')
    parser.add_argument('--linear', type=Path, default=HERE / 'linear_cost.json')
    parser.add_argument('--specs', type=Path, default=HERE / 'cost_specs.json')
    parser.add_argument('--selection', type=Path, default=HERE / 'final_selection.json')
    parser.add_argument('--output', type=Path, default=HERE / 'cost_summary.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve existing summary: {args.output}')
    result = build_summary(args.neural, args.linear, args.specs, args.selection)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status='complete', output=relative(args.output),
                          groups=len(result['groups']), fits=len(result['fits']))))


if __name__ == '__main__':
    main()
