"""Guarded first-stage Korean report from verified saved results and costs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'research/smallk_failure_followup_20261014'
sys.path[:0] = [str(HERE), str(OLD), str(ROOT)]
DATASETS = ('robin', 'beijing_aotizhongxin')
METHODS = ('PERSIST', 'MEAN', 'SELF33', 'JOINT33', 'SELF512', 'RAW_BOLT')
ROUTES = ('SELF33', 'JOINT33', 'SELF512')
COLORS = {'PERSIST': '#8793a0', 'MEAN': '#b391b5', 'SELF33': '#278e94',
          'JOINT33': '#d26f49', 'SELF512': '#4978b7', 'RAW_BOLT': '#a79142'}
FINAL = {'robin': 'val', 'beijing_aotizhongxin': 'eval'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def record(path, role):
    path = resolve(path)
    return {'path': str(path), 'sha256': sha(path), 'role': role}


def score(producer, method, split):
    if method == 'RAW_BOLT':
        return producer['references'][split]['RAW_BOLT']
    if method in ('PERSIST', 'MEAN'):
        return producer['reference_shared_controls'][method][split]
    return producer['families'][method]['splits'][split]


def validate_scientific(producers, producer_records, audit, auth_sha, run_sha):
    if len(producers) != 2 or {p.get('dataset') for p in producers} != set(DATASETS):
        raise ValueError('EXACT_TWO_UNIQUE_FIRST_READOUT_PRODUCERS_REQUIRED')
    if audit.get('status') != 'PASS' or audit.get('totals', {}).get('pool_candidates_checked') != 54:
        raise ValueError('54_POOL_INDEPENDENT_SAVED_ARRAY_AUDIT_PASS_REQUIRED')
    for p, r in zip(producers, producer_records):
        if (p.get('status') != 'COMPLETE' or p.get('protected_TEST_accessed') is not False
                or p.get('independent_holdout') is not False or p.get('pool_comparisons') != 27
                or p.get('permission_gate', {}).get('sha256') != auth_sha
                or p.get('run_manifest', {}).get('sha256') != run_sha):
            raise ValueError('APPROVED_COMPLETE_PAST_ONLY_READOUT_REQUIRED')
        bound = [v for v in audit.get('producer_records', []) if v.get('dataset') == p['dataset']]
        if len(bound) != 1 or bound[0]['sha256'] != r['sha256']:
            raise ValueError('AUDIT_PRODUCER_EXACT_SHA_BINDING_REQUIRED')
        for route in ROUTES:
            f = p['families'][route]
            pool = f['pool']
            if len(pool) != 9 or [v['kind'] for v in pool] != ['PERSIST', 'MEAN'] + ['RIDGE'] * 7:
                raise ValueError('EXACT_NINE_POOL_ORDER_REQUIRED')
            expected = min(range(9), key=lambda i: pool[i]['tune_scores']['normalized']['mse'])
            if f['selected_index'] != expected or f['selected_kind'] != pool[expected]['kind']:
                raise ValueError('EARLIEST_GLOBAL_TUNE_SELECTION_MISMATCH')
            if f['selected_alpha'] != pool[expected].get('alpha'):
                raise ValueError('SELECTED_ALPHA_MISMATCH')
    return True


def audit_cost_saved_arrays(cost, cost_record, producer_records, check):
    """Read saved outputs; no model, deployment helper or hardware replay."""
    import numpy as np
    if (cost.get('status') != 'COMPLETE' or cost.get('source_before_after_SHA_pass') is not True
            or cost.get('model_binary_before_after_SHA_pass') is not True
            or cost.get('F_before') != cost.get('F_after') or len(cost.get('rows', [])) != 24):
        raise ValueError('COMPLETE_PRESERVED_FROZEN_24_CELL_COST_REQUIRED')
    cohort = {(d, b, r) for d in DATASETS for b in (1, 8) for r in METHODS}
    if {(v['dataset'], v['batch'], v['route']) for v in cost['rows']} != cohort:
        raise ValueError('EXACT24_COST_COHORT_REQUIRED')
    if {v['sha256'] for v in cost.get('producer_records', [])} != {v['sha256'] for v in producer_records}:
        raise ValueError('COST_EXACT_SCIENTIFIC_PRODUCER_SHA_BINDING_REQUIRED')
    files, checks = 0, 0
    for row in cost['rows']:
        check()
        if (row['status'] != 'COMPLETE' or row['repeats'] != 6 or row['warmup_each_route'] != 2
                or set(row['positions']) != set(range(6)) or len(row['seconds']) != 6
                or any(not math.isfinite(v) or v <= 0 for v in row['seconds'])):
            raise ValueError('COMPLETE_POSITION_BALANCED_SIX_REPEAT_COST_REQUIRED')
        if not math.isclose(row['median_seconds'], float(np.median(row['seconds'])), rel_tol=1e-13):
            raise ValueError('COST_MEDIAN_METADATA_MISMATCH')
        if not math.isclose(row['median_seconds_per_window'], row['median_seconds']/row['batch'], rel_tol=1e-13):
            raise ValueError('COST_BATCH_NORMALIZATION_MISMATCH')
        if not np.allclose(np.asarray(row['paired_RAW_ratios']),
                           np.asarray(row['seconds'])/np.asarray(row['paired_RAW_seconds']), rtol=1e-13, atol=0):
            raise ValueError('COST_PAIRED_RATIO_MISMATCH')
        if not math.isclose(row['paired_RAW_median_ratio'], float(np.median(row['paired_RAW_ratios'])), rel_tol=1e-13):
            raise ValueError('COST_PAIRED_MEDIAN_RATIO_MISMATCH')
        loaded = {}
        for kind in ('timed_outputs', 'reference'):
            asset = row[kind]
            if sha(asset['path']) != asset['sha256']:
                raise ValueError('COST_PAYLOAD_FILE_SHA_MISMATCH')
            with np.load(asset['path'], allow_pickle=False) as opened:
                arrays = {k: opened[k].copy() for k in opened.files}
            for key, meta in asset['arrays'].items():
                value = arrays[key]
                digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                if list(value.shape) != meta['shape'] or str(value.dtype) != meta['dtype'] or digest != meta['sha256']:
                    raise ValueError('COST_PAYLOAD_ARRAY_SHA_MISMATCH')
                checks += 1
            loaded[kind] = arrays
            files += 1
        timed, reference = loaded['timed_outputs'], loaded['reference']
        if not np.array_equal(timed['origins'], reference['origins']) or set(timed['position'].tolist()) != set(range(6)):
            raise ValueError('TIMED_REFERENCE_ORIGIN_OR_POSITION_MISMATCH')
        for key, atol, rtol in (('normalized', 1e-5, 2e-6), ('physical', .002, 2e-5)):
            values = timed[key]
            expected = reference[row['route'] + '_' + key]
            if values.shape != (6, row['batch'], 48, expected.shape[-1]) or not np.isfinite(values).all():
                raise ValueError('FINITE_TIMED_OUTPUT_SHAPE_MISMATCH')
            if not np.all(np.abs(values.astype(np.float64)-expected.astype(np.float64)) <= atol+rtol*np.abs(expected)):
                raise ValueError('ACTUAL_TIMED_SAVED_REFERENCE_PARITY_FAILED')
            checks += 6
        if not all(all(entry['pass'] for entry in block.values()) for block in row['parity']):
            raise ValueError('COST_DECLARED_PARITY_FAILED')
    return {'status': 'PASS', 'producer': cost_record, 'whole_rows_checked': 24,
            'payload_files_checked_including_shared_reference_reopens': files,
            'saved_array_and_metadata_checks': checks, 'hardware_replayed': False,
            'source_before_after_SHA_pass': True, 'scope': 'saved timing outputs, same-reference parity, arithmetic and binding; not independent hardware trial'}


def validate_independent_cost(audit, cost_record):
    if (audit.get('status') != 'PASS' or audit.get('cost_producer_record', {}).get('sha256') != cost_record['sha256']
            or audit.get('rows_checked') != 24 or audit.get('timed_repeats_checked') != 144
            or audit.get('source_before_after_SHA_pass') is not True):
        raise ValueError('EXACT_INDEPENDENT_24_CELL144_REPEAT_COST_AUDIT_PASS_REQUIRED')


def validate_expanded(producer, producer_record, audit, first_records, auth_sha, run_sha):
    if (producer.get('status') != 'COMPLETE' or producer.get('dataset') != 'robin'
            or producer.get('phase') != 'MATCHED_FIT512_ONE_DIAGNOSTIC'
            or producer.get('expanded_FIT_origin_count') != 512 or producer.get('pool_comparisons') != 18
            or producer.get('permission_gate', {}).get('sha256') != auth_sha
            or producer.get('run_manifest', {}).get('sha256') != run_sha
            or producer.get('original256_reconstruction_exact_pass') is not True
            or producer.get('original_FIT_statistics_unchanged') is not True
            or producer.get('first256_result', {}).get('sha256') not in {r['sha256'] for r in first_records}):
        raise ValueError('EXACT_APPROVED_MATCHED_FIT512_DIAGNOSTIC_REQUIRED')
    if (audit.get('status') != 'PASS' or audit.get('producer_record', {}).get('sha256') != producer_record['sha256']
            or audit.get('pool_candidates_checked') != 18):
        raise ValueError('EXACT18_POOL_EXPANDED_PRODUCER_AUDIT_PASS_REQUIRED')
    if set(producer['families']) != {'SELF33', 'JOINT33'}:
        raise ValueError('ONLY_MATCHED_SELF_JOINT_FIT512_DIAGNOSTIC_ALLOWED')
    for route in ('SELF33', 'JOINT33'):
        f = producer['families'][route]
        chosen = min(range(9), key=lambda i: f['pool'][i]['tune_scores']['normalized']['mse'])
        if f['selected_index'] != chosen or f['selected_kind'] != f['pool'][chosen]['kind']:
            raise ValueError('EXPANDED_GLOBAL_TUNE_POOL_SELECTION_MISMATCH')


def expanded_metric_rows(producer):
    rows = []
    for route in ('SELF33', 'JOINT33'):
        f = producer['families'][route]
        for split in ('fit', 'tune', 'val'):
            for scale in ('normalized', 'physical_per_channel_diagnostic'):
                value = f['splits'][split][scale]
                rows.append({'dataset': 'robin', 'FIT_origins': 512, 'method': route, 'split': split,
                    'split_role': producer['split_roles'][split], 'metric_scale': scale,
                    'mse': value['mse'], 'mae': value['mae'], 'active_channels': value['active_channels'],
                    'selected_kind': f['selected_kind'], 'selected_alpha': f['selected_alpha'],
                    'observed_counts': value['observed_counts'], 'per_channel_mse': value['per_channel_mse'],
                    'per_channel_mae': value['per_channel_mae'], 'independent_holdout': False})
    return rows


def metric_rows(producers):
    rows = []
    for p in producers:
        for method in METHODS:
            family = p['families'].get(method, {})
            for split in ('fit', 'tune', FINAL[p['dataset']]):
                for scale in ('normalized', 'physical_per_channel_diagnostic'):
                    value = score(p, method, split)[scale]
                    rows.append({'dataset': p['dataset'], 'method': method, 'split': split,
                        'split_role': p['split_roles'][split], 'metric_scale': scale,
                        'mse': value['mse'], 'mae': value['mae'], 'active_channels': value['active_channels'],
                        'selected_kind': family.get('selected_kind', method),
                        'selected_alpha': family.get('selected_alpha'),
                        'observed_counts': value['observed_counts'],
                        'per_channel_mse': value['per_channel_mse'], 'per_channel_mae': value['per_channel_mae'],
                        'independent_holdout': False, 'physical_channel_units_comparable': False})
    return rows


def pool_rows(producers):
    rows = []
    for p in producers:
        for method in p['families']:
            f = p['families'][method]
            for i, candidate in enumerate(f['pool']):
                rows.append({'dataset': p['dataset'], 'method': method, 'candidate_index': i,
                    'candidate_kind': candidate['kind'], 'alpha': candidate.get('alpha'),
                    'selected': i == f['selected_index'], 'selected_by_global_TUNE_macro': True,
                    'fit_mse': candidate['fit_scores']['normalized']['mse'],
                    'fit_mae': candidate['fit_scores']['normalized']['mae'],
                    'tune_mse': candidate['tune_scores']['normalized']['mse'],
                    'tune_mae': candidate['tune_scores']['normalized']['mae'],
                    'model_sha256': candidate.get('model_sha256'),
                    'new_ridge': candidate['kind'] == 'RIDGE' and f['new_training']})
    return rows


def capacity_rows(producers):
    rows = []
    for p in producers:
        c = p['data_contract']['fit']['channels'] if 'channels' in p['data_contract']['fit'] else len(score(p, 'PERSIST', 'fit')['normalized']['observed_counts'])
        for route in p['families']:
            f, state = p['families'][route], p['feature_states'][route]
            meta = f.get('selected_model_metadata', {})
            rows.append({'dataset': p['dataset'], 'method': route, 'channels': c,
                'features_per_output_channel': state['feature_count'],
                'same_temporal_summary_channel_comparison': route in ('SELF33', 'JOINT33'),
                'selected_kind': f['selected_kind'], 'selected_alpha': f['selected_alpha'],
                'potential_registered_coefficients': state['feature_count']*48*c,
                'selected_registered_coefficients': meta.get('registered_coefficient_count', 0),
                'selected_registered_intercepts': meta.get('registered_intercept_count', 48*c if f['selected_kind'] == 'MEAN' else 0),
                'design_rank_min': meta.get('design_rank_min'), 'design_rank_max': meta.get('design_rank_max'),
                'factorization_groups': f['preparation']['factorization_groups'],
                'channel_submodels_per_alpha': f['preparation'].get('channel_submodels_per_alpha', 1),
                'tune_selection_margin_mse': f['tune_selection_margin_mse'],
                'tune_selection_margin_rel_pct': f.get('tune_selection_margin_rel_pct'),
                'deterministic_no_fake_seed_replication': True, 'pure_information_causal_claim': False})
    return rows


def cost_rows(cost):
    rows = []
    for r in cost['rows']:
        memories = r['memory']
        rows.append({'dataset': r['dataset'], 'method': r['route'], 'batch': r['batch'],
            'repeats': r['repeats'], 'warmup_each_route': r['warmup_each_route'],
            'whole_median_ms_per_batch': 1000*r['median_seconds'],
            'whole_median_ms_per_window': 1000*r['median_seconds_per_window'],
            'paired_RAW_median_ratio': r['paired_RAW_median_ratio'],
            'peak_allocated_vram_MiB': max(m['peak_allocated_vram_bytes'] for m in memories)/2**20,
            'peak_reserved_vram_MiB': max(m['peak_reserved_vram_bytes'] for m in memories)/2**20,
            'sampled_peak_rss_MiB': max(m['resident_rss_bytes'] for m in memories)/2**20,
            'selected_kind': r['selected_metadata']['selected_kind'],
            'selected_alpha': r['selected_metadata'].get('selected_alpha'),
            'single_local_process': True, 'method_exclusive_memory': False})
    return rows


def expanded_cost_rows(cost):
    return [{'dataset': r['dataset'], 'method': r['route'], 'FIT_origins': 512, 'batch': r['batch'],
             'warmup_each_route': r['warmup_each_route'], 'repeats': r['repeats'],
             'whole_median_ms_per_window': 1000*r['median_seconds_per_window'],
             'whole_median_ms_per_batch': 1000*r['median_seconds'],
             'sampled_peak_rss_MiB': r['sampled_peak_rss_bytes']/2**20,
             'GPU_execution': False, 'fresh_RAW_comparator': False,
             'cross_trial_speed_superiority_proven': False} for r in cost['rows']]


def validate_expanded_cost(cost, cost_record, audit, expanded_record):
    if (cost.get('status') != 'COMPLETE' or cost.get('producer_record', {}).get('sha256') != expanded_record['sha256']
            or cost.get('source_before_after_SHA_pass') is not True or cost.get('GPU_execution') is not False
            or len(cost.get('rows', [])) != 4):
        raise ValueError('EXACT_FOUR_CPU_EXPANDED_COST_ROWS_REQUIRED')
    if (audit.get('status') != 'PASS' or audit.get('cost_producer_record', {}).get('sha256') != cost_record['sha256']
            or audit.get('rows_checked') != 4 or audit.get('timed_repeats_checked') != 24
            or audit.get('source_before_after_SHA_pass') is not True):
        raise ValueError('EXACT_INDEPENDENT_FOUR_CELL24_REPEAT_CPU_COST_AUDIT_PASS_REQUIRED')


def execution_attempt_rows(accepted_producer_records):
    accepted = {r['sha256'] for r in accepted_producer_records}
    rows, runs = [], []
    for path in sorted((HERE/'resources_v1').glob('*.json')):
        if path.name == 'ACTIVE.json':
            continue
        value = read(path)
        error = value.get('check_failure') or value.get('caller_error') or {}
        if not isinstance(error, dict):
            error = {'message': str(error)}
        rows.append({'kind': 'metered_process', 'path': str(path.relative_to(HERE)).replace('\\', '/'),
            'tag': value.get('tag'), 'status': value.get('status'), 'gpu_process': value.get('gpu'),
            'wall_seconds': value.get('wall_seconds'), 'new_model_bundles': None,
            'reused_model_bundles': None, 'pool_comparisons': None,
            'accepted_scientific_producer_in_report': None,
            'error_type': error.get('type'), 'error_message': str(error.get('message', '')).replace('\n', ' ')[:600]})
    bootstrap = HERE/'BOOTSTRAP_FAILURE_v1.json'
    if bootstrap.exists():
        value = read(bootstrap)
        rows.append({'kind': 'administrative_bootstrap', 'path': bootstrap.name,
            'tag': 'manifest_bootstrap', 'status': value.get('status'), 'gpu_process': False,
            'wall_seconds': None, 'new_model_bundles': 0, 'reused_model_bundles': 0,
            'pool_comparisons': 0, 'accepted_scientific_producer_in_report': False,
            'error_type': 'AdministrativeManifestConstruction',
            'error_message': str(value.get('error', '')).replace('\n', ' ')[:600]})
    for path in sorted((HERE/'runs').glob('*/RESULT.json')):
        value = read(path)
        if not value.get('scientific_execution'):
            continue
        accepted_in_report = sha(path) in accepted
        item = {'kind': 'scientific_producer_attempt', 'path': str(path.relative_to(HERE)).replace('\\', '/'),
            'tag': path.parent.name, 'status': value.get('status'), 'gpu_process': value.get('GPU_execution'),
            'wall_seconds': value.get('wall_seconds'), 'new_model_bundles': value.get('new_model_bundles', 0),
            'reused_model_bundles': value.get('reused_model_bundles', 0),
            'pool_comparisons': value.get('pool_comparisons', 0),
            'accepted_scientific_producer_in_report': accepted_in_report,
            'error_type': (value.get('failure') or {}).get('type'),
            'error_message': str((value.get('failure') or {}).get('message', '')).replace('\n', ' ')[:600]}
        rows.append(item)
        runs.append(item)
    summary = {'all_recorded_new_model_bundles_including_unaccepted_attempts': sum(r['new_model_bundles'] for r in runs),
               'all_recorded_reused_model_bundles_including_unaccepted_attempts': sum(r['reused_model_bundles'] for r in runs),
               'all_recorded_pool_comparisons_including_unaccepted_attempts': sum(r['pool_comparisons'] for r in runs),
               'accepted_new_model_bundles': sum(r['new_model_bundles'] for r in runs if r['accepted_scientific_producer_in_report']),
               'accepted_reused_model_bundles': sum(r['reused_model_bundles'] for r in runs if r['accepted_scientific_producer_in_report']),
               'accepted_pool_comparisons': sum(r['pool_comparisons'] for r in runs if r['accepted_scientific_producer_in_report']),
               'failed_or_budget_held_process_receipts': sum(r['kind'] == 'metered_process' and r['status'] != 'COMPLETE' for r in rows),
               'administrative_bootstrap_failures': sum(r['kind'] == 'administrative_bootstrap' for r in rows),
               'counted_before_this_report_terminal_receipt': True,
               'unaccepted_COMPLETE_is_not_declared_invalid_without_bound_audit': True}
    return rows, summary


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows])


def make_figures(producers, costs, folder, check, expanded=None, expanded_costs=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'savefig.dpi': 180})
    mapping = {p['dataset']: p for p in producers}
    records = []
    def save(fig, name):
        check()
        fig.savefig(folder / (name + '.png'), bbox_inches='tight')
        fig.savefig(folder / (name + '.svg'), bbox_inches='tight')
        plt.close(fig)
        for suffix in ('png', 'svg'):
            records.append(record(folder / (name + '.' + suffix), 'figure'))
        check()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i, dataset in enumerate(DATASETS):
        p = mapping[dataset]
        for j, metric in enumerate(('mse', 'mae')):
            values = [score(p, m, FINAL[dataset])['normalized'][metric] for m in METHODS]
            ax = axes[i, j]
            ax.bar(np.arange(6), values, color=[COLORS[m] for m in METHODS])
            ax.set_xticks(np.arange(6), METHODS, rotation=30, ha='right')
            ax.set_title(dataset + ' / ' + FINAL[dataset] + ' / ' + metric.upper())
            ax.set_ylabel('FIT-normalized observed-channel macro ' + metric.upper())
            for x, y in enumerate(values):
                ax.text(x, y, f'{y:.4f}', ha='center', va='bottom', fontsize=8)
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('First diagnostic: selected past-only models; development / already exposed evaluation')
    save(fig, '01_selected_development_scores')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for i, dataset in enumerate(DATASETS):
        for j, route in enumerate(ROUTES):
            f = mapping[dataset]['families'][route]
            y = [r['tune_scores']['normalized']['mse'] for r in f['pool']]
            ax = axes[i, j]
            ax.plot(range(9), y, 'o-', color=COLORS[route])
            ax.scatter([f['selected_index']], [y[f['selected_index']]], marker='*', s=160, color='#151d29', zorder=5)
            ax.set_xticks(range(9), ['P', 'M', '0', '1e-4', '.001', '.01', '.1', '1', '10'], rotation=45)
            ax.set_title(dataset + ' / ' + route)
            ax.set_ylabel('Global TUNE macro MSE')
            ax.grid(alpha=.2)
    fig.suptitle('Same nine-candidate pool and global macro alpha selection; star = selected')
    save(fig, '02_same_pool_tune_selection')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        p = mapping[dataset]
        splits = ('fit', 'tune', FINAL[dataset])
        for method in METHODS:
            ax.plot(range(3), [score(p, method, s)['normalized']['mse'] for s in splits],
                    'o-', color=COLORS[method], label=method)
        ax.set_xticks(range(3), splits)
        ax.set_title(dataset)
        ax.set_ylabel('Observed-channel macro MSE')
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    fig.suptitle('FIT, TUNE and previously exposed development roles are distinct')
    save(fig, '03_fit_tune_development')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for i, dataset in enumerate(DATASETS):
        p = mapping[dataset]
        split = FINAL[dataset]
        for j, metric in enumerate(('mse', 'mae')):
            own = np.asarray(score(p, 'SELF33', split)['normalized']['per_channel_' + metric])
            joint = np.asarray(score(p, 'JOINT33', split)['normalized']['per_channel_' + metric])
            values = own-joint
            ax = axes[i, j]
            ax.bar(range(len(values)), values, color=np.where(values >= 0, '#278e94', '#d26f49'))
            ax.axhline(0, color='#333333', linewidth=.8)
            ax.set_xlabel('Original channel index')
            ax.set_ylabel('SELF33 - JOINT33 channel ' + metric.upper())
            ax.set_title(dataset + ' / ' + split + ' / positive = JOINT improvement')
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('Channel diagnostics: same temporal summaries; capacity differs')
    save(fig, '04_channel_information_diagnostics')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        g = mapping[dataset]['branch_lock']['common_information_gate']
        blocks = g['four_chronological_blocks']
        ax.plot(range(4), [b['self_mse'] for b in blocks], 'o-', label='SELF33', color=COLORS['SELF33'])
        ax.plot(range(4), [b['joint_mse'] for b in blocks], 'o-', label='JOINT33', color=COLORS['JOINT33'])
        ax.set_xticks(range(4), ['block1', 'block2', 'block3', 'block4'])
        ax.set_title(dataset + ' / exploratory gate ' + ('PASS' if g['passed'] else 'FAIL'))
        ax.set_ylabel('TUNE chronological-block macro MSE')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Overlapping origins are not independent samples; gate allocates resources only')
    save(fig, '05_chronological_tune_gate')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for i, dataset in enumerate(DATASETS):
        for j, batch in enumerate((1, 8)):
            selected = {r['method']: r for r in costs if r['dataset'] == dataset and r['batch'] == batch}
            values = [selected[m]['whole_median_ms_per_window'] for m in METHODS]
            ax = axes[i, j]
            ax.bar(np.arange(6), values, color=[COLORS[m] for m in METHODS])
            ax.set_xticks(np.arange(6), METHODS, rotation=30, ha='right')
            ax.set_yscale('log')
            ax.set_title(dataset + ' / B' + str(batch) + ' / median six balanced repeats')
            ax.set_ylabel('Whole milliseconds / window (log scale)')
            for x, y in enumerate(values):
                ax.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=8)
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('Whole physical-input-to-physical-CPU-return cost; one local hardware trial')
    save(fig, '06_whole_deployment_cost')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for index, dataset in enumerate(DATASETS):
        p = mapping[dataset]
        for col, metric in enumerate(('mse', 'mae')):
            ax = axes[index, col]
            for split, color in (('tune', '#278e94'), (FINAL[dataset], '#d26f49')):
                values = np.asarray(score(p, 'RAW_BOLT', split)['normalized']['per_channel_' + metric])
                ax.plot(np.arange(len(values)), values, 'o-', label=split, color=color, markersize=4)
            ax.set_yscale('symlog', linthresh=1e-8)
            ax.set_xlabel('Original channel index')
            ax.set_ylabel('Raw Bolt channel ' + metric.upper() + ' (symlog, linear <=1e-8)')
            ax.set_title(dataset + ' / same frozen Raw Bolt; different exposed periods')
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle('Observable TUNE/development error instability; no causal distribution-shift proof')
    save(fig, '08_raw_channel_period_uncertainty')
    if expanded is not None:
        first = mapping['robin']
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for ax, split in zip(axes[0], ('tune', 'val')):
            x = np.arange(2)
            for index, route in enumerate(('SELF33', 'JOINT33')):
                values = [first['families'][route]['splits'][split]['normalized']['mse'],
                          expanded['families'][route]['splits'][split]['normalized']['mse']]
                ax.plot(x, values, 'o-', label=route, color=COLORS[route])
                for position, value in enumerate(values):
                    ax.annotate(f'{value:.4f}', (position, value), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8)
            ax.set_xticks(x, ['FIT256', 'FIT512'])
            ax.set_title('Robin ' + split + ' / same feature and selection recipe')
            ax.set_ylabel('Observed-channel macro MSE')
            ax.grid(alpha=.2)
            ax.legend()
        g = expanded['branch_lock']['common_information_gate']
        blocks = g['four_chronological_blocks']
        axes[1, 0].plot(range(4), [b['self_mse'] for b in blocks], 'o-', color=COLORS['SELF33'], label='SELF33')
        axes[1, 0].plot(range(4), [b['joint_mse'] for b in blocks], 'o-', color=COLORS['JOINT33'], label='JOINT33')
        axes[1, 0].set_xticks(range(4), ['block1', 'block2', 'block3', 'block4'])
        axes[1, 0].set_title('FIT512 TUNE exploratory gate ' + ('PASS' if g['passed'] else 'FAIL'))
        axes[1, 0].set_ylabel('TUNE block macro MSE')
        axes[1, 0].grid(alpha=.2)
        axes[1, 0].legend()
        for offset, route in enumerate(('SELF33', 'JOINT33')):
            values = [next(r['whole_median_ms_per_window'] for r in expanded_costs if r['method'] == route and r['batch'] == b) for b in (1, 8)]
            axes[1, 1].bar(np.arange(2)+(offset-.5)*.3, values, width=.3, color=COLORS[route], label=route)
        axes[1, 1].set_xticks(range(2), ['B1', 'B8'])
        axes[1, 1].set_title('FIT512 selected CPU-only whole costs / separate trial')
        axes[1, 1].set_ylabel('Whole milliseconds / window')
        axes[1, 1].grid(axis='y', alpha=.2)
        axes[1, 1].legend()
        fig.suptitle('One matched FIT512 diagnostic; origins are overlapping and not a nested sample-count intervention')
        save(fig, '07_matched_FIT512_stop_diagnostic')
    return records


def make_text(producers, scientific_audit, cost_audit, costs, expanded=None, expanded_costs=None, execution_summary=None):
    mapping = {p['dataset']: p for p in producers}
    lines = ['# 공통 채널 정보와 고정 Bolt의 가치: 첫 공정 비교', '',
        '이 보고서는 승인된 새 캠페인의 첫 SELF33·JOINT33·SELF512 비교와 실제 배포 비용을 다룬다. 후속 표현 학습과 연구 전체의 완료를 뜻하지 않는다. 원수치 검산 통과는 성능 개선이나 독립 일반화 검증이 아니다.', '',
        '## 먼저 답할 질문', '',
        '**다른 채널이 도움이 되는가?** SELF33 대 JOINT33은 같은 32×16 평균+현재값을 쓰고, 같은 PERSIST·FIT 평균·7α의 전체 macro TUNE MSE로 선택했다. 입력 차원과 계수 용량은 다르므로 순수 정보의 인과 효과로 해석하지 않는다.', '']
    for dataset in DATASETS:
        p = mapping[dataset]
        g = p['branch_lock']['common_information_gate']
        a, b = g['self_tune'], g['joint_tune']
        blocks = sum(v['joint_improved'] for v in g['four_chronological_blocks'])
        lines.append(f"- {dataset}: TUNE MSE {a['mse']:.9f} → {b['mse']:.9f}, MAE {a['mae']:.9f} → {b['mae']:.9f}; MSE 감소 {g['mse_gain_pct']:.4f}%, MAE 변화 {g['mae_change_pct']:.4f}%, 시간순 4블록 중 {blocks}개 개선. 탐색 gate {'통과' if g['passed'] else '미통과'}. FIT/TUNE로 고정한 다음 분기는 `{p['branch_lock']['next_branch']}`다.")
    bself = score(mapping['beijing_aotizhongxin'], 'SELF33', 'eval')['normalized']
    bjoint = score(mapping['beijing_aotizhongxin'], 'JOINT33', 'eval')['normalized']
    lines += ['', f"반례도 남는다. Beijing의 기노출 EVAL에서 JOINT33 MSE는 {bjoint['mse']:.9f}, SELF33은 {bself['mse']:.9f}이며, MAE는 각각 {bjoint['mae']:.9f}, {bself['mae']:.9f}다. TUNE와 개발 EVAL의 이득 방향 및 MSE/MAE가 일치하지 않는 부분이 있으므로 ‘다른 채널이 모든 기간에서 쓸모없다’는 결론을 낼 수 없다. 이 EVAL 반례로 이미 고정한 TUNE gate를 사후에 뒤집지도 않았다."]
    raw_tune = score(mapping['robin'], 'RAW_BOLT', 'tune')['normalized']
    raw_dev = score(mapping['robin'], 'RAW_BOLT', 'val')['normalized']
    near_tune = sum(v <= 1e-8 for v in raw_tune['per_channel_mse'])
    near_dev = sum(v <= 1e-8 for v in raw_dev['per_channel_mse'])
    lines += ['', f"Robin Raw Bolt도 TUNE MSE/MAE {raw_tune['mse']:.9f}/{raw_tune['mae']:.9f}에서 VAL {raw_dev['mse']:.9f}/{raw_dev['mae']:.9f}로 달라진다. 채널 MSE ≤1e−8인 수는 TUNE {near_tune}/17, VAL {near_dev}/17이다. 같은 frozen F·FIT 통계에서 관측한 기간별 오차의 불안정성이지, 분포 변화의 인과 원인이나 누출을 이 수치만으로 확정한 것이 아니다.", '',
              '![Raw 채널별 기간 오차 변화](figures/08_raw_channel_period_uncertainty.png)']
    lines += ['', '**강한 자기 채널 대안과 Bolt는 어떠한가?** SELF512는 시간 해상도가 다르므로 공통 정보의 직접 대조가 아니라 강한 단순 기준이다. Raw Bolt는 같은 origins·FIT 정규화·H48의 원채널 예측을 재사용하고, 비용에서는 실제 캐시 모델을 새로 추론했다. 아래 개발 점수는 분기 선택에 쓰지 않았다.', '', '```text', 'dataset                method     split  selected/alpha       MSE          MAE']
    for dataset in DATASETS:
        p = mapping[dataset]
        for method in METHODS:
            value = score(p, method, FINAL[dataset])['normalized']
            family = p['families'].get(method, {})
            chosen = family.get('selected_kind', method)
            alpha = family.get('selected_alpha')
            description = chosen + (('/' + str(alpha)) if alpha is not None else '')
            lines.append(f"{dataset:<23}{method:<11}{FINAL[dataset]:<7}{description:<21}{value['mse']:.9f}  {value['mae']:.9f}")
    lines += ['```', '', '![선택된 방법의 개발 오차](figures/01_selected_development_scores.png)', '',
        '**누출은 있었는가?** 이 단계의 배포 readout과 Raw Bolt는 과거 history/current anchor만 받았다. FIT 특징 통계·FIT 평균·FIT ridge와 TUNE 선택, DEV payload 이전 분기 잠금을 저장했고 보호 TEST는 접근하지 않았다. 검산의 미래 교란과 산술 gate는 검사한 계약의 통과이며, 실제 모든 가능한 누출이나 독립 일반화를 증명하지 않는다.', '',
        '**압축기·잠재 예측·디코더 중 무엇이 해결되었는가?** 첫 비교에는 압축과 디코더를 넣지 않았다. 공통 정보에 대한 공정한 readout 근거를 확보한 단계다. 작은 K의 손실 보존과 Bolt/native의 추가 가치는 승인 계획의 다음 분기 또는 기존 동일 조건 근거로 판단해야 한다.', '',
        '**대조학습과 동적 그래프가 필요한가?** 이번 비교로 필요성을 입증할 수 없다. 단순 self/joint의 자원 gate에 따라 후속을 결정하며, 새로운 대조 손실이나 그래프를 임의로 추가하지 않는다.', '',
        '## 비교 계약과 원수치', '',
        'Robin C17·Beijing Aotizhongxin C11, L512/H48, FIT256/TUNE64/VAL30·기노출 EVAL146를 유지했다. 특징 center/population scale과 원자료 정규화는 FIT에서만 정했다. FP32 Y−현재값을 만든 뒤 FP64 centered masked ridge를 학습하며 λ=αNc, 절편 비규제, FP32 innovation+현재값 순서를 썼다. α는 0·1e−4·.001·.01·.1·1·10이고 세 경로 모두 하나의 전체 macro TUNE α를 선택했다.', '',
        '결정적 ridge를 신경망 seed 두 번의 독립 반복으로 세지 않았다. 새/재사용 모델 묶음과 내부 채널 submodel·mask factorization은 별도 기록한다. JOINT 후보는 이전 7개 선택을 덮지 않고 같은 두 통제를 추가해 새 9pool에서 선택했다.', '',
        '[전체 지표 CSV](tables/COMPARISON.csv) · [54개 후보 CSV](tables/POOL_SELECTION.csv) · [용량/선택 여유 CSV](tables/CAPACITY.csv) · [24개 비용 CSV](tables/COST.csv)', '',
        '![같은 9pool 선택](figures/02_same_pool_tune_selection.png)', '',
        '![FIT/TUNE/개발 역할](figures/03_fit_tune_development.png)', '',
        '![채널별 차이](figures/04_channel_information_diagnostics.png)', '',
        '![시간순 TUNE 자원 gate](figures/05_chronological_tune_gate.png)', '',
        '## 실제 추론 비용과 한계', '',
        '각 자료에서 같은 첫 1/8 origins로 B1/B8을 측정했다. 6경로 각각 warmup2회 뒤 순서를 균형 배치한 실제 6회 반복을 썼다. 타이머에는 원단위 과거의 창별 causal forward-fill→FIT 정규화→CPU selected readout 또는 H2D·고정 Bolt·D2H→동일 CPU inverse→원단위 CPU 반환·최종 동기화를 포함했다. 디스크/모델/계수 로드·참조 추론·SHA·파일 저장·guard는 타이머 밖이다.', '',
        'PERSIST/MEAN이 선택되면 그 경로의 실제 통제 비용을 재고 alias를 보존했다. ridge 계수는 비용 측정 중 재학습하지 않았다. 메모리는 공유 Bolt와 여러 CPU 계수가 상주하는 조건의 전체 peak이며 방법별 단독 allocation이나 일반적인 속도 우위가 아니다. 열/클록/carryover와 독립적인 하드웨어 반복은 확인하지 않았다. CPU readout은 현재 NumPy 구현으로, 최적화된 native 구현이라고 주장하지 않는다.', '',
        '![전체 배포 비용](figures/06_whole_deployment_cost.png)', '',
        '## 검산 범위와 다음 판단', '',
        f"첫 산술 독립 검산의 상태는 `{scientific_audit['status']}`이며 54개 후보의 저장 예측·관측 마스크·특징 통계·선택·정규방정식·origin·계보를 검사했다. 비용 저장 배열 검산은 `{cost_audit['status']}`이며 {cost_audit['whole_rows_checked']}조건을 확인했다. 비용 검산은 실제 타이밍을 새로 재측정한 독립 하드웨어 실험이 아니다.", '',
        'Raw Bolt의 실제 frozen state와 모델 바이너리 SHA, config 및 normalizer의 class/eps/arcsinh/forward parity를 비용 전후 기록했다. weights SHA만으로 동작이 같다고 대신 주장하지 않았다. 실제 모델은 캐시를 사용했고 설치·다운로드·백본 재학습은 없었다.', '',
        'TUNE gate의 1% MSE·1% MAE·3/4블록 기준은 자원 배분 기준이며 유의성 검정이 아니다. 중첩 origins와 작은 자료 수 때문에 효과의 불확실성이 남는다. 이 보고서의 다음 분기는 이미 FIT/TUNE에서 고정되었고, 전체 새 표현 계열·CPU/GPU·wall 상한 안에서만 후속을 진행한다. 독립 확증은 별도 설계가 필요하며 보호 TEST는 이번 승인 범위에 없다.', '',
        '[검산과 파일 계보](MANIFEST.json) · [비용 저장 배열 검산](COST_ARRAY_AUDIT.json) · [실행 계획](../../PHASE1_PLAN_KO.md) · [실행 상태와 실패 기록](../../RUN_STATUS.md)', '']
    if expanded is not None:
        g = expanded['branch_lock']['common_information_gate']
        lines += ['## Robin의 한 번뿐인 FIT512 표본·규제 진단', '',
            f"첫 FIT256에서 JOINT의 FIT 오차는 낮지만 TUNE가 나빠 승인된 1회 진단을 실행했다. 동일 FIT 시간 구간·L512/H48·FIT 정규화·TUNE64·VAL30·특징·9pool·α 선택을 유지하고 self/joint를 함께 FIT512로 학습했다. 기존256 origins와 새512 origins의 교집합은 {expanded['expanded_and_original_origin_overlap_count']}개이며 포함 관계가 아니다. 중첩 창도 독립 표본이 아니므로 ‘표본 수만 두 배로 바꾼 인과 실험’으로 해석하지 않는다.", '',
            '추가 Robin의 원 trainval x 멤버에는 이미 노출된 VAL 행도 들어 있다. 이 멤버를 읽어 FIT prefix만 특징·학습 목표로 구성하고 원256과 새512의 재구성을 검사했다. 새 DEV 점수화 전에 분기를 고정했지만, 해당 멤버의 VAL 바이트가 물리적으로 읽히지 않았다고 주장하지 않는다. 학습/분기에서 FIT/TUNE만 사용한 계약과 실제 파일 I/O의 범위를 구분한다.', '',
            f"확대 후 TUNE SELF33 MSE {g['self_tune']['mse']:.9f}, JOINT33 MSE {g['joint_tune']['mse']:.9f}; SELF33 MAE {g['self_tune']['mae']:.9f}, JOINT33 MAE {g['joint_tune']['mae']:.9f}. MSE 감소 {g['mse_gain_pct']:.4f}%, MAE 변화 {g['mae_change_pct']:.4f}%, 개선 블록 {sum(b['joint_improved'] for b in g['four_chronological_blocks'])}/4. gate는 {'통과' if g['passed'] else '미통과'}이고 다음 결정은 `{expanded['branch_lock']['next_branch']}`이다.", '',
            '```text', 'FIT origins  method   split  selected/alpha       MSE          MAE']
        for count, p in ((256, mapping['robin']), (512, expanded)):
            for route in ('SELF33', 'JOINT33'):
                f = p['families'][route]
                label = f['selected_kind'] + ('/' + str(f['selected_alpha']) if f['selected_alpha'] is not None else '')
                for split in ('fit', 'tune', 'val'):
                    value = f['splits'][split]['normalized']
                    lines.append(f"{count:<13}{route:<9}{split:<7}{label:<21}{value['mse']:.9f}  {value['mae']:.9f}")
        lines += ['```', '', '![FIT512 추가 진단과 중단 판단](figures/07_matched_FIT512_stop_diagnostic.png)', '',
            '[FIT512 원수치](tables/EXPANDED_COMPARISON.csv) · [추가18pool](tables/EXPANDED_POOL_SELECTION.csv) · [추가 CPU4비용](tables/EXPANDED_COST.csv)', '',
            '추가4비용은 실제 선택된 FIT512 SELF33/JOINT33 × B1/B8의 CPU 전용 별도 process다. 각각 warmup2·실제6repeat이고 두 순서 위치에3회씩 배치했다. 같은 실제 원단위 VAL 과거에서 whole 예측을 실행했다. 이 process에는 Bolt를 상주시켜 두지 않았고 새 Raw 비용을 재지 않았으므로 first24의 GPU 상주 조건과 메모리·속도 우위를 직접 비교하지 않는다.', '',
            '## 미실행 후속과 연구 판단', '',
            '공통 정보 gate가 새 작은-K 보존 비교의 선행 조건이므로 ForeCA 원리 projection과 예측 목적 linear E/D는 이번에 새 학습하지 않았다. 새 native head는 유효한 공통 E/D의 Bolt 추가 가치가 불명확할 때 여는 대조인데 선행 공통 경로가 선택되지 않았다. 추가 RAW33 잔차 결합은 유용한 공통 경로와 개별 정보를 분리하는 후속이며, 기존 MAIN84fit는 이미 있으므로 이름만 바꿔 반복하지 않았다. 긴 Raw L 입력은 입력 범위가 남은 경쟁 설명일 때의 대조다. 이번에는 동일 L512의 자기 채널보다 joint가 TUNE 자원 gate를 넘지 못했고, 허용된 표본·규제 진단 뒤 같은 계열 중단 조건에 도달해 긴 입력 분기로 확대하지 않았다. 이것은 각 미실행 방법의 성능 실패를 주장하는 것이 아니다. 기존 TS2Vec/GTConvAE/LEVEL/동일 E-D의 Bolt/native 결과는 [Phase0 감사](../../PHASE0_AUDIT_KO.md)에 별도로 남아 있다.', '',
            f"이 보고서의 PASS 감사에 묶인 새로운 비교는 {execution_summary['accepted_new_model_bundles']}개의 결정적 ridge 모델 묶음과 재사용 {execution_summary['accepted_reused_model_bundles']}개, {execution_summary['accepted_pool_comparisons']}pool 비교다. 보존한 미채택 시도까지 포함한 실제 기록은 새 {execution_summary['all_recorded_new_model_bundles_including_unaccepted_attempts']}개·재사용 {execution_summary['all_recorded_reused_model_bundles_including_unaccepted_attempts']}개·{execution_summary['all_recorded_pool_comparisons_including_unaccepted_attempts']}pool이다. 실행한 것과 검산 후 본 보고서에 채택한 것을 구분한다. 각 묶음 내부의 채널별/마스크별 factorization은 용량 CSV에 따로 남겼다. first2자료와 추가Robin은 서로 독립적인 세 데이터셋이나 세 seed 반복이 아니다.", '',
            '현재 결론은 **이 Robin·Beijing 계약에서 작은-K 공통 표현을 주연구로 밀어갈 단순 readout 근거가 부족하다**는 조건부 보류다. 재구성 목적의 표현이 예측에 부적합하다는 가설, 학습 최적화 문제, 비선형·지연 의존성의 존재를 이 선형 비교만으로 확정하거나 기각하지 않는다. Raw Bolt를 유지할 실용 근거와 새로운 압축 방법의 연구 기여는 구분해야 하며, 독립 평가와 다른 사전 선정 자료의 공통 정보 근거는 아직 남는다.', '',
            '## 보존한 실행 실패와 재개 조건', '',
            '첫 독립 검산 v1은 Beijing metadata의 list/dict 형식 처리에서 실패했다. 원 결과를 보존하고 새 검산 v2에서 고쳤다. 첫 GPU 비용 시도는 기본 Conda Python에 chronos 패키지가 없어 모델을 로드하지 못하고0개 비용 조건으로 실패했다. 설치하거나 기존 시계를 재설정하지 않고 이미 있던 .venv Python으로 새 tag에서 동일 소스를 실행했다. 첫 비용 배열 검산도 producer의 배열 SHA metadata 형식 처리에서 실패하여 새 검산에 정확한 형식을 적용했다. 실패 시도는 삭제하거나 완료로 바꾸지 않았다. 이 실패들은 과학적 방법의 성능 실패와 구분한다. NumPy·Python 환경이 다른 결과의 재현 범위도 exact saved-artifact·선택·parity gate로 제한했다.', '',
            '최초 dense 진단에는 별도의 실제 산술 문제가 있었다. 원 history의 바이트는 같았지만 FP32 시간 요약의 메모리 배치에 따라 reduction 값이 달랐다. center 최대 차이는5.25e−8이고, JOINT α0 예측의 최대 차이는4.37498e−5로 기존 parity 기준을 넘었다. 허용 오차를 느슨하게 하지 않고 canonical memory 배치를 명시해 같은 FIT/origins/마스크/통계/α/목표로 새 시도를 학습·검산했다. 이는 새 방법이나 범용적인 표현학습 실패 원인이 아니라 이 산술 구현의 재현 문제 수정이다. 최초14fit와 수선한14fit를 모두 실행 기록에 세되, 현재의18pool 감사 PASS producer만 유효 비교에 사용한다.', '',
            f"본 보고서 terminal receipt 기록 전까지 실패/예산 보류 process receipt는 {execution_summary['failed_or_budget_held_process_receipts']}개였다. 전체 시도와 실제 상태·오류는 [EXECUTION_ATTEMPTS.csv](tables/EXECUTION_ATTEMPTS.csv)에 남겼다. producer의 `COMPLETE` 실행 상태만으로 해당 시도의 독립 산술 검산이 PASS했다고 바꾸어 쓰지 않았다. 보존한 미채택 dense 시도나 metadata 수선은 현재 보고서의 정확한 accepted producer SHA와 감사에서 분리했다.", '',
            f"행정적인 manifest 생성 실패는 별도로 {execution_summary['administrative_bootstrap_failures']}건 보존했다. PowerShell의 List→array 변환 오류를 고치며 원 승인 시작/만료 시각은 유지했다. 이 행정 오류는 실험 실행이나 새 캠페인 시계 시작으로 세지 않았다.", '',
            '재개는 gate가 넘지 못한 방법을 자동으로 늘리는 방식이 아니다. 공통 정보를 잘 지원하는 자료/변수 정의 또는 비선형 대조에 대한 별도 목적·데이터 계약을 정할 때 새 범위를 합의한다. 이번 승인 안의 중단은 자원 소진이나 성능 개선 보장이 아니라 기록된 과학적 gate의 결정이다.', '']
    return '\n'.join(lines)


def make_presentation_text(producers, costs, expanded=None):
    by_dataset = {p['dataset']: p for p in producers}
    lines = ['# 교수님 미팅용 6장 발표 스토리', '',
        '실제 PPT가 아닌 검산된 원수치 기반 구성안이다. [상세 한국어 보고서](REPORT_KO.md)를 함께 사용한다.', '',
        '## 1. 문제: 작은 K가 무엇을 보존해야 하는가', '',
        'TSFM은 여러 시계열로 사전학습한 예측 모델이다. 현재 Chronos-Bolt wrapper는 C개 채널을 각각 같은 고정 모델에 넣는다. 압축을 연구하려면, 단순 자기 채널 예측보다 다른 채널을 함께 볼 실용적 이득이 있는지 먼저 확인해야 한다. 문제는 복원 오차를 줄이는 것만으로 미래 예측 정보를 보존했다는 보장이 없다는 점이다.', '',
        '질문은 ‘같은 과거 특징·표본·목표·선택에서 모든 C를 보면 실제 미래 C를 더 잘 맞히는가, 그 정보의 작은-K 경로가 Raw 대비 비용을 줄일 여지가 있는가’다. 이번 단계는 첫 문장부터 검사한다.', '',
        '## 2. 공정 비교 계약', '',
        'Robin C17·Beijing C11, L512/H48, FIT256/TUNE64, 관측 channel-macro normalized MSE를 유지했다. SELF33은 자기 채널32×16 평균+현재값, JOINT33은 같은 모든 채널 특징, SELF512는 강한 자기 채널 전체 과거 기준이다. 세 경로 모두 현재값 anchor·actual-Y innovation·FIT 특징 정규화·λ=αNc·동일9pool·글로벌 macro TUNE α 선택을 쓴다. Raw는 같은 조건의 강한 실용 기준이다.', '',
        '![같은 후보와 선택](figures/02_same_pool_tune_selection.png)', '',
        '계수·rank·특징 차원은 다르므로 채널 정보의 순수한 인과 효과나 독립 일반화로 과장하지 않는다.', '',
        '## 3. 공통 정보의 TUNE 결과와 반례', '']
    for dataset in DATASETS:
        g = by_dataset[dataset]['branch_lock']['common_information_gate']
        lines.append(f"- {dataset}: SELF33→JOINT33 TUNE MSE {g['self_tune']['mse']:.6f}→{g['joint_tune']['mse']:.6f}, MAE {g['self_tune']['mae']:.6f}→{g['joint_tune']['mae']:.6f}; 탐색 gate {'통과' if g['passed'] else '미통과'}.")
    lines += ['', 'Beijing의 노출 EVAL에는 JOINT의 MSE가 개선되고 MAE는 악화하는 반례가 있다. Robin Raw의 TUNE에는 거의0인 채널 오차가 많지만 VAL에서 달라진다. 기간별 불안정성을 남기고, DEV를 보고 TUNE 분기를 뒤집지 않는다.', '',
              '![Raw 기간별 채널 오차](figures/08_raw_channel_period_uncertainty.png)', '',
              '## 4. 작은 FIT의 과적합 설명을 한 번 검사', '']
    if expanded is not None:
        g = expanded['branch_lock']['common_information_gate']
        lines += [f"Robin self/joint를 동일 FIT 구간의512 origins로 함께 다시 학습했다. 원256과 교집합은 {expanded['expanded_and_original_origin_overlap_count']}개이고 중첩 창이며 포함 관계가 아니므로 독립 표본 수만의 효과로 주장하지 않는다. TUNE MSE SELF33 {g['self_tune']['mse']:.6f}, JOINT33 {g['joint_tune']['mse']:.6f}; 탐색 gate {'통과' if g['passed'] else '미통과'}다.", '',
                  '![한 번의 FIT512 진단](figures/07_matched_FIT512_stop_diagnostic.png)']
    else:
        lines.append('추가 진단의 실행 결과는 이 보고서 범위에 아직 포함되지 않았다.')
    lines += ['', '## 5. 실제 배포 비용을 정확도와 함께 읽기', '',
        'B1/B8 whole에는 실제 원단위 과거의 causal forward-fill·FIT 정규화·전송/예측·동일 CPU inverse·원단위 CPU 반환을 포함했다. 각각warmup2·순서 균형6repeat다. 모델 로드와 저장은 제외했다. 메모리는 Bolt/계수 co-resident 조건이고 단독 방법 메모리가 아니다.', '',
        '![실제 전체 추론 비용](figures/06_whole_deployment_cost.png)', '',
        'FIT512의4조건은 GPU가 없는 별도 CPU process이며 Raw를 새로 재지 않아 first24와 직접적인 속도 우위를 주장하지 않는다. 빠르기만 하거나 한 지표만 좋아지는 것은 방법 채택의 충분조건이 아니다.', '',
        '## 6. 이번 기여와 남은 질문', '',
        '이번에 확인한 것은 조건이 다른 기존 self/joint 비교를 동일 특징·글로벌 α·9pool로 맞추고, 단순 기준·실제 비용·선택 불안정성·한 번의 표본 진단을 연결한 근거다. 작은-K나 비선형 표현이 불가능하다고 증명한 연구 기여가 아니다.', '',
        'TUNE 선행 gate를 통과하지 못한 이 자료/계약에서는 새로운 ForeCA·예측 E/D 학습을 진행하지 않고 조건부 보류한다. 미실행 방법의 실패로 쓰지 않는다. 재구성 목표의 불일치, 학습 최적화, 비선형/지연 정보의 가능성은 남았다. 적합한 사전 선정 자료와 강한 self 대조·독립 평가 설계가 다음 판단 조건이다. 기존 AdaPTS와의 직접 중복도 [선행연구 검토](../../LITERATURE_REVIEW_KO.md)에 남긴다.', '']
    return '\n'.join(lines)


def execute(args, limit):
    output = HERE / 'reports' / args.tag
    output.parent.mkdir(exist_ok=True)
    output.mkdir(exist_ok=False)
    terminal = None
    result = {'status': 'RUNNING', 'protected_TEST_accessed': False, 'independent_holdout': False,
              'report_scope': 'first readout diagnostic and actual whole deployment costs', 'input_records': []}
    failure = None
    try:
        limit.checkfull()
        producer_records = [record(p, 'readout_producer') for p in args.readout_results]
        producers = [read(r['path']) for r in producer_records]
        audit_record = record(args.audit, 'independent_saved_array_audit')
        scientific_audit = read(audit_record['path'])
        validate_scientific(producers, producer_records, scientific_audit,
                            limit.authorization_record['sha256'], limit.run_manifest_record['sha256'])
        cost_record = record(args.cost, 'actual_whole_cost_producer')
        cost = read(cost_record['path'])
        independent_cost_record = record(args.cost_audit, 'independent_whole_cost_audit')
        independent_cost_audit = read(independent_cost_record['path'])
        validate_independent_cost(independent_cost_audit, cost_record)
        cost_audit = audit_cost_saved_arrays(cost, cost_record, producer_records, limit.shared.checkfast)
        result['input_records'] = producer_records + [audit_record, cost_record, independent_cost_record]
        expanded = expanded_costs = None
        expanded_records = []
        expanded_flags = (args.expanded_result, args.expanded_audit, args.expanded_cost, args.expanded_cost_audit)
        if any(expanded_flags) and not all(expanded_flags):
            raise ValueError('ALL_FOUR_EXPLICIT_EXPANDED_RESULT_AUDIT_COST_BINDINGS_REQUIRED')
        if all(expanded_flags):
            expanded_record = record(args.expanded_result, 'matched_FIT512_producer')
            expanded = read(expanded_record['path'])
            expanded_audit_record = record(args.expanded_audit, 'independent18_pool_expanded_audit')
            expanded_audit = read(expanded_audit_record['path'])
            validate_expanded(expanded, expanded_record, expanded_audit, producer_records,
                              limit.authorization_record['sha256'], limit.run_manifest_record['sha256'])
            expanded_cost_record = record(args.expanded_cost, 'actual_CPU_four_cell_expanded_cost')
            expanded_cost = read(expanded_cost_record['path'])
            expanded_cost_audit_record = record(args.expanded_cost_audit, 'independent_CPU_expanded_cost_audit')
            expanded_cost_audit = read(expanded_cost_audit_record['path'])
            validate_expanded_cost(expanded_cost, expanded_cost_record, expanded_cost_audit, expanded_record)
            expanded_costs = expanded_cost_rows(expanded_cost)
            expanded_records = [expanded_record, expanded_audit_record, expanded_cost_record, expanded_cost_audit_record]
            result['input_records'].extend(expanded_records)
        provenance = output / 'provenance'
        provenance.mkdir(exist_ok=False)
        generator_record = record(Path(__file__), 'actual_report_generator_source')
        shutil.copyfile(Path(__file__), provenance / Path(__file__).name)
        result['report_generator_source'] = {**generator_record, 'snapshot_path': str(provenance/Path(__file__).name)}
        result['input_snapshots'] = []
        for index, r in enumerate(result['input_records']):
            target = provenance / (str(index).zfill(2) + '_' + r['role'] + '.json')
            shutil.copyfile(r['path'], target)
            if sha(target) != r['sha256']:
                raise ValueError('REPORT_INPUT_JSON_SNAPSHOT_SHA_MISMATCH')
            result['input_snapshots'].append({'input_record': r, 'snapshot_path': str(target), 'sha256': sha(target)})
        accepted_producers = producer_records + ([expanded_records[0]] if expanded_records else [])
        attempts, execution_summary = execution_attempt_rows(accepted_producers)
        result['execution_summary'] = execution_summary
        tables, figures = output / 'tables', output / 'figures'
        tables.mkdir(exist_ok=False)
        figures.mkdir(exist_ok=False)
        metrics, pools, capacities, costs = metric_rows(producers), pool_rows(producers), capacity_rows(producers), cost_rows(cost)
        for name, rows in (('COMPARISON', metrics), ('POOL_SELECTION', pools), ('CAPACITY', capacities), ('COST', costs)):
            limit.checkfull()
            write_csv(tables / (name + '.csv'), rows)
        write_csv(tables/'EXECUTION_ATTEMPTS.csv', attempts)
        if expanded is not None:
            for name, rows in (('EXPANDED_COMPARISON', expanded_metric_rows(expanded)),
                               ('EXPANDED_POOL_SELECTION', pool_rows([expanded])),
                               ('EXPANDED_CAPACITY', capacity_rows([expanded])),
                               ('EXPANDED_COST', expanded_costs)):
                limit.checkfull()
                write_csv(tables / (name + '.csv'), rows)
        with (output / 'COST_ARRAY_AUDIT.json').open('x', encoding='utf-8') as handle:
            json.dump(cost_audit, handle, ensure_ascii=False, indent=2, allow_nan=False)
        figure_records = make_figures(producers, costs, figures, limit.checkfull, expanded, expanded_costs)
        with (output / 'REPORT_KO.md').open('x', encoding='utf-8') as handle:
            handle.write(make_text(producers, scientific_audit, cost_audit, costs, expanded, expanded_costs, execution_summary))
        with (output / 'PRESENTATION_BRIEF_KO.md').open('x', encoding='utf-8') as handle:
            handle.write(make_presentation_text(producers, costs, expanded))
        result.update(status='COMPLETE', scientific_audit_PASS=True, cost_saved_array_audit_PASS=True,
                      metric_rows=len(metrics), pool_rows=len(pools), capacity_rows=len(capacities), cost_rows=len(costs),
                      figures=figure_records, report=record(output/'REPORT_KO.md', 'Korean_first_stage_report'),
                      comparison=record(tables/'COMPARISON.csv', 'comparison'),
                      presentation_brief=record(output/'PRESENTATION_BRIEF_KO.md', 'six_slide_story_MD_not_actual_PPT'),
                      expanded_model_bundles=expanded.get('new_model_bundles', 0) if expanded is not None else 0,
                      expanded_pool_rows=expanded.get('pool_comparisons', 0) if expanded is not None else 0,
                      expanded_cost_rows=len(expanded_costs) if expanded_costs is not None else 0,
                      independent_cost_audit=independent_cost_record,
                      permission_gate=limit.authorization_record, run_manifest=limit.run_manifest_record,
                      phase1_whole_complete_claim=False,
                      model_or_GPU_executed_in_report=False, numeric_payload_access='cost saved outputs only')
        for r in result['input_records']:
            if sha(r['path']) != r['sha256']:
                raise ValueError('REPORT_SOURCE_CHANGED_DURING_GENERATION')
        if sha(Path(__file__)) != generator_record['sha256'] or sha(result['report_generator_source']['snapshot_path']) != generator_record['sha256']:
            raise ValueError('REPORT_GENERATOR_SOURCE_CHANGED_DURING_GENERATION')
        limit.checkfull()
    except Exception as error:
        import adaptive_runtime_v1 as runtime
        failure = {'type': type(error).__name__, 'message': str(error)}
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error, runtime.BudgetHeld) else 'FAILED_PRESERVED', failure=failure)
    finally:
        terminal = limit.close(result['status'], failure)
        result['resource_terminal'] = terminal
        result['status'] = terminal['terminal_status']
        with (output / 'MANIFEST.json').open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        print(json.dumps({'report': str(output/'REPORT_KO.md'), 'status': result['status']}), flush=True)
    return 0 if result['status'] == 'COMPLETE' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--readout-results', nargs=2, required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--cost', required=True)
    parser.add_argument('--cost-audit', required=True)
    parser.add_argument('--expanded-result')
    parser.add_argument('--expanded-audit')
    parser.add_argument('--expanded-cost')
    parser.add_argument('--expanded-cost-audit')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'actual_report_execution': False, 'figure_count_planned': 8 if args.expanded_result else 7}))
        return 0
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    import adaptive_runtime_v1 as runtime
    limit = runtime.ContinuationLimit(args.tag, gpu=False)
    try:
        return execute(args, limit)
    except BaseException as error:
        if not limit.closed:
            limit.close('FAILED_PRESERVED', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
