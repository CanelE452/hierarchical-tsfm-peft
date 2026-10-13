"""Bind unchanged science artifacts to repaired paths and clean cost records."""
from pathlib import Path
import argparse
import copy
import json
import time
import experiment as ex
from measure_resolved_cost import resolve


def artifact(path):
    return {'path': str(path.relative_to(ex.ROOT)), 'sha256': ex.sha(path)}


def run(tag, full_tag, repair_tag, cost_tag):
    start = time.perf_counter()
    process = ex.psutil.Process()
    cpu_start = sum(process.cpu_times()[:2])
    records, source_runs = resolve(full_tag, repair_tag)
    full_path, repair_path = [ex.ROOT/source_runs[k]['path'] for k in ('full', 'repair')]
    full, repair = ex.read(full_path), ex.read(repair_path)
    clean_path = ex.HERE/('run_'+cost_tag)/'RESULT.json'
    clean = ex.read(clean_path)
    if clean['status'] != 'COMPLETE' or len(clean['records']) != 236:
        raise RuntimeError('Complete236 clean cost records required')
    if clean['source_runs'] != source_runs:
        raise AssertionError('CLEAN_COST_SOURCE_RUN_BINDING_MISMATCH')
    cost_lookup = {(r['dataset'], r['label'], r['method']): r for r in clean['records']}
    cases = []
    for resolved in records:
        case = copy.deepcopy(resolved['case'])
        case['source_run'] = artifact(resolved['source_path'])
        for method in case['methods']:
            timing = cost_lookup[(case['dataset'], case['label'], method['method'])]
            if timing['science_prediction'] != method['prediction']:
                raise AssertionError('CLEAN_COST_SCIENCE_ARTIFACT_MISMATCH')
            method['clean_latency'] = timing['latency']
            method['clean_cost_source'] = artifact(clean_path)
            method['original_latency_preserved'] = True
        cases.append(case)
    core_keys = {(c['dataset'], c['seed'], c['objective'], c['config'], c['k'], c['length'], c['receptive_field'], c['initialization']['kind'])
                 for c in cases if c['stage'] in ('G1', 'G2')}
    if len(core_keys) != 44:
        raise AssertionError('CORE_EXPERIMENT_CONDITION_COVERAGE44')
    cache_runs = {name: ex.CACHE/('run_'+name) for name in (full_tag, repair_tag)}
    completion_evidence = {}
    for name, folder in cache_runs.items():
        completion_evidence[name] = {'core_u0_checkpoint_files': [artifact(p) for p in sorted(folder.rglob('*u0.pt'))],
            'core_u600_checkpoint_files': [artifact(p) for p in sorted(folder.rglob('*u600.pt'))],
            'native_completed_prediction_files': [artifact(p) for p in sorted(folder.rglob('NATIVE.npz'))],
            'all_prediction_npz_count': len(list(folder.rglob('*.npz')))}
    attempts = len(core_keys)+len(repair['attempted_specs'])
    core600 = sum(len(v['core_u600_checkpoint_files']) for v in completion_evidence.values())
    native600 = sum(len(v['native_completed_prediction_files']) for v in completion_evidence.values())
    resource_receipts = []
    for path in sorted((ex.HERE/'resources').glob('*.json')):
        value = ex.read(path)
        if value['status']=='RUNNING':
            raise RuntimeError('Study process resource still active')
        resource_receipts.append({**artifact(path), 'record': value})
    external = []
    external_folder = ex.ROOT/'research/forecast_metric_alignment_20261013'
    for path in sorted(external_folder.glob('run_*/RESOURCE_USAGE.json')):
        value = ex.read(path)
        if value['status'] not in ('COMPLETE', 'FAILED_PRESERVED'):
            raise RuntimeError('Supplement resource still active')
        external.append({**artifact(path), 'record': value})
    own_wall = sum(r['record']['wall_seconds'] for r in resource_receipts)
    external_wall = sum(r['record']['wall_seconds'] for r in external)
    total_wall = own_wall+external_wall
    if total_wall >= 2700:
        raise RuntimeError('GPU_PROCESS_CAP_EXCEEDED')
    storage = {str(root.relative_to(ex.ROOT)): sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
               for root in (ex.HERE, ex.CACHE, external_folder, ex.ROOT/'.cache/forecast_metric_alignment_20261013')}
    output = ex.HERE/('run_'+tag)
    output.mkdir(exist_ok=False)
    result = {'status': 'COMPLETE', 'aggregation_only': True, 'new_training_updates_in_aggregation': 0,
        'cases': cases, 'method_count': 236, 'case_count': 70, 'new_training_runs': 44,
        'new_native_training_conditions': 32, 'primary_metric': full['primary_metric'],
        'H': full['H'], 'FIT': full['FIT'], 'TUNE': full['TUNE'], 'device': full['device'], 'dtype': full['dtype'],
        'tf32_matmul': full['tf32_matmul'], 'tf32_cudnn': full['tf32_cudnn'], 'scope': full['scope'],
        'protected_test_guarded': full['protected_test_guarded'], 'contracts': full['contracts'], 'fixtures': full['fixtures'],
        'bolt_initial_hash': full['bolt_initial_hash'], 'bolt_final_hash': full['bolt_final_hash'],
        'source_hashes': full['source_hashes'], 'source_hashes_unchanged_at_completion': full['source_hashes_unchanged_at_completion'],
        'aggregation_source': artifact(Path(__file__)),
        'source_runs': {**source_runs, 'clean_cost': artifact(clean_path)},
        'source_full_status': full['status'], 'source_repair_status': repair['status'],
        'source_failure_history_preserved': full.get('failed_paths', []),
        'configuration_selection': repair['configuration_checks'], 'gradient_cuda_fixtures': repair['gradient_cuda_fixtures'],
        'gradient_cpu_diagnostic': repair['gradient_diagnostic'],
        'attempt_accounting': {'distinct_planned_core_conditions': 44, 'core_fit_attempts_including_repairs': attempts,
            'completed_core600_fit_runs_including_duplicate_repair': core600,
            'core600_optimizer_steps': core600*600,
            'native600_fit_runs_including_duplicate_repair': native600, 'native600_optimizer_steps': native600*600,
            'core_fit_attempts_without_u600_checkpoint': attempts-core600,
            'original_partial_prediction_npz_preserved_not_in_completed_science': completion_evidence[full_tag]['all_prediction_npz_count']-full['method_count'],
            'completion_evidence': completion_evidence,
            'note': 'u600 checkpoint proves saved completion; NATIVE.npz is written only after fixed600 native loop. Temporary restored preflight/diagnostic updates and supplemental fits are separate.'},
        'resources': {'own_gpu_process_wall_seconds': own_wall, 'supplement_gpu_process_wall_seconds': external_wall,
            'all_gpu_process_wall_seconds_summed_conservatively': total_wall, 'gpu_process_wall_cap_seconds': 2700,
            'remaining_gpu_process_wall_seconds': 2700-total_wall,
            'own_receipts': resource_receipts, 'supplement_receipts': external, 'storage_bytes_before_aggregation': storage,
            'total_storage_bytes_before_aggregation': sum(storage.values()), 'storage_cap_bytes': 4*1024**3,
            'peak_sampled_rss_bytes': max(r['record']['peak_sampled_rss_bytes'] for r in resource_receipts+external),
            'peak_allocated_vram_bytes': max(r['record']['peak_allocated_vram_bytes'] for r in resource_receipts+external)},
        'cost_protocol': {'primary': 'clean_latency', 'original_preserved_field': 'latency',
            'clean_cost_method_count': 236, 'clean_cost_batch_records': 472, 'new_training_updates': 0,
            'note': 'Supplement training overlapped initial timing. Main costs remeasured after all Python GPU training ended; ordinary desktop GPU apps remain.'}}
    result['aggregation_resources'] = {'wall_seconds': time.perf_counter()-start,
        'cpu_seconds': sum(process.cpu_times()[:2])-cpu_start, 'gpu_seconds': 0}
    ex.write_new(output/'RESULT.json', result)
    print(json.dumps({'status': 'COMPLETE', 'cases': 70, 'methods': 236, 'core_attempts': attempts,
                      'completed_core600_runs': core600, 'native600_runs': native600, 'all_gpu_process_wall_seconds': total_wall}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='resolved_v1')
    parser.add_argument('--full-tag', default='full_v1')
    parser.add_argument('--repair-tag', default='repair_v1')
    parser.add_argument('--cost-tag', default='clean_cost_v1')
    args = parser.parse_args()
    for tag in (args.tag, args.full_tag, args.repair_tag, args.cost_tag):
        if not tag.replace('_', '').isalnum():
            raise ValueError('Alphanumeric/underscore tags required')
    run(args.tag, args.full_tag, args.repair_tag, args.cost_tag)
