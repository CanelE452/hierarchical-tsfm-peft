"""Derive manuscript/report values from completed JSON only; no models or arrays."""
import argparse
import json
from statistics import mean

from runtime_v6 import HERE, ROOT, Job, digest, save_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def receipt(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)}


def costs(report):
    result = {}
    for group in report['summary']['groups']:
        family, batch = group['family'], str(group['batch_origins'])
        rows = [r for r in report['rows'] if r['include_in_primary'] and
                r['family'] == family and str(r['batch_origins']) == batch]
        blocks = [mean(r['origins_per_second']['median'] for r in rows if r['block'] == i) for i in range(3)]
        item = {'throughput': group['mean_fit_median_origins_per_second'],
                'ms_per_origin': group['mean_fit_median_ms_per_origin'],
                'block_seed_mean_throughput': blocks,
                'deployed_parameters': rows[0]['total_deployed_parameters'],
                'deployed_tensor_bytes': rows[0]['total_deployed_tensor_bytes'],
                'trainable_before_merge': rows[0]['registered_trainable_parameters_before_merge'],
                'fitted_coefficients': 24624 if family == 'shared' else rows[0]['registered_trainable_parameters_before_merge']}
        for field in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            if field in rows[0]:
                values = [r[field] for r in rows]
                item[field] = max(values)
                item[field + '_min'] = min(values)
                item[field.replace('_bytes', '_mib')] = max(values) / 2**20
        result.setdefault(family, {})[batch] = item
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    output = HERE / 'report_values.json'
    if output.exists():
        raise FileExistsError('Preserve prior reporting arithmetic')
    with Job(args.name, category='cpu_analysis', reserve_seconds=30):
        paths = [HERE / name for name in ('evaluation01.json', 'cost_gpu01.json', 'cost_cpu01.json')]
        evaluation, gpu, cpu = [read(path) for path in paths]
        assert all(value['status'] == 'complete' for value in (evaluation, gpu, cpu))
        assert gpu['stage'] == 'gpu' and cpu['stage'] == 'cpu'
        assert gpu['seal'] == cpu['seal'] == evaluation['seal']
        result = {'status': 'complete', 'sources': [receipt(p) for p in paths],
                  'source_code': receipt(HERE / 'summarize_v6.py'),
                  'gpu': costs(gpu), 'cpu': costs(cpu), 'accuracy': evaluation['role_scores'],
                  'comparisons': evaluation['comparisons'], 'channel_directions': {}, 'training': []}
        models = evaluation['models']
        channel_means = {}
        for family, ids in evaluation['roles'].items():
            channel_means[family] = [mean(models[i]['scores']['combined']['channel_mse'][c] for i in ids)
                                     for c in range(len(evaluation['channels']))]
        for family, errors in channel_means.items():
            differences = [a - b for a, b in zip(channel_means['level_res'], errors)]
            result['channel_directions'][family] = {'lower': sum(v < 0 for v in differences),
                'higher': sum(v > 0 for v in differences), 'ties': sum(v == 0 for v in differences),
                'delta_mse_by_channel': differences, 'channel_order': evaluation['channels']}
        result['cost_relative_to'] = {}
        level = result['gpu']['level_res']['4']
        for family in ('old_res', 'level_raw', 'f0', 'lora', 'level_linear_res'):
            ref = result['gpu'][family]['4']
            result['cost_relative_to'][family] = {
                'batch4_throughput_ratio': level['throughput'] / ref['throughput'],
                'batch4_throughput_change_pct': 100 * (level['throughput'] / ref['throughput'] - 1),
                'peak_allocated_change_pct': 100 * (level['peak_allocated_bytes'] / ref['peak_allocated_bytes'] - 1),
                'batch1_latency_change_pct': 100 * (result['gpu']['level_res']['1']['ms_per_origin'] /
                                                   result['gpu'][family]['1']['ms_per_origin'] - 1)}
        for spec in read(HERE / 'protocol.json')['fits']:
            path = HERE / 'runs' / spec['id'] / 'result.json'
            row = read(path)
            result['training'].append({key: row[key] for key in ('id', 'trainable_parameters', 'total_parameters',
                'selected_epoch', 'epochs_completed', 'total_steps', 'elapsed_s', 'initial_val_mse', 'best_val_mse',
                'training_peak', 'initial_train_probe', 'selected_train_probe', 'replay_error')})
            result['sources'].append(receipt(path))
        result['definitions'] = {'accuracy': 'Individual seed losses averaged, no prediction ensemble.',
            'cost_center': gpu['summary']['definition'],
            'memory': 'Maximum observed peak across the three blocks and selected seeds; minimum retained separately.',
            'training': 'Observed selected runs with different stopping epochs; elapsed times are not controlled speed ratios.',
            'shared_parameters': 'Fitted coefficients are stored as buffers at inference; zero registered nn.Parameters does not mean no fitted model.'}
        save_json(output, result)
        print(json.dumps({'output': str(output), 'cost_relative_to': result['cost_relative_to']}))


if __name__ == '__main__':
    main()
