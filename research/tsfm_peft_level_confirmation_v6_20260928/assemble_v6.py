"""Assemble already measured evidence; no data arrays, model import or forecasts."""
import argparse
import csv
import json
from pathlib import Path

from runtime_v6 import HERE, ROOT, Job, budget_snapshot, digest, save_json


def receipt(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)}


def complete(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value['status'] != 'complete':
        raise ValueError(f'Incomplete evidence: {path}')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--gpu-cost', type=Path, required=True)
    parser.add_argument('--cpu-cost', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    output = HERE / 'final_comparison.json'
    if output.exists():
        raise FileExistsError('Preserve the previous evidence assembly')
    with Job(args.job, category='cpu_analysis', reserve_seconds=30):
        evaluation = complete(args.evaluation)
        gpu, cpu = complete(args.gpu_cost), complete(args.cpu_cost)
        if gpu.get('stage') != 'gpu' or cpu.get('stage') != 'cpu':
            raise ValueError('Separate GPU and CPU measurement stages are required')
        if evaluation['seal'] != gpu['seal'] or evaluation['seal'] != cpu['seal']:
            raise ValueError('Accuracy/cost selection seals differ')
        result = dict(evaluation)
        result.update(evidence_sources={'evaluation': receipt(args.evaluation),
                                        'gpu_cost': receipt(args.gpu_cost), 'cpu_cost': receipt(args.cpu_cost)},
                      gpu_cost_summary=gpu['summary'], cpu_cost_summary=cpu['summary'],
                      cost_drift=gpu['drift_checks'], budget_at_assembly=budget_snapshot(),
                      verification_scope='Measured evidence assembled; final verification is a separate artifact.',
                      seed_aggregation='mean individual seed losses, not prediction ensemble')
        save_json(output, result)
        with (HERE / 'final_comparison.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['family', 'model_id', 'seed', 'period', 'mse', 'mae', 'aggregation'])
            writer.writeheader()
            for family, periods in evaluation['role_scores'].items():
                for period, score in periods.items():
                    writer.writerow(dict(family=family, model_id='', seed='', period=period,
                                         mse=score['mse'], mae=score['mae'], aggregation='mean individual model loss'))
            for identifier, model in evaluation['models'].items():
                for period, score in model['scores'].items():
                    writer.writerow(dict(family=model['spec']['family'], model_id=identifier,
                                         seed=model['spec'].get('seed'), period=period,
                                         mse=score['mse'], mae=score['mae'], aggregation='individual model'))
        channel_path = HERE / 'final_channel_metrics.csv'
        with channel_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['family', 'model_id', 'seed', 'period', 'channel',
                                                        'mse', 'mae', 'target_count', 'squared_error_sum', 'absolute_error_sum'])
            writer.writeheader()
            for identifier, model in evaluation['models'].items():
                for period, score in model['scores'].items():
                    for index, channel in enumerate(evaluation['channels']):
                        writer.writerow(dict(family=model['spec']['family'], model_id=identifier,
                            seed=model['spec'].get('seed'), period=period, channel=channel,
                            mse=score['channel_mse'][index], mae=score['channel_mae'][index],
                            target_count=score['channel_target_count'][index],
                            squared_error_sum=score['channel_squared_error_sum'][index],
                            absolute_error_sum=score['channel_absolute_error_sum'][index]))
        save_json(HERE / 'assembly_receipt.json', {'status': 'complete', 'source': receipt(__file__),
                  'inputs': result['evidence_sources'], 'outputs': [receipt(output), receipt(HERE / 'final_comparison.csv'), receipt(channel_path)],
                  'new_predictions': 0, 'new_fits': 0})
        print(json.dumps({'output': str(output), 'roles': len(evaluation['role_scores'])}))


if __name__ == '__main__':
    main()
