"""Sequential approved v9 fits; preserve failures and reuse only completed receipts."""
import argparse
import json
import subprocess
import sys

from runtime_v9 import CACHE, HERE, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['robin', 'jena'], required=True)
    parser.add_argument('--reserve-seconds', type=float, default=1200)
    args = parser.parse_args()
    if any(HERE.glob('test_exposure_*.json')):
        raise RuntimeError('Do not launch fitting after either v9 TEST exposure')
    protocol = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
    for spec in (s for s in protocol['fits'] if s['dataset'] == args.dataset):
        out = HERE / 'runs' / spec['id']
        result_path = out / 'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding='utf-8'))
            if result['status'] != 'complete' or result['spec'] != spec:
                raise RuntimeError('Existing result is not reusable: ' + spec['id'])
            for record in (result, result['best_trained'], result['alpha_selected']):
                if digest(record['checkpoint']) != record['checkpoint_sha256']:
                    raise RuntimeError('Completed checkpoint changed: ' + spec['id'])
            for record in (result['best_val_prediction'], result['best_trained']['best_val_prediction'], result['alpha_selected']['val_prediction']):
                if digest(record['path']) != record['sha256']:
                    raise RuntimeError('Completed VAL prediction changed: ' + spec['id'])
            if digest(result['alpha_selected']['selection_path']) != result['alpha_selected']['selection_sha256']:
                raise RuntimeError('Completed alpha selection changed: ' + spec['id'])
            print('Reused completed attempt ' + spec['id'], flush=True)
            continue
        if out.exists() or (CACHE / 'runs' / spec['id']).exists():
            raise RuntimeError('Interrupted/failed attempt needs an explicit new retry spec: ' + spec['id'])
        command = [sys.executable, '-B', str(HERE / 'train_v9.py'), '--dataset', args.dataset, '--id', spec['id'],
                   '--job', 'gpu_' + spec['id'], '--reserve-seconds', str(args.reserve_seconds)]
        print('Launching ' + spec['id'], flush=True)
        logs = CACHE / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / (spec['id'] + '.log')
        with log.open('x', encoding='utf-8') as handle:
            outcome = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        print('Finished ' + spec['id'] + ' exit=' + str(outcome.returncode) + ' log=' + str(log), flush=True)
        if outcome.returncode:
            raise SystemExit(outcome.returncode)
    print('All requested base fits complete', flush=True)


if __name__ == '__main__':
    main()
