"""Reproducible finite stages; preserve existing attempts and exact identities."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from research.b2_revin_20261018_v1.contracts import atomic_json, read, sha
from . import train


DATASETS = ('uci274_sml2010', 'uci235_household_hourly')


def paths(repo, dataset):
    base = Path(repo) / '.cache' / train.NAME / 'data' / dataset
    return base / 'DATA_DIAGNOSTIC.json', base / 'PREFIX_PHYSICAL.npz'


def cost_requests(repo, model_dir):
    requests = []
    for dataset in DATASETS:
        config, prefix = paths(repo, dataset)
        k = (read(config)['C'] + 3) // 4
        common = dict(dataset=dataset, K=k, config_path=str(config.resolve()),
                      prefix_path=str(prefix.resolve()), model_dir=str(Path(model_dir).resolve()))
        requests.append(dict(common, family='RAW', run_id='Raw_' + dataset,
                             seed=92601, factory='train:cost_inputs'))
        rows = read(Path(repo) / '.cache' / train.NAME / 'datasets' / dataset / 'VERIFIED_ROWS.json')
        for row in rows:
            requests.append(dict(common, family='B2', run_id=row['run_id'],
                                 seed=row['seed'], factory='train:cost_inputs'))
        for seed in train.SEEDS:
            requests.append(dict(common, family='DIRECT_G',
                                 run_id=f'DirectMLP_{dataset}_s{seed}', seed=seed,
                                 factory='baselines:cost_inputs'))
    return requests


def run_costs(repo, model_dir):
    repo = Path(repo).resolve()
    root = repo / '.cache' / train.NAME
    completed = []
    for request in cost_requests(repo, model_dir):
        dataset, run_id = request['dataset'], request['run_id']
        cache = root / 'datasets' / dataset
        report = repo / 'reports' / train.NAME / 'datasets' / dataset
        existing = report / 'costs' / (run_id + '_B8.json')
        if existing.exists():
            value = read(existing)
            if (value.get('status') != 'COMPLETE' or value.get('run_id') != run_id
                    or value.get('dataset') != dataset or value.get('family') != request['family']):
                raise ValueError('Existing incomplete cost attempt must be inspected and preserved')
            private = cache / 'costs' / (run_id + '_B8_predictions.npz')
            if sha(private) != value['prediction_file_sha256']:
                raise ValueError('Completed cost private output bytes changed')
            completed.append(value)
            continue
        request_dir = root / 'cost_requests'
        request_dir.mkdir(parents=True, exist_ok=True)
        target = request_dir / (run_id + '.json')
        if target.exists() and read(target) != request:
            raise ValueError('Cost request changed during resume')
        if not target.exists():
            atomic_json(target, request)
        print('COST ' + run_id, flush=True)
        subprocess.run([sys.executable, '-m', 'research.' + train.NAME + '.costs',
                        '--request', str(target), '--report-dir', str(report),
                        '--cache-dir', str(cache)], cwd=repo, check=True)
        completed.append(read(existing))
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train', 'baselines', 'costs'))
    parser.add_argument('--model-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    repo = train.REPO
    cache = repo / '.cache' / train.NAME
    cache.mkdir(parents=True, exist_ok=True)
    log = cache / 'PROCESS_ATTEMPTS.jsonl'
    started = time.time()
    base = dict(stage=args.stage, pid=os.getpid(), started_unix=started,
                producer_sha256=sha(__file__))

    def event(value):
        with log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(base, **value), sort_keys=True) + '\n')

    event(dict(status='STARTED'))
    try:
        if args.stage == 'train':
            train.execute_many([paths(repo, ds) for ds in DATASETS], args.model_dir)
        elif args.stage == 'baselines':
            from .baselines import fit_direct, direct_id
            for dataset in DATASETS:
                config, prefix = paths(repo, dataset)
                ctx = train.build_context(config, prefix, args.model_dir)
                for seed in train.SEEDS:
                    print('DIRECT_SEARCH_REFIT ' + direct_id(ctx['task'], seed), flush=True)
                    fit_direct(ctx['task'], ctx, seed)
        else:
            run_costs(repo, args.model_dir)
    except BaseException as error:
        event(dict(status='FAILED', finished_unix=time.time(),
                   wall_seconds=time.time() - started, error_type=type(error).__name__))
        raise
    event(dict(status='COMPLETE', finished_unix=time.time(), wall_seconds=time.time() - started))
    print('STAGE_COMPLETE ' + args.stage, flush=True)


if __name__ == '__main__':
    main()
