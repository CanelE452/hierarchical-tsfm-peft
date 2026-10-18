"""Run one registered dataset batch, keeping a durable stage journal."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

from research.b2_revin_20261018_v1.contracts import read, sealed_json, sha


NAME = 'b2_dataset_search_20261019_v1'
REPO = Path(__file__).resolve().parents[2]
POLICIES = ('DEFAULT128', 'COMMON_CAP')


def paths(repo, dataset):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', dataset):
        raise ValueError('Unsafe dataset identifier')
    folder = Path(repo) / '.cache' / NAME / 'data' / dataset
    return folder / 'DATA_DIAGNOSTIC.json', folder / 'PREFIX_PHYSICAL.npz'


def load_wave(repo, wave):
    if type(wave) is not int or wave < 1:
        raise ValueError('Positive integer wave required')
    path = Path(repo) / 'reports' / NAME / 'waves' / f'WAVE_{wave:03d}.json'
    value = read(path)
    entries = value.get('datasets', [])
    if value.get('status') != 'REGISTERED_BEFORE_TRAINING' or not entries:
        raise ValueError('A sealed nonempty batch is required')
    names = [entry['dataset'] for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError('A dataset cannot repeat inside a wave')
    for entry in entries:
        config, prefix = paths(repo, entry['dataset'])
        metadata = read(config)
        if (metadata.get('status') != 'READY' or sha(config) != entry['config_sha256']
                or sha(prefix) != entry['prefix_sha256']
                or metadata.get('prefix_only_approved') is not True
                or metadata['dataset'] != entry['dataset']):
            raise ValueError('Registered data readiness or byte identity changed')
    return path, value


def cost_requests(repo, model_dir, wave):
    from . import train
    requests = []
    for entry in wave['datasets']:
        dataset = entry['dataset']
        config, prefix = paths(repo, dataset)
        metadata = read(config)
        q = (metadata['C'] + 3) // 4
        rows = read(Path(repo) / '.cache' / NAME / 'datasets' / dataset / 'VERIFIED_ROWS.json')
        expected = {(k, seed) for k in entry['K_grid'] for seed in train.SEEDS}
        actual = {(int(row['K']), int(row['seed'])) for row in rows}
        if len(rows) != len(expected) or actual != expected:
            raise ValueError('Every registered K and seed must have a verified refit')
        common = dict(dataset=dataset, config_path=str(config.resolve()),
                      prefix_path=str(prefix.resolve()), model_dir=str(Path(model_dir).resolve()),
                      factory='train:cost_inputs')
        models = [dict(family='RAW', K=q, run_id='Raw_' + dataset, seed=92601)]
        models.extend(dict(family='B2', K=int(row['K']), run_id=row['run_id'], seed=int(row['seed']))
                      for row in rows)
        models.extend(dict(family='DIRECT_G', K=q, run_id=f'DirectMLP_{dataset}_s{seed}', seed=seed)
                      for seed in train.SEEDS)
        requests.extend(dict(common, **model, row_policy=policy)
                        for model in models for policy in POLICIES)
    return requests


def run_costs(repo, model_dir, wave):
    root = Path(repo) / '.cache' / NAME
    directory = root / 'cost_requests'
    directory.mkdir(parents=True, exist_ok=True)
    requests = cost_requests(repo, model_dir, wave)
    for index, request in enumerate(requests, 1):
        name = request['run_id'] + '_' + request['row_policy']
        target = directory / (name + '.json')
        sealed_json(target, request)
        print(f'COST {index}/{len(requests)} {name}', flush=True)
        cache = root / 'datasets' / request['dataset']
        report = Path(repo) / 'reports' / NAME / 'datasets' / request['dataset']
        subprocess.run([sys.executable, '-u', '-m', 'research.' + NAME + '.costs',
                        '--request', str(target), '--report-dir', str(report),
                        '--cache-dir', str(cache)], cwd=repo, check=True)
    return len(requests)


def fit_direct_or_reuse(task, ctx, seed):
    from research.b2_dataset_extension_20261018_v1 import baselines
    from research.b2_revin_20261018_v1.contracts import binding, resolve_binding
    run_id = baselines.direct_id(task, seed)
    path = Path(ctx['report']) / (run_id + '_FIT.json')
    if not path.exists():
        return baselines.fit_direct(task, ctx, seed)
    evidence = read(path)
    expected_axes = dict(status='COMPLETE_VALID', family='DIRECT_G', run_id=run_id,
        dataset=task['dataset'], seed=int(seed), C=len(task['columns']), L=512, H=48,
        T_access=False, V_independent_test=False, F_instances=0)
    if (any(evidence.get(key) != value for key, value in expected_axes.items())
            or type(evidence.get('reused_complete_artifact')) is not bool):
        raise ValueError('Completed Direct MLP evidence identity changed')
    bundle = resolve_binding(evidence['bundle_binding'], ctx)
    relative = evidence['artifact_cache_relative_path']
    expected_bundle = (Path(ctx['cache']) / relative).resolve()
    if (bundle != expected_bundle or not bundle.is_relative_to(Path(ctx['cache']).resolve())
            or sha(bundle) != evidence['artifact_sha256']):
        raise ValueError('Completed Direct MLP bundle binding changed')
    public = Path(ctx['report']) / 'runs' / run_id
    configuration = Path(ctx['cache']) / 'runs' / run_id / 'RUN_CONTRACT.json'
    if not all(target.is_file() for target in (configuration, public / 'BASE_FIT_RECEIPT.json', public / 'SOURCE_CONTRACT.json')):
        raise ValueError('Completed Direct MLP is missing its source/completion contracts')
    from research.adapter_diagnostics_20261017_v2.existing_jobs import _modules
    modules = ctx.get('modules') or _modules(ctx)
    spec = dict(run_id=run_id, family='DIRECT_G', dataset=task['dataset'], seed=int(seed),
        device='cuda', fold='primary', epochs=list(baselines.train.EPOCHS), lr=.001,
        effective_batch=16, microbatch=4, F_row_microbatch=128,
        checkpoint_inputs=False, core_dtype='float32')
    with baselines.direct_scope(modules, task):
        _, files = modules['fit']._source_contract()
        expected_source = dict(**read(configuration), files=files, spec=spec)
        if read(public / 'SOURCE_CONTRACT.json') != expected_source:
            raise ValueError('Completed Direct MLP public source contract changed')
        # All completion files exist: the immutable core can only validate and
        # reuse its source/config/data contract and artifact, never start fitting.
        refreshed = modules['fit'].fit_base(spec, task, ctx['cache'], ctx['report'], None, ctx['deadline'])
    if refreshed.get('status') != 'COMPLETE_VALID' or refreshed.get('reused_complete_artifact') is not True:
        raise ValueError('Completed Direct MLP did not pass the immutable completion guard')
    current = dict(refreshed, dataset=task['dataset'], C=len(task['columns']), L=512, H=48,
        bundle_binding=binding(bundle, ctx), T_access=False, V_independent_test=False, F_instances=0)
    original_values = {key: value for key, value in evidence.items() if key != 'reused_complete_artifact'}
    current_values = {key: value for key, value in current.items() if key != 'reused_complete_artifact'}
    if original_values != current_values:
        raise ValueError('Completed Direct MLP fit metadata changed')
    return current


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train', 'baselines', 'costs', 'all'))
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--wave', default=1, type=int)
    args = parser.parse_args(argv)
    from . import train
    wave_path, wave = load_wave(REPO, args.wave)
    root = REPO / '.cache' / NAME
    root.mkdir(parents=True, exist_ok=True)
    journal = root / 'PROCESS_ATTEMPTS.jsonl'
    producer = sha(__file__)
    base = dict(pid=os.getpid(), process_started_unix=time.time(), wave=args.wave,
                wave_sha256=sha(wave_path), producer_sha256=producer)

    def event(stage, status, **values):
        if sha(__file__) != producer:
            raise ValueError('Coordinator source changed during execution')
        record = dict(base, stage=stage, status=status, observed_unix=time.time(), **values)
        with journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')

    stages = ('train', 'baselines', 'costs') if args.stage == 'all' else (args.stage,)
    for stage in stages:
        started = time.time()
        event(stage, 'STARTED')
        try:
            if stage == 'train':
                selection = REPO / 'reports' / NAME / 'waves' / f'WAVE_{args.wave:03d}_SEARCHES_SEALED.json'
                train.execute_many([paths(REPO, entry['dataset']) for entry in wave['datasets']],
                                   args.model_dir, selection_path=selection)
            elif stage == 'baselines':
                from research.b2_dataset_extension_20261018_v1.baselines import direct_id
                for entry in wave['datasets']:
                    ctx = train.build_context(*paths(REPO, entry['dataset']), args.model_dir)
                    for seed in train.SEEDS:
                        print('DIRECT_SEARCH_REFIT ' + direct_id(ctx['task'], seed), flush=True)
                        fit_direct_or_reuse(ctx['task'], ctx, seed)
            else:
                count = run_costs(REPO, args.model_dir, wave)
                print(f'COST_ROUTES_COMPLETE {count}', flush=True)
        except BaseException as error:
            event(stage, 'FAILED', wall_seconds=time.time() - started,
                  error_type=type(error).__name__)
            raise
        event(stage, 'COMPLETE', wall_seconds=time.time() - started)
        print('STAGE_COMPLETE ' + stage, flush=True)


if __name__ == '__main__':
    main()
