"""Prefix-bound B2 searches and fresh refits in the new extension namespace."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from research.b2_revin_20261018_v1.contracts import (
    binding, fingerprint, read, resolve_binding, sealed_json, sha,
)


NAME = 'b2_dataset_extension_20261018_v1'
REPO = Path(__file__).resolve().parents[2]
SEEDS = (92601, 92602)
EPOCHS = (.25, 1., 3.)
SPLITS = ('A_fit', 'A_stop', 'M_cal', 'V_select', 'T_final')
DEPENDENCIES = (
    'research/b2_revin_20261018_v1/contracts.py',
    'research/b2_revin_20261018_v1/training_jobs.py',
    'research/b2_revin_20261018_v1/revin.py',
    'research/b2_revin_20261018_v1/deployment.py',
    'research/adapter_improvements_20261018_v1/training_jobs.py',
    'research/adapter_diagnostics_20261017_v2/existing_jobs.py',
    'research/informs_broad_validation_20261015_v1/fit_base.py',
    'research/informs_broad_validation_20261015_v1/training.py',
    'research/informs_broad_validation_20261015_v1/data_contract.py',
    'research/informs_broad_validation_20261015_v1/networks.py',
    'research/informs_broad_validation_20261015_v1/evaluation.py',
    'research/informs_broad_validation_20261015_v1/gpu_runtime.py',
    'research/temporal_frontend_campaign_20261015_v1/runtime.py',
)


def _array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _header(archive, name):
    with archive.zip.open(name + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(stream)
        if version == (2, 0):
            return np.lib.format.read_array_header_2_0(stream)
        raise ValueError('Unsupported approved prefix NPY header')


def load_task(config_path, prefix_path):
    config = read(config_path)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', str(config.get('dataset', ''))):
        raise ValueError('Safe dataset identifier required')
    if config.get('prefix_only_approved') is not True:
        raise ValueError('An explicitly approved TRAIN/development prefix is required')
    if (config.get('L'), config.get('H')) != (512, 48):
        raise ValueError('The fixed extension requires L512/H48')
    ranges = config.get('split_ranges', {})
    if set(ranges) not in (set(SPLITS), set(SPLITS) | {'A60'}):
        raise ValueError('Complete chronological split metadata required')
    bounds = {key: tuple(map(int, ranges[key])) for key in SPLITS}
    for index, key in enumerate(SPLITS):
        lo, hi = bounds[key]
        if lo >= hi or lo != (0 if index == 0 else bounds[SPLITS[index-1]][1]):
            raise ValueError('Split gaps, overlaps or empty ranges are forbidden')
    if 'A60' in ranges and tuple(ranges['A60']) != (0, bounds['A_stop'][1]):
        raise ValueError('A60 metadata differs from the fresh refit prefix')
    rows = config.get('prefix_rows')
    if type(rows) is not int or rows != bounds['V_select'][1]:
        raise ValueError('Approved numeric prefix must stop exactly before T_final')
    prefix_path = Path(prefix_path)
    with np.load(prefix_path, allow_pickle=False) as archive:
        if set(archive.files) != {'physical', 'observed', 'times', 'columns'}:
            raise ValueError('Approved prefix must contain only four declared arrays')
        headers = {name: _header(archive, name) for name in archive.files}
        shape, order, dtype = headers['physical']
        if len(shape) != 2 or shape[0] != rows or dtype != np.dtype('float32') or order:
            raise ValueError('Physical header is outside approved FP32 prefix')
        channels = shape[1]
        if (headers['observed'] != (shape, False, np.dtype('bool'))
                or headers['times'] != ((rows,), False, np.dtype('int64'))
                or headers['columns'][0] != (channels,) or headers['columns'][2].kind != 'U'):
            raise ValueError('Prefix mask/time/channel headers are inconsistent')
        if sha(prefix_path) != config.get('prefix_sha256'):
            raise ValueError('Approved prefix bytes changed')
        values = {name: archive[name].copy() for name in archive.files}
    physical, observed, times, columns = (values[key] for key in ('physical', 'observed', 'times', 'columns'))
    interval = config.get('interval_seconds')
    if (type(interval) is not int or interval < 1 or not np.all(np.diff(times) == interval * 1_000_000_000)
            or len(set(columns.tolist())) != channels or channels < 2
            or not np.isfinite(physical[observed]).all()):
        raise ValueError('Invalid regular prefix coordinates or observed values')
    if bounds['A_fit'][1] < config['L'] + config['H']:
        raise ValueError('A_fit has no complete training window')
    for key in ('A_stop', 'V_select'):
        if bounds[key][1] - max(bounds[key][0], config['L']) < config['H']:
            raise ValueError(key + ' has no complete forecast origin')
    physical[~observed] = np.nan
    hashes = {name: _array_sha(value) for name, value in dict(values, physical=physical).items()}
    task = dict(name=config['dataset'], dataset=config['dataset'], physical=physical,
        observed=observed, times=times, columns=columns.tolist(), L=512, H=48,
        interval_seconds=interval, period=int(config.get('period', 24)),
        split_ranges=bounds, array_sha256=hashes,
        data_sha256=fingerprint(dict(arrays=hashes, split_ranges=bounds, L=512, H=48)),
        units=config.get('units', 'source_physical_units'),
        metadata=dict(prefix_only=True, T_access=False, prefix_rows=rows,
                      prefix_sha256=config['prefix_sha256'], source_config_sha256=sha(config_path)))
    return config, task


def jobs(task):
    k = math.ceil(len(task['columns']) / 4)
    return [dict(job_id=f'B2_{task["dataset"]}_K{k}_OFF_s{seed}', cohort=task['dataset'],
        family='B2', revin=False, init_kind='PCA', K=k, seed=seed, device='cuda') for seed in SEEDS]


def build_context(config_path, prefix_path, model_dir, *, repo=REPO, cache_dir=None, report_dir=None, deadline=None):
    config, task = load_task(config_path, prefix_path)
    repo = Path(repo).resolve()
    roots = dict(cache=repo / '.cache' / NAME, report=repo / 'reports' / NAME)
    cache = Path(cache_dir).resolve() if cache_dir else roots['cache'] / 'datasets' / task['dataset']
    report = Path(report_dir).resolve() if report_dir else roots['report'] / 'datasets' / task['dataset']
    if not cache.is_relative_to(roots['cache']) or not report.is_relative_to(roots['report']):
        raise ValueError('Training outputs must remain inside the new namespace')
    cache.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    sources = {f'research/{NAME}/{name}': sha(Path(__file__).parent / name)
               for name in ('train.py', '__init__.py')}
    sources.update({relative: sha(REPO / relative) for relative in DEPENDENCIES})
    return dict(repo=repo, asset=Path(model_dir).resolve(), cache=cache, report=report,
        deadline=time.time() + 24 * 3600 if deadline is None else float(deadline),
        scientific_source_hashes=sources, config=config, task=task,
        config_path=Path(config_path).resolve(), prefix_path=Path(prefix_path).resolve(),
        model_dir=Path(model_dir).resolve())


def runtime(ctx, task, model_dir):
    from research.adapter_diagnostics_20261017_v2.existing_jobs import _modules
    from research.b2_revin_20261018_v1 import training_jobs
    modules = _modules(ctx)
    from gpu_runtime import FrozenBolt

    def foundation_loader(context, pinned_modules, horizon, device='cuda'):
        foundation = FrozenBolt(model_dir, horizon, row_microbatch=128,
                                checkpoint_inputs=False, device=device)
        if foundation.initial_hash != pinned_modules['legacy'].F_SHA:
            raise ValueError('Frozen foundation tensor state differs from the fixed B2 reference')
        return foundation

    foundation = foundation_loader(ctx, modules, task['H'])
    return training_jobs, dict(ctx, modules=modules, task=task, foundation=foundation,
                              foundation_loader=foundation_loader)


def _verify_search(job, result):
    if result.get('status') != 'COMPLETE':
        raise RuntimeError('Search incomplete; preserve checkpoints and resume the same identity')
    expected = dict(dataset=job['cohort'], family='B2', revin=False, seed=job['seed'],
                    latent_k=job['K'], epochs=list(EPOCHS))
    if (result.get('cap') != 3 or result.get('V_used') is not False
            or any(result.get('spec', {}).get(key) != value for key, value in expected.items())):
        raise ValueError('Completed search scientific axes changed')
    curve = [(float(row['epoch']), float(row['astop_mse'])) for row in result['curves']
             if row.get('astop_mse') is not None]
    if [epoch for epoch, value in curve] != [0., *EPOCHS] or not all(math.isfinite(value) for _, value in curve):
        raise ValueError('All fixed A_stop checkpoints are required')
    selected = min(curve, key=lambda pair: (pair[1], pair[0]))
    if float(result['selected_epoch']) != selected[0]:
        raise ValueError('Strict minimum A_stop with earlier tie is required')
    return selected


def search_dataset(config_path, prefix_path, model_dir, *, repo=REPO, cache_dir=None,
                   report_dir=None, deadline=None, runtime_factory=None):
    ctx = build_context(config_path, prefix_path, model_dir, repo=repo,
                        cache_dir=cache_dir, report_dir=report_dir, deadline=deadline)
    task = ctx['task']
    fixed_jobs = jobs(task)
    identity = dict(schema_version=1, dataset=task['dataset'], jobs=fixed_jobs, epochs=list(EPOCHS),
        cap=3, K=fixed_jobs[0]['K'], revin=False, data_sha256=task['data_sha256'],
        prefix_sha256=ctx['config']['prefix_sha256'], input_config_sha256=sha(config_path),
        split_ranges=task['split_ranges'], source_hashes=ctx['scientific_source_hashes'],
        recipe=dict(lr=.001, effective_batch=16, microbatch=4, F_row_microbatch=128,
                    checkpoint_inputs=False, optimizer='AdamW', weight_decay=0., gradient_clip=1.),
        selection='A_fit only; strict minimum A_stop at 0/.25/1/3 with earlier tie',
        T_access=False, V_independent_test=False)
    sealed_json(ctx['report'] / 'TRAIN_IDENTITY.json', identity)
    helper, active = (runtime_factory or runtime)(ctx, task, ctx['model_dir'])
    results = []
    try:
        for job in fixed_jobs:
            print('SEARCH ' + job['job_id'], flush=True)
            result = helper.search(job, active, epochs=EPOCHS)
            _verify_search(job, result)
            results.append(result)
        sealed_json(ctx['cache'] / 'SEARCH_RESULTS.json', results)
        public = dict(status='COMPLETE', dataset=task['dataset'], cap=3, V_used=False,
            identity_binding=binding(ctx['report'] / 'TRAIN_IDENTITY.json', ctx),
            searches={job['job_id']: dict(seed=job['seed'], epoch=result['selected_epoch'],
                astop_mse=result['summary']['selected_astop_mse'], checkpoint=result['checkpoint_binding'])
                for job, result in zip(fixed_jobs, results)})
        sealed_json(ctx['report'] / 'TRAIN_SEARCH_SUMMARY.json', public)
        return dict(ctx=ctx, jobs=fixed_jobs, searches=results, summary=public)
    finally:
        active['foundation'].assert_frozen_unchanged()
        active.pop('foundation', None)


def seal_searches(states, output_path):
    if not states or len({state['ctx']['task']['dataset'] for state in states}) != len(states):
        raise ValueError('One completed search group per dataset required')
    selections = {}
    datasets = {}
    for state in states:
        ctx = state['ctx']
        identifiers = []
        for job, result in zip(state['jobs'], state['searches']):
            epoch, value = _verify_search(job, result)
            checkpoint = resolve_binding(result['checkpoint_binding'], ctx)
            # Cross-dataset decisions use the shared repository root.
            selections[job['job_id']] = dict(epoch=epoch, astop_mse=value,
                checkpoint=dict(reference='repo://' + checkpoint.relative_to(ctx['repo']).as_posix(),
                                sha256=sha(checkpoint)))
            identifiers.append(job['job_id'])
        if len(identifiers) != 2:
            raise ValueError('Both fixed seeds must finish before any fresh refit')
        datasets[ctx['task']['dataset']] = dict(matched_arms=identifiers,
            identity_binding=dict(reference='repo://' + (ctx['report'] / 'TRAIN_IDENTITY.json').relative_to(ctx['repo']).as_posix(),
                                  sha256=sha(ctx['report'] / 'TRAIN_IDENTITY.json')))
    decision = dict(status='SEALED', cap=3, V_used=False, selection_split='A_stop',
        all_searches_complete=True, matched_arms=sorted(selections), datasets=datasets, selections=selections)
    output_path = Path(output_path).resolve()
    root = states[0]['ctx']['repo'] / 'reports' / NAME
    if not output_path.is_relative_to(root):
        raise ValueError('Selection seal must remain inside the new public namespace')
    sealed_json(output_path, decision)
    return decision


def _selection_decision(job, ctx, cap):
    value = ctx.get('selection_decision', {})
    path = Path(value.get('path', ''))
    if not path.is_file() or sha(path) != value.get('sha256'):
        raise ValueError('Exact all-search selection seal required before V')
    decision = read(path)
    group = decision.get('datasets', {}).get(job['cohort'], {})
    expected = {f'B2_{job["cohort"]}_K{job["K"]}_OFF_s{seed}' for seed in SEEDS}
    if (cap != 3 or decision.get('cap') != 3 or decision.get('status') != 'SEALED'
            or decision.get('V_used') is not False or decision.get('all_searches_complete') is not True
            or decision.get('selection_split') != 'A_stop' or set(group.get('matched_arms', [])) != expected
            or job['job_id'] not in decision.get('matched_arms', [])):
        raise ValueError('The fixed two-seed group is absent from the all-search seal')
    for entry in decision['datasets'].values():
        resolve_binding(entry['identity_binding'], ctx)
    for entry in decision['selections'].values():
        resolve_binding(entry['checkpoint'], ctx)
    return binding(path, ctx)


@contextmanager
def selection_scope(helper):
    original = getattr(helper, '_selection_decision', None)
    helper._selection_decision = _selection_decision
    try:
        yield
    finally:
        if original is None:
            del helper._selection_decision
        else:
            helper._selection_decision = original


def refit_dataset(config_path, prefix_path, model_dir, selection_path, *, repo=REPO,
                  cache_dir=None, report_dir=None, deadline=None, runtime_factory=None):
    ctx = build_context(config_path, prefix_path, model_dir, repo=repo,
                        cache_dir=cache_dir, report_dir=report_dir, deadline=deadline)
    searches = read(ctx['cache'] / 'SEARCH_RESULTS.json')
    ctx['selection_decision'] = dict(path=Path(selection_path).resolve(), sha256=sha(selection_path))
    fixed_jobs = jobs(ctx['task'])
    helper, active = (runtime_factory or runtime)(ctx, ctx['task'], ctx['model_dir'])
    rows = []
    try:
        with selection_scope(helper):
            for job, search in zip(fixed_jobs, searches):
                _verify_search(job, search)
                _selection_decision(job, active, 3)
                print('REFIT ' + job['job_id'], flush=True)
                result = helper._refit_predict(job, active, search, 3)
                if result.get('status') != 'COMPLETE' or len(result.get('rows', [])) != 1:
                    raise RuntimeError('Fresh refit incomplete; resume the same identity')
                row = dict(result['rows'][0], final=True, status='COMPLETE',
                    bundle_binding=result['bundle_binding'], cap=3, revin=False,
                    V_independent_test=False, all_searches_sealed_before_V=True)
                rows.append(row)
        sealed_json(ctx['cache'] / 'VERIFIED_ROWS.json', rows)
        sealed_json(ctx['report'] / 'VERIFIED_ROWS.json', rows)
        summary = dict(status='COMPLETE', dataset=ctx['task']['dataset'], new_searches=2,
            fresh_refits=2, cap=3, rows=rows,
            selection_binding=binding(selection_path, ctx), T_access=False)
        sealed_json(ctx['report'] / 'TRAIN_SUMMARY.json', summary)
        return summary
    finally:
        active['foundation'].assert_frozen_unchanged()
        active.pop('foundation', None)


def execute_many(inputs, model_dir, *, repo=REPO, deadline=None, runtime_factory=None):
    states = [search_dataset(config, prefix, model_dir, repo=repo, deadline=deadline,
                             runtime_factory=runtime_factory) for config, prefix in inputs]
    selection = Path(repo).resolve() / 'reports' / NAME / 'ALL_SEARCHES_SEALED.json'
    seal_searches(states, selection)
    results = [refit_dataset(config, prefix, model_dir, selection, repo=repo, deadline=deadline,
                            runtime_factory=runtime_factory) for config, prefix in inputs]
    return dict(status='COMPLETE', new_searches=2 * len(inputs), fresh_refits=2 * len(inputs), datasets=results)


def execute(config_path, prefix_path, model_dir, **kwargs):
    return execute_many([(config_path, prefix_path)], model_dir, **kwargs)


def cost_inputs(request, report_dir, cache_dir):
    ctx = build_context(request['config_path'], request['prefix_path'], request['model_dir'],
                        report_dir=report_dir, cache_dir=cache_dir)
    from research.adapter_diagnostics_20261017_v2.existing_jobs import _modules
    from research.b2_revin_20261018_v1.costs import route_loader
    modules = _modules(ctx)
    from gpu_runtime import FrozenBolt

    def foundation_loader(context, pinned_modules, horizon, device='cuda'):
        foundation = FrozenBolt(ctx['model_dir'], horizon, row_microbatch=128,
                                checkpoint_inputs=False, device=device)
        if foundation.initial_hash != pinned_modules['legacy'].F_SHA:
            raise ValueError('Cost foundation tensor state differs from the B2 reference')
        return foundation

    ctx.update(modules=modules, foundation_loader=foundation_loader)
    if request['family'] == 'B2':
        rows = read(ctx['cache'] / 'VERIFIED_ROWS.json')
        matches = [row for row in rows if row['run_id'] == request['run_id']]
        if len(matches) != 1:
            raise ValueError('One exact trained accuracy identity required')
        candidate = matches[0]
        saved_path = resolve_binding(candidate['prediction_binding'], ctx)
        with np.load(saved_path, allow_pickle=False) as archive:
            saved = {key: archive[key].copy() for key in archive.files}
    elif request['family'] == 'RAW':
        candidate = dict(family='RAW', dataset=ctx['task']['dataset'], run_id=request['run_id'],
                         seed=int(request.get('seed', 92601)), C=len(ctx['task']['columns']),
                         K=len(ctx['task']['columns']), L=512, H=48)
        task = ctx['task']
        mean, std, _ = modules['data'].fit_statistics(task, task['split_ranges']['A_stop'][1])
        origins = modules['data'].origins(task['split_ranges']['V_select'], 512, 48, 48)
        target_rows = origins[:, None] + np.arange(48)[None]
        observed = task['observed'][target_rows]
        target = ((np.where(observed, task['physical'][target_rows], mean) - mean) / std).astype(np.float32)
        saved = dict(target_normalized=target, observed=observed.copy(), origins=origins,
                     mean=mean, std=std, channel_ids=np.asarray(task['columns']), pred_normalized=None)
    else:
        raise ValueError('Train cost factory handles B2 and Raw only')
    loader = route_loader(candidate, ctx['task'], ctx, inventory=dict(baselines=[]))
    return dict(route_loader=loader, loader=loader, task=ctx['task'], saved=saved,
                expected_f_sha=modules['legacy'].F_SHA, expected_f_params=47_718_016,
                source_ctx=ctx, candidate=candidate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('search', 'refit'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args(argv)
    if args.phase == 'search':
        result = search_dataset(args.config, args.prefix, args.model_dir, deadline=args.deadline)
        print(json.dumps(result['summary'], ensure_ascii=False))
    else:
        if args.selection is None:
            parser.error('--selection is required for fresh refit')
        print(json.dumps(refit_dataset(args.config, args.prefix, args.model_dir,
                                      args.selection, deadline=args.deadline), ensure_ascii=False))


if __name__ == '__main__':
    main()
