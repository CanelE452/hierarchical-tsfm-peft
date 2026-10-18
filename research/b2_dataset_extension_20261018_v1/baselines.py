"""Matched DirectG baseline using the unchanged search/refit and deployment code."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np

from research.b2_revin_20261018_v1.contracts import binding, read, sealed_json, sha
from . import train


def direct_id(task, seed):
    if int(seed) not in train.SEEDS:
        raise ValueError('The two fixed seeds are required')
    return f'DirectMLP_{task["dataset"]}_s{seed}'


@contextmanager
def direct_scope(modules, task):
    data, fit = modules['data'], modules['fit']
    original_split, original_source = data.split_ranges, fit._source_contract
    own = Path(__file__).resolve()
    own_sha = sha(own)

    def split(value, fold='primary'):
        if value is not task or fold != 'primary':
            raise ValueError('Direct baseline must use its exact registered task')
        return task['split_ranges']

    def sources():
        if sha(own) != own_sha:
            raise ValueError('Direct producer source changed during fitting')
        _, files = original_source()
        files = dict(files, **{'research/' + train.NAME + '/baselines.py': own_sha})
        return modules['training'].fingerprint(files), files

    data.split_ranges, fit._source_contract = split, sources
    try:
        yield
    finally:
        data.split_ranges, fit._source_contract = original_split, original_source


def fit_direct(task, ctx, seed):
    from research.adapter_diagnostics_20261017_v2.existing_jobs import _modules
    modules = ctx.get('modules') or _modules(ctx)
    run_id = direct_id(task, seed)
    spec = dict(run_id=run_id, family='DIRECT_G', dataset=task['dataset'], seed=int(seed),
                device='cuda', fold='primary', epochs=list(train.EPOCHS), lr=.001,
                effective_batch=16, microbatch=4, F_row_microbatch=128,
                checkpoint_inputs=False, core_dtype='float32')
    with direct_scope(modules, task):
        result = modules['fit'].fit_base(spec, task, ctx['cache'], ctx['report'],
                                         None, ctx['deadline'])
    if result.get('status') != 'COMPLETE_VALID':
        raise RuntimeError('Direct baseline incomplete; retain checkpoints and resume identity')
    bundle = Path(ctx['cache']) / result['artifact_cache_relative_path']
    evidence = dict(result, dataset=task['dataset'], C=len(task['columns']), L=512, H=48,
                    bundle_binding=binding(bundle, ctx), T_access=False,
                    V_independent_test=False, F_instances=0)
    sealed_json(Path(ctx['report']) / (run_id + '_FIT.json'), evidence)
    return evidence


def cost_inputs(request, report_dir, cache_dir):
    if request['family'] != 'DIRECT_G':
        raise ValueError('This factory is the fixed DirectG baseline only')
    ctx = train.build_context(request['config_path'], request['prefix_path'],
                              request['model_dir'], report_dir=report_dir, cache_dir=cache_dir)
    from research.adapter_diagnostics_20261017_v2.existing_jobs import _modules
    modules = _modules(ctx)
    task = ctx['task']
    run_id = direct_id(task, int(request['seed']))
    if request['run_id'] != run_id:
        raise ValueError('Direct cost run must match the exact fixed seed identity')
    evidence = read(Path(ctx['report']) / (run_id + '_FIT.json'))
    relative = evidence['artifact_cache_relative_path']
    if sha(Path(ctx['cache']) / relative) != evidence['artifact_sha256']:
        raise ValueError('Direct baseline artifact changed before cost measurement')
    spec = dict(family='DIRECT_G', run_id=run_id,
                artifact_cache_relative_path=relative,
                artifact_sha256=evidence['artifact_sha256'],
                canonical_fit_end=task['split_ranges']['A_stop'][1])

    def loader():
        return modules['deployment'].Route(spec, task, ctx['cache'], device='cuda')

    mean, std, _ = modules['data'].fit_statistics(task, task['split_ranges']['A_stop'][1])
    origins = modules['data'].origins(task['split_ranges']['V_select'], 512, 48, 48)
    rows = origins[:, None] + np.arange(48)[None]
    observed = task['observed'][rows]
    target = ((np.where(observed, task['physical'][rows], mean) - mean) / std).astype(np.float32)
    saved = dict(target_normalized=target, observed=observed.copy(), origins=origins,
                 mean=mean, std=std, channel_ids=np.asarray(task['columns']), pred_normalized=None)
    return dict(route_loader=loader, loader=loader, task=task, saved=saved,
                source_ctx=ctx, candidate=evidence)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args(argv)
    ctx = train.build_context(args.config, args.prefix, args.model_dir, deadline=args.deadline)
    for seed in train.SEEDS:
        print('DIRECT_SEARCH_REFIT ' + direct_id(ctx['task'], seed), flush=True)
        result = fit_direct(ctx['task'], ctx, seed)
        print(json.dumps(dict(run_id=result['run_id'], selected_epoch=result['selected_epoch'],
                              search_updates=result['search_updates'],
                              refit_updates=result['refit_updates'])), flush=True)


if __name__ == '__main__':
    main()
