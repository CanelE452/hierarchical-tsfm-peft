"""Registered multi-width searches using immutable B2 fit and deployment cores."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

import numpy as np

from research.b2_dataset_extension_20261018_v1 import train as core
from research.b2_revin_20261018_v1.contracts import binding, fingerprint, read, resolve_binding, sealed_json, sha


NAME = 'b2_dataset_search_20261019_v1'
REPO = Path(__file__).resolve().parents[2]
SEEDS, EPOCHS = core.SEEDS, core.EPOCHS
load_task, runtime, _verify_search = core.load_task, core.runtime, core._verify_search
DEPENDENCIES = (*core.DEPENDENCIES,
    'research/b2_dataset_extension_20261018_v1/train.py',
    'research/b2_dataset_extension_20261018_v1/baselines.py',
    'research/b2_revin_20261018_v1/costs.py',
    'research/informs_broad_validation_20261015_v1/deployment.py')


def k_grid(channels):
    if type(channels) is not int or channels < 2:
        raise ValueError('At least two original channels required')
    q = math.ceil(channels / 4)
    return tuple(sorted({max(1, math.ceil(q / 2)), q, min(channels, 2 * q)}))


def jobs(task):
    return [dict(job_id=f'B2_{task["dataset"]}_K{k}_OFF_s{seed}', cohort=task['dataset'],
        family='B2', revin=False, init_kind='PCA', K=k, seed=seed, device='cuda')
        for k in k_grid(len(task['columns'])) for seed in SEEDS]


def source_hashes():
    files = [Path(__file__), Path(__file__).with_name('__init__.py')]
    return dict({path.relative_to(REPO).as_posix(): sha(path) for path in files},
                **{relative: sha(REPO / relative) for relative in DEPENDENCIES})


def build_context(config_path, prefix_path, model_dir, *, repo=REPO, cache_dir=None, report_dir=None, deadline=None):
    config, task = load_task(config_path, prefix_path)
    repo = Path(repo).resolve()
    cache_root, report_root = repo / '.cache' / NAME, repo / 'reports' / NAME
    cache = Path(cache_dir).resolve() if cache_dir else cache_root / 'datasets' / task['dataset']
    report = Path(report_dir).resolve() if report_dir else report_root / 'datasets' / task['dataset']
    if not cache.is_relative_to(cache_root) or not report.is_relative_to(report_root):
        raise ValueError('Outputs must stay inside the new search namespace')
    protocol = report_root / 'PROTOCOL.md'
    if not protocol.is_file():
        raise ValueError('The registered new protocol must exist before training')
    cache.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    return dict(repo=repo, asset=Path(model_dir).resolve(), cache=cache, report=report,
        task=task, config=config, model_dir=Path(model_dir).resolve(),
        config_path=Path(config_path).resolve(), prefix_path=Path(prefix_path).resolve(),
        deadline=time.time() + 24 * 3600 if deadline is None else float(deadline),
        scientific_source_hashes=source_hashes(), protocol_sha256=sha(protocol))


def _registration(ctx):
    task = ctx['task']
    return dict(dataset=task['dataset'], jobs=jobs(task), K_grid=list(k_grid(len(task['columns']))),
        data_sha256=task['data_sha256'], input_config_sha256=sha(ctx['config_path']),
        prefix_sha256=ctx['config']['prefix_sha256'], sources=ctx['scientific_source_hashes'])


def seal_wave_plan(contexts, output_path):
    if not contexts or len({ctx['task']['dataset'] for ctx in contexts}) != len(contexts):
        raise ValueError('One context per registered wave dataset required')
    repo = contexts[0]['repo']
    if any(ctx['repo'] != repo or ctx['protocol_sha256'] != contexts[0]['protocol_sha256'] for ctx in contexts):
        raise ValueError('A wave must share its repository and protocol')
    path = Path(output_path).resolve()
    if not path.is_relative_to(repo / 'reports' / NAME):
        raise ValueError('Wave plan must stay inside the new namespace')
    entries = {ctx['task']['dataset']: _registration(ctx) for ctx in contexts}
    payload = dict(status='SEALED', phase='BEFORE_SEARCH', cap=3, epochs=list(EPOCHS),
        V_used=False, dataset_order=[ctx['task']['dataset'] for ctx in contexts], datasets=entries,
        matched_arms=[job['job_id'] for ctx in contexts for job in jobs(ctx['task'])],
        protocol_binding=dict(reference=f'repo://reports/{NAME}/PROTOCOL.md', sha256=contexts[0]['protocol_sha256']))
    sealed_json(path, payload)
    return payload


def _wave(ctx, path):
    path = Path(path).resolve()
    if not path.is_relative_to(ctx['repo'] / 'reports' / NAME):
        raise ValueError('Wave plan escaped the new namespace')
    plan = read(path)
    if (plan.get('status') != 'SEALED' or plan.get('phase') != 'BEFORE_SEARCH'
            or plan.get('V_used') is not False or plan.get('epochs') != list(EPOCHS)
            or plan.get('datasets', {}).get(ctx['task']['dataset']) != _registration(ctx)):
        raise ValueError('Registered wave, task or producer source changed')
    resolve_binding(plan['protocol_binding'], ctx)
    return plan


def _check_search(job, result, ctx):
    selected = _verify_search(job, result)
    if result.get('contract', {}).get('data_sha') != ctx['task']['data_sha256']:
        raise ValueError('Search data binding differs from the approved task')
    model_sha = result.get('contract', {}).get('model_sha')
    if not isinstance(model_sha, str) or len(model_sha) != 64:
        raise ValueError('Search must bind the actual frozen F state')
    resolve_binding(result['checkpoint_binding'], ctx)
    return selected


def search_dataset(config_path, prefix_path, model_dir, *, repo=REPO, cache_dir=None,
                   report_dir=None, deadline=None, runtime_factory=None, wave_plan=None):
    ctx = build_context(config_path, prefix_path, model_dir, repo=repo,
        cache_dir=cache_dir, report_dir=report_dir, deadline=deadline)
    if wave_plan is None:
        wave_plan = ctx['report'] / 'WAVE_PLAN.json'
        seal_wave_plan([ctx], wave_plan)
    _wave(ctx, wave_plan)
    ctx['wave_plan'] = Path(wave_plan).resolve()
    fixed_jobs = jobs(ctx['task'])
    identity = dict(_registration(ctx), schema_version=1, cap=3, epochs=list(EPOCHS),
        revin=False, split_ranges=ctx['task']['split_ranges'],
        wave_plan_binding=binding(wave_plan, ctx), protocol_sha256=ctx['protocol_sha256'],
        recipe=dict(lr=.001, effective_batch=16, microbatch=4, F_row_microbatch=128,
                    checkpoint_inputs=False, optimizer='AdamW', weight_decay=0., gradient_clip=1.),
        TRAIN_stride=1, evaluation_stride=48, T_access=False, V_independent_test=False)
    sealed_json(ctx['report'] / 'TRAIN_IDENTITY.json', identity)
    helper, active = (runtime_factory or runtime)(ctx, ctx['task'], ctx['model_dir'])
    results = []
    try:
        for job in fixed_jobs:
            print('SEARCH ' + job['job_id'], flush=True)
            result = helper.search(job, active, epochs=EPOCHS)
            _check_search(job, result, ctx)
            results.append(result)
        sealed_json(ctx['cache'] / 'SEARCH_RESULTS.json', results)
        summary = dict(status='COMPLETE', dataset=ctx['task']['dataset'], cap=3, V_used=False,
            identity_binding=binding(ctx['report'] / 'TRAIN_IDENTITY.json', ctx),
            wave_plan_binding=binding(wave_plan, ctx),
            searches={job['job_id']: dict(seed=job['seed'], K=job['K'], epoch=result['selected_epoch'],
                astop_mse=result['summary']['selected_astop_mse'], checkpoint=result['checkpoint_binding'])
                for job, result in zip(fixed_jobs, results)})
        sealed_json(ctx['report'] / 'TRAIN_SEARCH_SUMMARY.json', summary)
        return dict(ctx=ctx, jobs=fixed_jobs, searches=results, summary=summary)
    finally:
        active['foundation'].assert_frozen_unchanged()
        active.pop('foundation', None)


def seal_searches(states, output_path):
    if not states:
        raise ValueError('Complete registered wave required')
    first = states[0]['ctx']
    plan_path = first['wave_plan']
    plan = _wave(first, plan_path)
    actual = [job['job_id'] for state in states for job in state['jobs']]
    if (len(actual) != len(set(actual)) or set(actual) != set(plan['matched_arms'])
            or [state['ctx']['task']['dataset'] for state in states] != plan['dataset_order']
            or any(state['ctx']['wave_plan'] != plan_path for state in states)):
        raise ValueError('Every registered wave search must finish before any fresh refit/V')
    selections, datasets, model_states = {}, {}, set()
    for state in states:
        ctx = state['ctx']
        _wave(ctx, plan_path)
        if state['jobs'] != jobs(ctx['task']) or len(state['searches']) != len(state['jobs']):
            raise ValueError('Registered wave width/seed coverage is incomplete')
        for job, result in zip(state['jobs'], state['searches']):
            epoch, value = _check_search(job, result, ctx)
            checkpoint = resolve_binding(result['checkpoint_binding'], ctx)
            model_states.add(result['contract']['model_sha'])
            selections[job['job_id']] = dict(epoch=epoch, astop_mse=value, K=job['K'], seed=job['seed'],
                model_sha=result['contract']['model_sha'],
                checkpoint=dict(reference='repo://' + checkpoint.relative_to(ctx['repo']).as_posix(), sha256=sha(checkpoint)))
        identity = ctx['report'] / 'TRAIN_IDENTITY.json'
        datasets[ctx['task']['dataset']] = dict(matched_arms=[job['job_id'] for job in state['jobs']],
            identity_binding=dict(reference='repo://' + identity.relative_to(ctx['repo']).as_posix(), sha256=sha(identity)))
    if len(model_states) != 1:
        raise ValueError('The whole wave must share the exact pinned F state')
    path = Path(output_path).resolve()
    if not path.is_relative_to(first['repo'] / 'reports' / NAME):
        raise ValueError('Selection seal escaped the new namespace')
    decision = dict(status='SEALED', cap=3, V_used=False, selection_split='A_stop',
        all_searches_complete=True, matched_arms=plan['matched_arms'], datasets=datasets,
        selections=selections, wave_plan_binding=binding(plan_path, first))
    sealed_json(path, decision)
    return decision


def _selection_decision(job, ctx, cap):
    descriptor = ctx.get('selection_decision', {})
    path = Path(descriptor.get('path', ''))
    if not path.is_relative_to(ctx['repo'] / 'reports' / NAME) or sha(path) != descriptor.get('sha256'):
        raise ValueError('Exact registered all-search seal required before V')
    decision = read(path)
    plan_path = resolve_binding(decision['wave_plan_binding'], ctx)
    plan = _wave(ctx, plan_path)
    own = jobs(ctx['task'])
    if (cap != 3 or decision.get('cap') != 3 or decision.get('status') != 'SEALED'
            or decision.get('V_used') is not False or decision.get('all_searches_complete') is not True
            or decision.get('selection_split') != 'A_stop' or decision.get('matched_arms') != plan['matched_arms']
            or set(decision.get('selections', {})) != set(plan['matched_arms'])
            or decision.get('datasets', {}).get(job['cohort'], {}).get('matched_arms') != [item['job_id'] for item in own]
            or job not in own):
        raise ValueError('Complete wave width/seed group required before V')
    for item in decision['datasets'].values():
        resolve_binding(item['identity_binding'], ctx)
    for item in decision['selections'].values():
        resolve_binding(item['checkpoint'], ctx)
    return binding(path, ctx)


@contextmanager
def selection_scope(helper):
    original = helper._selection_decision
    helper._selection_decision = _selection_decision
    try:
        yield
    finally:
        helper._selection_decision = original


def refit_dataset(config_path, prefix_path, model_dir, selection_path, *, repo=REPO,
                  cache_dir=None, report_dir=None, deadline=None, runtime_factory=None):
    ctx = build_context(config_path, prefix_path, model_dir, repo=repo,
        cache_dir=cache_dir, report_dir=report_dir, deadline=deadline)
    results = read(ctx['cache'] / 'SEARCH_RESULTS.json')
    fixed_jobs = jobs(ctx['task'])
    if len(results) != len(fixed_jobs):
        raise ValueError('All registered K/seed searches are required')
    ctx['selection_decision'] = dict(path=Path(selection_path).resolve(), sha256=sha(selection_path))
    for job, result in zip(fixed_jobs, results):
        _check_search(job, result, ctx)
        _selection_decision(job, ctx, 3)
        if read(selection_path)['selections'][job['job_id']]['epoch'] != result['selected_epoch']:
            raise ValueError('Fresh refit epoch differs from the all-search seal')
    helper, active = (runtime_factory or runtime)(ctx, ctx['task'], ctx['model_dir'])
    rows = []
    try:
        with selection_scope(helper):
            for job, search in zip(fixed_jobs, results):
                print('REFIT ' + job['job_id'], flush=True)
                result = helper._refit_predict(job, active, search, 3)
                if result.get('status') != 'COMPLETE' or len(result.get('rows', [])) != 1:
                    raise RuntimeError('Fresh refit incomplete; preserve checkpoint and resume identity')
                row = dict(result['rows'][0])
                expected = dict(family='B2', dataset=job['cohort'], revin=False, seed=job['seed'],
                    K=job['K'], C=len(ctx['task']['columns']), L=512, H=48, cap=3,
                    selected_epoch=search['selected_epoch'], independent_FP64='PASS', evaluation='V_select',
                    fit='fresh_A60', data_sha=ctx['task']['data_sha256'], model_sha=search['contract']['model_sha'])
                if any(row.get(key) != value for key, value in expected.items()):
                    raise ValueError('Fresh refit model/data/scoring identity differs')
                resolve_binding(result['bundle_binding'], ctx)
                resolve_binding(row['prediction_binding'], ctx)
                scores = [float(value['astop_mse']) for value in search['curves'] if value.get('astop_mse') is not None]
                last_best = scores[-1] < min(scores[:-1])
                row.update(final=True, status='COMPLETE', bundle_binding=result['bundle_binding'],
                    V_independent_test=False, all_searches_sealed_before_V=True,
                    selection_decision_binding=binding(selection_path, ctx),
                    last_checkpoint_strictly_best=last_best, last_scores_still_falling=scores[-1] < scores[-2],
                    cap_reached=True, selected_at_cap=search['selected_epoch'] == 3,
                    training_status='CAP3_STILL_IMPROVING' if last_best else 'SELECTED_EPOCH_ZERO'
                        if search['selected_epoch'] == 0 else 'CAP3_CONVERGENCE_UNRESOLVED')
                rows.append(row)
        sealed_json(ctx['cache'] / 'VERIFIED_ROWS.json', rows)
        sealed_json(ctx['report'] / 'VERIFIED_ROWS.json', rows)
        summary = dict(status='COMPLETE', dataset=ctx['task']['dataset'], new_searches=len(fixed_jobs),
            fresh_refits=len(rows), cap=3, K_grid=list(k_grid(len(ctx['task']['columns']))), rows=rows,
            selection_binding=binding(selection_path, ctx), T_access=False)
        sealed_json(ctx['report'] / 'TRAIN_SUMMARY.json', summary)
        return summary
    finally:
        active['foundation'].assert_frozen_unchanged()
        active.pop('foundation', None)


def execute_many(inputs, model_dir, *, repo=REPO, deadline=None, runtime_factory=None, selection_path=None):
    repo = Path(repo).resolve()
    selection = Path(selection_path).resolve() if selection_path else repo / 'reports' / NAME / 'ALL_SEARCHES_SEALED.json'
    plan_path = selection.with_name(selection.stem + '_WAVE_PLAN.json')
    contexts = [build_context(config, prefix, model_dir, repo=repo, deadline=deadline) for config, prefix in inputs]
    seal_wave_plan(contexts, plan_path)
    states = [search_dataset(config, prefix, model_dir, repo=repo, deadline=deadline,
        runtime_factory=runtime_factory, wave_plan=plan_path) for config, prefix in inputs]
    seal_searches(states, selection)
    results = [refit_dataset(config, prefix, model_dir, selection, repo=repo, deadline=deadline,
                            runtime_factory=runtime_factory) for config, prefix in inputs]
    return dict(status='COMPLETE', new_searches=sum(item['new_searches'] for item in results),
                fresh_refits=sum(item['fresh_refits'] for item in results), datasets=results, selection_path=str(selection))


def cost_inputs(request, report_dir, cache_dir):
    ctx = build_context(request['config_path'], request['prefix_path'], request['model_dir'],
                        report_dir=report_dir, cache_dir=cache_dir)
    task = ctx['task']
    if request['dataset'] != task['dataset'] or request['seed'] not in SEEDS:
        raise ValueError('Cost dataset/seed differs from its exact task')
    from research.adapter_diagnostics_20261017_v2.existing_jobs import _modules
    from research.b2_revin_20261018_v1.costs import route_loader
    modules = _modules(ctx)
    from gpu_runtime import FrozenBolt
    def foundation_loader(context, pinned_modules, horizon, device='cuda'):
        foundation = FrozenBolt(ctx['model_dir'], horizon, row_microbatch=128, checkpoint_inputs=False, device=device)
        if foundation.initial_hash != pinned_modules['legacy'].F_SHA:
            raise ValueError('Cost frozen F differs from the registered parent')
        return foundation
    ctx.update(modules=modules, foundation_loader=foundation_loader)
    saved, family, run_id = None, request['family'], request['run_id']
    if family == 'B2':
        if request['K'] not in k_grid(len(task['columns'])):
            raise ValueError('Cost K was not registered')
        expected_id = f'B2_{task["dataset"]}_K{request["K"]}_OFF_s{request["seed"]}'
        rows = read(ctx['cache'] / 'VERIFIED_ROWS.json')
        matches = [row for row in rows if row['run_id'] == run_id]
        if run_id != expected_id or len(matches) != 1:
            raise ValueError('Cost requires one exact trained K/seed')
        candidate = matches[0]
        if candidate['data_sha'] != task['data_sha256'] or candidate['K'] != request['K'] or candidate['seed'] != request['seed']:
            raise ValueError('Scored B2 task/width/seed differs')
        prediction = resolve_binding(candidate['prediction_binding'], ctx)
        resolve_binding(candidate['bundle_binding'], ctx)
        with np.load(prediction, allow_pickle=False) as archive:
            saved = {key: archive[key].copy() for key in archive.files}
        loader = route_loader(candidate, task, ctx, inventory=dict(baselines=[]))
    elif family == 'RAW':
        if run_id != 'Raw_' + task['dataset'] or request['seed'] != SEEDS[0]:
            raise ValueError('Raw must use its deterministic registered identity')
        candidate = dict(family='RAW', dataset=task['dataset'], run_id=run_id, seed=request['seed'],
                         C=len(task['columns']), K=len(task['columns']), L=512, H=48)
        loader = route_loader(candidate, task, ctx, inventory=dict(baselines=[]))
    elif family == 'DIRECT_G':
        from research.b2_dataset_extension_20261018_v1.baselines import direct_id
        if run_id != direct_id(task, request['seed']):
            raise ValueError('Direct MLP must use its exact registered seed identity')
        candidate = read(ctx['report'] / (run_id + '_FIT.json'))
        bundle = resolve_binding(candidate['bundle_binding'], ctx)
        expected = dict(family='DIRECT_G', dataset=task['dataset'], seed=request['seed'],
                        C=len(task['columns']), L=512, H=48, epochs=list(EPOCHS), status='COMPLETE_VALID')
        if (any(candidate.get(key) != value for key, value in expected.items())
                or candidate['artifact_sha256'] != sha(bundle)):
            raise ValueError('Direct MLP fit/checkpoint/scientific axes differ')
        candidate['K'] = 0
        spec = dict(family='DIRECT_G', run_id=run_id, artifact_cache_relative_path=bundle.relative_to(ctx['cache']).as_posix(),
                    artifact_sha256=candidate['artifact_sha256'], canonical_fit_end=task['split_ranges']['A_stop'][1])
        def loader():
            return modules['deployment'].Route(spec, task, ctx['cache'], device='cuda')
    else:
        raise ValueError('Unregistered cost family')
    if saved is None:
        mean, std, _ = modules['data'].fit_statistics(task, task['split_ranges']['A_stop'][1])
        origins = modules['data'].origins(task['split_ranges']['V_select'], 512, 48, 48)
        rows = origins[:, None] + np.arange(48)[None]
        observed = task['observed'][rows]
        target = ((np.where(observed, task['physical'][rows], mean) - mean) / std).astype(np.float32)
        saved = dict(target_normalized=target, observed=observed.copy(), origins=origins, mean=mean, std=std,
                     channel_ids=np.asarray(task['columns']), pred_normalized=None)
    return dict(route_loader=loader, loader=loader, task=task, saved=saved,
        expected_f_sha=modules['legacy'].F_SHA, expected_f_params=47_718_016, candidate=candidate, source_ctx=ctx,
        source_bindings=dict(config_sha256=sha(ctx['config_path']), prefix_sha256=ctx['config']['prefix_sha256'],
                             data_sha256=task['data_sha256'], sources=ctx['scientific_source_hashes']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('search', 'refit'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--prefix', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--wave-plan', type=Path)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args(argv)
    if args.phase == 'search':
        value = search_dataset(args.config, args.prefix, args.model_dir, wave_plan=args.wave_plan)['summary']
    else:
        if args.selection is None:
            parser.error('--selection is required')
        value = refit_dataset(args.config, args.prefix, args.model_dir, args.selection)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
