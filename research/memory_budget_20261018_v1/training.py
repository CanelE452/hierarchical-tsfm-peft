"""Gate-bound four-model Jena search/refit; importing starts no experiment."""
import argparse
import json
from pathlib import Path
import time

from . import common


def registered_jobs():
    return [dict(job_id=f'B2_jena_K{k}_OFF_s{seed}', cohort='jena', family='B2',
                 revin=False, init_kind='PCA', K=k, seed=seed, device='cuda')
            for k in (3, 12) for seed in (92601, 92602)]


def validate_gate(path, ctx, now=None):
    path = Path(path).resolve()
    if path != (Path(ctx['report'])/'PHASE_B_DECISION.json').resolve():
        raise ValueError('Phase B gate must be the registered report decision')
    gate = common.read(path)
    identity = gate.get('decision_identity')
    if identity != common.fingerprint({k: v for k, v in gate.items() if k != 'decision_identity'}):
        raise ValueError('Phase B decision identity changed')
    if (gate.get('status') != 'PROCEED' or gate.get('eligible') is not True
            or gate.get('phase_a_all_confirmed') is not True):
        raise ValueError('Phase B requires eligible and complete confirmed Phase A')
    protocol = common.resolve_binding(gate['protocol_binding'], ctx)
    if protocol != (Path(ctx['report'])/'PROTOCOL.md').resolve():
        raise ValueError('Phase B protocol reference differs from registered protocol')
    evidence = gate.get('phase_a_evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('Phase B requires bound Phase A evidence')
    for value in evidence:
        common.resolve_binding(value, ctx)
    budget = gate.get('budget', {})
    if budget.get('scientific_deadline_unix') != common.SCIENTIFIC_DEADLINE:
        raise ValueError('Phase B scientific deadline changed')
    estimates = [budget.get(key) for key in ('estimated_training_seconds', 'estimated_measurement_seconds')]
    if any(type(value) not in (int, float) or not 0 < value < float('inf') for value in estimates):
        raise ValueError('Phase B requires finite positive full-phase cost estimates')
    remaining = common.SCIENTIFIC_DEADLINE - (time.time() if now is None else now)
    if remaining < sum(estimates):
        raise TimeoutError('Insufficient remaining budget for the full Phase B')
    return gate


def _sources(ctx):
    sources = dict(ctx['scientific_source_hashes'])
    if not sources:
        raise ValueError('Phase B producer sources must be registered')
    for relative, expected in sources.items():
        path = (Path(ctx['repo'])/relative).resolve()
        if not path.is_relative_to(Path(ctx['repo']).resolve()) or common.sha(path) != expected:
            raise ValueError('Phase B producer source bytes changed: ' + relative)
    return sources


def _check(ctx):
    if time.time() >= ctx['deadline']:
        raise TimeoutError('Phase B scientific deadline reached')
    ctx['lease'].check()
    _sources(ctx)


def _event(ctx, value):
    path = Path(ctx['cache'])/'B_TRAINING_ATTEMPTS.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')


def _runtime(ctx):
    from research.adapter_improvements_20261018_v1.audit import load_task, load_foundation
    from research.b2_revin_20261018_v1 import training_jobs
    task = load_task(ctx, 'jena', ctx['modules'])
    foundation = load_foundation(ctx, ctx['modules'], task['H'], 'cuda')
    return training_jobs, dict(ctx, task=task, foundation=foundation)


def _canonical_row(job, result, search, ctx):
    if result.get('status') != 'COMPLETE' or len(result.get('rows', [])) != 1:
        raise RuntimeError('Phase B fresh refit is incomplete; resume the same command')
    row = dict(result['rows'][0])
    expected = dict(family='B2', revin=False, seed=job['seed'], dataset='jena', K=job['K'], cap=3,
        C=21, L=512, H=48, device='cuda', evaluation='V_select', fit='fresh_A60',
        independent_FP64='PASS', core_dtype='float32', prediction_dtype='float32',
        selected_epoch=float(search['selected_epoch']))
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError('Phase B refit row has unregistered model axes')
    common.resolve_binding(row['prediction_binding'], ctx)
    common.resolve_binding(result['bundle_binding'], ctx)
    curve = [v for v in search['curves'] if v.get('astop_mse') is not None]
    last_best = (len(curve) > 1 and curve[-1]['epoch'] == 3.
                 and curve[-1]['astop_mse'] < min(v['astop_mse'] for v in curve[:-1]))
    row.update(run_id=job['job_id']+'_cap3', job_id=job['job_id'], final=True,
        source_kind='NEW_TRAINED', status='COMPLETE', bundle_binding=result['bundle_binding'],
        selection_receipt_binding=result['selection_receipt_binding'],
        selection_decision_binding=result['selection_decision_binding'],
        training_status='TRAINING_CAP_STILL_IMPROVING' if last_best else 'SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED',
        last_checkpoint_strictly_best=last_best, V_independent_test=False,
        scaler_provenance='SAVED_AND_FIXED_A60_VERIFIED',
        normalization_fit_end=result['fit_receipt']['normalization_fit_end'],
        model_state_sha256=result['fit_receipt']['model_state_sha'],
        F_snapshot_sha256=result['model_sha'],
        initialization_sha=result['refit_summary']['contract']['initialization_sha'],
        training_recipe=dict(result['recipe'], seed=job['seed'], epochs=[.25, 1., 3.],
                             init_kind='PCA', revin=False),
        campaign_namespace=common.NAME, source_context='MEMORY_BUDGET_PHASE_B',
        artifact_context=dict(cache_reference='repo://.cache/'+common.NAME,
                              report_reference='repo://reports/'+common.NAME))
    return row


def execute(ctx, gate_path, runtime_factory=None):
    gate = validate_gate(gate_path, ctx)
    _check(ctx)
    gate_binding = common.binding(gate_path, ctx)
    sources = _sources(ctx)
    jobs = registered_jobs()
    for job in jobs:
        job['scientific_identity'] = common.fingerprint(dict(
            job=job, gate_binding=gate_binding, protocol_binding=gate['protocol_binding'],
            scientific_source_hashes=sources, epochs=[.25, 1., 3.]))
    identity = dict(schema_version=1, jobs=jobs, gate_binding=gate_binding,
        protocol_binding=gate['protocol_binding'], scientific_source_hashes=sources,
        scientific_deadline_unix=common.SCIENTIFIC_DEADLINE,
        selection='A_fit train; strict minimum A_stop at 0/.25/1/3; fresh A60 refit',
        new_K=[3, 12], reused_K=6, revin=False, max_new_models=4)
    common.sealed_json(Path(ctx['report'])/'B_TRAINING_IDENTITY.json', identity)
    started = time.time()
    _event(ctx, dict(event='STARTED', started_unix=started, gate_sha256=gate_binding['sha256']))
    active = 'PREPARE'
    resource_ctx = None
    try:
        helper, resource_ctx = (runtime_factory or _runtime)(ctx)
        searches = []
        for job in jobs:
            active = 'SEARCH_'+job['job_id']
            _check(ctx)
            print(active, flush=True)
            search = helper.search(job, resource_ctx, epochs=(.25, 1., 3.))
            expected_spec = dict(latent_k=job['K'], family='B2', revin=False,
                seed=job['seed'], dataset='jena', device='cuda', epochs=[.25, 1., 3.])
            if (search.get('status') != 'COMPLETE' or search.get('cap') != 3
                    or any(search.get('spec', {}).get(key) != value for key, value in expected_spec.items())
                    or search.get('V_used') is not False):
                raise RuntimeError('Phase B search incomplete or identity changed; resume the same command')
            searches.append(search)
        decision = dict(status='SEALED', cohort='jena', cap=3, selection_split='A_stop', V_used=False,
            matched_arms=[job['job_id'] for job in jobs], phase_a_gate_binding=gate_binding,
            scientific_source_hashes=sources, all_four_searches_complete=True,
            selections={job['job_id']: dict(epoch=result['selected_epoch'],
                astop_mse=result['summary']['selected_astop_mse'], checkpoint=result['checkpoint_binding'])
                for job, result in zip(jobs, searches)})
        selection_path = Path(ctx['report'])/'B_SELECTIONS_SEALED.json'
        common.sealed_json(selection_path, decision)
        resource_ctx['selection_decision'] = dict(path=selection_path, sha256=common.sha(selection_path))
        rows, summaries = [], []
        for job, search in zip(jobs, searches):
            active = 'REFIT_'+job['job_id']
            _check(ctx)
            print(active, flush=True)
            result = helper._refit_predict(job, resource_ctx, search, 3)
            row = _canonical_row(job, result, search, ctx)
            rows.append(row)
            summaries.append({key: row[key] for key in (
                'run_id', 'job_id', 'K', 'seed', 'MSE', 'MAE', 'cap', 'selected_epoch',
                'target_trained_params', 'P_params', 'E_params', 'D_params', 'search_seconds',
                'refit_seconds', 'training_status', 'last_checkpoint_strictly_best',
                'bundle_binding', 'prediction_binding', 'selection_receipt_binding')})
        _check(ctx)
        resource_ctx['foundation'].assert_frozen_unchanged()
        common.sealed_json(Path(ctx['cache'])/'B_PHASE_ROWS.json', rows)
        summary = dict(schema_version=1, status='COMPLETE', trained_models=4,
            gate_binding=gate_binding, protocol_binding=gate['protocol_binding'],
            source_identity_binding=common.binding(Path(ctx['report'])/'B_TRAINING_IDENTITY.json', ctx),
            selection_binding=common.binding(selection_path, ctx), rows=summaries,
            independent_test=False, all_searches_sealed_before_new_V=True,
            new_training_search_seconds=sum(row['search_seconds'] for row in rows),
            new_training_refit_seconds=sum(row['refit_seconds'] for row in rows),
            resume_command='python -m research.memory_budget_20261018_v1.training --asset-root <ASSET_ROOT> '
                '--gate reports/memory_budget_20261018_v1/PHASE_B_DECISION.json '
                '--resource-cache .cache/memory_budget_20261018_v1')
        common.sealed_json(Path(ctx['report'])/'B_TRAINING_SUMMARY.json', summary)
        _event(ctx, dict(event='COMPLETE', started_unix=started, finished_unix=time.time(),
                        actual_process_wall_seconds=time.time()-started, new_models=4))
        print('PHASE_B_TRAINING_COMPLETE', flush=True)
        return summary
    except BaseException as error:
        _event(ctx, dict(event='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
            phase=active, exception_type=type(error).__name__, started_unix=started,
            finished_unix=time.time(), actual_process_wall_seconds=time.time()-started,
            resume='same CLI; preserve checkpoints and exact producer sources'))
        raise
    finally:
        if resource_ctx is not None:
            foundation = resource_ctx.pop('foundation', None)
            if foundation is not None:
                foundation.assert_frozen_unchanged()
            del foundation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--resource-cache', type=Path, required=True)
    args = parser.parse_args(argv)
    paths = common.paths()
    if args.resource_cache.resolve() != paths['cache'].resolve():
        parser.error('--resource-cache must be the registered new campaign cache')
    parent = common.source_context(args.asset_root)
    sources = dict(parent['scientific_source_hashes'], **common.source_hashes())
    ctx = dict(parent, **paths, scientific_source_hashes=sources, deadline=common.SCIENTIFIC_DEADLINE)
    validate_gate(args.gate, ctx)
    from research.informs_broad_validation_20261015_v1.resources import validate_gpu_child
    with validate_gpu_child(args.resource_cache) as lease:
        ctx['lease'] = lease
        ctx['deadline'] = min(common.SCIENTIFIC_DEADLINE, lease.deadline)
        execute(ctx, args.gate)


if __name__ == '__main__':
    main()
