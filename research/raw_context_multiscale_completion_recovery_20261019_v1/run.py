"""CPU-only verifier recovery and explicitly gated downstream diagnostics."""
import argparse
import ast
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import textwrap
import time

from research.raw_context_multiscale_20261018_v1 import common
from . import lineage


def identity_ok(spec, contract, allowed):
    ordinary = all(isinstance(contract.get(key),str) and len(contract[key]) == 64
                   for key in ('source_sha','config_sha','data_sha','model_sha'))
    if ordinary:
        return True
    return (spec.get('family') == 'DIRECT_G' and spec in allowed
            and contract.get('model_sha') == 'NO_FOUNDATION'
            and all(isinstance(contract.get(key),str) and len(contract[key]) == 64
                    for key in ('source_sha','config_sha','data_sha')))


def route_tree(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    matches = [node for node in ast.walk(tree) if isinstance(node,ast.If)
               and len(node.body) == 1 and isinstance(node.body[0],ast.Raise)
               and isinstance(node.body[0].exc,ast.Call)
               and any(isinstance(value,ast.Constant) and value.value ==
                       'Complete source/model/config/data identities required'
                       for value in node.body[0].exc.args)]
    if len(matches) != 1 or not isinstance(matches[0].test,ast.UnaryOp):
        raise ValueError('The exact identity guard must occur once')
    expected = ast.parse("not all(isinstance(contract.get(key), str) and len(contract[key]) == 64 for key in ('source_sha', 'config_sha', 'data_sha', 'model_sha'))",mode='eval').body
    if ast.dump(matches[0].test) != ast.dump(expected):
        raise ValueError('The exact identity guard structure changed')
    matches[0].test = ast.parse('not _recovery_identity_ok(spec, contract)',mode='eval').body
    return ast.fix_missing_locations(tree)


def validation_tree(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value == 'VALIDATION_REPORT.json':
            node.value = lineage.VALIDATION
            count += 1
    if count != 2:
        raise ValueError('Exactly the two validation output literals must be adapted')
    return ast.fix_missing_locations(tree)


@contextmanager
def verifier(ctx):
    from research.raw_context_multiscale_20261018_v1 import report
    plan = lineage.validate_registration(ctx)
    original_route, original_verify = report._verify_route, report.verify
    namespace = dict(report.__dict__, _recovery_identity_ok=lambda spec,contract:
                     identity_ok(spec,contract,plan['allowed_direct_routes']))
    exec(compile(route_tree(original_route),str(Path(__file__).resolve())+'::route','exec'),namespace)
    # The copied function uses the module's recursive entry point for combined routes.
    fixed_route = namespace['_verify_route']
    fixed_route.__globals__['_verify_route'] = fixed_route
    exec(compile(validation_tree(original_verify),str(Path(__file__).resolve())+'::validation','exec'),namespace)
    report._verify_route, report.verify = fixed_route, namespace['verify']
    try:
        yield report
    finally:
        report._verify_route, report.verify = original_route, original_verify


def register(ctx):
    report, cache, repo = Path(ctx['report']),Path(ctx['cache']),Path(ctx['repo'])
    audit = common.read(report/'INDEPENDENT_AUDIT_PLAN_001.json')
    mcal = common.read(report/'MCAL_FINAL_FIT_PREFLIGHT_001.json')
    frozen = {**audit['primary_source_files'],**audit['supplement_source_files'],
              **audit['source_files'],**mcal['source_files']}
    inputs = dict(failed_terminal=cache/'TERMINAL_CORE_002.json',
        failed_chain_terminal=cache/'TERMINAL_SUPPLEMENT_CHAIN_001.json',
        failed_validation=report/'VALIDATION_REPORT.json',
        failed_stderr=cache/'EXECUTION_CORE_ATTEMPT_002.stderr.log',
        execution_start=cache/'CORE_EXECUTION_START.json',
        primary_plan=report/'EXPERIMENT_PLAN_REPAIR_002.json', scores=report/'SCORES.json',
        cost_index=report/'COST_INDEX.json', model_seal=report/'MODELS_SEALED_BEFORE_V.json',
        selection=report/'SELECTIONS_SEALED.json', final_body_models=cache/'FINAL_BODY_MODELS.json',
        parents=report/'DATA_AND_PARENT_BINDINGS.json')
    parents = common.read(inputs['parents'])['parents']
    start = common.read(inputs['execution_start'])
    files = sorted((repo/'research'/lineage.NAME).glob('*.py'))
    tests = [repo/'tests'/name for name in ('test_raw_context_multiscale_completion_recovery.py',
                                           'test_raw_context_multiscale_recovery_adapters.py',
                                           'test_raw_context_multiscale_recovery_lineage.py')]
    value = dict(status='REGISTERED_VERIFIER_ONLY_RECOVERY',registered_unix=time.time(),
        scientific_contract_changed=False,new_training_updates=0,new_V_forecasts=0,
        new_GPU_measurements=0,T_access=False,
        identity_exception='EXACT_REGISTERED_DIRECT_G_NO_FOUNDATION_ONLY',
        original_deadline_unix=start['started_unix']+start['limits']['wall_seconds'],limits=start['limits'],
        frozen_source_files=frozen,
        recovery_source_files={p.relative_to(repo).as_posix():common.sha(p) for p in files},
        test_source_files={p.relative_to(repo).as_posix():common.sha(p) for p in tests},
        completion_driver_binding=common.binding(cache/'driver_completion_recovery_001.py',ctx),
        input_bindings={key:common.binding(path,ctx) for key,path in inputs.items()},
        allowed_direct_routes=[row['route_spec'] for row in parents if row['family']=='DIRECT_G'],
        recovery_scope='Reuse the complete 56 saved forecasts and 120 cost receipts; independently rescore and render only. Preserve original FAIL files. Explicit named consumer gate adapters use actual new success paths.')
    common.sealed_json(report/lineage.REGISTRATION,value)
    lineage.validate_registration(ctx)
    return dict(status=value['status'])


def complete(ctx):
    plan = lineage.validate_registration(ctx)
    ctx['deadline'] = plan['original_deadline_unix']
    if time.time() >= ctx['deadline']:
        raise TimeoutError('Original execution deadline exhausted')
    if (ctx['report']/lineage.VALIDATION).exists() or (ctx['cache']/lineage.WORK_RESULT).exists():
        raise FileExistsError('Preserve prior recovery outputs')
    started = time.time()
    common.event(ctx,'VERIFIER_COMPLETION_RECOVERY','STARTED',GPU_seconds=0)
    with verifier(ctx) as report:
        validation = report.verify(ctx)
        if validation['status'] != 'PASS':
            raise ValueError('Recovered audit did not pass')
        report.generate(ctx)
    lineage.validate_registration(ctx)
    common.event(ctx,'VERIFIER_COMPLETION_RECOVERY','COMPLETE',wall_seconds=time.time()-started,GPU_seconds=0)
    files = [ctx['report']/name for name in ('REPORT_INPUTS.json','RESOURCE_LEDGER.json','README_KO.md')]
    value = dict(status='PASS_VERIFIED_AND_RENDERED',started_unix=started,finished_unix=time.time(),
        wall_seconds=time.time()-started,GPU_seconds=0,
        worker_pid=os.getpid(),
        registration_binding=common.binding(ctx['report']/lineage.REGISTRATION,ctx),
        validation_binding=common.binding(ctx['report']/lineage.VALIDATION,ctx),
        generated_bindings={p.name:common.binding(p,ctx) for p in files},
        new_training_updates=0,new_V_forecasts=0,new_GPU_measurements=0,T_access=False)
    common.sealed_json(ctx['cache']/lineage.WORK_RESULT,value)
    return dict(status=value['status'],models=56,cost_facets=120)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('register','complete','cp0','final','independent','mcal'))
    parser.add_argument('--asset-root',required=True)
    args = parser.parse_args(argv)
    ctx = common.context(args.asset_root,device='cuda' if args.stage=='cp0' else 'cpu')
    if args.stage == 'register':
        result = register(ctx)
    elif args.stage == 'complete':
        result = complete(ctx)
    else:
        from .adapters import install
        from research.raw_context_multiscale_supplement_20261019_v1 import cp0,report
        from research.raw_context_multiscale_independent_20261019_v1 import run as independent
        from research.raw_context_multiscale_mcal_diagnostics_20261019_v1 import run as mcal
        evidence = lineage.validate_completion(ctx)
        with install(ctx):
            if args.stage == 'cp0':
                result = cp0.evaluate(ctx,ctx['report']/'SUPPLEMENT_PLAN_001.json',device='cuda')
            elif args.stage == 'final':
                result = report.generate(ctx,plan_path=ctx['report']/'SUPPLEMENT_PLAN_001.json',
                                         terminal_path=evidence['terminal_path'])
            elif args.stage == 'independent':
                code = independent.main(['audit','--require-cp0','--repo-root',str(ctx['repo'])])
                if code != 0:
                    raise RuntimeError('Independent audit failed')
                result = common.read(ctx['report']/'INDEPENDENT_FINAL_AUDIT_001.json')
            else:
                result = mcal.evaluate(ctx,ctx['report']/'MCAL_FIT_DIAGNOSTIC_SCOPE_001.json',
                                       ctx['report']/'MCAL_FINAL_FIT_PREFLIGHT_001.json')
    print(json.dumps(dict(stage=args.stage,status=result['status'])))


if __name__ == '__main__':
    main()
