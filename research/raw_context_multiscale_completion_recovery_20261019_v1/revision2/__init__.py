"""Second verifier recovery with exact historical config-schema validation."""
import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import textwrap
import time

from research.raw_context_multiscale_20261018_v1 import common
from .. import adapters, lineage, run as first


REGISTRATION = 'COMPLETION_RECOVERY_REGISTRATION_002.json'
VALIDATION = 'RECOVERY_VALIDATION_REPORT_002.json'
TERMINAL = 'TERMINAL_COMPLETION_RECOVERY_002.json'
WORK_RESULT = 'COMPLETION_RECOVERY_WORK_RESULT_002.json'
ROOT = 'research/informs_broad_validation_20261015_v1/'
PRODUCERS = {
    ROOT+'priority_valid_family_20261016_v1/family_minimum_runner.py':
        '12462b8f1085e346750a24cd37fd52191f8780799e2f9e5903b4a57bcbc570c1',
    ROOT+'priority_pems_subset_20261016_v1/subset_minimum_runner.py':
        'caf332f445121ae03249c23023fde916fefb17ab0eb7685dbd9cefe7be566381',
    ROOT+'priority_b2_validation_20261016_v1/safe_jena_prefix.py':
        '6b496dcbbb6cef48acb9010c630ed81359a2d60bff89aaec9e9e77c4b6d0b541',
}
LEGACY = {
    'jena_lora': ('priority_valid_family_20261016_v1',
                  'postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1', 'LORA', 'jena'),
    'pems_direct': ('priority_pems_subset_20261016_v1',
                    'postT_cpu_pems08_E1_SUBSET_DIRECT_G_k0_s92601_C32_priority_v1', 'DIRECT_G', 'pems32'),
}


def _source_files(ctx):
    package = Path(ctx['repo'])/'research'/lineage.NAME/'revision2'
    return {path.relative_to(ctx['repo']).as_posix():common.sha(path)
            for path in (package/'__init__.py', package/'__main__.py')}


def _legacy_records(ctx, original):
    parents = common.read(common.resolve_binding(original['input_bindings']['parents'],ctx))['parents']
    result = {}
    for kind,(namespace,run_id,family,dataset) in LEGACY.items():
        selected = [row for row in parents if row.get('run_id') == run_id and row.get('family') == family
                    and row.get('dataset') == dataset]
        if len(selected) != 1:
            raise ValueError('Exactly the registered historical schema route is required')
        base = Path(ctx['repo'])/ROOT/namespace
        source_path, plan_path = base/'runs'/run_id/'SOURCE_CONTRACT.json',base/'PLAN.json'
        source = common.read(source_path)
        result[kind] = dict(route=selected[0]['route_spec'], source_binding=common.binding(source_path,ctx),
            legacy_plan_binding=common.binding(plan_path,ctx), spec=source['spec'], config_sha=source['config_sha'],
            actual_prefix_sha256=source.get('actual_prefix_sha256'),
            subset_identity_payload=source.get('subset_identity_payload'))
    return result


def register(ctx):
    original = lineage.validate_registration(ctx)
    report,cache = Path(ctx['report']),Path(ctx['cache'])
    failed_terminal,failed_validation = cache/lineage.TERMINAL,report/lineage.VALIDATION
    terminal,validation = common.read(failed_terminal),common.read(failed_validation)
    if terminal.get('status') != 'FAILED' or terminal.get('exit_code') != 1 or validation.get('reason') != 'Bundle config/axis identity differs from its training contract':
        raise ValueError('Preserved first recovery config-schema failure required')
    test = Path(ctx['repo'])/'tests/test_raw_context_multiscale_recovery_revision2.py'
    value = dict(status='REGISTERED_EXACT_CONFIG_SCHEMA_RECOVERY',registered_unix=time.time(),
        previous_registration_binding=common.binding(report/lineage.REGISTRATION,ctx),
        failed_recovery_terminal_binding=common.binding(failed_terminal,ctx),
        failed_recovery_validation_binding=common.binding(failed_validation,ctx),
        completion_driver_binding=common.binding(cache/'driver_completion_recovery_002.py',ctx),
        source_files=_source_files(ctx),test_source_files={test.relative_to(ctx['repo']).as_posix():common.sha(test)},
        producer_source_files=dict(PRODUCERS),legacy_routes=_legacy_records(ctx,original),
        original_deadline_unix=original['original_deadline_unix'],limits=original['limits'],
        new_training_updates=0,new_V_forecasts=0,new_GPU_measurements=0,GPU_seconds=0,T_access=False,
        scientific_contract_changed=False,
        scope='Verify original exact config schemas for two fixed historical artifacts; preserve all original checks, predictions, costs, failures and execution budget.')
    common.sealed_json(report/REGISTRATION,value)
    validate_registration(ctx)
    return dict(status=value['status'])


def validate_registration(ctx):
    original = lineage.validate_registration(ctx)
    report,cache,repo = Path(ctx['report']),Path(ctx['cache']),Path(ctx['repo']).resolve()
    plan = common.read(report/REGISTRATION)
    if (plan.get('status') != 'REGISTERED_EXACT_CONFIG_SCHEMA_RECOVERY'
            or any(plan.get(key) != 0 for key in ('new_training_updates','new_V_forecasts','new_GPU_measurements','GPU_seconds'))
            or plan.get('T_access') is not False or plan.get('scientific_contract_changed') is not False
            or plan.get('original_deadline_unix') != original['original_deadline_unix'] or plan.get('limits') != original['limits']):
        raise ValueError('Exact schema-only recovery and original budget required')
    expected = {
        'previous_registration_binding': report/lineage.REGISTRATION,
        'failed_recovery_terminal_binding': cache/lineage.TERMINAL,
        'failed_recovery_validation_binding': report/lineage.VALIDATION,
        'completion_driver_binding': cache/'driver_completion_recovery_002.py'}
    for key,path in expected.items():
        if common.resolve_binding(plan[key],ctx).resolve() != path.resolve():
            raise ValueError('Exact preceding failure and new driver paths must be bound')
    failure = common.read(cache/lineage.TERMINAL)
    validation = common.read(report/lineage.VALIDATION)
    if (failure.get('status') != 'FAILED' or failure.get('exit_code') != 1
            or validation.get('status') != 'FAIL'
            or validation.get('reason') != 'Bundle config/axis identity differs from its training contract'
            or not failure['finished_unix'] < plan['registered_unix'] < plan['original_deadline_unix']):
        raise ValueError('Revision two must follow the actual unchanged first recovery failure')
    expected_test = 'tests/test_raw_context_multiscale_recovery_revision2.py'
    if (plan.get('source_files') != _source_files(ctx)
            or set(plan.get('test_source_files',{})) != {expected_test}
            or plan.get('producer_source_files') != PRODUCERS):
        raise ValueError('Exact revision-two and historical producer closure required')
    for group in ('source_files','test_source_files','producer_source_files'):
        for name,digest in plan[group].items():
            path = (repo/name).resolve()
            if not path.is_relative_to(repo) or common.sha(path) != digest:
                raise ValueError('Registered recovery or historical producer changed: '+name)
    if plan.get('legacy_routes') != _legacy_records(ctx,original):
        raise ValueError('The exact two legacy artifact/spec/config/plan bindings changed')
    return plan


def config_ok(spec,contract,saved,task,axes,legacy):
    for kind,record in legacy.items():
        if spec != record['route']:
            continue
        if saved['spec'] != record['spec'] or saved.get('plan_sha256') != record['legacy_plan_binding']['sha256']:
            return False
        if kind == 'jena_lora':
            prefix = hashlib.sha256(json.dumps(task['materialized_prefix_array_sha256'],sort_keys=True).encode()).hexdigest()
            if prefix != saved.get('actual_prefix_sha256') or prefix != record['actual_prefix_sha256']:
                return False
            identity = dict(spec=saved['spec'],L=int(task['L']),H=int(task['H']),C=len(task['columns']),
                columns=list(task['columns']),inherited_array_sha256=task.get('array_sha256',{}),materialized_prefix_sha256=prefix)
        elif kind == 'pems_direct':
            payload = task['subset_identity_payload']
            if (saved.get('subset_identity_payload') != payload or record['subset_identity_payload'] != payload
                    or common.fingerprint(payload) != task['data_sha256']):
                return False
            identity = dict(spec=saved['spec'],subset_identity=payload)
        else:
            return False
        return contract['config_sha'] == record['config_sha'] == common.fingerprint(identity)
    return contract['config_sha'] == common.fingerprint(dict(spec=saved['spec'],axes=axes))


def route_tree(function):
    tree = first.route_tree(function)
    matches = [node for node in ast.walk(tree) if isinstance(node,ast.If) and len(node.body)==1
               and isinstance(node.body[0],ast.Raise) and isinstance(node.body[0].exc,ast.Call)
               and any(isinstance(value,ast.Constant) and value.value=='Bundle config/axis identity differs from its training contract'
                       for value in node.body[0].exc.args)]
    expected = ast.parse("contract['config_sha'] != common.fingerprint(dict(spec=saved['spec'], axes=axes))",mode='eval').body
    if len(matches)!=1 or ast.dump(matches[0].test)!=ast.dump(expected):
        raise ValueError('Exact original config guard required')
    matches[0].test = ast.parse('not _recovery_config_ok(spec,contract,saved,task,axes)',mode='eval').body
    return ast.fix_missing_locations(tree)


def validation_tree(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='VALIDATION_REPORT.json':
            node.value=VALIDATION
            count+=1
    if count!=2:
        raise ValueError('Exactly two original validation output literals required')
    return ast.fix_missing_locations(tree)


@contextmanager
def verifier(ctx):
    from research.raw_context_multiscale_20261018_v1 import report
    plan = validate_registration(ctx)
    original_plan = lineage.validate_registration(ctx)
    original_route,original_verify = report._verify_route,report.verify
    namespace = dict(report.__dict__,_recovery_identity_ok=lambda spec,contract:
        first.identity_ok(spec,contract,original_plan['allowed_direct_routes']),
        _recovery_config_ok=lambda spec,contract,saved,task,axes:
        config_ok(spec,contract,saved,task,axes,plan['legacy_routes']))
    exec(compile(route_tree(original_route),str(Path(__file__))+'::route','exec'),namespace)
    fixed = namespace['_verify_route']
    fixed.__globals__['_verify_route']=fixed
    exec(compile(validation_tree(original_verify),str(Path(__file__))+'::verify','exec'),namespace)
    report._verify_route,report.verify = fixed,namespace['verify']
    try:
        yield report
    finally:
        report._verify_route,report.verify = original_route,original_verify


def complete(ctx):
    plan = validate_registration(ctx)
    ctx['deadline']=plan['original_deadline_unix']
    if time.time()>=ctx['deadline']:
        raise TimeoutError('Original execution deadline exhausted')
    if (ctx['report']/VALIDATION).exists() or (ctx['cache']/WORK_RESULT).exists():
        raise FileExistsError('Preserve existing second recovery outputs')
    started=time.time()
    common.event(ctx,'VERIFIER_COMPLETION_RECOVERY_002','STARTED',GPU_seconds=0)
    with verifier(ctx) as report:
        result=report.verify(ctx)
        if result['status']!='PASS':
            raise ValueError('Full independent numerical audit must pass')
        report.generate(ctx)
    validate_registration(ctx)
    finished=time.time()
    common.event(ctx,'VERIFIER_COMPLETION_RECOVERY_002','COMPLETE',wall_seconds=finished-started,GPU_seconds=0)
    names=('REPORT_INPUTS.json','RESOURCE_LEDGER.json','README_KO.md')
    value=dict(status='PASS_VERIFIED_AND_RENDERED',started_unix=started,finished_unix=finished,
        wall_seconds=finished-started,GPU_seconds=0,worker_pid=os.getpid(),
        registration_binding=common.binding(ctx['report']/REGISTRATION,ctx),
        validation_binding=common.binding(ctx['report']/VALIDATION,ctx),
        generated_bindings={name:common.binding(ctx['report']/name,ctx) for name in names},
        new_training_updates=0,new_V_forecasts=0,new_GPU_measurements=0,T_access=False)
    common.sealed_json(ctx['cache']/WORK_RESULT,value)
    return dict(status=value['status'],models=56,cost_facets=120)


def validate_completion(ctx):
    plan=validate_registration(ctx)
    original=lineage.validate_registration(ctx)
    report,cache=Path(ctx['report']),Path(ctx['cache'])
    terminal_path,validation_path,work_path=cache/TERMINAL,report/VALIDATION,cache/WORK_RESULT
    terminal,validation,work=common.read(terminal_path),common.read(validation_path),common.read(work_path)
    registration_binding=common.binding(report/REGISTRATION,ctx)
    validation_binding=common.binding(validation_path,ctx)
    if (terminal.get('status')!='COMPLETE' or terminal.get('exit_code')!=0 or terminal.get('GPU_seconds')!=0
            or terminal.get('registration_binding')!=registration_binding
            or terminal.get('validation_binding')!=validation_binding
            or terminal.get('work_result_binding')!=common.binding(work_path,ctx)
            or terminal.get('driver_binding')!=plan['completion_driver_binding']
            or terminal.get('original_deadline_unix')!=plan['original_deadline_unix']
            or work.get('status')!='PASS_VERIFIED_AND_RENDERED'
            or work.get('registration_binding')!=registration_binding or work.get('validation_binding')!=validation_binding
            or any(work.get(key)!=0 for key in ('new_training_updates','new_V_forecasts','new_GPU_measurements','GPU_seconds'))
            or work.get('T_access') is not False or validation.get('status')!='PASS'
            or validation.get('models')!=56 or validation.get('cost_facets')!=120
            or validation.get('T_access') is not False or validation.get('V_independent_test') is not False):
        raise ValueError('Actual completed second CPU recovery and full PASS validation required')
    for key,input_key in (('SCORES.json','scores'),('COST_INDEX.json','cost_index'),
                          ('MODELS_SEALED_BEFORE_V.json','model_seal'),('registered_plan','primary_plan')):
        if validation.get('manifest_bindings',{}).get(key)!=original['input_bindings'][input_key]:
            raise ValueError('All original scientific manifests must remain bound')
    for row in (terminal,work):
        if (any(not math.isfinite(float(row[key])) for key in ('started_unix','finished_unix','wall_seconds'))
                or row['wall_seconds']<0 or not math.isclose(row['wall_seconds'],row['finished_unix']-row['started_unix'],rel_tol=0,abs_tol=.001)):
            raise ValueError('Finite consistent actual process wall accounting required')
    if not (plan['registered_unix']<=terminal['started_unix']<=work['started_unix']<=work['finished_unix']
            <=terminal['finished_unix']<=plan['original_deadline_unix']):
        raise ValueError('Original deadline and actual recovery chronology required')
    if set(work.get('generated_bindings',{}))!={'REPORT_INPUTS.json','RESOURCE_LEDGER.json','README_KO.md'}:
        raise ValueError('Exact complete rendered output closure required')
    for name,value in work['generated_bindings'].items():
        if common.resolve_binding(value,ctx).resolve()!=(report/name).resolve():
            raise ValueError('Exact public generated output path required')
    return dict(terminal_path=terminal_path,validation_path=validation_path)


@contextmanager
def install(ctx):
    validate_completion(ctx)
    old=(adapters.validate_completion,adapters.TERMINAL_NAME,adapters.VALIDATION_NAME,adapters.FUNCTIONS)
    changes=[]
    for module,name,replacements in adapters.FUNCTIONS:
        changes.append((module,name,{key:(value.replace('_001.json','_002.json'),count)
                                    for key,(value,count) in replacements.items()}))
    adapters.validate_completion=validate_completion
    adapters.TERMINAL_NAME,adapters.VALIDATION_NAME=TERMINAL,VALIDATION
    adapters.FUNCTIONS=tuple(changes)
    try:
        with adapters.install(ctx) as manifest:
            yield manifest
    finally:
        adapters.validate_completion,adapters.TERMINAL_NAME,adapters.VALIDATION_NAME,adapters.FUNCTIONS=old


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('register','complete','cp0','final','independent','mcal'))
    parser.add_argument('--asset-root',required=True)
    args=parser.parse_args(argv)
    ctx=common.context(args.asset_root,device='cuda' if args.stage=='cp0' else 'cpu')
    if args.stage=='register':
        result=register(ctx)
    elif args.stage=='complete':
        result=complete(ctx)
    else:
        from research.raw_context_multiscale_supplement_20261019_v1 import cp0,report
        from research.raw_context_multiscale_independent_20261019_v1 import run as independent
        from research.raw_context_multiscale_mcal_diagnostics_20261019_v1 import run as mcal
        evidence=validate_completion(ctx)
        with install(ctx):
            if args.stage=='cp0':
                result=cp0.evaluate(ctx,ctx['report']/'SUPPLEMENT_PLAN_001.json',device='cuda')
            elif args.stage=='final':
                result=report.generate(ctx,plan_path=ctx['report']/'SUPPLEMENT_PLAN_001.json',terminal_path=evidence['terminal_path'])
            elif args.stage=='independent':
                if independent.main(['audit','--require-cp0','--repo-root',str(ctx['repo'])])!=0:
                    raise RuntimeError('Independent numerical audit failed')
                result=common.read(ctx['report']/'INDEPENDENT_FINAL_AUDIT_001.json')
            else:
                result=mcal.evaluate(ctx,ctx['report']/'MCAL_FIT_DIAGNOSTIC_SCOPE_001.json',ctx['report']/'MCAL_FINAL_FIT_PREFLIGHT_001.json')
    print(json.dumps(dict(stage=args.stage,status=result['status'])))

