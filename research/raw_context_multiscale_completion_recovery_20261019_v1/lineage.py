"""Keep the failed attempt immutable and require a separately completed CPU audit."""
from pathlib import Path
import math

from research.raw_context_multiscale_20261018_v1 import common


NAME = 'raw_context_multiscale_completion_recovery_20261019_v1'
REGISTRATION = 'COMPLETION_RECOVERY_REGISTRATION_001.json'
VALIDATION = 'RECOVERY_VALIDATION_REPORT_001.json'
TERMINAL = 'TERMINAL_COMPLETION_RECOVERY_001.json'
WORK_RESULT = 'COMPLETION_RECOVERY_WORK_RESULT_001.json'
SOURCE_NAMES = {'__init__.py','lineage.py','run.py','adapters.py'}
TEST_NAMES = {'test_raw_context_multiscale_completion_recovery.py',
              'test_raw_context_multiscale_recovery_adapters.py',
              'test_raw_context_multiscale_recovery_lineage.py'}


def validate_registration(ctx):
    path = Path(ctx['report']) / REGISTRATION
    plan = common.read(path)
    if (plan.get('status') != 'REGISTERED_VERIFIER_ONLY_RECOVERY'
            or plan.get('scientific_contract_changed') is not False
            or plan.get('new_training_updates') != 0 or plan.get('new_V_forecasts') != 0
            or plan.get('new_GPU_measurements') != 0 or plan.get('T_access') is not False
            or plan.get('identity_exception') != 'EXACT_REGISTERED_DIRECT_G_NO_FOUNDATION_ONLY'):
        raise ValueError('Exact verifier-only recovery registration required')
    repo = Path(ctx['repo']).resolve()
    expected_frozen = {
        **{'research/raw_context_multiscale_20261018_v1/'+name:None for name in
           ('__init__.py','audit.py','calibration.py','common.py','costs.py','data.py',
            'deployment.py','gates.py','models.py','pipeline.py','report.py','run.py','training.py')},
        **{'research/raw_context_multiscale_supplement_20261019_v1/'+name:None
           for name in ('__init__.py','cp0.py','report.py')},
        **{'research/'+namespace+'/'+name:None for namespace in
           ('raw_context_multiscale_independent_20261019_v1','raw_context_multiscale_mcal_diagnostics_20261019_v1')
           for name in ('__init__.py','run.py')}}
    expected_groups = dict(frozen_source_files=set(expected_frozen),
        recovery_source_files={'research/'+NAME+'/'+name for name in SOURCE_NAMES},
        test_source_files={'tests/'+name for name in TEST_NAMES})
    for group in ('frozen_source_files', 'recovery_source_files', 'test_source_files'):
        files = plan.get(group, {})
        if set(files) != expected_groups[group]:
            raise ValueError('Complete frozen source closure required')
        for name, digest in files.items():
            source = (repo/name).resolve()
            if not source.is_relative_to(repo) or common.sha(source) != digest:
                raise ValueError('Recovery or frozen producer source changed: ' + name)
    driver = common.resolve_binding(plan['completion_driver_binding'],ctx)
    if driver.resolve() != (Path(ctx['cache'])/'driver_completion_recovery_001.py').resolve():
        raise ValueError('The actual separately exited CPU driver must be bound')
    required = {'failed_terminal', 'failed_chain_terminal', 'failed_validation', 'failed_stderr',
                'execution_start', 'primary_plan', 'scores', 'cost_index', 'model_seal',
                'selection', 'final_body_models', 'parents'}
    if set(plan.get('input_bindings', {})) != required:
        raise ValueError('Exact recovery input lineage required')
    inputs = {key: common.resolve_binding(value, ctx) for key,value in plan['input_bindings'].items()}
    expected_paths = dict(failed_terminal=Path(ctx['cache'])/'TERMINAL_CORE_002.json',
        failed_chain_terminal=Path(ctx['cache'])/'TERMINAL_SUPPLEMENT_CHAIN_001.json',
        failed_validation=Path(ctx['report'])/'VALIDATION_REPORT.json',
        execution_start=Path(ctx['cache'])/'CORE_EXECUTION_START.json')
    if any(inputs[key].resolve() != value.resolve() for key,value in expected_paths.items()):
        raise ValueError('The actual original failed attempt paths must remain explicit')
    failure = common.read(inputs['failed_terminal'])
    chain = common.read(inputs['failed_chain_terminal'])
    failed_validation = common.read(inputs['failed_validation'])
    if (failure.get('status') != 'FAILED' or failure.get('exit_code') != 1
            or chain.get('status') != 'FAILED_PRESERVED_FOR_ROOT_DIAGNOSIS' or chain.get('stages') != []
            or failed_validation.get('status') != 'FAIL'
            or failed_validation.get('reason') != 'Complete source/model/config/data identities required'):
        raise ValueError('Preserve the actual metadata-verifier failure, without relabeling it')
    start = common.read(inputs['execution_start'])
    deadline = start['started_unix'] + start['limits']['wall_seconds']
    if (start.get('status') != 'STARTED' or plan.get('original_deadline_unix') != deadline
            or plan.get('limits') != start['limits']
            or plan['registered_unix'] <= failure['finished_unix']):
        raise ValueError('Recovery must retain the original execution budget and follow failure')
    parents = common.read(inputs['parents'])['parents']
    allowed = [row['route_spec'] for row in parents if row['family'] == 'DIRECT_G']
    if len(allowed) != 2 or plan.get('allowed_direct_routes') != allowed:
        raise ValueError('Exactly the two originally bound Direct G parents are admissible')
    for route in allowed:
        artifact = Path(ctx['cache']) / route['artifact_cache_relative_path']
        if common.sha(artifact) != route['artifact_sha256']:
            raise ValueError('The registered Direct artifact changed')
    return plan


def validate_completion(ctx):
    plan = validate_registration(ctx)
    report, cache = Path(ctx['report']), Path(ctx['cache'])
    terminal_path, validation_path = cache/TERMINAL, report/VALIDATION
    terminal, validation = common.read(terminal_path), common.read(validation_path)
    work_path = cache/WORK_RESULT
    work = common.read(work_path)
    if (terminal.get('status') != 'COMPLETE' or terminal.get('exit_code') != 0
            or terminal.get('registration_binding') != common.binding(report/REGISTRATION,ctx)
            or terminal.get('work_result_binding') != common.binding(work_path,ctx)
            or terminal.get('validation_binding') != common.binding(validation_path,ctx)
            or terminal.get('original_deadline_unix') != plan['original_deadline_unix']
            or terminal.get('driver_binding') != plan['completion_driver_binding']
            or terminal.get('GPU_seconds') != 0
            or work.get('status') != 'PASS_VERIFIED_AND_RENDERED'
            or work.get('registration_binding') != terminal['registration_binding']
            or work.get('validation_binding') != terminal['validation_binding']
            or validation.get('status') != 'PASS' or validation.get('models') != 56
            or validation.get('cost_facets') != 120 or validation.get('T_access') is not False
            or validation.get('V_independent_test') is not False):
        raise ValueError('Actual completed CPU recovery and separate PASS validation required')
    if (any(work.get(key) != 0 for key in ('new_training_updates','new_V_forecasts','new_GPU_measurements','GPU_seconds'))
            or work.get('T_access') is not False
            or set(work.get('generated_bindings',{})) != {'REPORT_INPUTS.json','RESOURCE_LEDGER.json','README_KO.md'}):
        raise ValueError('Full CPU-only result and exact rendered output closure required')
    for key, input_key in (('SCORES.json','scores'),('COST_INDEX.json','cost_index'),
                           ('MODELS_SEALED_BEFORE_V.json','model_seal'),('registered_plan','primary_plan')):
        if validation.get('manifest_bindings',{}).get(key) != plan['input_bindings'][input_key]:
            raise ValueError('Recovered PASS must bind the original scientific manifests')
    for value in (terminal,work):
        if (any(not math.isfinite(float(value[key])) for key in ('started_unix','finished_unix','wall_seconds'))
                or value['wall_seconds'] < 0
                or not math.isclose(value['wall_seconds'],value['finished_unix']-value['started_unix'],
                                    rel_tol=0,abs_tol=.001)):
            raise ValueError('Finite, consistent actual process wall accounting required')
    if not (plan['registered_unix'] <= terminal['started_unix'] <= work['started_unix'] <= work['finished_unix']
            <= terminal['finished_unix'] <= plan['original_deadline_unix']):
        raise ValueError('Recovery chronology or original deadline is invalid')
    for name,value in work['generated_bindings'].items():
        if common.resolve_binding(value,ctx).resolve() != (report/name).resolve():
            raise ValueError('Recovered output must resolve to its exact public path')
    return dict(terminal_path=terminal_path, validation_path=validation_path)
