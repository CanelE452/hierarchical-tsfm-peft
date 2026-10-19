"""Synthetic revision-two contract, immutable failure, and adapter checks."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from research.raw_context_multiscale_20261018_v1 import common
from research.raw_context_multiscale_completion_recovery_20261019_v1 import adapters, lineage
from research.raw_context_multiscale_completion_recovery_20261019_v1 import revision2 as second


def fixture_config(kind):
    spec=dict(family='LORA' if kind=='jena_lora' else 'DIRECT_G',run_id=kind,artifact_sha256='a'*64)
    task=dict(L=512,H=48 if kind=='jena_lora' else 12,columns=['a','b'],array_sha256={'physical':'b'*64},
              materialized_prefix_array_sha256={'physical':'c'*64,'observed':'d'*64,'times':'e'*64,'columns':'f'*64})
    saved=dict(spec=dict(run_id=kind,seed=92601),plan_sha256='9'*64)
    record=dict(route=copy.deepcopy(spec),spec=copy.deepcopy(saved['spec']),legacy_plan_binding=dict(sha256='9'*64))
    if kind=='jena_lora':
        prefix=hashlib.sha256(json.dumps(task['materialized_prefix_array_sha256'],sort_keys=True).encode()).hexdigest()
        saved['actual_prefix_sha256']=record['actual_prefix_sha256']=prefix
        identity=dict(spec=saved['spec'],L=512,H=48,C=2,columns=task['columns'],
                      inherited_array_sha256=task['array_sha256'],materialized_prefix_sha256=prefix)
    else:
        payload=dict(schema='synthetic_subset',columns=task['columns'],prefix=task['materialized_prefix_array_sha256'])
        task['subset_identity_payload']=copy.deepcopy(payload)
        saved['subset_identity_payload']=copy.deepcopy(payload)
        record['subset_identity_payload']=copy.deepcopy(payload)
        task['data_sha256']=common.fingerprint(payload)
        identity=dict(spec=saved['spec'],subset_identity=payload)
    record['config_sha']=common.fingerprint(identity)
    return spec,dict(config_sha=record['config_sha']),saved,task,{}, {kind:record}


@pytest.mark.parametrize('kind',('jena_lora','pems_direct'))
def test_historical_schemas_are_recomputed_exactly(kind):
    assert second.config_ok(*fixture_config(kind))


@pytest.mark.parametrize('kind',('jena_lora','pems_direct'))
@pytest.mark.parametrize('field',('artifact','spec','plan','config','prefix_or_subset'))
def test_legacy_exception_rejects_changed_identity(kind,field):
    spec,contract,saved,task,axes,records=fixture_config(kind)
    if field=='artifact':spec['artifact_sha256']='0'*64
    elif field=='spec':saved['spec']['seed']=1
    elif field=='plan':saved['plan_sha256']='0'*64
    elif field=='config':contract['config_sha']='0'*64
    elif kind=='jena_lora':task['materialized_prefix_array_sha256']['physical']='0'*64
    else:task['subset_identity_payload']['schema']='changed'
    assert not second.config_ok(spec,contract,saved,task,axes,records)


def test_other_models_retain_exact_generic_config_rule():
    saved=dict(spec=dict(seed=92601,family='B2'))
    axes=dict(L=512,H=48,C=2,columns=['a','b'])
    contract=dict(config_sha=common.fingerprint(dict(spec=saved['spec'],axes=axes)))
    assert second.config_ok(dict(family='B2'),contract,saved,{},axes,{})
    axes['H']=12
    assert not second.config_ok(dict(family='B2'),contract,saved,{},axes,{})


def test_entire_route_ast_changes_only_two_guard_expressions():
    from research.raw_context_multiscale_20261018_v1 import report
    original=ast.parse(inspect.getsource(report._verify_route))
    changed=second.route_tree(report._verify_route)
    pairs=list(zip([x for x in ast.walk(original) if isinstance(x,ast.If)],
                   [x for x in ast.walk(changed) if isinstance(x,ast.If)]))
    differences=[(a,b) for a,b in pairs if ast.dump(a.test)!=ast.dump(b.test)]
    assert len(differences)==2
    for a,b in differences:b.test=copy.deepcopy(a.test)
    assert ast.dump(changed)==ast.dump(original)
    text=ast.unparse(changed)
    for guard in ('model_state_sha','scaler_fingerprint','new_scientific_source_hashes','data_sha'):
        assert guard in text


def test_second_validation_changes_only_two_output_path_literals():
    from research.raw_context_multiscale_20261018_v1 import report
    original=ast.parse(inspect.getsource(report.verify))
    changed=second.validation_tree(report.verify)
    count=0
    for node in ast.walk(changed):
        if isinstance(node,ast.Constant) and node.value==second.VALIDATION:
            node.value='VALIDATION_REPORT.json';count+=1
    assert count==2
    assert ast.dump(changed)==ast.dump(original)


@pytest.fixture
def completion(tmp_path,monkeypatch):
    ctx=common.paths(tmp_path)
    ctx['report'].mkdir(parents=True);ctx['cache'].mkdir(parents=True)
    def write(path,value):
        common.atomic_json(path,value);return path
    original=dict(input_bindings={})
    for key,name in (('scores','SCORES.json'),('cost_index','COST_INDEX.json'),
                     ('model_seal','MODELS_SEALED_BEFORE_V.json'),('primary_plan','EXPERIMENT_PLAN_REPAIR_002.json')):
        original['input_bindings'][key]=common.binding(write(ctx['report']/name,{}),ctx)
    plan=dict(registered_unix=1002.,original_deadline_unix=22600.,completion_driver_binding=dict(reference='cache://driver2.py',sha256='a'*64))
    write(ctx['report']/second.REGISTRATION,plan)
    registration_binding=common.binding(ctx['report']/second.REGISTRATION,ctx)
    validation=dict(status='PASS',models=56,cost_facets=120,T_access=False,V_independent_test=False,
        manifest_bindings={name:original['input_bindings'][key] for key,name in
                          (('scores','SCORES.json'),('cost_index','COST_INDEX.json'),
                           ('model_seal','MODELS_SEALED_BEFORE_V.json'),('primary_plan','registered_plan'))})
    write(ctx['report']/second.VALIDATION,validation)
    work=dict(status='PASS_VERIFIED_AND_RENDERED',started_unix=1003.,finished_unix=1004.,wall_seconds=1.,
        registration_binding=registration_binding,validation_binding=common.binding(ctx['report']/second.VALIDATION,ctx),
        GPU_seconds=0,new_training_updates=0,new_V_forecasts=0,new_GPU_measurements=0,T_access=False,
        generated_bindings={name:common.binding(write(ctx['report']/name,{}),ctx)
                            for name in ('REPORT_INPUTS.json','RESOURCE_LEDGER.json','README_KO.md')})
    write(ctx['cache']/second.WORK_RESULT,work)
    terminal=dict(status='COMPLETE',exit_code=0,GPU_seconds=0,started_unix=1003.,finished_unix=1005.,wall_seconds=2.,
        registration_binding=registration_binding,validation_binding=work['validation_binding'],
        work_result_binding=common.binding(ctx['cache']/second.WORK_RESULT,ctx),
        driver_binding=plan['completion_driver_binding'],original_deadline_unix=plan['original_deadline_unix'])
    write(ctx['cache']/second.TERMINAL,terminal)
    for name,folder in (('TERMINAL_CORE_002.json','cache'),(lineage.TERMINAL,'cache'),
                         ('VALIDATION_REPORT.json','report'),(lineage.VALIDATION,'report')):
        write(ctx[folder]/name,dict(status='FAILED' if folder=='cache' else 'FAIL'))
    monkeypatch.setattr(second,'validate_registration',lambda _:plan)
    monkeypatch.setattr(lineage,'validate_registration',lambda _:original)
    return ctx,plan,validation,work,terminal,write


def test_second_completion_passes_and_all_prior_failure_bytes_survive(completion):
    ctx,_,_,_,_,_=completion
    paths=[ctx['cache']/'TERMINAL_CORE_002.json',ctx['cache']/lineage.TERMINAL,
           ctx['report']/'VALIDATION_REPORT.json',ctx['report']/lineage.VALIDATION]
    before=[p.read_bytes() for p in paths]
    result=second.validate_completion(ctx)
    assert result['terminal_path']==ctx['cache']/second.TERMINAL
    assert result['validation_path']==ctx['report']/second.VALIDATION
    assert [p.read_bytes() for p in paths]==before


@pytest.mark.parametrize('field',('models','cost_facets','new_training_updates','new_V_forecasts','GPU_seconds',
                                  'T_access','empty_outputs','wall_seconds','chronology','driver'))
def test_second_completion_rejects_semantically_rebound_corruption(completion,field):
    ctx,_,validation,work,terminal,write=completion
    if field in ('models','cost_facets'):validation[field]-=1
    elif field in ('new_training_updates','new_V_forecasts','GPU_seconds'):work[field]=1
    elif field=='T_access':work['T_access']=True
    elif field=='empty_outputs':work['generated_bindings']={}
    elif field=='wall_seconds':work['wall_seconds']=99
    elif field=='chronology':work['started_unix']=1001.;work['wall_seconds']=3.
    elif field=='driver':terminal['driver_binding']=terminal['registration_binding']
    write(ctx['report']/second.VALIDATION,validation)
    work['validation_binding']=common.binding(ctx['report']/second.VALIDATION,ctx)
    write(ctx['cache']/second.WORK_RESULT,work)
    terminal['validation_binding']=work['validation_binding']
    terminal['work_result_binding']=common.binding(ctx['cache']/second.WORK_RESULT,ctx)
    write(ctx['cache']/second.TERMINAL,terminal)
    with pytest.raises(ValueError):second.validate_completion(ctx)


def test_second_adapters_restore_original_metadata_after_failure(tmp_path,monkeypatch):
    ctx=common.paths(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(second,'validate_completion',lambda _:dict(
        terminal_path=ctx['cache']/second.TERMINAL,validation_path=ctx['report']/second.VALIDATION))
    before=(adapters.validate_completion,adapters.TERMINAL_NAME,adapters.VALIDATION_NAME,adapters.FUNCTIONS)
    with pytest.raises(RuntimeError,match='synthetic'):
        with second.install(ctx) as manifest:
            assert manifest['terminal_name']==second.TERMINAL
            assert manifest['validation_name']==second.VALIDATION
            assert all('_001.json' not in change['new'] for row in manifest['functions']
                       for change in row['literal_changes'].values())
            raise RuntimeError('synthetic')
    assert (adapters.validate_completion,adapters.TERMINAL_NAME,adapters.VALIDATION_NAME,adapters.FUNCTIONS)==before


def test_registration_requires_original_failure_before_any_success_read(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)):
        second.validate_completion(common.paths(tmp_path))


def test_legacy_inventory_accepts_raw_records_without_top_level_run_id(tmp_path):
    ctx=common.paths(tmp_path)
    ctx['report'].mkdir(parents=True)
    parents=[dict(dataset='jena',family='RAW',role='RAW_BASELINE',route_spec=dict(family='RAW',run_id='Raw_jena'))]
    for kind,(namespace,run_id,family,dataset) in second.LEGACY.items():
        route=dict(run_id=run_id,family=family,artifact_sha256='a'*64)
        parents.append(dict(dataset=dataset,run_id=run_id,family=family,route_spec=route))
        base=tmp_path/second.ROOT/namespace
        (base/'runs'/run_id).mkdir(parents=True)
        common.atomic_json(base/'PLAN.json',dict(synthetic=True))
        common.atomic_json(base/'runs'/run_id/'SOURCE_CONTRACT.json',
                           dict(spec=dict(run_id=run_id),config_sha='b'*64,
                                actual_prefix_sha256='c'*64,subset_identity_payload=dict(schema='synthetic')))
    path=ctx['report']/'DATA_AND_PARENT_BINDINGS.json'
    common.atomic_json(path,dict(parents=parents))
    original=dict(input_bindings=dict(parents=common.binding(path,ctx)))
    result=second._legacy_records(ctx,original)
    assert set(result)=={'jena_lora','pems_direct'}
    assert all(result[kind]['route']['run_id']==recipe[1] for kind,recipe in second.LEGACY.items())
