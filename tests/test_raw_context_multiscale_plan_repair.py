"""A reporting registration repair preserves the original scientific contract."""
import copy
from pathlib import Path

import pytest

from research.raw_context_multiscale_20261018_v1 import common,run


@pytest.fixture
def repair_context(tmp_path,monkeypatch):
    ctx=dict(repo=tmp_path,report=tmp_path/'reports',cache=tmp_path/'cache',asset=tmp_path/'asset')
    for name in ('report','cache','asset'):
        ctx[name].mkdir()
    files={}
    for name in ('run.py','report.py','models.py','deployment.py'):
        relative='research/'+common.NAME+'/'+name
        target=tmp_path/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('original '+name,encoding='utf8')
        files[relative]=common.sha(target)
    original=dict(status='REGISTERED_BEFORE_NEW_V',T_access=False,registered_unix=10.,started_unix=1.,
                  source_files=files,limits=dict(wall_seconds=21600,GPU_reservation_seconds=14400),
                  seeds=[92601,92602],primary_metric='MSE',recipe=dict(microbatch=4))
    original_path=ctx['report']/'EXPERIMENT_PLAN.json'
    common.sealed_json(original_path,original)
    revised=copy.deepcopy(original)
    changed=['research/'+common.NAME+'/'+name for name in ('run.py','report.py')]
    for relative in changed:
        (tmp_path/relative).write_text('report coverage repair '+relative,encoding='utf8')
        revised['source_files'][relative]=common.sha(tmp_path/relative)
    revised.update(registered_unix=20.,supersedes=common.binding(original_path,ctx),
        repair=dict(status='REGISTERED_BEFORE_NEW_BODY_SEARCH',scientific_contract_changed=False,
                    new_body_searches_or_V_scores=0,RawContext_GPU_sessions_started=0,
                    changed_source_paths=changed))
    common.sealed_json(ctx['report']/'EXPERIMENT_PLAN_REPAIR_001.json',revised)
    monkeypatch.setattr(common,'model_sources',lambda repo:{})
    return ctx,original,revised


def test_original_plan_is_preserved_and_old_cli_path_resolves_registered_repair(repair_context):
    ctx,original,revised=repair_context
    path=ctx['report']/'EXPERIMENT_PLAN.json'
    before=path.read_bytes()
    assert run.check_plan(ctx,path)==revised
    assert ctx['plan_path']==ctx['report']/'EXPERIMENT_PLAN_REPAIR_001.json'
    assert common.read(path)==original and path.read_bytes()==before
    assert run.make_plan(ctx)==revised


@pytest.mark.parametrize('field,value',[
    ('limits',dict(wall_seconds=22000,GPU_reservation_seconds=14400)),
    ('seeds',[92601]),('primary_metric','MAE'),('recipe',dict(microbatch=8))])
def test_reporting_repair_cannot_change_scientific_contract(repair_context,field,value):
    ctx,_,revised=repair_context
    common.atomic_json(run.registered_plan_path(ctx),dict(revised,**{field:value}))
    with pytest.raises(ValueError,match='scientific contract'):
        run.load_registered_plan(ctx)


def test_reporting_repair_cannot_change_body_producer(repair_context):
    ctx,_,revised=repair_context
    relative='research/'+common.NAME+'/models.py'
    (ctx['repo']/relative).write_text('changed body',encoding='utf8')
    revised['source_files'][relative]=common.sha(ctx['repo']/relative)
    revised['repair']['changed_source_paths'].append(relative)
    common.atomic_json(run.registered_plan_path(ctx),revised)
    with pytest.raises(ValueError,match='Only the registered reporting'):
        run.load_registered_plan(ctx)


def test_repair_requires_exact_original_plan_bytes_and_current_sources(repair_context):
    ctx,_,_=repair_context
    original=ctx['report']/'EXPERIMENT_PLAN.json'
    original.write_text(original.read_text(encoding='utf8')+' ',encoding='utf8')
    with pytest.raises(ValueError,match='scientific contract'):
        run.load_registered_plan(ctx)


def test_source_edit_after_repair_registration_is_rejected(repair_context):
    ctx,_,_=repair_context
    source=ctx['repo']/'research'/common.NAME/'report.py'
    source.write_text('unregistered later edit',encoding='utf8')
    with pytest.raises(ValueError,match='register a repair'):
        run.check_plan(ctx,run.registered_plan_path(ctx))


def test_execution_start_binds_current_repair_without_restarting_budget(repair_context,monkeypatch):
    ctx,_,_=repair_context
    run.check_plan(ctx,ctx['report']/'EXPERIMENT_PLAN.json')
    proof=dict(commit='a'*40)
    common.sealed_json(ctx['cache']/'PRECEDING_DATASET_PUBLICATION.json',proof)
    monkeypatch.setattr(run.time,'time',lambda:100.)
    first=run.start_execution(ctx,proof)
    assert first['plan_binding']==common.binding(run.registered_plan_path(ctx),ctx)
    monkeypatch.setattr(run.time,'time',lambda:200.)
    assert run.start_execution(ctx,proof)==first and ctx['deadline']==21700.


@pytest.fixture
def execution_repair_context(repair_context,monkeypatch):
    ctx,original,previous=repair_context
    run.check_plan(ctx,ctx['report']/'EXPERIMENT_PLAN.json')
    proof=dict(commit='a'*40)
    common.sealed_json(ctx['cache']/'PRECEDING_DATASET_PUBLICATION.json',proof)
    monkeypatch.setattr(run.time,'time',lambda:100.)
    run.start_execution(ctx,proof)
    terminal=ctx['cache']/'TERMINAL_CORE_001.json'
    common.sealed_json(terminal,dict(status='FAILED',exit_code=1,started_unix=98.,finished_unix=116.))
    revised=copy.deepcopy(previous)
    changed=['research/'+common.NAME+'/'+name for name in ('run.py','models.py')]
    for relative in changed:
        (ctx['repo']/relative).write_text('legacy parent schema repair '+relative,encoding='utf8')
        revised['source_files'][relative]=common.sha(ctx['repo']/relative)
    revised.update(registered_unix=120.,supersedes=common.binding(
        ctx['report']/'EXPERIMENT_PLAN_REPAIR_001.json',ctx),repair=dict(
        status='REGISTERED_BEFORE_NEW_BODY_SEARCH',scientific_contract_changed=False,
        new_body_searches_or_V_scores=0,RawContext_GPU_sessions_started=2,
        changed_source_paths=changed,correction='LEGACY_PARENT_LOADING_SCHEMA_ONLY',
        preserve_execution_start_and_deadline=True,
        original_execution_start_binding=common.binding(ctx['cache']/'CORE_EXECUTION_START.json',ctx),
        failed_attempt_terminal=common.binding(terminal,ctx),GPU_profile_output_id='GPU_REPAIR_002'))
    common.sealed_json(ctx['report']/'EXPERIMENT_PLAN_REPAIR_002.json',revised)
    return ctx,original,previous,revised,proof


def test_second_repair_keeps_original_and_first_repair_bytes_and_redirects_old_cli(execution_repair_context):
    ctx,_,_,revised,_=execution_repair_context
    paths=[ctx['report']/name for name in ('EXPERIMENT_PLAN.json','EXPERIMENT_PLAN_REPAIR_001.json')]
    before=[path.read_bytes() for path in paths]
    for path in paths:
        assert run.check_plan(ctx,path)==revised
        assert ctx['plan_path']==ctx['report']/'EXPERIMENT_PLAN_REPAIR_002.json'
    assert [path.read_bytes() for path in paths]==before


@pytest.mark.parametrize('field,value',[
    ('limits',dict(wall_seconds=22000,GPU_reservation_seconds=14400)),
    ('seeds',[92601]),('primary_metric','MAE'),('recipe',dict(microbatch=8))])
def test_second_repair_rejects_scientific_or_budget_changes(execution_repair_context,field,value):
    ctx,_,_,revised,_=execution_repair_context
    common.atomic_json(run.registered_plan_path(ctx),dict(revised,**{field:value}))
    with pytest.raises(ValueError,match='scientific contract'):
        run.load_registered_plan(ctx)


@pytest.mark.parametrize('name',['EXPERIMENT_PLAN.json','EXPERIMENT_PLAN_REPAIR_001.json'])
def test_second_repair_requires_intact_registration_chain(execution_repair_context,name):
    ctx,*_=execution_repair_context
    path=ctx['report']/name
    path.write_text(path.read_text(encoding='utf8')+' ',encoding='utf8')
    with pytest.raises(ValueError,match='scientific contract'):
        run.load_registered_plan(ctx)


def test_second_repair_requires_first_repair_file(execution_repair_context):
    ctx,*_=execution_repair_context
    (ctx['report']/'EXPERIMENT_PLAN_REPAIR_001.json').unlink()
    with pytest.raises(ValueError,match='first repair'):
        run.load_registered_plan(ctx)


def test_second_repair_declares_exact_changed_sources_and_current_models(execution_repair_context):
    ctx,_,_,revised,_=execution_repair_context
    revised['repair']['changed_source_paths'].remove('research/'+common.NAME+'/models.py')
    common.atomic_json(run.registered_plan_path(ctx),revised)
    with pytest.raises(ValueError,match='execution repair'):
        run.load_registered_plan(ctx)


def test_second_repair_rejects_reporting_edit(execution_repair_context):
    ctx,_,_,revised,_=execution_repair_context
    relative='research/'+common.NAME+'/report.py'
    (ctx['repo']/relative).write_text('unrelated reporting edit',encoding='utf8')
    revised['source_files'][relative]=common.sha(ctx['repo']/relative)
    revised['repair']['changed_source_paths'].append(relative)
    common.atomic_json(run.registered_plan_path(ctx),revised)
    with pytest.raises(ValueError,match='execution repair'):
        run.load_registered_plan(ctx)


def test_second_repair_admits_registered_parent_deployment_loading_fix(execution_repair_context):
    ctx,_,_,revised,_=execution_repair_context
    relative='research/'+common.NAME+'/deployment.py'
    (ctx['repo']/relative).write_text('copy missing Raw foundation pin from inherited constant',encoding='utf8')
    revised['source_files'][relative]=common.sha(ctx['repo']/relative)
    revised['repair']['changed_source_paths'].append(relative)
    common.atomic_json(run.registered_plan_path(ctx),revised)
    assert run.check_plan(ctx,ctx['report']/'EXPERIMENT_PLAN.json')==revised


def test_second_repair_rejects_model_edit_after_registration(execution_repair_context):
    ctx,*_=execution_repair_context
    (ctx['repo']/'research'/common.NAME/'models.py').write_text('later unregistered model edit',encoding='utf8')
    with pytest.raises(ValueError,match='register a repair'):
        run.check_plan(ctx,ctx['report']/'EXPERIMENT_PLAN.json')


@pytest.mark.parametrize('field,value',[
    ('preserve_execution_start_and_deadline',False),('correction','OTHER'),
    ('GPU_profile_output_id','GPU'),('new_body_searches_or_V_scores',1)])
def test_second_repair_requires_narrow_correction_metadata(execution_repair_context,field,value):
    ctx,_,_,revised,_=execution_repair_context
    revised['repair'][field]=value
    common.atomic_json(run.registered_plan_path(ctx),revised)
    with pytest.raises(ValueError,match='scientific contract|execution repair'):
        run.load_registered_plan(ctx)


@pytest.mark.parametrize('name',['CORE_EXECUTION_START.json','TERMINAL_CORE_001.json'])
def test_second_repair_rejects_changed_execution_evidence(execution_repair_context,name):
    ctx,*_=execution_repair_context
    path=ctx['cache']/name
    path.write_text(path.read_text(encoding='utf8')+' ',encoding='utf8')
    with pytest.raises(ValueError,match='execution repair'):
        run.load_registered_plan(ctx)


@pytest.mark.parametrize('status,code',[('COMPLETE',0),('FAILED',0),('RUNNING',1)])
def test_second_repair_requires_actual_failed_terminal(execution_repair_context,status,code):
    ctx,_,_,revised,_=execution_repair_context
    path=ctx['cache']/'TERMINAL_CORE_001.json'
    common.atomic_json(path,dict(status=status,exit_code=code))
    revised['repair']['failed_attempt_terminal']=common.binding(path,ctx)
    common.atomic_json(run.registered_plan_path(ctx),revised)
    with pytest.raises(ValueError,match='execution repair'):
        run.load_registered_plan(ctx)


def test_second_repair_resume_preserves_start_and_deadline_and_records_separate_admission(execution_repair_context,monkeypatch):
    ctx,_,_,revised,proof=execution_repair_context
    paths=[ctx['cache']/'CORE_EXECUTION_START.json',ctx['report']/'EXECUTION_START.json']
    before=[path.read_bytes() for path in paths]
    run.check_plan(ctx,ctx['report']/'EXPERIMENT_PLAN_REPAIR_001.json')
    monkeypatch.setattr(run.time,'time',lambda:200.)
    first=run.start_execution(ctx,proof)
    assert first['started_unix']==100. and ctx['deadline']==21700.
    assert [path.read_bytes() for path in paths]==before
    admission=ctx['report']/'EXECUTION_REPAIR_ADMISSION_002.json'
    value=common.read(admission)
    assert value['active_plan_binding']==common.binding(run.registered_plan_path(ctx),ctx)
    assert value['original_execution_start_binding']==revised['repair']['original_execution_start_binding']
    assert value['failed_attempt_terminal']==revised['repair']['failed_attempt_terminal']
    assert value['deadline_unix']==21700.
    preserved=admission.read_bytes()
    monkeypatch.setattr(run.time,'time',lambda:300.)
    assert run.start_execution(ctx,proof)==first
    assert admission.read_bytes()==preserved and ctx['deadline']==21700.
    monkeypatch.setattr(run.time,'time',lambda:21701.)
    with pytest.raises(TimeoutError,match='original core execution'):
        run.start_execution(ctx,proof)


def test_second_repair_start_rejects_changed_publication(execution_repair_context,monkeypatch):
    ctx,_,_,_,proof=execution_repair_context
    run.check_plan(ctx,ctx['report']/'EXPERIMENT_PLAN_REPAIR_001.json')
    path=ctx['cache']/'PRECEDING_DATASET_PUBLICATION.json'
    common.atomic_json(path,dict(proof,commit='b'*40))
    monkeypatch.setattr(run.time,'time',lambda:200.)
    with pytest.raises(ValueError,match='publication identity'):
        run.start_execution(ctx,dict(proof,commit='b'*40))
