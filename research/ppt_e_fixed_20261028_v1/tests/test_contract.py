"""Synthetic CPU-only study and shared-GPU gate tests."""
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.ppt_e_fixed_20261028_v1 import controller, models, prepare, run_one, runtime

BROAD = runtime.REPO / 'research/informs_broad_validation_20261015_v1'
sys.path.insert(0, str(BROAD))
import fit_base
import networks
import training

torch.set_num_threads(2)


class CPUFoundation:
    initial_hash = 'SYNTHETIC_CPU_ONLY'
    device = 'cpu'
    streams = row_calls = physical_streams = physical_calls = 0

    def __call__(self, z):
        self.streams += z.shape[0] * z.shape[-1]
        return z.mean(1, keepdim=True).expand(-1, 48, -1)

    def assert_frozen_unchanged(self):
        pass


def task():
    rng = np.random.default_rng(4107)
    values = rng.normal(size=(760, 21)).astype(np.float32)
    values[576:, 0] += 3 * values[576:, 1]
    return dict(physical=values, observed=np.ones(values.shape, bool), L=512, H=48,
                dataset='synthetic', columns=[f'c{i}' for i in range(21)])


def spec():
    return dict(run_id='synthetic-E-fixed', method_id='E_FIXED', family='B2', latent_k=6,
                seed=92601, device='cpu', lr=.001, microbatch=4, effective_batch=16)


def contract():
    source, _ = fit_base._source_contract()
    return dict(source_sha=source, config_sha='synthetic', data_sha='synthetic', model_sha='SYNTHETIC_CPU_ONLY')


def stage(tmp_path, *, fit_end=576, stop=None, epochs=(.25, 1.), suffix='one'):
    return fit_base._stage(spec(), task(), fit_end, stop, list(epochs),
                           tmp_path/suffix/'cache', tmp_path/suffix/'public', CPUFoundation(), time.time()+30, contract())


def test_factory_changes_only_E_trainability_and_keeps_all_tensors_and_rng():
    data, foundation = task(), CPUFoundation()
    original, norm = fit_base.create_model(spec(), data, 576, foundation)
    state, rng = copy.deepcopy(original.state_dict()), torch.get_rng_state().clone()
    flags = {name: parameter.requires_grad for name, parameter in original.named_parameters()}
    with models.patch_core(fit_base, training):
        fixed, fixed_norm = fit_base.create_model(spec(), data, 576, foundation)
    assert set(fixed.state_dict()) == set(state)
    assert all(torch.equal(value, fixed.state_dict()[name]) for name, value in state.items())
    assert torch.equal(torch.get_rng_state(), rng)
    assert np.array_equal(norm['basis'], fixed_norm['basis'])
    assert {name for name, value in fixed.named_parameters() if value.requires_grad != flags[name]} == {'E'}
    assert sum(p.numel() for p in fixed.parameters()) == 35717
    assert sum(p.numel() for p in fixed.parameters() if p.requires_grad) == 35591


def test_actual_core_stage_updates_P_and_D_but_keeps_E_and_excludes_optimizer(tmp_path):
    original, _ = fit_base.create_model(spec(), task(), 576, CPUFoundation())
    initial = copy.deepcopy(original.state_dict())
    with models.patch_core(fit_base, training) as audits:
        result = stage(tmp_path, epochs=(.25, 1., 3.))
    assert result['summary']['updates'] > 0
    assert torch.equal(initial['E'], result['model']['E'])
    assert not torch.equal(initial['D'], result['model']['D'])
    assert any(not torch.equal(value, result['model'][name]) for name, value in initial.items() if name.startswith('front.'))
    assert len(audits) == 1 and audits[0]['status'] == 'PASS'
    assert audits[0]['optimizer_parameter_group_checks'] == 3
    assert audits[0]['initial_E_sha256'] == audits[0]['live_E_sha256'] == audits[0]['selected_E_sha256']
    assert audits[0]['frozen_parameters'] == 126 and audits[0]['trainable_parameters'] == 35591


def test_search_and_A60_are_fresh_separately_fitted_PCA_stages(tmp_path):
    with models.patch_core(fit_base, training) as audits:
        search = stage(tmp_path, fit_end=576, stop=(576, 672), suffix='search')
        refit = stage(tmp_path, fit_end=672, epochs=(.25,), suffix='refit')
    assert [a['phase'] for a in audits] == ['search', 'refit']
    assert [a['fit_end'] for a in audits] == [576, 672]
    assert not torch.equal(search['model']['E'], refit['model']['E'])
    assert all(a['initial_E_sha256'] == a['selected_E_sha256'] for a in audits)


def test_zero_update_refit_is_valid_and_still_preserves_E(tmp_path):
    with models.patch_core(fit_base, training) as audits:
        result = stage(tmp_path, epochs=())
    assert result['summary']['updates'] == 0
    assert audits[0]['optimizer_parameter_group_checks'] == 0
    assert audits[0]['E_requires_grad'] is False and audits[0]['E_gradient_absent'] is True


def test_frozen_E_mutation_is_a_scientific_failure(tmp_path, monkeypatch):
    def corrupted(*args):
        model = args[-2]
        with torch.no_grad():
            model.E.add_(1)
        return dict(model=training.cpu_state(model), summary={})
    monkeypatch.setattr(fit_base, '_stage_body', corrupted)
    with models.patch_core(fit_base, training):
        with pytest.raises(ValueError, match='E differs'):
            stage(tmp_path)


def test_optimizer_including_E_is_rejected_before_training(tmp_path, monkeypatch):
    def invalid_body(*args):
        model = args[-2]
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        return training.train_segment(model, None, None, None, optimizer, 1, None)
    monkeypatch.setattr(fit_base, '_stage_body', invalid_body)
    with models.patch_core(fit_base, training):
        with pytest.raises(ValueError, match='included in optimizer'):
            stage(tmp_path)


def test_patch_restores_hooks_after_failure():
    before = fit_base.create_model, fit_base._stage_body, training.train_segment
    with pytest.raises(RuntimeError):
        with models.patch_core(fit_base, training):
            raise RuntimeError('synthetic')
    assert (fit_base.create_model, fit_base._stage_body, training.train_segment) == before


@pytest.mark.parametrize('change', [dict(seed=92606),dict(K=5),dict(dataset='appliances'),
                                  dict(method_id='OURS'),dict(extended=1),dict(job_id='old')])
def test_unregistered_jobs_are_rejected(change):
    with pytest.raises(ValueError):
        run_one.validate_job(dict(prepare.jobs()[0], **change))


def test_exact_family_seeds_and_epoch_schedules():
    all_jobs = prepare.jobs()+prepare.jobs(True)
    assert len(all_jobs) == len({j['job_id'] for j in all_jobs}) == 10
    for job in all_jobs:
        run_one.validate_job(job)
        assert run_one.schedule(job) == ([.25,1.,3.,6.,10.] if job['extended'] else [.25,1.,3.])
        assert run_one.build_spec(job)['E_trainable'] is False


def test_full_study_admission_reserves_possible_extension(tmp_path, monkeypatch):
    instance = controller.Controller.__new__(controller.Controller)
    instance.jobs, instance.extensions = prepare.jobs(), prepare.jobs(True)
    instance.base_estimate, instance.extension_estimate, instance.margin = 1080., 1800., 120.
    instance.state = dict(run_statuses={})
    instance.save = lambda: None
    monkeypatch.setattr(runtime, 'REPORT', tmp_path)
    monkeypatch.setattr(controller.time, 'time', lambda: 1000.)
    instance.deadline = 15519.
    assert instance.reserve_campaign() is False
    assert instance.state['campaign_admission']['reserved_seconds'] == 14520
    instance.deadline = 15520.
    assert instance.reserve_campaign() is True


def test_extension_is_based_on_three_caps_and_not_V_values(tmp_path, monkeypatch):
    instance = controller.Controller.__new__(controller.Controller)
    instance.jobs, instance.extensions = prepare.jobs(), prepare.jobs(True)
    instance.base_estimate, instance.extension_estimate, instance.margin = 1080., 1800., 120.
    instance.state = dict(run_statuses={j['job_id']: 'COMPLETE' for j in instance.jobs})
    instance.deadline = 10000.
    selected = {j['job_id']: 3. if i < 3 else 1. for i,j in enumerate(instance.jobs)}
    monkeypatch.setattr(runtime, 'REPORT', tmp_path)
    monkeypatch.setattr(runtime, 'read', lambda path: {'row': {'selected_epoch':selected[path.parent.name], 'MSE':float('nan')}})
    monkeypatch.setattr(controller.time, 'time', lambda: 0.)
    assert instance.extension_decision('E_FIXED:jena')['status'] == 'ADMITTED'
    selected[instance.jobs[2]['job_id']] = 1.
    assert instance.extension_decision('E_FIXED:jena')['status'] == 'NOT_REQUIRED'


def setup_lease(tmp_path, monkeypatch):
    report, cache, upstream = tmp_path/'report', tmp_path/'cache', tmp_path/'upstream'
    for path in (report, cache, upstream): path.mkdir()
    monkeypatch.setattr(runtime, 'REPORT', report)
    monkeypatch.setattr(runtime, 'CACHE', cache)
    monkeypatch.setattr(runtime, 'UPSTREAM_REPORT', upstream)
    (report/'REGISTRATION_SEAL.json').write_text('{}',encoding='utf8')
    seal_sha = runtime.sha(report/'REGISTRATION_SEAL.json')
    (report/'ROOT_LAUNCH_GATE.json').write_text(json.dumps(dict(status='PASS',allow_gpu=True,registration_seal_sha256=seal_sha)))
    (upstream/'EXECUTION_COMPLETION.json').write_text(json.dumps(dict(status='TRAINING_COMPLETE')))
    lease = dict(status='ACTIVE',token='token-fixed',pid=100,created_utc='same-time',registration_seal_sha256=seal_sha)
    (cache/'GPU_LEASE.json').write_text(json.dumps(lease))
    line = f'python -m research.{runtime.NAME}.run_one --job fixture --lease-token token-fixed'
    snapshot = {100:dict(pid=100,ppid=99,created_utc='same-time',command_line=f'python -m research.{runtime.NAME}.controller'),
                200:dict(pid=200,ppid=100,created_utc='second',command_line=line),
                300:dict(pid=300,ppid=200,created_utc='third',command_line=line)}
    monkeypatch.setattr(runtime.os, 'getpid', lambda: 300)
    monkeypatch.setattr(runtime, 'process_snapshot', lambda: snapshot)
    return snapshot


def test_windows_venv_launcher_ancestry_is_accepted(tmp_path, monkeypatch):
    setup_lease(tmp_path, monkeypatch)
    assert runtime.require_active_lease('token-fixed')['pid'] == 100


def test_controller_launcher_is_bound_only_when_in_owner_ancestry_with_same_args(tmp_path,monkeypatch):
    snapshot=setup_lease(tmp_path,monkeypatch)
    snapshot[100]['ppid']=50
    snapshot[50]=dict(pid=50,ppid=1,created_utc='controller-wrapper',command_line=snapshot[100]['command_line'])
    snapshot[900]=dict(pid=900,ppid=1,created_utc='unrelated-controller',command_line=snapshot[100]['command_line'])
    identities=runtime.require_active_lease('token-fixed')['verified_ancestry']
    assert [r['pid'] for r in identities]==[300,200,100,50]
    snapshot[50]['command_line']+=' --unregistered-option'
    assert [r['pid'] for r in runtime.require_active_lease('token-fixed')['verified_ancestry']]==[300,200,100]


@pytest.mark.parametrize('mutation', ['token','pid_reused','intermediate','unrelated_parent','upstream_live'])
def test_direct_or_stale_or_conflicting_worker_is_rejected(tmp_path, monkeypatch, mutation):
    snapshot = setup_lease(tmp_path, monkeypatch)
    token = 'token-fixed'
    if mutation == 'token': token = 'wrong'
    if mutation == 'pid_reused': snapshot[100]['created_utc'] = 'new-process'
    if mutation == 'intermediate': snapshot[200]['command_line'] = 'python unrelated.py'
    if mutation == 'unrelated_parent': snapshot[200]['ppid'] = 777
    if mutation == 'upstream_live':
        snapshot[400] = dict(pid=400,ppid=1,created_utc='now',command_line=f'python -m research.{runtime.UPSTREAM_NAME}.run_one')
    with pytest.raises((ValueError, RuntimeError)):
        runtime.require_active_lease(token)


def test_missing_root_gate_prevents_execution(tmp_path, monkeypatch):
    setup_lease(tmp_path, monkeypatch)
    (runtime.REPORT/'ROOT_LAUNCH_GATE.json').unlink()
    with pytest.raises(FileNotFoundError): runtime.require_active_lease('token-fixed')


def mock_host_helper(monkeypatch):
    binding={'mode':'identity_only','synthetic':True}
    lease={'verified_ancestry':[dict(pid=300,created_utc='child'),dict(pid=100,created_utc='parent')]}
    calls=[]
    def creation_times(pids):
        calls.append(('identities',pids));return {300:3.,100:1.}
    def admit(contract,*,allowed_processes):
        calls.append(('admit',str(contract),allowed_processes))
        return dict(status='PASS',mode='identity_only',unexpected_gpu_pids=[],desktop_concurrent_load=True)
    helper=SimpleNamespace(__file__=str(runtime.REPO/runtime.HOST_GPU_HELPER_REL),creation_times=creation_times,admit_identity=admit)
    monkeypatch.setattr(runtime,'require_active_lease',lambda token:lease)
    monkeypatch.setattr(runtime,'verify_host_gpu_binding',lambda:binding)
    monkeypatch.setattr(runtime,'sha',lambda path:'synthetic-seal')
    monkeypatch.setitem(sys.modules,f'research.{runtime.UPSTREAM_NAME}.host_gpu_gate',helper)
    return helper,lease,calls,binding


def test_host_identity_gate_uses_only_verified_ancestry_and_preserves_receipt(monkeypatch):
    _,_,calls,binding=mock_host_helper(monkeypatch)
    receipt=runtime.require_idle_gpu('synthetic-token')
    assert calls[0]==('identities',[300,100])
    assert calls[1][2]=={300:3.,100:1.}
    assert receipt['host_gpu_binding']==binding and receipt['registration_seal_sha256']=='synthetic-seal'
    assert receipt['desktop_concurrent_load'] is True


@pytest.mark.parametrize('mutation',['new_context','failed_receipt','unexpected_pid','wrong_mode','wrong_module','ancestry_changed'])
def test_host_identity_gate_fails_closed_before_any_cuda(monkeypatch,mutation):
    helper,lease,calls,_=mock_host_helper(monkeypatch)
    if mutation=='new_context':
        def deny(*args,**kwargs): raise RuntimeError('Unknown Python/WSL/new context')
        helper.admit_identity=deny
    elif mutation=='wrong_module': helper.__file__='another_checkout/host_gpu_gate.py'
    elif mutation=='ancestry_changed':
        snapshots=iter([lease,{'verified_ancestry':[dict(pid=300,created_utc='new identity')]}])
        monkeypatch.setattr(runtime,'require_active_lease',lambda token:next(snapshots))
    else:
        receipt=dict(status='PASS',mode='identity_only',unexpected_gpu_pids=[])
        receipt[{'failed_receipt':'status','unexpected_pid':'unexpected_gpu_pids','wrong_mode':'mode'}[mutation]]={'failed_receipt':'FAILED','unexpected_pid':[99],'wrong_mode':'benchmark'}[mutation]
        helper.admit_identity=lambda *args,**kwargs:receipt
    with pytest.raises((ValueError,RuntimeError)): runtime.require_idle_gpu('synthetic-token')


@pytest.mark.parametrize('mutation',[None,'helper','contract','mode'])
def test_host_helper_and_contract_are_pinned_in_registration(tmp_path,monkeypatch,mutation):
    monkeypatch.setattr(runtime,'REPO',tmp_path)
    for name in (runtime.HOST_GPU_HELPER_REL,runtime.HOST_GPU_CONTRACT_REL):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic')
    binding=runtime.host_gpu_registration();protocol={'host_gpu_gate':binding}
    if mutation=='helper': (tmp_path/runtime.HOST_GPU_HELPER_REL).write_text('changed')
    elif mutation=='contract': (tmp_path/runtime.HOST_GPU_CONTRACT_REL).write_text('changed')
    elif mutation=='mode': binding['mode']='benchmark'
    if mutation:
        with pytest.raises(ValueError): runtime.verify_host_gpu_binding(protocol)
    else: assert runtime.verify_host_gpu_binding(protocol)==binding


def test_worker_reservations_remain_live_after_main_returns(tmp_path, monkeypatch):
    job_path=tmp_path/'job.json';job_path.write_text(json.dumps(prepare.jobs()[0]))
    monkeypatch.setattr(runtime,'REPORT',tmp_path/'report')
    monkeypatch.setattr(runtime,'CACHE',tmp_path/'cache')
    monkeypatch.setattr(runtime,'SHARED_CACHE',tmp_path/'shared')
    acquired=[];released=[]
    class Lock:
        def __init__(self,path): self.path=path
        def __enter__(self): acquired.append(self.path);return self
        def __exit__(self,*args): released.append(self.path)
    monkeypatch.setattr(runtime,'CampaignLock',Lock)
    for name in ('install_write_guard','require_idle_gpu','guard_preserved'):
        monkeypatch.setattr(runtime,name,lambda *args: None)
    monkeypatch.setattr(runtime,'require_active_lease',lambda token: None)
    monkeypatch.setattr(runtime,'atomic',lambda *args: None)
    monkeypatch.setattr(runtime,'write_new',lambda *args: None)
    monkeypatch.setattr(runtime,'assert_protected_T',lambda *args: None)
    monkeypatch.setattr(run_one,'run_job',lambda *args,**kwargs: {'status':'COMPLETE'})
    before=len(run_one.WORKER_RESERVATIONS)
    try:
        assert run_one.main(['--job',str(job_path),'--lease-token','synthetic']) == 0
        assert acquired == [runtime.SHARED_CACHE/'worker.lock',runtime.CACHE/'worker.lock']
        assert released == [] and len(run_one.WORKER_RESERVATIONS) == before+2
    finally:
        del run_one.WORKER_RESERVATIONS[before:]


def test_host_receipt_is_saved_after_new_attempt_creation_before_model_loading(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'REPORT',tmp_path/'reports');monkeypatch.setattr(runtime,'CACHE',tmp_path/'cache')
    monkeypatch.setattr(run_one,'validate_registration',lambda job:{'seal':'synthetic'})
    monkeypatch.setattr(runtime,'guard_preserved',lambda:{'changed':[]})
    monkeypatch.setattr(runtime,'record_opens',lambda:set())
    job=prepare.jobs()[0];public=runtime.REPORT/'jobs'/job['job_id']/'attempt_001'
    gate=dict(status='PASS',mode='identity_only')
    def stop_before_loading():
        assert runtime.read(public/'HOST_GPU_GATE.json')==gate
        raise RuntimeError('synthetic stop before any model or CUDA loading')
    monkeypatch.setattr(runtime,'load_modules',stop_before_loading)
    with pytest.raises(RuntimeError,match='synthetic stop'):
        run_one.run_job(job,host_gate=gate)
    digest=runtime.sha(public/'HOST_GPU_GATE.json')
    with pytest.raises(FileExistsError): run_one.run_job(job,host_gate=gate)
    assert runtime.sha(public/'HOST_GPU_GATE.json')==digest


def test_T_boundary_receipt_requires_zero_tail_parsing():
    access=dict(new_T_numeric_values_parsed=0,members=[dict(member=name,tail_numeric_values_parsed=0)
                for name in ('x.npy','finite.npy','times.npy','mean.npy','std.npy','columns.npy')])
    runtime.assert_protected_T(dict(job=prepare.jobs()[0],task_access=access))
    access['members'][0]['tail_numeric_values_parsed']=1
    with pytest.raises(ValueError): runtime.assert_protected_T(dict(job=prepare.jobs()[0],task_access=access))


@pytest.mark.parametrize('case',['original','amendment001','amendment002','missing001','missing002','extra','unknown','changed002'])
def test_registration_binds_exact_prospective_amendment_chain(tmp_path,monkeypatch,case):
    monkeypatch.setattr(runtime,'REPORT',tmp_path)
    monkeypatch.setattr(runtime,'guard_preserved',lambda:{'changed':[]})
    monkeypatch.setattr(runtime,'verify_host_gpu_binding',lambda protocol:{'mode':'identity_only'})
    monkeypatch.setattr(run_one,'source_hashes',lambda:{})
    job=prepare.jobs()[0]
    runtime.atomic(tmp_path/'RUN_PLAN.json',dict(jobs=[job]))
    runtime.atomic(tmp_path/'UNIT_TESTS.json',dict(status='PASS',synthetic=True))
    protocol=dict(registered_sources={},unit_test_receipt_sha256=runtime.sha(tmp_path/'UNIT_TESTS.json'))
    names={'PREREGISTRATION.json','PREREGISTRATION.md','RUN_PLAN.json','PRESERVATION_BASELINE.json','REFERENCE_RUNS.json'}
    if case=='amendment001':
        protocol['amendment']='AMENDMENT_001.json';names.add('AMENDMENT_001.json')
    elif case=='unknown':
        protocol['amendment']='AMENDMENT_UNKNOWN.json'
    elif case!='original':
        protocol['amendment']='AMENDMENT_002.json';names.update({'AMENDMENT_001.json','AMENDMENT_002.json'})
    if case=='missing001':names.remove('AMENDMENT_001.json')
    if case=='missing002':names.remove('AMENDMENT_002.json')
    if case=='extra':names.add('UNREGISTERED.json')
    runtime.atomic(tmp_path/'PREREGISTRATION.json',protocol)
    for name in names-{'PREREGISTRATION.json','RUN_PLAN.json'}:
        (tmp_path/name).write_text('synthetic-only',encoding='utf-8')
    runtime.atomic(tmp_path/'REGISTRATION_SEAL.json',dict(created_utc='2026-01-01T00:00:00+00:00',
        files={name:runtime.sha(tmp_path/name) for name in names}))
    if case=='changed002':(tmp_path/'AMENDMENT_002.json').write_text('unsealed-change',encoding='utf-8')
    if case in ('original','amendment001','amendment002'):
        assert set(run_one.validate_registration(job)['files'])==names
    else:
        with pytest.raises(ValueError):run_one.validate_registration(job)
