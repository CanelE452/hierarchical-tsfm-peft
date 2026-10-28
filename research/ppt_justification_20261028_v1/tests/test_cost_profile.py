"""Mock-only cost-core integration: no CUDA access, foundation load, or training."""
import math
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.memory_budget_20261018_v1 import worker as core
from research.ppt_justification_20261028_v1 import cost_profile as cost, runtime


class Sampler:
    driver='CPU_MOCK'
    latest_gpu_mib=None

    def __init__(self,*args,**kwargs):
        pass

    def start(self):
        pass

    def set_phase(self,phase):
        pass

    def stop(self):
        return dict(process_GPU_sampled_peak_MiB=None,process_GPU_sampling_status='CPU_MOCK',
                    process_GPU_sampling_reason='Synthetic test; no GPU queried',CPU_RSS_sampled_peak_MiB=0.)


class Route:
    def __init__(self,inputs,mutate=False,drift=False):
        candidate=inputs['candidate']
        self.family,self.latent_k=candidate['family'],candidate['K']
        self.device='cpu'
        self.spec=dict(row_microbatch=128,F_row_microbatch=128)
        self.mean,self.std=inputs['saved']['mean'],inputs['saved']['std']
        self.artifact_sha256='c'*64 if self.family=='B2' else None
        self.model=torch.nn.Linear(1,1,bias=False).eval() if self.family=='B2' else None
        frozen=torch.nn.Linear(1,1,bias=False).eval().requires_grad_(False)
        self.pool=SimpleNamespace(instances_created=1,load_seconds=0.,
                                  foundation=SimpleNamespace(initial_hash='f'*64,checkpoint_inputs=False,model=frozen))
        self.last_receipt={}
        self.calls=[]
        self.closed=False
        self.mutate,self.drift=mutate,drift

    def predict_physical(self,physical,observed):
        batch=len(physical)
        streams=batch*self.latent_k
        cap=self.spec['F_row_microbatch']
        self.last_receipt=dict(F_logical_streams=streams,F_physical_streams=streams,
                               F_calls=math.ceil(streams/cap),whole_seconds=.001)
        self.calls.append((batch,streams,self.last_receipt['F_calls']))
        if self.mutate and len(self.calls)==1:
            with torch.no_grad():
                self.pool.foundation.model.weight.add_(1.)
        return np.full((batch,48,26),2. if self.drift else .25,np.float64)

    def memory_state(self):
        return dict(resident_weights_gpu_MiB=0.,resident_weights_cpu_MiB=.001,active_adapter_MiB=0.)

    def close(self):
        self.closed=True


def inputs(family='B2',mutate=False,drift=False):
    task=dict(L=512,H=48,columns=[f'c{i}' for i in range(26)],
              physical=np.zeros((15788,26),np.float32),observed=np.ones((15788,26),bool),
              split_ranges=dict(V_select=[13814,15788]))
    saved=dict(pred_normalized=np.full((41,48,26),.25,np.float32),
               target_normalized=np.zeros((41,48,26),np.float32),observed=np.ones((41,48,26),bool),
               origins=np.arange(13814,15741,48,dtype=np.int64),channel_ids=np.asarray(task['columns']),
               mean=np.zeros(26,np.float32),std=np.ones(26,np.float32))
    candidate=dict(run_id='Raw_appliances' if family=='RAW' else 'KTAU_main_appliances_K10_s92601',
                   family=family,K=26 if family=='RAW' else 10,C=26,L=512,H=48,seed=92601,
                   dataset='appliances',MSE=.0625,MAE=.25,prediction_binding={'sha256':'a'*64})
    if family=='B2':
        candidate['bundle_binding']={'sha256':'c'*64}
    value=dict(task=task,saved=saved,candidate=candidate,expected_f_sha='f'*64,expected_f_params=1)
    routes=[]
    def loader():
        route=Route(value,mutate,drift)
        routes.append(route)
        return route
    value['loader']=loader
    return value,routes


@pytest.mark.parametrize('family,streams,calls,effective_cap',[('RAW',208,2,128),('B2',80,1,80)])
def test_real_measurement_core_contract_with_cpu_mock(family,streams,calls,effective_cap):
    value,routes=inputs(family)
    result,saved=cost.measure(value,'0'*64,device='cpu',sampler_factory=Sampler)
    assert len(routes)==1 and routes[0].closed
    assert len(routes[0].calls)==6+10+20
    assert [r['phase'] for r in result['trials']]==['warmup']*10+['timed']*20
    assert result['parity']['status']=='PASS' and result['parity']['full_V_origins']==41
    assert result['parity']['last_batch_size']==1
    assert result['summary']['requested_row_microbatch']==128
    assert result['summary']['effective_row_microbatch']==effective_cap
    assert all(r['F_logical_streams']==streams and r['F_calls']==calls for r in result['trials'])
    assert result['states_before']==result['states_after']
    assert result['summary']['model_load_with_state_hash_seconds']>=result['summary']['model_load_seconds']>=0
    assert result['training_executed'] is False
    assert saved['pred_normalized'].shape==(41,48,26)


def test_mutated_foundation_state_is_rejected():
    value,routes=inputs(mutate=True)
    with pytest.raises(ValueError,match='tensor states changed'):
        cost.measure(value,'0'*64,device='cpu',sampler_factory=Sampler)
    assert routes[0].closed


def test_full_V_prediction_drift_is_rejected_before_timing():
    value,routes=inputs(drift=True)
    with pytest.raises(ValueError,match='Full V chunk prediction parity failed'):
        cost.measure(value,'0'*64,device='cpu',sampler_factory=Sampler)
    assert routes[0].closed and len(routes[0].calls)==6


def test_reference_validation_uses_full_cohort_and_macro_fp64():
    value,_=inputs()
    row=value['candidate']
    assert cost.verify_reference(row,value['saved'])['aggregation']=='equal-channel observed macro'
    value['saved']['origins'][0]+=1
    with pytest.raises(ValueError,match='origin grid'):
        cost.verify_reference(row,value['saved'])


def test_reference_target_or_scaler_mismatch_is_rejected():
    value,_=inputs()
    other={k:v.copy() for k,v in value['saved'].items()}
    other['std'][0]=2
    with pytest.raises(ValueError,match='cohort/scaler'):
        cost.verify_reference(value['candidate'],value['saved'],other)


@pytest.mark.parametrize('state',[None,'RUNNING','STOPPED','READY'])
def test_measurement_is_blocked_until_training_terminal(tmp_path,monkeypatch,state):
    monkeypatch.setattr(runtime,'REPORT',tmp_path)
    if state is not None:
        (tmp_path/'EXECUTION_COMPLETION.json').write_text('{"status":"'+state+'"}')
    with pytest.raises(RuntimeError,match='forbidden until'):
        cost.ensure_training_finished()


def test_training_terminal_is_required_without_loading_gpu(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'REPORT',tmp_path)
    (tmp_path/'EXECUTION_COMPLETION.json').write_text('{"status":"TRAINING_COMPLETE"}')
    cost.ensure_training_finished()


def test_measure_never_loads_foundation_or_creates_optimizer(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('GPU/train call is forbidden in CPU/mock validation')
    monkeypatch.setattr(runtime,'load_foundation',forbidden)
    monkeypatch.setattr(torch.optim,'AdamW',forbidden)
    monkeypatch.setattr(torch.cuda,'synchronize',forbidden)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',forbidden)
    value,_=inputs()
    result,_=cost.measure(value,'0'*64,device='cpu',sampler_factory=Sampler)
    assert result['status']=='COMPLETE'


@pytest.fixture
def lease_context(tmp_path,monkeypatch):
    public=tmp_path/'reports'
    cache=tmp_path/'cache'
    public.mkdir()
    cache.mkdir()
    monkeypatch.setattr(cost,'COST_REPORT',public)
    monkeypatch.setattr(cost,'COST_CACHE',cache)
    monkeypatch.setattr(runtime,'CACHE',tmp_path/'main_cache')
    monkeypatch.setattr(runtime,'REPORT',tmp_path/'main_reports')
    (public/'COST_SEAL.json').write_text('{"revision":2}')
    gate=dict(status='PASS',allow_gpu=True,registration_seal_sha256=runtime.sha(public/'COST_SEAL.json'))
    (public/'ROOT_LAUNCH_GATE.json').write_text(json.dumps(gate))
    module='research.ppt_justification_20261028_v1.cost_profile'
    run_id=cost.FACETS[0]
    command=['python.exe','-B','-m',module,'--worker',run_id,'--lease-token','TOKEN']
    records={100:dict(pid=100,ppid=99,created=10.,command=['python.exe','-B','-m',module,'--run'],exe='python.exe'),
             200:dict(pid=200,ppid=100,created=11.,command=command.copy(),exe='python.exe'),
             300:dict(pid=300,ppid=200,created=12.,command=command.copy(),exe='python.exe')}
    parent=records[100]
    lease=dict(status='ACTIVE',pid=100,created=10.,command=parent['command'],token='TOKEN',active_facet=run_id,
               registration_seal_sha256=gate['registration_seal_sha256'],root_gate_sha256=runtime.sha(public/'ROOT_LAUNCH_GATE.json'))
    (cache/'GPU_LEASE.json').write_text(json.dumps(lease))
    monkeypatch.setattr(cost.os,'getpid',lambda:300)
    monkeypatch.setattr(cost,'process_snapshot',lambda:records)
    return records,run_id


def test_windows_python_launcher_ancestry_is_accepted(lease_context):
    records,run_id=lease_context
    lease,allowed=cost.validate_worker_lease(run_id,'TOKEN')
    assert lease['pid']==100 and allowed=={100,200,300}


@pytest.mark.parametrize('fault',['dead_parent','pid_reuse','wrong_parent_command','unrelated_parent','wrong_wrapper_token','wrong_facet'])
def test_worker_rejects_absent_or_mismatched_live_parent(lease_context,fault):
    records,run_id=lease_context
    if fault=='dead_parent':
        del records[100]
    elif fault=='pid_reuse':
        records[100]['created']=99.
    elif fault=='wrong_parent_command':
        records[100]['command']=records[100]['command']+['--different']
    elif fault=='unrelated_parent':
        records[200]['ppid']=999
    elif fault=='wrong_wrapper_token':
        records[200]['command'][-1]='WRONG'
    else:
        run_id=cost.FACETS[1]
    with pytest.raises(RuntimeError,match='parent|ancestry'):
        cost.validate_worker_lease(run_id,'TOKEN')


def test_direct_worker_has_no_gpu_side_effects_without_parent_token(monkeypatch):
    monkeypatch.setattr(cost,'make_inputs',lambda *a:pytest.fail('Inputs/GPU must never be loaded'))
    monkeypatch.setattr(cost,'process_snapshot',lambda:pytest.fail('No-token child rejects immediately'))
    with pytest.raises(RuntimeError,match='Direct cost worker'):
        cost.run_worker(cost.FACETS[0])


def test_root_gate_must_match_seal(lease_context):
    _,run_id=lease_context
    (cost.COST_REPORT/'COST_SEAL.json').write_text('{"revision":3}')
    with pytest.raises(RuntimeError,match='launch gate'):
        cost.validate_worker_lease(run_id,'TOKEN')


@pytest.mark.parametrize('module',['research.ppt_justification_20261028_v1.controller',
                                   'research.ppt_justification_20261028_v1.run_one',
                                   'research.ppt_e_fixed_20261028_v1.controller',
                                   'research.ppt_e_fixed_20261028_v1.run_one'])
def test_stale_terminal_file_cannot_override_live_training_process(lease_context,module,monkeypatch):
    records,_=lease_context
    runtime.REPORT.mkdir()
    (runtime.REPORT/'EXECUTION_COMPLETION.json').write_text('{"status":"TRAINING_COMPLETE"}')
    cost.ensure_training_finished()
    records[400]=dict(pid=400,ppid=1,created=13.,command=['python.exe','-m',module],exe='python.exe')
    monkeypatch.setattr(cost.subprocess,'run',lambda *a,**k:pytest.fail('Live process rejected before GPU query'))
    with pytest.raises(RuntimeError,match='Live main/E/cost'):
        cost.assert_no_gpu_conflicts({100,200,300})


def test_external_gpu_compute_process_blocks_cost(lease_context,monkeypatch):
    def reject(*args,**kwargs):
        raise RuntimeError('Unknown GPU context PID: 9876')
    monkeypatch.setattr(cost.host_gpu_gate,'admit_identity',reject)
    with pytest.raises(RuntimeError,match='Unknown GPU context'):
        cost.assert_no_gpu_conflicts({100,200,300})


def test_idle_gpu_snapshot_passes_without_cuda_calls(lease_context,monkeypatch):
    seen=[]
    def admit(path,**kwargs):
        seen.append(kwargs['allowed_processes'])
        return dict(status='PASS',desktop_concurrent_load=True)
    monkeypatch.setattr(cost.host_gpu_gate,'admit_identity',admit)
    monkeypatch.setattr(torch.cuda,'init',lambda:pytest.fail('CPU preflight must not initialize CUDA'))
    cost.assert_no_gpu_conflicts({100,200,300})
    assert seen==[{100:10.,200:11.,300:12.}]


def test_missing_parent_shared_reservation_rejects_worker(lease_context):
    with pytest.raises(RuntimeError,match='no longer holds'):
        cost.verify_parent_reservation()


def test_duplicate_cost_worker_is_blocked_by_actual_child_lock(lease_context,monkeypatch):
    _,run_id=lease_context
    monkeypatch.setattr(cost,'make_inputs',lambda *a:pytest.fail('Blocked worker cannot load GPU'))
    with runtime.CampaignLock(runtime.CACHE/'controller.lock'), runtime.CampaignLock(cost.COST_CACHE/'worker.lock'):
        with pytest.raises(RuntimeError,match='Another live process holds worker.lock'):
            cost.run_worker(run_id,'TOKEN')


def test_actual_child_retains_shared_lock_on_parent_loss(lease_context,monkeypatch):
    _,run_id=lease_context
    initial=len(cost.WORKER_RESERVATIONS)
    monkeypatch.setattr(cost,'ensure_training_finished',lambda:(_ for _ in ()).throw(RuntimeError('parent died after locks acquired')))
    try:
        with runtime.CampaignLock(runtime.CACHE/'controller.lock'):
            with pytest.raises(RuntimeError,match='parent died'):
                cost.run_worker(run_id,'TOKEN')
        with pytest.raises(RuntimeError,match='Another live process holds worker.lock'):
            with runtime.CampaignLock(runtime.CACHE/'worker.lock'):
                pytest.fail('Orphan child still owns this GPU reservation')
    finally:
        for lock in cost.WORKER_RESERVATIONS[initial:]:
            lock.__exit__(None,None,None)
        del cost.WORKER_RESERVATIONS[initial:]


def test_host_contamination_preserves_results_and_forbids_silent_repeat(lease_context,monkeypatch):
    _,run_id=lease_context
    monkeypatch.setattr(runtime,'REPORT',cost.COST_REPORT.parent)
    (cost.COST_REPORT/'COST_PROTOCOL.json').write_text('{}')
    monkeypatch.setattr(cost,'ensure_training_finished',lambda:None)
    monkeypatch.setattr(cost,'assert_no_gpu_conflicts',lambda *a,**k:dict(status='PASS'))
    monkeypatch.setattr(cost,'validate_seal',lambda:({},[dict(run_id=run_id)]))
    monkeypatch.setattr(cost,'make_inputs',lambda *a:None)
    monkeypatch.setattr(cost,'measure',lambda *a,**k:(dict(status='COMPLETE',summary=dict(status='COMPLETE')),dict(synthetic=np.zeros(1))))
    class Observer:
        def __init__(self,*a,**k): pass
        def start(self): return self
        def wait_preflight(self): return dict(status='PASS')
        def finish(self,*a,**k): return dict(status='CONTAMINATED',errors=['background_load_limit_exceeded'])
    monkeypatch.setattr(cost.host_gpu_gate,'HostObservation',Observer)
    monkeypatch.setattr(torch.cuda,'init',lambda:pytest.fail('No GPU access in synthetic test'))
    initial=len(cost.WORKER_RESERVATIONS)
    try:
        with runtime.CampaignLock(runtime.CACHE/'controller.lock'):
            with pytest.raises(RuntimeError,match='host condition was contaminated'):
                cost.run_worker(run_id,'TOKEN')
        receipt=runtime.read(cost.COST_REPORT/(run_id+'_MEASUREMENT.json'))
        assert receipt['status']==receipt['summary']['status']=='CONTAMINATED'
        assert (cost.COST_CACHE/(run_id+'_predictions.npz')).is_file()
        assert (cost.COST_REPORT/(run_id+'_HOST_OBSERVATION.json')).is_file()
    finally:
        for lock in cost.WORKER_RESERVATIONS[initial:]: lock.__exit__(None,None,None)
        del cost.WORKER_RESERVATIONS[initial:]
    try:
        with runtime.CampaignLock(runtime.CACHE/'controller.lock'):
            with pytest.raises(FileExistsError,match='never overwrite'):
                cost.run_worker(run_id,'TOKEN')
    finally:
        for lock in cost.WORKER_RESERVATIONS[initial:]: lock.__exit__(None,None,None)
        del cost.WORKER_RESERVATIONS[initial:]
