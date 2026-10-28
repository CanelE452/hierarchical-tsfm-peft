"""Frozen Appliances K10/Raw B8 inference costs; never trains a model."""
import argparse
from datetime import datetime, timezone
import csv
import importlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

import numpy as np
import psutil

from . import runtime, host_gpu_gate


COST_REPORT = runtime.REPORT / 'cost_profile'
COST_CACHE = runtime.CACHE / 'cost_profile'
FACETS = ('Raw_appliances', 'KTAU_main_appliances_K10_s92601', 'KTAU_main_appliances_K10_s92602')
WORKER_RESERVATIONS = []
SOURCE_FILES = [
    f'research/{runtime.NAME}/cost_profile.py', f'research/{runtime.NAME}/runtime.py',
    f'research/{runtime.NAME}/tests/test_cost_profile.py',
    f'research/{runtime.NAME}/host_gpu_gate.py', f'research/{runtime.NAME}/tests/test_host_gpu_gate.py',
    f'research/{runtime.NAME}/verify_cost_results.py', f'research/{runtime.NAME}/tests/test_cost_validation.py',
    'research/memory_budget_20261018_v1/worker.py',
    'research/b2_dataset_extension_20261018_v1/costs.py',
    *[f'research/informs_broad_validation_20261015_v1/{name}.py' for name in
      ('deployment', 'combination', 'data_contract', 'fit_base', 'training', 'networks', 'gpu_runtime', 'evaluation')]]


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in
                ('pred_normalized', 'target_normalized', 'observed', 'origins', 'channel_ids', 'mean', 'std')}


def verify_reference(row, saved, canonical=None):
    from research.memory_budget_20261018_v1.worker import independent_metrics
    if saved['pred_normalized'].shape != (41, 48, 26) or saved['target_normalized'].shape != (41, 48, 26):
        raise ValueError('Appliances full-V shape differs')
    expected = np.arange(13814, 15788-48+1, 48, dtype=np.int64)
    if not np.array_equal(saved['origins'], expected) or int(saved['origins'].max()) + 48 >= 15788:
        raise ValueError('Appliances V origin grid/last target differs')
    if canonical is not None:
        for key in ('target_normalized', 'observed', 'origins', 'channel_ids', 'mean', 'std'):
            if not np.array_equal(saved[key], canonical[key], equal_nan=saved[key].dtype.kind in 'fc'):
                raise ValueError('Cost models do not share exact cohort/scaler: ' + key)
    metrics = independent_metrics(saved)
    if max(abs(metrics[key] - float(row[key])) for key in ('MSE', 'MAE')) > 1e-12:
        raise ValueError('Cost accuracy binding differs from channel-macro FP64 metrics')
    return metrics


def prepare(*, amend=False):
    if amend:
        archived = COST_REPORT/'revisions/preflight_001'
        amendment = runtime.read(COST_REPORT/'AMENDMENT_001.json')
        if (list(COST_REPORT.glob('*_MEASUREMENT.json')) or (COST_REPORT/'COST_LAUNCH.json').exists()
                or (COST_REPORT/'ROOT_LAUNCH_GATE.json').exists()):
            raise ValueError('Cost amendment is allowed only before any launch/measurement')
        for name,digest in amendment['archived_files'].items():
            if runtime.sha(archived/name)!=digest:
                raise ValueError('Preserved cost v1 evidence changed')
        for name in ('COST_INPUTS.json','COST_PROTOCOL.json','COST_SEAL.json'):
            if runtime.sha(COST_REPORT/name)!=runtime.sha(archived/name):
                raise ValueError('Only the preserved first cost preflight may be amended')
    elif (COST_REPORT / 'COST_SEAL.json').exists():
        validate_seal()
        return dict(status='EXISTING_COST_SEAL_VERIFIED')
    source = runtime.REPO / 'reports/comparative_ablation_followup_20261027_v1/RUNS.csv'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    references, canonical = [], None
    for run_id in FACETS:
        raw = run_id == 'Raw_appliances'
        seed = 92601 if raw else int(run_id[-5:])
        matching = [row for row in rows if row['dataset'] == 'appliances'
                    and row['method_id'] == ('RAW' if raw else 'OURS') and (raw or int(row['seed']) == seed)]
        if len(matching) != 1:
            raise ValueError('Cost source must bind one fixed accuracy row: ' + run_id)
        row = matching[0]
        if raw:
            prediction = runtime.DEFAULT_SOURCE / '.cache/b2_dataset_search_20261019_v1/datasets/appliances/costs/Raw_appliances_DEFAULT128_B8_predictions.npz'
            bundle = None
        else:
            prediction = Path(row['prediction_path'])
            bundle = runtime.DEFAULT_SOURCE / '.cache/k_tau_rule_20261023_v1/datasets/appliances/runs' / run_id / 'cap3/selected_bundle.pt'
        saved = arrays(prediction)
        metrics = verify_reference(row, saved, canonical)
        if canonical is None:
            canonical = saved
        if runtime.sha(prediction) != row['prediction_sha256']:
            raise ValueError('Source row prediction hash differs: ' + run_id)
        entry = dict(run_id=run_id, family='RAW' if raw else 'B2', dataset='appliances', seed=seed,
                     K=26 if raw else 10, C=26, L=512, H=48, MSE=float(row['MSE']), MAE=float(row['MAE']),
                     prediction_path=str(prediction), prediction_sha256=runtime.sha(prediction),
                     bundle_path=None if raw else str(bundle), bundle_sha256=None if raw else runtime.sha(bundle),
                     selected_epoch=None if raw else float(row['selected_epoch']), training_cap=0 if raw else 3,
                     validation=dict(status='PASS', shape=[41,48,26], T_start=15788, target_end=15782,
                                     metric_aggregation=metrics['aggregation'], T_numeric_values_parsed=0))
        if not raw:
            import torch
            artifact = torch.load(bundle, map_location='cpu', weights_only=False)
            if (artifact['spec']['family'] != 'B2' or artifact['spec']['latent_k'] != 10
                    or artifact['spec']['seed'] != seed or artifact['spec']['revin'] is not False
                    or artifact['normalization']['fit_end'] != 11841
                    or artifact['contract']['model_sha'] != runtime.F_SHA
                    or artifact['selected_search_epoch'] != entry['selected_epoch']):
                raise ValueError('Cost checkpoint is not the selected original Full-P K10 model')
            for key in ('mean', 'std'):
                if not np.array_equal(artifact['normalization'][key], saved[key]):
                    raise ValueError('Selected checkpoint scaler differs from scored prediction')
            entry.update(bundle_spec=artifact['spec'], bundle_contract=artifact['contract'],
                         model_state_sha=artifact['model_state_sha'])
        references.append(entry)
    prefix = runtime.DEFAULT_SOURCE / '.cache/b2_dataset_search_20261019_v1/data/appliances/PREFIX_PHYSICAL.npz'
    config = prefix.with_name('DATA_DIAGNOSTIC.json')
    protocol = dict(status='REGISTERED_BEFORE_MEASUREMENT', revision=2, created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Existing selected Full-P Appliances K10 seeds92601/92602 and deterministic Raw; no training, refit, or model selection.',
        facet_order=list(FACETS), fresh_process_per_facet=True, concurrency=1,
        device='cuda', dtype='float32', threads=2, TF32=False, autocast=False, quantization=False, compile=False,
        batch_size=8, requested_F_row_microbatch=128, warmups=10, timed_trials=20,
        expected_work=dict(RAW=dict(streams=208,calls=2,row_sizes=[128,80]), B2=dict(streams=80,calls=1,row_sizes=[80])),
        core='Unmodified research.memory_budget_20261018_v1.worker.evaluate_facet(stage=confirm)',
        deployment='Unmodified broad-validation deployment.Route with selected exact original bundle, fixed A60 scaler, frozen Bolt.',
        timing_scope='physical history and mask -> local causal fill -> fixed A60 scale -> P/E/F/D or Raw -> inverse -> CPU output; excludes window slicing, load, full-V parity, hashes, metric scoring.',
        parity=dict(full_V_origins=41, complete_target_shape=[41,48,26], prediction_atol=1e-5,prediction_rtol=1e-4,
                    source_metric_atol=1e-12, regenerated_metric_atol=1e-5,regenerated_metric_rtol=1e-4,
                    metric='equal-channel observed macro FP64',state='Exact adapter/F tensor hashes before and after inference'),
        synchronization='Before and after every warmup/timed whole-path call; CUDA peak reset before each timed call.',
        memory_scope='Core empty_cache once after deployed model load and before loaded baseline/full-V parity; peak allocated=max20 timed trials, reserved reported separately, resident weights included.',
        summary='median/IQR/p95 from20 repeated first-B8 calls; descriptive p95, not service-tail guarantee.',
        restrictions='Never run while the scientific controller/worker is active. No GPU action in preparation. Root authorizes measurement after training.',
        launch_gate=dict(root_gate='ROOT_LAUNCH_GATE.json: status PASS, allow_gpu true, registration_seal_sha256 equals COST_SEAL hash',
                         parent='Live PID/create_time/exact command plus ACTIVE facet/token lease; shared campaign controller.lock for whole parent lifetime.',
                         child='At most five Python process ancestry links through the exact worker command; shared campaign worker.lock and cost worker.lock for whole actual GPU child lifetime.',
                         handoff='Parent holds shared worker.lock only between children, releases immediately before spawning; child reacquires before any CUDA work and retains through process exit.',
                         exclusion='Terminal training JSON plus live main/E/cost process snapshot and nvidia-smi compute PID exclusion. Parent death leaves the actual GPU child holding shared worker.lock.'),
        model_load_seconds_scope='Actual deployment loader time excludes pre-inference adapter/foundation state hashes; wrapper including hash time reported separately.',
        estimated_minutes=[5,15], training_deadline='2026-10-28T17:30:00+09:00', delivery_deadline='2026-10-28T20:00:00+09:00',
        reference_source=dict(path=str(source),sha256=runtime.sha(source)),
        safe_data=dict(prefix_path=str(prefix),prefix_sha256=runtime.sha(prefix),config_path=str(config),config_sha256=runtime.sha(config),
                       A60_fit_end=11841,V=[13814,15788],T_start=15788,T_numeric_values_parsed=0),
        foundation=dict(state_sha256=runtime.F_SHA,parameters=47718016,model_path=str(runtime.DEFAULT_ASSET/runtime.MODEL_REL)),
        source_hashes={name:runtime.sha(runtime.REPO/name) for name in SOURCE_FILES})
    if amend:
        protocol['amendment']=dict(path='AMENDMENT_001.json',sha256=runtime.sha(COST_REPORT/'AMENDMENT_001.json'),
                                  previous_seal_sha256=runtime.sha(COST_REPORT/'revisions/preflight_001/COST_SEAL.json'))
    writer=runtime.atomic if amend else runtime.write_new
    writer(COST_REPORT/'COST_INPUTS.json',dict(rows=references))
    writer(COST_REPORT/'COST_PROTOCOL.json',protocol)
    writer(COST_REPORT/'COST_SEAL.json',dict(revision=2,created_utc=datetime.now(timezone.utc).isoformat(),
                      files={name:runtime.sha(COST_REPORT/name) for name in ('COST_INPUTS.json','COST_PROTOCOL.json')}))
    return dict(status='SEALED_NOT_MEASURED',facets=3,CPU_reference_checks=3,estimated_minutes=[5,15])


def validate_seal():
    seal=runtime.read(COST_REPORT/'COST_SEAL.json')
    for name,digest in seal['files'].items():
        if runtime.sha(COST_REPORT/name)!=digest:
            raise ValueError('Cost input/protocol seal changed: '+name)
    protocol=runtime.read(COST_REPORT/'COST_PROTOCOL.json')
    if protocol.get('revision')!=4 or seal.get('revision')!=4:
        raise ValueError('System creation admission requires the prospective cost-v4 amendment and seal')
    host=protocol['host_gpu_gate']
    if runtime.sha(COST_REPORT/host['contract_path'])!=host['contract_sha256']:
        raise ValueError('Registered host contract changed')
    contract,_=host_gpu_gate.load_policy(COST_REPORT/host['contract_path'])
    if contract['policy_sha256']!=host['policy_sha256']:
        raise ValueError('Registered immutable desktop policy changed')
    if runtime.sha(COST_REPORT/'HOST_GATE_CPU_TESTS.json')!=protocol['host_cpu_test_receipt_sha256']:
        raise ValueError('Bound host CPU test receipt changed')
    if 'amendment' in protocol:
        amendment=protocol['amendment']
        if runtime.sha(COST_REPORT/amendment['path'])!=amendment['sha256']:
            raise ValueError('Cost pre-measurement amendment changed')
    for name,digest in protocol['source_hashes'].items():
        if runtime.sha(runtime.REPO/name)!=digest:
            raise ValueError('Cost producer source changed: '+name)
    for key in ('prefix','config'):
        if runtime.sha(protocol['safe_data'][key+'_path'])!=protocol['safe_data'][key+'_sha256']:
            raise ValueError('Bounded physical prefix changed')
    rows=runtime.read(COST_REPORT/'COST_INPUTS.json')['rows']
    for row in rows:
        for label in ('prediction','bundle'):
            if row[label+'_path'] is not None and runtime.sha(row[label+'_path'])!=row[label+'_sha256']:
                raise ValueError('Selected cost artifact changed: '+row['run_id'])
    return protocol, rows


def ensure_training_finished():
    path=runtime.REPORT/'EXECUTION_COMPLETION.json'
    if not path.is_file() or runtime.read(path).get('status') not in ('TRAINING_COMPLETE','TRAINING_INCOMPLETE'):
        raise RuntimeError('GPU cost measurement is forbidden until scientific training has ended')


def process_snapshot():
    records={}
    for process in psutil.process_iter(['pid','ppid','create_time','cmdline','exe']):
        try:
            info=process.info
            records[info['pid']]=dict(pid=info['pid'],ppid=info['ppid'],created=info['create_time'],
                                     command=info['cmdline'] or [],exe=info['exe'] or '')
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    return records


def module_name(record):
    command=record['command']
    return command[command.index('-m')+1] if '-m' in command and command.index('-m')+1<len(command) else None


def cost_command(record,flag):
    return module_name(record)==f'research.{runtime.NAME}.cost_profile' and flag in record['command']


def parent_python_chain(records,parent_pid):
    allowed=set()
    current=parent_pid
    for _ in range(5):
        record=records.get(current)
        if record is None or not cost_command(record,'--run'):
            break
        allowed.add(current)
        current=record['ppid']
    return allowed


def validate_root_gate():
    path=COST_REPORT/'ROOT_LAUNCH_GATE.json'
    if not path.is_file():
        raise RuntimeError('Root measurement launch gate is absent')
    gate=runtime.read(path)
    if (gate.get('status')!='PASS' or gate.get('allow_gpu') is not True
            or gate.get('registration_seal_sha256')!=runtime.sha(COST_REPORT/'COST_SEAL.json')):
        raise RuntimeError('Root measurement launch gate does not match the cost seal')
    return runtime.sha(path)


def validate_worker_lease(run_id,token,records=None):
    if not token:
        raise RuntimeError('Direct cost worker launch has no owning parent lease')
    gate_sha=validate_root_gate()
    path=COST_CACHE/'GPU_LEASE.json'
    if not path.is_file():
        raise RuntimeError('Cost worker has no owning parent lease')
    lease=runtime.read(path)
    if (lease.get('status')!='ACTIVE' or lease.get('token')!=token or lease.get('active_facet')!=run_id
            or lease.get('registration_seal_sha256')!=runtime.sha(COST_REPORT/'COST_SEAL.json')
            or lease.get('root_gate_sha256')!=gate_sha):
        raise RuntimeError('Cost worker parent lease identity differs')
    records=process_snapshot() if records is None else records
    parent=records.get(lease['pid'])
    if (parent is None or parent['created']!=lease['created'] or parent['command']!=lease['command']
            or not cost_command(parent,'--run')):
        raise RuntimeError('Cost worker parent is dead or its creation/command identity differs')
    allowed=parent_python_chain(records,parent['pid'])
    current=os.getpid()
    for _ in range(5):
        if current==parent['pid']:
            return lease,allowed
        record=records.get(current)
        if (record is None or not record['command']
                or Path(record['exe'] or record['command'][0]).stem.lower() not in ('python','pythonw')
                or not cost_command(record,'--worker') or '--lease-token' not in record['command']
                or record['command'][record['command'].index('--worker')+1]!=run_id
                or record['command'][record['command'].index('--lease-token')+1]!=token):
            break
        allowed.add(current)
        current=record['ppid']
    raise RuntimeError('Cost worker ancestry does not reach its registered live parent')


def verify_parent_reservation():
    reservation=runtime.CampaignLock(runtime.CACHE/'controller.lock')
    try:
        reservation.__enter__()
    except RuntimeError as exc:
        if 'Another live process holds' in str(exc):
            return
        raise
    else:
        reservation.__exit__(None,None,None)
        raise RuntimeError('Cost parent no longer holds the shared controller reservation')


def assert_no_gpu_conflicts(allowed_pids,records=None):
    records=process_snapshot() if records is None else records
    modules={f'research.{runtime.NAME}.controller',f'research.{runtime.NAME}.run_one',
             'research.ppt_e_fixed_20261028_v1.controller','research.ppt_e_fixed_20261028_v1.run_one'}
    conflicts=[r['pid'] for r in records.values() if r['pid'] not in allowed_pids and
               (module_name(r) in modules or cost_command(r,'--run') or cost_command(r,'--worker'))]
    if conflicts:
        raise RuntimeError('Live main/E/cost processes prohibit measurement: '+str(conflicts))
    allowed={pid:records[pid]['created'] for pid in allowed_pids}
    return host_gpu_gate.admit_identity(COST_REPORT/'HOST_GPU_CONTRACT.json',allowed_processes=allowed)


class StateCheckedRoute:
    def __init__(self, route, inputs, device):
        from research.b2_dataset_extension_20261018_v1.costs import _states
        self.route,self.inputs,self.device=route,inputs,device
        self.request={'family':inputs['candidate']['family']}
        self.before=_states(route,inputs,self.request,device)
        self.after=None
        self.closed=False
        self.predictions=[]
        self.parity_batches=math.ceil(len(inputs['saved']['origins'])/8)

    def __getattr__(self,name):
        return getattr(self.route,name)

    def predict_physical(self,physical,observed):
        output=self.route.predict_physical(physical,observed)
        if len(self.predictions)<self.parity_batches:
            self.predictions.append(((output-self.route.mean)/self.route.std).copy())
        return output

    def close(self):
        if self.closed:
            return
        self.closed=True
        from research.b2_dataset_extension_20261018_v1.costs import _states
        try:
            self.after=_states(self.route,self.inputs,self.request,self.device)
        finally:
            self.route.close()


def measure(inputs, protocol_sha, *, device='cuda', sampler_factory=None):
    from research.memory_budget_20261018_v1 import worker as core
    wrapped=[]
    load_seconds=[]
    original_loader=inputs['loader']
    def loader():
        started=time.perf_counter()
        route=original_loader()
        load_seconds.append(time.perf_counter()-started)
        try:
            value=StateCheckedRoute(route,inputs,device)
            wrapped.append(value)
            return value
        except BaseException:
            route.close()
            raise
    active=dict(inputs,loader=loader,registered_rows=(128,))
    candidate=inputs['candidate']
    request=dict(asset_root='REGISTERED_SELECTED_INPUTS',accuracy_run_id=candidate['run_id'],
                 batch_size=8,row_microbatch=128,stage='confirm',protocol_sha256=protocol_sha,
                 facet_id=candidate['run_id']+'_DEFAULT128_B8')
    kwargs={'device':device}
    if sampler_factory is not None:
        kwargs['sampler_factory']=sampler_factory
    result=core.evaluate_facet(request,active,**kwargs)
    result['summary']['model_load_with_state_hash_seconds']=result['summary']['model_load_seconds']
    result['summary']['model_load_seconds']=load_seconds[0]
    checked=wrapped[0]
    if checked.before!=checked.after:
        raise ValueError('Selected adapter/foundation tensor states changed across inference')
    predictions=np.concatenate(checked.predictions)
    if core._fingerprint(predictions)!=result['parity']['output_fingerprints']['regenerated_normalized']:
        raise ValueError('Captured complete-V outputs differ from measurement parity')
    result.update(states_before=checked.before,states_after=checked.after,
                  T_numeric_values_parsed=0,training_executed=False)
    return result,dict(inputs['saved'],pred_normalized=predictions,
                       reference_pred_normalized=inputs['saved']['pred_normalized'])


def make_inputs(row, protocol):
    modules=runtime.load_modules()
    deployment=importlib.import_module('deployment')
    if Path(deployment.__file__).resolve()!=runtime.REPO/'research/informs_broad_validation_20261015_v1/deployment.py':
        raise ValueError('Wrong deployment source root')
    task=runtime.load_task('appliances')
    runtime.assert_protected_T(dict(job={'dataset':'appliances'},task_access=task['_campaign_access']))
    saved=arrays(row['prediction_path'])
    verify_reference(row,saved)
    mean,std,_=modules['data'].fit_statistics(task,11841)
    if not np.array_equal(mean,saved['mean']) or not np.array_equal(std,saved['std']):
        raise ValueError('Physical task normalization differs from cost accuracy')
    target_rows=saved['origins'][:,None]+np.arange(48)[None]
    observed=task['observed'][target_rows]
    targets=((task['physical'][target_rows]-mean)/std).astype(np.float32)
    if not np.array_equal(observed,saved['observed']) or not np.allclose(targets[observed],saved['target_normalized'][observed],atol=1e-5,rtol=1e-4):
        raise ValueError('Physical V targets/mask differ from bound score arrays')
    candidate={key:row[key] for key in ('run_id','family','dataset','seed','K','C','L','H','MSE','MAE')}
    candidate['prediction_binding']=dict(path=row['prediction_path'],sha256=row['prediction_sha256'])
    if row['bundle_path']:
        candidate['bundle_binding']=dict(path=row['bundle_path'],sha256=row['bundle_sha256'])
    def loader():
        if row['bundle_path']:
            bundle=Path(row['bundle_path'])
            spec=dict(row['bundle_spec'],artifact_cache_relative_path=bundle.name,artifact_sha256=row['bundle_sha256'])
            cache_root=bundle.parent
        else:
            spec=dict(family='RAW',run_id=row['run_id'],dataset='appliances',foundation_sha=runtime.F_SHA)
            cache_root=COST_CACHE
        spec.update(device='cuda',canonical_fit_end=11841,row_microbatch=128,F_row_microbatch=128,checkpoint_inputs=False)
        pool=deployment.FoundationPool(runtime.DEFAULT_ASSET/runtime.MODEL_REL,'cuda',
                                       factory=lambda p,h,m,d:runtime.load_foundation(device=d))
        try:
            route=deployment.Route(spec,task,cache_root,device='cuda',foundation_pool=pool)
            route.owns_pool=True
            return route
        except BaseException:
            pool.close()
            raise
    return dict(candidate=candidate,loader=loader,task=task,saved=saved,
                expected_f_sha=runtime.F_SHA,expected_f_params=47718016,
                source_ctx=dict(repo=str(runtime.REPO)),
                source_bindings=dict(runtime_source_sha256=protocol['source_hashes']))


def run_worker(run_id,lease_token=None):
    lease,allowed=validate_worker_lease(run_id,lease_token)
    verify_parent_reservation()
    own=runtime.CampaignLock(COST_CACHE/'worker.lock')
    shared=runtime.CampaignLock(runtime.CACHE/'worker.lock')
    own.__enter__()
    try:
        shared.__enter__()
    except BaseException:
        own.__exit__(None,None,None)
        raise
    # These live handles remain locked until actual Python process exit, including orphan shutdown.
    WORKER_RESERVATIONS.extend((own,shared))
    validate_worker_lease(run_id,lease_token)
    verify_parent_reservation()
    ensure_training_finished()
    host_admission=assert_no_gpu_conflicts(allowed)
    protocol,rows=validate_seal()
    row=next(r for r in rows if r['run_id']==run_id)
    receipt=COST_REPORT/(run_id+'_MEASUREMENT.json')
    prediction=COST_CACHE/(run_id+'_predictions.npz')
    if receipt.exists() or prediction.exists():
        raise FileExistsError('Preserve existing cost attempt; never overwrite a measured facet')
    identity_records=process_snapshot()
    identities={pid:identity_records[pid]['created'] for pid in allowed}
    preflight=host_gpu_gate.HostObservation(COST_REPORT/'HOST_GPU_CONTRACT.json',run_id+'_preflight',identities).start().wait_preflight()
    runtime.write_new(COST_REPORT/(run_id+'_HOST_PREFLIGHT.json'),preflight)
    if preflight['status']!='PASS':
        raise RuntimeError('Cost host preflight failed before inference; preserve all evidence')
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    observer=host_gpu_gate.HostObservation(COST_REPORT/'HOST_GPU_CONTRACT.json',run_id+'_measurement',identities,gpu_worker=True).start()
    measured_from=time.time()
    try:
        result,values=measure(make_inputs(row,protocol),runtime.sha(COST_REPORT/'COST_PROTOCOL.json'))
    finally:
        observation=observer.finish(measured_from,time.time())
        runtime.write_new(COST_REPORT/(run_id+'_HOST_OBSERVATION.json'),observation)
    if observation['status']!='PASS':
        result['status']=result['summary']['status']='CONTAMINATED'
    result.update(host_admission=host_admission,host_preflight=preflight,host_observation=observation)
    validate_seal()
    prediction.parent.mkdir(parents=True,exist_ok=True)
    with prediction.open('xb') as stream:
        np.savez_compressed(stream,**values)
    result.update(prediction_path=str(prediction),prediction_sha256=runtime.sha(prediction),
                  protocol_sha256=runtime.sha(COST_REPORT/'COST_PROTOCOL.json'),
                  parent_lease_identity={key:value for key,value in lease.items() if key!='token'},actual_worker_pid=os.getpid(),
                  CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    runtime.write_new(receipt,result)
    if observation['status']!='PASS':
        raise RuntimeError('Completed cost measurements retained but host condition was contaminated; no comparison PASS or automatic repeat')
    return dict(status=result['status'],run_id=run_id,receipt=str(receipt))


def run_all():
    gate_sha=validate_root_gate()
    with runtime.CampaignLock(runtime.CACHE/'controller.lock'), runtime.CampaignLock(COST_CACHE/'cost.lock'):
        ensure_training_finished()
        validate_seal()
        records=process_snapshot()
        parent=records[os.getpid()]
        if not cost_command(parent,'--run'):
            raise RuntimeError('Cost parent must use the registered --run module entry point')
        allowed=parent_python_chain(records,parent['pid'])
        with runtime.CampaignLock(runtime.CACHE/'worker.lock'):
            assert_no_gpu_conflicts(allowed,records)
        runtime.write_new(COST_REPORT/'COST_LAUNCH.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),facets=list(FACETS)))
        lease=dict(status='ACTIVE',token=secrets.token_hex(24),pid=parent['pid'],created=parent['created'],
                   command=parent['command'],created_utc=datetime.now(timezone.utc).isoformat(),active_facet=None,
                   registration_seal_sha256=runtime.sha(COST_REPORT/'COST_SEAL.json'),root_gate_sha256=gate_sha)
        runtime.write_new(COST_CACHE/'GPU_LEASE.json',lease)
        try:
            for run_id in FACETS:
                with runtime.CampaignLock(runtime.CACHE/'worker.lock'):
                    ensure_training_finished()
                    assert_no_gpu_conflicts(allowed)
                    lease['active_facet']=run_id
                    runtime.atomic(COST_CACHE/'GPU_LEASE.json',lease)
                log=COST_CACHE/(run_id+'.log')
                log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('x',encoding='utf8') as stream:
                    process=subprocess.run([sys.executable,'-B','-m',f'research.{runtime.NAME}.cost_profile',
                        '--worker',run_id,'--lease-token',lease['token']],
                        cwd=runtime.REPO,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if process.returncode!=0:
                    raise RuntimeError('Cost facet failed; preserve log: '+str(log))
        finally:
            runtime.atomic(COST_CACHE/'GPU_LEASE.json',dict(lease,status='CLOSED',closed_utc=datetime.now(timezone.utc).isoformat()))
        with runtime.CampaignLock(runtime.CACHE/'worker.lock'):
            assert_no_gpu_conflicts(allowed)
        rows=[runtime.read(COST_REPORT/(run_id+'_MEASUREMENT.json'))['summary'] for run_id in FACETS]
        raw=rows[0]
        for row in rows:
            row['latency_ratio_to_raw']=row['median_batch_ms']/raw['median_batch_ms']
            row['peak_allocated_ratio_to_raw']=row['peak_allocated_MiB']/raw['peak_allocated_MiB']
        runtime.write_new(COST_REPORT/'COST_SUMMARY.json',dict(status='COMPLETE',rows=rows,
                          completed_utc=datetime.now(timezone.utc).isoformat(),training_executed=False,T_numeric_values_parsed=0))
    return dict(status='COMPLETE',facets=3)


def prepare_host_amendment(*, seal=False):
    archive=COST_REPORT/'revisions/preflight_002/ARCHIVE_RECEIPT.json'
    preserved=runtime.read(archive)
    if (COST_REPORT/'COST_LAUNCH.json').exists() or list(COST_REPORT.glob('*_MEASUREMENT.json')):
        raise RuntimeError('Host admission can only be amended before any cost measurement')
    for row in preserved['files']:
        if runtime.sha(runtime.REPO/row['archived'])!=row['sha256']:
            raise ValueError('Archived cost-v2 evidence changed')
        if Path(row['source']).name in ('COST_INPUTS.json','COST_PROTOCOL.json','COST_SEAL.json'):
            if runtime.sha(runtime.REPO/row['source'])!=row['sha256']:
                raise ValueError('Current cost-v2 registration differs from its preserved original')
    old=runtime.REPO/next(row['archived'] for row in preserved['files'] if row['source'].endswith('/COST_PROTOCOL.json'))
    old_seal=next(row['sha256'] for row in preserved['files'] if row['source'].endswith('/COST_SEAL.json'))
    if runtime.sha(COST_REPORT/'COST_SEAL.json')!=old_seal:
        raise ValueError('Only the preserved cost-v2 registration may receive this prospective amendment')
    contract,_=host_gpu_gate.load_policy(COST_REPORT/'HOST_GPU_CONTRACT.json')
    test_receipt=runtime.read(COST_REPORT/'HOST_GATE_CPU_TESTS.json')
    current_sources={name:runtime.sha(runtime.REPO/name) for name in SOURCE_FILES}
    if test_receipt.get('status')!='PASS' or test_receipt.get('source_sha256')!=current_sources:
        raise ValueError('Current cost host source has no matching successful CPU test receipt')
    amendment_path=COST_REPORT/'AMENDMENT_002.json'
    draft=COST_REPORT/'COST_PROTOCOL_v3_DRAFT.json'
    if not seal:
        amendment=dict(status='PROSPECTIVE_HOST_ADMISSION_AMENDMENT',revision=3,created_utc=datetime.now(timezone.utc).isoformat(),
            previous_seal_sha256=old_seal,archive_receipt_sha256=runtime.sha(archive),cost_measurements_before_amendment=0,
            reason='Windows WDDM lists registered desktop compute+graphics contexts; context existence alone is not an independent experiment workload.',
            changes='Retain scientific process/lease/locks; exact immutable desktop identities plus System PID4 boot-bound counter attribution; cost-only 60s/1Hz all-engine preflight and continuous observation, p95<=5% and max<=15%.',
            unchanged='Model/data/scaler/seed/facet order/B8/F128/FP32/10warmup20timed/scientific selection/deadlines and measurement core.',
            contamination='Preserve every result and raw host trace; any host violation blocks comparative PASS, with no selective trial deletion or automatic rerun.',
            public_contract_sha256=runtime.sha(COST_REPORT/'HOST_GPU_CONTRACT.json'),immutable_private_policy_sha256=contract['policy_sha256'],
            limitations=contract['limits'])
        runtime.write_new(amendment_path,amendment)
        protocol=runtime.read(old)
        protocol.update(revision=3,created_utc=datetime.now(timezone.utc).isoformat(),
            source_hashes=current_sources,host_cpu_test_receipt_sha256=runtime.sha(COST_REPORT/'HOST_GATE_CPU_TESTS.json'),
            estimated_minutes=[8,18],host_preflight_overhead_seconds=180,
            host_gpu_gate=dict(contract_path='HOST_GPU_CONTRACT.json',contract_sha256=runtime.sha(COST_REPORT/'HOST_GPU_CONTRACT.json'),
                               policy_sha256=contract['policy_sha256'],condition='Registered desktop concurrency; not idle/exclusive GPU'),
            amendment=dict(path='AMENDMENT_002.json',sha256=runtime.sha(amendment_path),previous_seal_sha256=old_seal))
        protocol['launch_gate']['exclusion']='Existing terminal training/live experiment/lease/shared-lock exclusion retained; exact registered WDDM desktop identities with cost-only bounded all-engine host telemetry.'
        runtime.write_new(draft,protocol)
        return dict(status='DRAFT_READY_NOT_SEALED',draft=str(draft),previous_seal_sha256=old_seal)
    protocol=runtime.read(draft)
    if (protocol['source_hashes']!={name:runtime.sha(runtime.REPO/name) for name in SOURCE_FILES}
            or protocol['host_gpu_gate']['contract_sha256']!=runtime.sha(COST_REPORT/'HOST_GPU_CONTRACT.json')
            or protocol['amendment']['sha256']!=runtime.sha(amendment_path)):
        raise ValueError('Reviewed host amendment draft/source/contract changed before sealing')
    runtime.atomic(COST_REPORT/'COST_PROTOCOL.json',protocol)
    runtime.atomic(COST_REPORT/'COST_SEAL.json',dict(revision=3,created_utc=datetime.now(timezone.utc).isoformat(),
        previous_seal_sha256=old_seal,files={name:runtime.sha(COST_REPORT/name) for name in
            ('COST_INPUTS.json','COST_PROTOCOL.json','HOST_GPU_CONTRACT.json','AMENDMENT_002.json')}))
    validate_seal()
    return dict(status='SEALED_NOT_MEASURED',revision=3,seal_sha256=runtime.sha(COST_REPORT/'COST_SEAL.json'))


def prepare_system_identity_amendment(*, seal=False):
    archive=COST_REPORT/'revisions/preflight_003/ARCHIVE_RECEIPT.json'
    preserved=runtime.read(archive)
    if (COST_REPORT/'COST_LAUNCH.json').exists() or list(COST_REPORT.glob('*_MEASUREMENT.json')):
        raise RuntimeError('System identity can only be amended before any cost measurement')
    for row in preserved['files']:
        if runtime.sha(runtime.REPO/row['archived'])!=row['sha256']:
            raise ValueError('Archived cost-v3 evidence changed')
        if Path(row['source']).name in ('COST_INPUTS.json','COST_PROTOCOL.json','COST_SEAL.json','HOST_GPU_CONTRACT.json'):
            if runtime.sha(runtime.REPO/row['source'])!=row['sha256']:
                raise ValueError('Current cost-v3 registration differs from its preserved original')
    private_receipt=runtime.REPO/preserved['private_archive_receipt_relative_path']
    if runtime.sha(private_receipt)!=preserved['private_archive_receipt_sha256']:
        raise ValueError('Private preflight archive receipt changed')
    if any(runtime.sha(runtime.REPO/row['archived'])!=row['sha256'] for row in runtime.read(private_receipt)['files']):
        raise ValueError('Private failed preflight evidence changed')
    failure=runtime.read(COST_REPORT/'HOST_ADMISSION_FAILURE_001.json')
    if failure['phase']!='PREFLIGHT' or failure['measurement_count']!=0 or failure['scientific_retries_used']!=0:
        raise ValueError('System identity amendment requires the preserved pre-measurement failure')
    contract_draft=COST_REPORT/'HOST_GPU_CONTRACT_v2_DRAFT.json'
    contract,_=host_gpu_gate.load_policy(contract_draft)
    current_sources={name:runtime.sha(runtime.REPO/name) for name in SOURCE_FILES}
    tests=runtime.read(COST_REPORT/'HOST_GATE_CPU_TESTS.json')
    if tests.get('status')!='PASS' or tests.get('source_sha256')!=current_sources:
        raise ValueError('Current system identity source has no matching CPU test receipt')
    old_seal=preserved['previous_seal_sha256']
    if runtime.sha(COST_REPORT/'COST_SEAL.json')!=old_seal:
        raise ValueError('Only the preserved cost-v3 seal can receive the system identity amendment')
    amendment_path=COST_REPORT/'AMENDMENT_003.json'
    draft=COST_REPORT/'COST_PROTOCOL_v4_DRAFT.json'
    if not seal:
        runtime.write_new(amendment_path,dict(status='PROSPECTIVE_SYSTEM_IDENTITY_AMENDMENT',revision=4,
            created_utc=datetime.now(timezone.utc).isoformat(),previous_seal_sha256=old_seal,
            archive_receipt_sha256=runtime.sha(archive),failure_receipt_sha256=runtime.sha(COST_REPORT/'HOST_ADMISSION_FAILURE_001.json'),
            phase='PREFLIGHT',cost_measurements_before_amendment=0,scientific_retries_used=0,
            reason='Installed psutil Windows boot_time is a wall-clock-minus-uptime estimate cached per Python process; exact cross-process equality is not a stable kernel identity.',
            changes='Explicit windows_system_creation_counter_only provider checks exact System PID4 creation time at every identity snapshot and replay; boot estimate is provenance only. Old boot-only provider policies are rejected.',
            unchanged='Exact39 desktop identities, device/engine set, all process/lease/lock exclusions, 60s/1Hz/p95<=5%/max<=15% observation, scientific methods, inputs, seeds, facet order, measurement core and deadlines.',
            public_contract_sha256=runtime.sha(contract_draft),immutable_private_policy_sha256=contract['policy_sha256']))
        protocol=runtime.read(COST_REPORT/'COST_PROTOCOL.json')
        protocol.update(revision=4,created_utc=datetime.now(timezone.utc).isoformat(),source_hashes=current_sources,
            host_cpu_test_receipt_sha256=runtime.sha(COST_REPORT/'HOST_GATE_CPU_TESTS.json'),
            amendment=dict(path='AMENDMENT_003.json',sha256=runtime.sha(amendment_path),previous_seal_sha256=old_seal))
        protocol['host_gpu_gate'].update(contract_sha256=runtime.sha(contract_draft),policy_sha256=contract['policy_sha256'])
        runtime.write_new(draft,protocol)
        return dict(status='DRAFT_READY_NOT_SEALED',revision=4,draft=str(draft),previous_seal_sha256=old_seal)
    protocol=runtime.read(draft)
    if (protocol['source_hashes']!=current_sources or
            protocol['host_gpu_gate']['contract_sha256']!=runtime.sha(contract_draft) or
            protocol['amendment']['sha256']!=runtime.sha(amendment_path) or
            protocol['host_cpu_test_receipt_sha256']!=runtime.sha(COST_REPORT/'HOST_GATE_CPU_TESTS.json')):
        raise ValueError('Reviewed system identity draft/source/evidence changed before sealing')
    runtime.atomic(COST_REPORT/'HOST_GPU_CONTRACT.json',contract)
    if runtime.sha(COST_REPORT/'HOST_GPU_CONTRACT.json')!=runtime.sha(contract_draft):
        raise ValueError('Activated host contract bytes differ from reviewed draft')
    runtime.atomic(COST_REPORT/'COST_PROTOCOL.json',protocol)
    runtime.atomic(COST_REPORT/'COST_SEAL.json',dict(revision=4,created_utc=datetime.now(timezone.utc).isoformat(),
        previous_seal_sha256=old_seal,files={name:runtime.sha(COST_REPORT/name) for name in
            ('COST_INPUTS.json','COST_PROTOCOL.json','HOST_GPU_CONTRACT.json','AMENDMENT_003.json','HOST_ADMISSION_FAILURE_001.json')}))
    validate_seal()
    return dict(status='SEALED_NOT_MEASURED',revision=4,seal_sha256=runtime.sha(COST_REPORT/'COST_SEAL.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--amend-preflight',action='store_true')
    group.add_argument('--run',action='store_true')
    group.add_argument('--worker',choices=FACETS)
    group.add_argument('--prepare-host-amendment',action='store_true')
    group.add_argument('--seal-host-amendment',action='store_true')
    group.add_argument('--prepare-system-identity-amendment',action='store_true')
    group.add_argument('--seal-system-identity-amendment',action='store_true')
    parser.add_argument('--lease-token')
    args=parser.parse_args()
    result=(prepare_system_identity_amendment(seal=args.seal_system_identity_amendment) if args.prepare_system_identity_amendment or args.seal_system_identity_amendment else
            prepare_host_amendment(seal=args.seal_host_amendment) if args.prepare_host_amendment or args.seal_host_amendment else
            prepare(amend=args.amend_preflight) if args.prepare or args.amend_preflight else
            run_worker(args.worker,args.lease_token) if args.worker else run_all())
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
