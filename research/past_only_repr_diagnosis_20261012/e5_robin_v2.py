"""Robin E5 extension: unavailable TUNE objective monitoring, unchanged training."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import sys
import time

import numpy as np
import psutil
import torch

import e5

BASE_E5_SHA = '35d64fa782591dcad2c60ba9863a280cd74eca4040aca1c8cc43adf4e103f1e6'
VERSION = 'E5_ROBIN_MONITOR_ONLY_V2'
_BASE_SOURCE_PINS = e5.source_pins
_BASE_WRITE = e5.write


def monitor(model,head,condition,data,batches,device,seed,objective_runner=None):
    objective_runner = objective_runner or e5.run_objective
    tune_origins = e5.training_origins(data['tune_bounds'])[:len(batches[0])]
    with torch.no_grad():
        fit_loss,_ = objective_runner(model,head,condition,data,batches[0],device,seed+200000,
                                     target_floor=data['fit_bounds'][0])
        tune_loss = None
        if len(tune_origins)>=2:
            tune_loss,_ = objective_runner(model,head,condition,data,tune_origins,device,seed+200001,
                                           target_floor=data['tune_bounds'][0])
    values = [float(fit_loss)] + ([] if tune_loss is None else [float(tune_loss)])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite objective curve')
    return {'fit_fixed_batch_objective_loss':float(fit_loss),
            'tune_fixed_batch_objective_loss':None if tune_loss is None else float(tune_loss),
            'tune_objective_curve_status':'UNAVAILABLE' if tune_loss is None else 'AVAILABLE',
            'tune_objective_curve_reason':None if tune_loss is not None else
                f'TUNE interval {data["tune_bounds"][1]-data["tune_bounds"][0]} < L512; fewer than two nonoverlapping512-history windows',
            'tune_nonoverlap_objective_monitor_windows':len(tune_origins),
            'cross_objective_loss_comparison_valid':False,
            'encoder_selection':'final600 fixed; no TUNE objective-loss selection',
            'TUNE_ridge_and_reconstruction_probe_selection':'unchanged original legal TUNE origins and rules'}


def train_encoder_v2(condition,seed,initial,data,batches,output,cache,budget,device):
    """Original train_encoder loop, with only its monitoring block changed."""
    torch.manual_seed(seed)
    model = e5.TinyEncoder(data['x'].shape[1],causal=condition!='FULL_REC').to(device)
    model.load_state_dict(initial)
    initial_hash = e5.state_hash(model.state_dict())
    torch.manual_seed(seed+17)
    head = e5.objective_head(condition,data['x'].shape[1]).to(device)
    optimizer = torch.optim.Adam(list(model.parameters())+list(head.parameters()),lr=.001)
    curves,exposures,checkpoints = [],[],[]
    started = time.perf_counter()
    for step in range(e5.UPDATES+1):
        if step%100==0:
            model.eval(); head.eval()
            curve = monitor(model,head,condition,data,batches,device,seed)
            curve.update({'update':step,'wall_seconds':time.perf_counter()-started})
            curves.append(curve)
            cp = cache/f'E_u{step:04}.pt'
            digest = e5.save_torch(cp,{'encoder':model.state_dict(),'head':head.state_dict(),
                'optimizer':optimizer.state_dict(),'step':step,'condition':condition,'seed':seed,
                'encoder_initial_hash':initial_hash,'batch_order_hash':e5.ahash(batches)})
            checkpoints.append({'update':step,'path':str(cp),'sha256':digest})
            e5.write(output/f'E_curve_u{step:04}.json',curve)
            budget.check()
        if step==e5.UPDATES:
            break
        model.train(); head.train(); optimizer.zero_grad(set_to_none=True)
        loss,exposure = e5.run_objective(model,head,condition,data,batches[step],device,seed+step+1)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward(); optimizer.step()
        exposure.update({'update':step+1,'origins':batches[step].tolist(),'loss':float(loss.detach()),
                         'history_end_inclusive':(batches[step]-1).tolist(),
                         'target_end_exclusive':(batches[step]+e5.H).tolist() if condition.endswith('FORECAST') else batches[step].tolist()})
        exposures.append(exposure)
        if (step+1)%10==0:
            budget.check()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    e5.write(output/'E_training.json',{'condition':condition,'seed':seed,'actual_updates':e5.UPDATES,
        'encoder_parameters':model.parameter_count,'head_parameters':sum(p.numel() for p in head.parameters()),
        'encoder_initial_hash':initial_hash,'encoder_final_hash':e5.state_hash(model.state_dict()),
        'batch_order_hash':e5.ahash(batches),'curves':curves,'checkpoints':checkpoints,
        'actual_sample_exposures':sum(v['input_samples'] for v in exposures),
        'actual_encoder_timestep_exposures':sum(v['encoder_input_timesteps'] for v in exposures),
        'actual_future_target_values':sum(v['future_target_values'] for v in exposures),
        'actual_future_target_timesteps':sum(v['future_target_timesteps'] for v in exposures),
        'wall_seconds':time.perf_counter()-started,'training_exposures':exposures,
        'monitoring_wrapper':VERSION,'TUNE_objective_monitor_is_not_required_for_final600_encoder':True})
    return model


def source_pins_v2():
    return {**_BASE_SOURCE_PINS(),str(Path(__file__).resolve()):e5.sha(Path(__file__)),
            str(e5.HERE/'test_e5_robin_v2.py'):e5.sha(e5.HERE/'test_e5_robin_v2.py')}


def write_v2(path,value,replace=False):
    if isinstance(value,dict) and value.get('experiment')=='E5' and value.get('status')=='PLANNED_NOT_EXECUTED':
        if value['dataset']!='robin':
            raise ValueError('This wrapper is only the approved balanced Robin extension')
        value.update({'wrapper_version':VERSION,'wrapper_sha256':e5.sha(Path(__file__)),
                      'original_E5_sha256_unchanged':BASE_E5_SHA,
                      'purpose':'Optional Robin four-objective/two-seed extension with unchanged FIT objectives and valid TUNE readout/probe selection',
                      'TUNE_objective_monitor':'Unavailable where399<L512 prevents independent512-history validation pairs; FIT curves remain at0/100..600',
                      'monitor_not_a_model_selection_rule':True})
    return _BASE_WRITE(path,value,replace=replace)


@contextmanager
def install():
    if e5.sha(e5.HERE/'e5.py')!=BASE_E5_SHA:
        raise ValueError('Original frozen E5 source bytes changed')
    previous = e5.train_encoder,e5.source_pins,e5.write
    e5.train_encoder,e5.source_pins,e5.write = train_encoder_v2,source_pins_v2,write_v2
    try:
        yield
    finally:
        e5.train_encoder,e5.source_pins,e5.write = previous


def option(argv,name,default=None):
    return argv[argv.index(name)+1] if name in argv else default


@contextmanager
def execution_lease(argv):
    gate_path,gate_sha = option(argv,'--gate'),option(argv,'--gate-sha256')
    if gate_path is None or gate_sha is None or e5.sha(gate_path)!=gate_sha:
        raise ValueError('Root gate with immutable SHA is required for profile/run')
    gate = e5.read(gate_path)
    if gate.get('experiment')!='E5' or gate.get('safety_pass') is not True or gate.get('execute') is not True:
        raise ValueError('Root E5 execution/safety gate required')
    if e5.read(e5.HERE/'E1_GATE.json').get('safety_pass') is not True:
        raise ValueError('E1 safety gate not passed')
    gpu = option(argv,'--device','cuda')=='cuda'
    lock = None
    if gpu:
        if gate.get('legacy_gpu_clear') is not True or gate.get('gpu_exclusive') is not True:
            raise ValueError('Actual legacy/new GPU exclusion gate required')
        for p in (e5.HERE/'resources').glob('*.json'):
            r = e5.read(p)
            if r.get('gpu') and r.get('status')=='RUNNING_CONFIRMED' and psutil.pid_exists(r['pid']):
                raise ValueError('Another new-campaign GPU process remains active')
        lock = e5.HERE/'resources'/'E5_ROBIN_V2_GPU_EXCLUSIVE.lock'
        with lock.open('x',encoding='utf-8') as f:
            json.dump({'pid':os.getpid(),'gate_sha256':gate_sha,'created_unix':time.time()},f)
    try:
        yield
    finally:
        if lock is not None and lock.exists() and e5.read(lock)['pid']==os.getpid():
            lock.unlink()


def main():
    argv = sys.argv[1:]
    if '--dataset' not in argv:
        argv += ['--dataset','robin']
    if option(argv,'--dataset')!='robin':
        raise ValueError('Only Robin is authorized in this extension')
    dispatch = '--profile' in argv or '--run' in argv
    if dispatch:
        plan_path = option(argv,'--frozen-plan')
        if plan_path is None or e5.read(plan_path).get('wrapper_version')!=VERSION:
            raise ValueError('A wrapper-version-pinned Robin plan is required')
    old = sys.argv
    sys.argv = [str(Path(__file__).resolve()),*argv]
    try:
        with install():
            if dispatch:
                with execution_lease(argv):
                    e5.main()
            else:
                e5.main()
    finally:
        sys.argv = old


if __name__=='__main__':
    main()
