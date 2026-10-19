"""Explicit, journaled entry points for registration, audit and execution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil

from . import common


def context(asset, device='cuda'):
    return common.inherited.context(asset, repo=common.REPO, device=device)


def searches(dataset):
    results = {}
    for seed in common.SEEDS:
        for method in common.METHODS:
            path = common.CACHE/'runs'/dataset/f'{method}_s{seed}'/'search'/'RESULT.json'
            if not path.exists():
                return None
            row = common.read(path)
            if row['status'] != 'COMPLETE':
                return None
            results[(method, seed)] = row
    return results


def train_B(asset):
    import math
    if (common.REPORT/'C_ADMISSION.json').exists():
        saved = common.read(common.REPORT/'B_DECISIONS.json')
        for key, row in saved['models'].items():
            dataset, method, seed_text = key.split('/')
            path = common.CACHE/'runs'/dataset/f'{method}_{seed_text}'/'search'/'RESULT.json'
            if common.sha(path) != row['search_result_sha256']:
                raise ValueError('Sealed B search evidence changed')
        return common.read(common.REPORT/'C_ADMISSION.json')
    for dataset in common.DATASETS:
        if searches(dataset) is None:
            launch('search', asset, dataset=dataset, extra=['--cap','3'])
    decisions = {}
    sealed_B=common.read(common.REPORT/'B_DECISIONS.json') if (common.REPORT/'B_DECISIONS.json').exists() else None
    for dataset in common.DATASETS:
        if sealed_B is not None:
            decisions[dataset]=sealed_B['datasets'][dataset]
            continue
        rows = searches(dataset)
        if rows is None:
            raise ValueError('Complete mandatory paired group required')
        need = any(r['curves'][-1]['M_select_mse'] < min(v['M_select_mse'] for v in r['curves'][:-1])
                   for r in rows.values())
        cap = int(rows[('RESIDUAL_F',common.SEEDS[0])]['curves'][-1]['epoch'])
        estimate = max(300., 2*sum(r['training_seconds']+r['evaluation_seconds'] for r in rows.values()))
        available = common.deadline()-time.time()
        extend = need and cap == 3 and available > estimate+1800
        decision = dict(any_last_strictly_best=need, completed_cap_before_decision=cap,
            extension_conservative_seconds=estimate, available_training_seconds=available,
            extension_rule='all four methods and both seeds together', new_V_used=False)
        extension_path = common.REPORT/f'B_EXTENSION_{dataset}.json'
        if extension_path.exists():
            decision = common.read(extension_path)
            extend = cap == 3
        if extend:
            common.seal(extension_path,dict(decision,action='EXTEND_ALL_TO_6'))
            launch('search',asset,dataset=dataset,extra=['--cap','6'])
            cap = 6
        decision.update(completed_cap=cap, convergence_status=(
            'BUDGET_LIMITED_CONVERGENCE_UNRESOLVED' if need and cap == 3 else
            'CAP6_CONVERGENCE_UNRESOLVED' if cap == 6 and any(
                r['curves'][-1]['M_select_mse'] < min(v['M_select_mse'] for v in r['curves'][:-1])
                for r in searches(dataset).values()) else 'REGISTERED_ENDPOINT_SELECTION_COMPLETE'))
        decisions[dataset] = decision
    result = dict(status='B_SELECTION_SEALED', datasets=decisions, new_V_used=False,
        models={f'{d}/{m}/s{s}':dict(selected_epoch=r['selected_epoch'],
            selected_M_select_mse=r['selected_M_select_mse'],checkpoint_sha256=common.sha(r['selected_checkpoint']),
            search_result_sha256=common.sha(common.CACHE/'runs'/d/f'{m}_s{s}'/'search'/'RESULT.json'))
            for d in common.DATASETS for (m,s),r in searches(d).items()})
    common.seal(common.REPORT/'B_DECISIONS.json',result)
    gates = {}
    for dataset in common.DATASETS:
        rows = searches(dataset)
        scores = {m: math.fsum(rows[(m,s)]['selected_M_select_mse'] for s in common.SEEDS)/2
                  for m in common.METHODS}
        scores['ALPHA_fitted'] = math.fsum(next(r['M_select_mse'] for r in common.read(
            common.CACHE/'probes'/dataset/f'alpha_s{s}.json')['rows'] if r['label']=='fitted') for s in common.SEEDS)/2
        g = common.read(common.CACHE/'probes'/dataset/'RESULT.json')['selected']['ZERO']['M_select']['mse']
        eps = 1e-12
        both_beat_G = all(rows[('RESIDUAL_F',s)]['selected_M_select_mse'] < g-eps for s in common.SEEDS)
        alternatives = min(v for m,v in scores.items() if m != 'RESIDUAL_F')
        scientific = both_beat_G and scores['RESIDUAL_F'] < alternatives-eps
        budget = common.deadline()-time.time() > 4200
        gates[dataset] = dict(status='ADMITTED' if scientific and budget else 'NOT_RUN_BUDGET' if scientific else 'NOT_ADMITTED_NO_INCREMENTAL_EVIDENCE',
            G_M_select_mse=g, means_M_select_mse=scores, both_F_seeds_beat_G=both_beat_G,
            best_alternative_M_select_mse=alternatives, FP64_recombination_uncertainty=eps,
            conservative_complete_C_group_and_B_refit_cost_reserve_seconds=4200,
            remaining_training_seconds=common.deadline()-time.time(), new_V_used=False,
            completed_B_cap=decisions[dataset]['completed_cap'])
    return common.seal(common.REPORT/'C_ADMISSION.json',dict(status='M_SELECT_ONLY_DECISIONS',datasets=gates,
        priority=['jena','pems32'], all_mandatory_B_models=16, new_V_used=False))


def finish_training(asset):
    admission = train_B(asset)
    for dataset in common.DATASETS:
        cap = admission['datasets'][dataset]['completed_B_cap']
        complete = all((common.CACHE/'runs'/dataset/f'{m}_s{s}'/'refit'/'RESULT.json').exists()
            and common.read(common.CACHE/'runs'/dataset/f'{m}_s{s}'/'refit'/'RESULT.json')['status']=='COMPLETE'
            for m in common.METHODS for s in common.SEEDS)
        if not complete:
            launch('refit',asset,dataset=dataset,extra=['--cap',str(cap)])
    c_sources = {name:common.sha(Path(__file__).parent/name) for name in ('blocks.py','c_stage.py','models.py','engine.py')}
    if not (common.REPORT/'C_SOURCE_AND_SELECTION_REGISTRATION.json').exists():
        common.seal(common.REPORT/'C_SOURCE_AND_SELECTION_REGISTRATION.json',dict(
            sources=c_sources, registration_unix=time.time(), new_V_used=False,
            checkpoint_rule='minimum full-path M_select MSE; SEQ stage1 half-path; exact tie earlier epoch',
            prefix_rule='at fixed selected checkpoints, minimum M_select MSE over 0/half/full; exact tie smaller K',
            justification='same predeclared order for all C procedures; no joint prefix/epoch grid',
            selected_prefix_remains_fixed_after_full_Mcal_refit=True))
    elif common.read(common.REPORT/'C_SOURCE_AND_SELECTION_REGISTRATION.json')['sources'] != c_sources:
        raise ValueError('C registered scientific sources changed')
    decisions = {}
    for dataset in common.DATASETS:
        gate = admission['datasets'][dataset]
        if gate['status'] != 'ADMITTED':
            decisions[dataset] = dict(status=gate['status'], admission=gate, models=[])
            continue
        names = ('SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX')
        complete = all((common.CACHE/'C'/dataset/f'{m}_s{s}'/'RESULT.json').exists()
            and common.read(common.CACHE/'C'/dataset/f'{m}_s{s}'/'RESULT.json')['status']=='COMPLETE'
            for m in names for s in common.SEEDS)
        if not complete:
            launch('c',asset,dataset=dataset,extra=['--cap',str(gate['completed_B_cap'])])
        records=[]
        for seed in common.SEEDS:
            for method in names:
                directory=common.CACHE/'C'/dataset/f'{method}_s{seed}'
                result=common.read(directory/'RESULT.json')
                if result['status']!='COMPLETE':
                    raise TimeoutError('C group partial; do not expose V before completion or explicit budget disposition')
                records.append(dict(dataset=dataset,method=method,seed=seed,
                    selected_prefix=result['selected_prefix'],selected_epoch=result['selected_epoch'],
                    selected_M_select_mse=result['selected_M_select_mse'],
                    selection_seal_sha256=result['selection_seal_sha256'],
                    model_file_sha256=result['model_file_sha256'], result_sha256=common.sha(directory/'RESULT.json')))
        decisions[dataset]=dict(status='COMPLETE_C_COMPARISON_GROUP',admission=gate,models=records)
    return common.seal(common.REPORT/'C_DECISIONS.json',dict(status='SEALED_BEFORE_V',datasets=decisions,
        all_B_and_C_selection_complete=True,new_V_predictions_or_scores_before_seal=0,
        B_decisions_sha256=common.sha(common.REPORT/'B_DECISIONS.json'),
        C_admission_sha256=common.sha(common.REPORT/'C_ADMISSION.json')))


def final_accuracy(asset):
    from . import publish_tables
    if not (common.REPORT/'CUDA_RESUME_VALIDATION.json').exists():
        launch('validate',asset)
    for dataset in common.DATASETS:
        if not (common.CACHE/f'FINAL_MANIFEST_{dataset}.json').exists():
            launch('evaluate',asset,dataset=dataset)
    return publish_tables.finalize_manifest()


def measure(asset):
    from . import publish_tables
    entries=common.read(common.CACHE/'FINAL_MANIFEST.json')
    requests=publish_tables.cost_requests(entries)
    public=[]
    for cost_id,path,request in requests:
        public.append(dict(cost_id=cost_id,accuracy_run_id=request['accuracy_run_id'],
            dataset=request['dataset'],method=request['method'],seed=request['seed'],
            prefix=request.get('prefix'),batch_size=request['batch_size'],device=request['device'],
            request_sha256=common.sha(path)))
    common.seal(common.REPORT/'COST_REQUESTS.json',dict(expected_count=len(public),requests=public,
        protocol='each request in a fresh process, ten warmups, twenty timed trials; physical full-path'))
    for cost_id,path,request in requests:
        output=common.CACHE/'cost_results'/(cost_id+'.json')
        if output.exists():
            saved=common.read(output)
            if saved['request']!=request or saved['full_V_parity']['status']!='PASS':
                raise ValueError('Saved cost evidence changed')
            continue
        if common.deadline(False)-time.time()<120:
            raise TimeoutError('Conservative whole-worker reserve exhausted; preserve cost resume queue')
        launch('cost-cpu' if request['device']=='cpu' else 'cost',asset,dataset=request['dataset'],
            extra=['--request',str(path),'--output',str(output)])
        publish_tables.cost_tables()
    return publish_tables.cost_tables()


def launch(stage, asset, *, dataset='jena', method=None, seed=92601, extra=()):
    gpu = stage not in ('audit', 'probes', 'report', 'verify','cost-cpu')
    if gpu:
        for process in psutil.process_iter(['pid','name','cmdline']):
            text=' '.join(process.info.get('cmdline') or [])
            if process.pid!=os.getpid() and 'python' in (process.info.get('name') or '').lower() and common.NAME in text and ('_worker' in text or '.costs' in text):
                raise RuntimeError('Another campaign worker is live; wait rather than overlap GPU execution')
    session = uuid.uuid4().hex
    directory = common.CACHE/'process_receipts'
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-X', 'utf8', '-m', __package__+'.run', '_worker', '--stage', stage,
               '--asset-root', str(asset), '--dataset', dataset, '--seed', str(seed), *extra]
    if method:
        command += ['--method', method]
    began = time.time()
    active = dict(session_id=session, stage=stage, dataset=dataset, method=method, seed=seed,
                  gpu_owned=gpu, start_unix=began, command=command)
    common.atomic(common.CACHE/'ACTIVE_PROCESS.json', active)
    log = common.CACHE/(f'{stage}_{dataset}_{method or "none"}_{seed}_{session[:8]}.log')
    code = -1
    process = None
    try:
        with log.open('w', encoding='utf8') as stream:
            process = subprocess.Popen(command, cwd=common.REPO, stdout=stream, stderr=subprocess.STDOUT)
            active.update(wrapper_pid=process.pid, wrapper_create_time=psutil.Process(process.pid).create_time())
            common.atomic(common.CACHE/'ACTIVE_PROCESS.json', active)
            common.event('PROCESS_START', session=active, log=str(log.relative_to(common.REPO)))
            code = process.wait()
    finally:
        if process is not None:
            code=process.wait()
        receipt = dict(active, end_unix=time.time(), exit_code=code, log=str(log.relative_to(common.REPO)))
        common.atomic(directory/(session+'.json'), receipt)
        (common.CACHE/'ACTIVE_PROCESS.json').unlink(missing_ok=True)
        common.atomic(common.REPORT/'RESOURCE_LEDGER.json', common.ledger())
        common.event('PROCESS_EXIT', stage_name=stage, dataset=dataset, method=method, seed=seed,
                     exit_code=code, seconds=receipt['end_unix']-began, log=receipt['log'])
    if code:
        print(log.read_text(encoding='utf8')[-10000:], flush=True)
        raise RuntimeError(f'{stage} failed with exit {code}; preserved {log}')
    return receipt


def worker(args):
    import torch
    torch.set_num_threads(2)
    common.event('WORKER_IDENTIFIED', stage_name=args.stage, pid_actual=os.getpid(),
                 create_time=psutil.Process().create_time(), python=sys.executable)
    ctx = context(args.asset_root, 'cpu' if args.stage in ('audit','probes') else 'cuda')
    if args.stage == 'audit':
        from . import data_contract
        result = data_contract.audit(ctx)
        common.atomic(common.REPORT/'PARENT_AND_DATA_CONTRACT.json', result)
        common.event('AUDIT', status=result['status'])
    elif args.stage == 'smoke':
        from . import smoke
        smoke.run(ctx, args.dataset)
    elif args.stage == 'probes':
        from . import data_contract, probes
        for seed in common.SEEDS:
            data = data_contract.load(ctx, args.dataset, seed)
            probes.alpha(data, args.dataset, seed)
            if seed == common.SEEDS[0]:
                probes.run(data, args.dataset)
    elif args.stage in ('search', 'refit'):
        from . import data_contract, engine, models
        from research.adapter_diagnostics_20261017_v2.existing_jobs import _foundation
        from research.informs_broad_validation_20261015_v1.networks import module_state_hash
        import numpy as np
        source_names = ('models.py', 'engine.py', 'data_contract.py')
        science = {name: common.sha(Path(__file__).parent/name) for name in source_names}
        pool, foundation = _foundation(ctx, ctx['modules'], common.inherited.EXPECTED[args.dataset]['H'], 'cuda')
        foundation.row_microbatch, foundation.checkpoint_inputs = 128, False
        endpoints = (0,.25,1,3) if args.cap == 3 else (0,.25,1,3,4,5,6)
        for seed in common.SEEDS:
            data = data_contract.load(ctx, args.dataset, seed)
            for method in common.METHODS:
                directory = common.CACHE/'runs'/args.dataset/f'{method}_s{seed}'
                model = models.make(data['s_model'], method, seed).to('cuda')
                before_g = module_state_hash(data['g_model'])
                before_s = module_state_hash(data['s_model'])
                active_data = dict(data)
                cache_seconds = 0.
                if method == 'READOUT_ONLY':
                    began = time.perf_counter()
                    pieces = []
                    with torch.no_grad():
                        for lo in range(0, len(data['x']), 4):
                            x = torch.as_tensor(data['x'][lo:lo+4], device='cuda')
                            pieces.append(model.features(x, foundation).cpu().numpy())
                    active_data['phi'] = np.concatenate(pieces)
                    cache_seconds = time.perf_counter()-began
                    common.atomic(directory/'READOUT_CACHE.json', dict(seconds=cache_seconds,
                        feature_identity=common.identity(active_data['phi']),
                        input_parent_sha256=data['parent_s']['model_state_sha256'],
                        scope='full Mcal fixed original S P/E/F lag features; no trainable P/E cache',
                        cache_used_for_deployment=False))
                contract = dict(scientific_sources=science, document_sha256=common.start()['document_sha256'],
                    parent_S=data['parent_s']['mirror']['sha256'], parent_G=data['parent_g']['mirror']['sha256'],
                    scaler_sha=data['parent_s']['scaler_sha256'], method=method, dataset=args.dataset,
                    F_tensor_sha256=foundation.initial_hash, F_row_microbatch=128, checkpoint_inputs=False)
                torch.cuda.reset_peak_memory_stats()
                began = time.perf_counter()
                common.event('METHOD_START', phase=args.stage, dataset=args.dataset, method=method, seed=seed, cap=args.cap)
                if args.stage == 'search':
                    result = engine.run_search(model, active_data, data['split']['fit_indices'],
                        data['split']['select_indices'], seed, foundation, directory/'search',
                        endpoints, common.deadline(), contract=contract, device='cuda')
                else:
                    search = common.read(directory/'search'/'RESULT.json')
                    if search['status'] != 'COMPLETE':
                        raise ValueError('Fresh refit requires completed search')
                    result = engine.run_refit(model, active_data, seed, foundation, directory/'refit',
                        search['selected_epoch'], common.deadline(), contract=contract, device='cuda')
                foundation.assert_frozen_unchanged()
                if before_g != module_state_hash(data['g_model']) or before_s != module_state_hash(data['s_model']):
                    raise AssertionError('Fixed parent state mutated')
                public = {k:v for k,v in result.items() if k not in ('contract','selected_checkpoint','checkpoint_path','state_path','checkpoints')}
                public.update(dataset=args.dataset, method=method, seed=seed, cap=args.cap,
                    **models.parameter_counts(model), new_V_used=False,
                    initial_parent_S=data['parent_s']['mirror']['sha256'], G_shared_seed=92601,
                    G_parent_sha256=data['parent_g']['mirror']['sha256'], F_frozen_unchanged=True,
                    G_S_frozen_unchanged=True, cache_generation_seconds=cache_seconds,
                    worker_method_seconds=time.perf_counter()-began,
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
                common.atomic(common.REPORT/'runs'/args.dataset/f'{method}_s{seed}'/(args.stage+f'_cap{args.cap}.json'), public)
                common.event('METHOD_END', phase=args.stage, dataset=args.dataset, method=method, seed=seed,
                    status=result['status'], selected_epoch=result['selected_epoch'], updates=result['updates'],
                    selected_M_select_mse=result.get('selected_M_select_mse'), seconds=public['worker_method_seconds'])
                del model, active_data
                torch.cuda.empty_cache()
                if result['status'] != 'COMPLETE':
                    raise TimeoutError('Partial checkpoint retained; resume same command')
        pool.close()
    elif args.stage == 'c':
        from . import c_stage, data_contract
        from research.adapter_diagnostics_20261017_v2.existing_jobs import _foundation
        from research.informs_broad_validation_20261015_v1.networks import module_state_hash
        gate=common.read(common.REPORT/'C_ADMISSION.json')['datasets'][args.dataset]
        if gate['status']!='ADMITTED':
            raise ValueError('C requires M_select admission')
        pool,foundation=_foundation(ctx,ctx['modules'],common.inherited.EXPECTED[args.dataset]['H'],'cuda')
        foundation.row_microbatch,foundation.checkpoint_inputs=128,False
        for seed in common.SEEDS:
            data=data_contract.load(ctx,args.dataset,seed)
            for method in ('SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX'):
                directory=common.CACHE/'C'/args.dataset/f'{method}_s{seed}'
                if (directory/'RESULT.json').exists() and common.read(directory/'RESULT.json')['status']=='COMPLETE':
                    continue
                contract=dict(parent_S=data['parent_s']['mirror']['sha256'],parent_G=data['parent_g']['mirror']['sha256'],
                    scaler_sha=data['parent_s']['scaler_sha256'],F_tensor_sha256=foundation.initial_hash,
                    document_sha256=common.start()['document_sha256'],F_row_microbatch=128,checkpoint_inputs=False)
                parent_hashes=(module_state_hash(data['s_model']),module_state_hash(data['g_model']))
                common.event('C_METHOD_START',dataset=args.dataset,method=method,seed=seed,cap=args.cap)
                result=c_stage.run(ctx,data,args.dataset,seed,method,args.cap,foundation,directory,
                                   common.deadline(),contract)
                foundation.assert_frozen_unchanged()
                if parent_hashes!=(module_state_hash(data['s_model']),module_state_hash(data['g_model'])):
                    raise AssertionError('C changed fixed original parents')
                if result['status']=='COMPLETE':
                    from .blocks import restore
                    deployed=restore(data['s_model'],result['state_path'])
                    if module_state_hash(deployed.front)!=module_state_hash(data['s_model'].front) or not torch.equal(deployed.center,data['s_model'].center):
                        raise AssertionError('C changed fixed A60 P/center')
                    common.atomic(directory/'FIXED_STATE_VALIDATION.json',dict(status='PASS',
                        F_frozen_unchanged=True,G_S_parent_unchanged=True,
                        P_state_sha256=module_state_hash(deployed.front),center_unchanged=True))
                    del deployed
                def public(value):
                    if isinstance(value,dict):
                        return {k:public(v) for k,v in value.items() if k not in ('contract','sampled_indices')}
                    if isinstance(value,list):
                        return [public(v) for v in value]
                    if isinstance(value,str) and value.startswith(str(common.REPO)):
                        return Path(value).relative_to(common.REPO).as_posix()
                    return value
                common.atomic(common.REPORT/'C'/args.dataset/f'{method}_s{seed}.json',public(result))
                common.event('C_METHOD_END',dataset=args.dataset,method=method,seed=seed,status=result['status'],
                    selected_prefix=result['selected_prefix'],selected_epoch=result.get('selected_epoch'),seconds=result['seconds'])
                if result['status']!='COMPLETE':
                    raise TimeoutError('C partial checkpoint preserved')
                torch.cuda.empty_cache()
        pool.close()
    elif args.stage == 'evaluate':
        from . import evaluate
        evaluate.run(ctx,args.dataset)
    elif args.stage == 'validate':
        from . import validation
        validation.audit_training_states(ctx)
        validation.run(ctx)
    elif args.stage in ('cost','cost-cpu'):
        from . import costs
        result=costs.worker(common.read(args.request),args.asset_root,request_path=args.request)
        common.atomic(args.output,result)
        common.event('COST',cost_id=result['request']['cost_id'],status=result['status'],
            median_batch_ms=result['median_batch_ms'],batch_size=result['batch_size'],
            peak_allocated_MiB=max((r['peak_allocated_MiB'] or 0) for r in result['trials']))
    else:
        raise ValueError('Unimplemented stage '+args.stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['register','audit','smoke','probes','search','refit','train-B','finish-training','evaluate','validate','final-accuracy','measure','_worker'])
    parser.add_argument('--asset-root', default=r'E:\CODING\proj\hierarchical-tsfm-peft')
    parser.add_argument('--dataset', choices=common.DATASETS, default='jena')
    parser.add_argument('--method', choices=common.METHODS)
    parser.add_argument('--seed', type=int, default=92601)
    parser.add_argument('--stage')
    parser.add_argument('--cap', type=int, choices=(3,6), default=3)
    parser.add_argument('--request')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.action == '_worker':
        worker(args)
    elif args.action == 'register':
        common.register()
    elif args.action == 'train-B':
        train_B(args.asset_root)
    elif args.action == 'finish-training':
        finish_training(args.asset_root)
    elif args.action == 'final-accuracy':
        final_accuracy(args.asset_root)
    elif args.action == 'measure':
        measure(args.asset_root)
    else:
        launch(args.action, args.asset_root, dataset=args.dataset, method=args.method, seed=args.seed,
               extra=['--cap', str(args.cap)])


if __name__ == '__main__':
    main()
