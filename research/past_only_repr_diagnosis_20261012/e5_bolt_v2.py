"""Read-only E5 causal encoder/probe attachment to the local frozen Bolt."""
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
import os
import time
import json

import numpy as np
import psutil
import torch

from common import (HERE,CACHE,H,L,SEEDS,Budget,BudgetHeld,access_guard,read,write,sha,ahash,
                    load_data,windows,masks,score,physical_score)
from model_neural import TinyEncoder,ReconstructionProbe

DEFAULT_RESULT = HERE/'results'/'E5_jena_balanced600_cpu_v1'/'RESULT.json'
CANDIDATES = ('CAUSAL_CONTRAST','CAUSAL_FORECAST')


def pins():
    return {str(p):sha(p) for p in [Path(__file__),HERE/'common.py',HERE/'model_neural.py',HERE/'e5.py',
                                    HERE/'models_io.py',HERE/'APPROVED_INSTRUCTIONS.md']}


def select_objective(records):
    means = {}
    for objective in CANDIDATES:
        rows = [r for r in records if r['condition']==objective]
        if sorted(r['seed'] for r in rows) != SEEDS:
            raise ValueError('Balanced nonREC objectives are required for TUNE-only selection')
        means[objective] = float(np.mean([r['tune']['mse'] for r in rows]))
    chosen = min(CANDIDATES,key=lambda k:means[k])
    return chosen,{'criterion':'two-seed mean TUNE direct-ridge physical macroMSE',
                   'tie_order':list(CANDIDATES),'candidate_mean_TUNE_mse':means,
                   'chosen':chosen,'uses_VAL_for_choice':False}


def source_records(top_path):
    top = read(top_path)
    if top.get('experiment')!='E5' or top.get('status')!='COMPLETE_BALANCED' or top.get('dataset')!='jena':
        raise ValueError('Expected actual completed balanced Jena E5')
    for p,digest in top['source_pins'].items():
        if sha(p)!=digest:
            raise ValueError('Original E5 implementation changed after actual training')
    records = []
    for item in top['conditions']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('E5 case bytes changed')
        record = read(item['path'])
        record['_case_path'],record['_case_sha256'] = item['path'],item['sha256']
        if record['encoder_updates']!=600 or record['probe_updates']!=600:
            raise ValueError('Actual E5 training/probe updates are not the declared600')
        records.append(record)
    selected,proof = select_objective(records)
    proof['source_bindings'] = [{'condition':r['condition'],'seed':r['seed'],'path':r['_case_path'],
                                'sha256':r['_case_sha256'],'TUNE_mse':r['tune']['mse']}
                               for r in records if r['condition'] in CANDIDATES]
    rows = sorted([r for r in records if r['condition'] in ('CAUSAL_REC',selected)],
                  key=lambda r:(r['seed'],r['condition']))
    if len(rows)!=4 or len({(r['condition'],r['seed']) for r in rows})!=4:
        raise ValueError('The REC/nonREC x two-seed bundle is incomplete')
    return top,rows,proof


def selected_assets(record):
    probe = record['reconstruction_probe']
    chosen = next(c for c in probe['checkpoints'] if c['update']==probe['selected_update'])
    assets = {record['_case_path']:record['_case_sha256'],
              record['encoder_final_path']:record['encoder_final_sha256'],chosen['path']:chosen['sha256'],
              record['prediction_path']:record['prediction_sha256']}
    for p,digest in assets.items():
        if sha(p)!=digest:
            raise ValueError('E5 E/D/direct-readout bytes changed')
    return assets,chosen


def plan(top_path,tag):
    top,rows,proof = source_records(top_path)
    data = load_data('jena')
    preflight_path = HERE/'ASSETS_AND_PREFLIGHT.json'
    foundation = read(preflight_path)['Bolt']
    foundation_pins = {str(Path(foundation['weights_actual_target'])):foundation['weights_sha'],
                       str(Path(foundation['directory'])/'config.json'):foundation['config_sha']}
    if any(sha(p)!=digest for p,digest in foundation_pins.items()):
        raise ValueError('Local Bolt config/weights changed after actual asset preflight')
    bindings = []
    for record in rows:
        if not np.array_equal(data['mean'],np.asarray(record['fit_normalization_mean'],dtype=np.float32)) or \
           not np.array_equal(data['std'],np.asarray(record['fit_normalization_std'],dtype=np.float32)):
            raise ValueError('New FIT normalization differs from the actual E5 encoder inputs')
        for split in ('fit','tune','val'):
            if data['origins'][split].tolist()!=record['row_metadata'][split]['origins']:
                raise ValueError('E5 origins differ')
        assets,chosen = selected_assets(record)
        bindings.append({'condition':record['condition'],'seed':record['seed'],
                         'case_path':record['_case_path'],'case_sha256':record['_case_sha256'],
                         'E_path':record['encoder_final_path'],'E_sha256':record['encoder_final_sha256'],
                         'D_path':chosen['path'],'D_sha256':chosen['sha256'],'D_update':chosen['update'],
                         'asset_pins':assets,'direct_ridge_reused':{'val':record['val'],'tune':record['tune'],
                                                                  'prediction_path':record['prediction_path'],
                                                                  'prediction_sha256':record['prediction_sha256']}})
    source = {'experiment':'E5_BOLT','status':'PLANNED_NOT_EXECUTED','dataset':'jena','scope':'FULL_VAL371_DEVELOPMENT',
              'purpose':'Test frozen Bolt attachment for causal reconstruction and one TUNE-selected nonREC objective',
              'simple_alternative':'Same E+selected D latent persistence, plus original same-origin direct ridge',
              'objective_selection':proof,'top_E5_path':str(top_path),'top_E5_sha256':sha(top_path),
              'source_pins':pins(),'data_source':data['source'],'data_sha256':data['source_hash'],
              'foundation_asset_pins':foundation_pins,'foundation_preflight_path':str(preflight_path),
              'foundation_preflight_sha256':sha(preflight_path),
              'bindings':bindings,'origin_hashes':{s:ahash(oo) for s,oo in data['origins'].items()},
              'origin_counts':{s:len(oo) for s,oo in data['origins'].items()},
              'fixed':{'L':L,'H':H,'latent_dim':16,'seeds':SEEDS,'device':'cuda','CPU_threads':2,
                       'E_updates':0,'D_updates':0,'F_updates':0,'E_from':'E5 final600',
                       'D_from':'E5 same600-update reconstruction probe, originally chosen by TUNE REC MSE',
                       'Bolt':'local Chronos-Bolt Small, median native64 truncated48',
                       'imputation':'NEW_FIT: each512-history starts from FIT mean, forward fill only',
                       'latency':'B1/B32,5warmups20timed CPUrawwindow/ffill/normalize/E/F/D/CPUoutput; load/disk/guards excluded'},
              'decisions':{'improved':'Conditional attachment candidate; no independent generalization claim',
                           'unchanged_or_worse':'Current frozen attachment has not established added value',
                           'stop':'memory/resource/concurrency/asset/source/nonfinite failure'},
              'limitation':'Bolt512 versus direct ridge33 latent ticks; fixed pretraining versus FIT supervised ridge. Across-objective E/D differ.'}
    path = HERE/'plans'/f'E5_BOLT_{tag}.json'; write(path,source)
    return path


def prepare(raw,observed,mean,std):
    if raw.ndim!=3 or raw.shape[1]!=L or raw.shape!=observed.shape:
        raise ValueError('Only [B,512,C] history and its observed mask are accepted')
    filled = np.empty_like(raw,dtype=np.float32)
    last = np.broadcast_to(np.asarray(mean,dtype=np.float32),(len(raw),len(mean))).copy()
    for t in range(L):
        last[observed[:,t]] = raw[:,t][observed[:,t]]
        filled[:,t] = last
    return ((filled-mean)/std).astype(np.float32)


def model_assets(binding,channels):
    from e5 import state_hash
    record = read(binding['case_path'])
    encoder = TinyEncoder(channels,causal=True)
    saved = torch.load(binding['E_path'],map_location='cpu',weights_only=True)
    if saved['step']!=600 or saved['condition']!=binding['condition'] or saved['seed']!=binding['seed']:
        raise ValueError('E5 saved encoder metadata differs')
    encoder.load_state_dict(saved['encoder']); encoder.eval().requires_grad_(False)
    decoder = ReconstructionProbe(channels)
    saved_d = torch.load(binding['D_path'],map_location='cpu',weights_only=True)
    if saved_d['step']!=binding['D_update']:
        raise ValueError('Selected E5 reconstruction probe metadata differs')
    decoder.load_state_dict(saved_d['decoder']); decoder.eval().requires_grad_(False)
    if state_hash(decoder.state_dict())!=record['reconstruction_probe']['selected_hash']:
        raise ValueError('Selected D tensor digest differs')
    return encoder.to('cuda'),decoder.to('cuda')


@torch.no_grad()
def predict_history(raw,observed,mean,std,encoder,decoder,bolt,persistence=False):
    normalized = prepare(raw,observed,mean,std)
    z = encoder(torch.from_numpy(normalized).to('cuda'))
    if persistence:
        future = z[:,-1:].expand(-1,H,-1)
    else:
        # The frozen official wrapper accepts only this encoded history, not labels.
        future = torch.from_numpy(bolt.forecast(z.cpu().numpy())).to('cuda')
    raw_prediction = decoder(future).cpu().numpy()
    return raw_prediction,future.cpu().numpy()


def raw_windows(data,origins):
    return (np.stack([data['physical'][o-L:o] for o in origins]),
            np.stack([data['mask'][o-L:o] for o in origins]))


def source_check(frozen):
    for p,digest in frozen['source_pins'].items():
        if sha(p)!=digest:
            raise ValueError(f'Frozen helper source changed: {p}')
    if sha(frozen['top_E5_path'])!=frozen['top_E5_sha256'] or sha(frozen['data_source'])!=frozen['data_sha256']:
        raise ValueError('E5 result/data bytes changed')
    if sha(frozen['foundation_preflight_path'])!=frozen['foundation_preflight_sha256'] or \
       any(sha(p)!=digest for p,digest in frozen['foundation_asset_pins'].items()):
        raise ValueError('Foundation asset receipt/config/weights bytes changed')
    for binding in frozen['bindings']:
        for p,digest in binding['asset_pins'].items():
            if sha(p)!=digest:
                raise ValueError('E5 attachment asset bytes changed')
    for binding in frozen['objective_selection']['source_bindings']:
        if sha(binding['path'])!=binding['sha256']:
            raise ValueError('TUNE-only objective choice source bytes changed')


@contextmanager
def gpu_lease(gate_path,gate_sha):
    if sha(gate_path)!=gate_sha:
        raise ValueError('Root execution/concurrency gate changed')
    gate = read(gate_path)
    if gate.get('experiment')!='E5_BOLT' or any(gate.get(k) is not True for k in ('execute','legacy_gpu_clear','gpu_exclusive')):
        raise ValueError('Root actual legacy/new GPU exclusion gate is required')
    if read(HERE/'E1_GATE.json').get('safety_pass') is not True:
        raise ValueError('E1 safety gate')
    for p in (HERE/'resources').glob('*.json'):
        r = read(p)
        if r.get('gpu') and r.get('status')=='RUNNING_CONFIRMED' and psutil.pid_exists(r['pid']):
            raise ValueError('Another new-campaign GPU process remains active')
    lock = HERE/'resources'/'E5_BOLT_GPU_EXCLUSIVE.lock'
    with lock.open('x',encoding='utf-8') as f:
        json.dump({'pid':os.getpid(),'created_unix':time.time(),'gate_sha256':gate_sha},f)
    try:
        yield
    finally:
        if lock.exists() and read(lock)['pid']==os.getpid():
            lock.unlink()


def remaining_gpu():
    return 5400-sum(read(p)['wall_seconds'] for p in (HERE/'resources').glob('*.json')
                    if read(p)['gpu'] and read(p)['phase']==2)


def profile_bundle(frozen,data,bolt,budget,folder):
    rows = []
    for binding in frozen['bindings']:
        encoder,decoder = model_assets(binding,len(data['columns']))
        oo = data['origins']['fit'][:8]
        samples = []
        for rep in range(4):
            budget.check(); torch.cuda.synchronize(); start = time.perf_counter()
            raw,observed = raw_windows(data,oo)
            predict_history(raw,observed,data['mean'],data['std'],encoder,decoder,bolt)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            if rep:
                samples.append(elapsed)
            budget.check()
        rows.append({'condition':binding['condition'],'seed':binding['seed'],'batch':8,
                     'samples_seconds':samples,'p50_seconds':float(np.median(samples))})
        del encoder,decoder; torch.cuda.empty_cache()
    # Benchmark B32/B1 scaling is approximated conservatively from the actual B8 path.
    estimated = 2*sum(r['p50_seconds']*(np.ceil(len(data['origins']['val'])/8)+25*4+25) for r in rows)+60
    receipt = {'status':'COST_ONLY_COMPLETE_NOT_SCIENCE','actual_optimizer_updates':0,
               'measured_four_bundles':rows,'estimated_balanced_seconds':float(estimated),
               'remaining_phase2_gpu_seconds':remaining_gpu(),'margin_multiplier':2,
               'fixed_load_scoring_IO_margin_seconds':60,'forecast_not_completion_guarantee':True,
               'new_VAL_scores_computed':False}
    write(folder/'COST_PREFLIGHT.json',receipt)
    return receipt


def benchmark(data,encoder,decoder,bolt,budget):
    rows = []
    for n in (1,32):
        oo = data['origins']['val'][np.arange(n)%len(data['origins']['val'])]
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        samples = []
        for rep in range(25):
            budget.check(); torch.cuda.synchronize(); start = time.perf_counter()
            raw,observed = raw_windows(data,oo)
            prediction,_ = predict_history(raw,observed,data['mean'],data['std'],encoder,decoder,bolt)
            prediction.copy(); torch.cuda.synchronize()
            if rep>=5:
                samples.append(time.perf_counter()-start)
            budget.check()
        rows.append({'batch':n,'unique_origins':min(n,len(data['origins']['val'])),
                     'repeat_origins':max(0,n-len(data['origins']['val'])),'warmups':5,'timed':20,
                     'samples_seconds':samples,'p50_seconds':float(np.median(samples)),
                     'scope':'CPU physical512-window/mask extraction + FIT-mean initialized causal ffill + fixed normalization + E + Bolt + fixed D + CPUoutput',
                     'excludes':'model load/disk/guards/scoring','peak_allocated_vram':torch.cuda.max_memory_allocated(),
                     'peak_reserved_vram':torch.cuda.max_memory_reserved()})
    return rows


def dispatch(plan_path,plan_sha,gate_path,gate_sha,tag,profile_only):
    if sha(plan_path)!=plan_sha:
        raise ValueError('Frozen plan bytes changed')
    frozen = read(plan_path); source_check(frozen)
    folder = HERE/'results'/f'E5_BOLT_{"PROFILE_" if profile_only else ""}{tag}'
    folder.mkdir(parents=True,exist_ok=False)
    result = {'experiment':'E5_BOLT','status':'RUNNING_CONFIRMED','scope':frozen['scope'],
              'plan_path':str(plan_path),'plan_sha256':plan_sha,'gate_path':str(gate_path),'gate_sha256':gate_sha,
              'objective_selection':frozen['objective_selection'],'conditions':[],'source_pins':pins()}
    try:
        with gpu_lease(gate_path,gate_sha),Budget(2,gpu=True,label='E5 Bolt attachment '+tag) as budget:
            from models_io import Bolt
            from e5 import state_hash
            data = load_data('jena'); bolt = Bolt('cuda'); result['Bolt_parity'] = bolt.parity()
            cost = profile_bundle(frozen,data,bolt,budget,folder); result['cost'] = cost
            if not profile_only and cost['estimated_balanced_seconds']>=cost['remaining_phase2_gpu_seconds']:
                raise BudgetHeld('Four attachment bundles plus matched B1/B32 benchmarks exceed phase2 remaining')
            if not profile_only:
                cache = CACHE/f'E5_BOLT_{tag}'; cache.mkdir(parents=True,exist_ok=False)
                for binding in frozen['bindings']:
                    encoder,decoder = model_assets(binding,len(data['columns']))
                    e_before,d_before = state_hash(encoder.state_dict()),state_hash(decoder.state_dict())
                    predictions = {name:[] for name in ('BOLT_FIXED_D','LATENT_PERSISTENCE_FIXED_D')}
                    latents = {name:[] for name in predictions}
                    for start in range(0,len(data['origins']['val']),8):
                        oo = data['origins']['val'][start:start+8]
                        raw,observed = raw_windows(data,oo)
                        for name,persistence in [('BOLT_FIXED_D',False),('LATENT_PERSISTENCE_FIXED_D',True)]:
                            p,z = predict_history(raw,observed,data['mean'],data['std'],encoder,decoder,bolt,persistence)
                            predictions[name].append(p); latents[name].append(z)
                        budget.check()
                    latency = benchmark(data,encoder,decoder,bolt,budget)
                    methods = []
                    for name in predictions:
                        p,z = np.concatenate(predictions[name]),np.concatenate(latents[name])
                        path = cache/f'{binding["condition"]}_s{binding["seed"]}_{name}.npz'
                        with path.open('xb') as f:
                            np.savez_compressed(f,prediction=p,latent_prediction=z,target=windows(data,'val',True),
                                mask=masks(data,'val'),origins=data['origins']['val'],mean=data['mean'],std=data['std'])
                        methods.append({'method':name,'val':physical_score(data,p,'val'),
                                        'prediction_path':str(path),'prediction_sha256':sha(path)})
                    direct = binding['direct_ridge_reused']
                    methods.append({'method':'DIRECT_RAW_RIDGE_REUSED_SAME_ORIGINS','val':direct['val'],
                                    'prediction_path':direct['prediction_path'],'prediction_sha256':direct['prediction_sha256'],
                                    'cost_status':'not newly benchmarked in this attachment; original E5 training/readout costs separate'})
                    if e_before!=state_hash(encoder.state_dict()) or d_before!=state_hash(decoder.state_dict()):
                        raise ValueError('Frozen E/D tensors changed')
                    record = {'condition':binding['condition'],'seed':binding['seed'],'methods':methods,
                              'end_to_end_BOLT_cost':latency,'E_hash_before_after':e_before,'D_hash_before_after':d_before,
                              'asset_pins':binding['asset_pins'],'optimizer_updates':0,
                              'foundation_asset_pins':frozen['foundation_asset_pins'],
                              'comparability':'Same371VAL/originaltarget/mask/FITstats/E tensors; fixed selected D shared between Bolt and latent persistence; reused direct ridge ran onCPU FP32 E and predicts raw from33 latent ticks; new E/F/D executeCUDA FP32',
                              'physical_scale_note':'Macro across heterogeneous channel units; normalized/per-channel/horizon metrics accompany it'}
                    path = folder/f'{binding["condition"]}_s{binding["seed"]}.json'; write(path,record)
                    result['conditions'].append({'condition':binding['condition'],'seed':binding['seed'],
                                                  'path':str(path),'sha256':sha(path),'methods':methods})
                    write(folder/'PROGRESS.json',result,replace=True)
                    del encoder,decoder; torch.cuda.empty_cache()
                if state_hash(bolt.model.state_dict())!=bolt.initial:
                    raise ValueError('Frozen Bolt tensor state changed')
            source_check(frozen); budget.check(); result['budget_ledger'] = str(budget.path)
        result.update({'status':'COST_ONLY_COMPLETE_NOT_SCIENCE' if profile_only else 'COMPLETE_BALANCED_UNVERIFIED',
                       'budget_ledger_sha256':sha(result['budget_ledger']),'actual_optimizer_updates':0,
                       'limitation':frozen['limitation']})
        write(folder/'RESULT.json',result)
        return folder/'RESULT.json'
    except BaseException as error:
        result.update({'status':'BUDGET_HELD_PARTIAL' if isinstance(error,BudgetHeld) else 'FAILED_PRESERVED',
                       'reason':str(error)})
        write(folder/'FAILURE.json',result)
        raise


def unit_test():
    records = [{'condition':c,'seed':s,'tune':{'mse':1 if c=='CAUSAL_CONTRAST' else 2},
                'val':{'mse':10000 if c=='CAUSAL_CONTRAST' else 0}}
               for c in CANDIDATES for s in SEEDS]
    chosen,proof = select_objective(records)
    assert chosen=='CAUSAL_CONTRAST' and not proof['uses_VAL_for_choice']
    raw = np.ones((2,L,3),np.float32); mask = np.ones_like(raw,dtype=bool)
    raw[:,0,0] = np.nan; mask[:,0,0] = False
    normalized = prepare(raw,mask,np.array([3.,4.,5.],np.float32),np.ones(3,np.float32))
    assert np.isfinite(normalized).all() and np.all(normalized[:,0,0]==0)
    assert np.all(normalized[:,1,0]==-2)
    try:
        prepare(raw[:,:511],mask[:,:511],np.zeros(3),np.ones(3))
    except ValueError:
        pass
    else:
        raise AssertionError('A truncated/future-extended history must be rejected')
    return {'status':'UNIT_PASS','checks':4,'science_dispatches':0,'GPU_calls':0,'source_pins':pins()}


def main():
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('unit','plan','profile','run'):
        group.add_argument('--'+name,action='store_true')
    parser.add_argument('--e5-result',type=Path,default=DEFAULT_RESULT)
    parser.add_argument('--tag'); parser.add_argument('--frozen-plan',type=Path); parser.add_argument('--plan-sha256')
    parser.add_argument('--gate',type=Path); parser.add_argument('--gate-sha256')
    args = parser.parse_args(); access_guard(); torch.set_num_threads(2)
    if args.unit:
        with Budget(2,gpu=False,label='E5 Bolt helper CPU contract unit') as budget:
            result = unit_test()
        result['budget_ledger'],result['budget_ledger_sha256'] = str(budget.path),sha(budget.path)
        path = HERE/'receipts'/f'E5_BOLT_UNIT_{budget.path.stem}.json'; write(path,result)
    else:
        if not args.tag or not args.tag.replace('_','').replace('-','').isalnum():
            parser.error('A simple fresh --tag is required')
        if args.plan:
            with Budget(2,gpu=False,label='E5 Bolt fixed attachment plan '+args.tag):
                path = plan(args.e5_result,args.tag)
        else:
            if any(v is None for v in (args.frozen_plan,args.plan_sha256,args.gate,args.gate_sha256)):
                parser.error('--profile/--run require frozen plan,Root GPU gate and both SHA256')
            path = dispatch(args.frozen_plan,args.plan_sha256,args.gate,args.gate_sha256,args.tag,args.profile)
    print(json.dumps({'path':str(path),'sha256':sha(path)},ensure_ascii=False))


if __name__=='__main__':
    main()
