"""Separate the approved CPU graph preparation from the later exclusive GPU fit."""
import argparse
import os
import time
import numpy as np
from pathlib import Path
import e6_v2 as impl
from common import *

BASE_PINS=impl.pins


def pins():
    return {**BASE_PINS(),str(Path(__file__)):sha(__file__)}


def check_plan(path,digest):
    if sha(path)!=digest:
        raise ValueError('Parent E6 plan changed')
    plan=read(path)
    if plan['dataset']!='jena' or plan['compute_device']!='cuda' or plan['status']!='READY':
        raise ValueError('Only the approved ready Jena GPU plan may be staged')
    for p,v in plan['source_pins'].items():
        if sha(p)!=v:
            raise ValueError('Frozen E6 implementation changed')
    for ref in (plan['E1_gate'],plan['E2_summary']):
        if sha(ref['path'])!=ref['sha256']:
            raise ValueError('Predecessor evidence changed')
    objective=plan['objective_decision']
    if objective.get('e5_complete') and sha(objective['path'])!=objective['sha256']:
        raise ValueError('TUNE-selected E5 objective evidence changed')
    return plan


def prepare(path,digest,tag,allocation):
    plan=check_plan(path,digest)
    output=HERE/'results'/f'E6_PREPARED_jena_{tag}'
    cache=CACHE/f'E6_PREPARED_jena_{tag}'
    output.mkdir(exist_ok=False);cache.mkdir(exist_ok=False)
    with impl.cpu_thread_contract(),impl.AllocationBudget(2,False,'E6 Jena complete CPU graph preparation '+tag,allocation) as budget:
        records,_=impl.scope_records('jena',plan['E2_summary']['path'])
        pairs={s:np.asarray(v,np.int64) for s,v in plan['pairs'].items()}
        if any(ahash(p)!=plan['pair_hashes'][s] for s,p in pairs.items()):
            raise ValueError('Staged origin order differs')
        lam,fixed,options=impl.selected_graph_lambda(records,pairs['tune'],budget)
        fixed_ref=impl.save_npz(cache/'FIXED_GRAPHS.npz',**{f'record{i}_{k}':np.asarray(v)
                                   for i,g in enumerate(fixed) for k,v in g.items()})
        blocks,traces,diagnostics=impl.precompute(records,pairs,fixed,lam,cache,output,budget)
        baselines=impl.matched_baselines(records,pairs,blocks,cache,output,budget)
        budget.check()
        started=time.perf_counter()
        for ref in traces.values():
            with np.load(ref['path'],allow_pickle=False) as z:
                _=z['input_history']
                for mode in impl.MODES[1:]:
                    _=z[f'{mode}_neighbor_message']
        measured=time.perf_counter()-started
        proof={'experiment':'E6_PREPARED','status':'COMPLETE_GRAPH_AND_MATCHED_BASELINES_ONLY',
            'dataset':'jena','parent_plan':str(path),'parent_plan_sha256':digest,'source_pins':pins(),
            'purpose':'Complete independent CPU-only past graph estimation while preserved legacy GPU remains busy',
            'comparison_conditions_unchanged':True,'objective':plan['objective'],'pairs':plan['pairs'],
            'pair_hashes':plan['pair_hashes'],'selected_graph_lambda':lam,'TUNE_graph_grid':options,
            'fixed_graph_arrays':fixed_ref,'graph_arrays':traces,'graph_diagnostics':diagnostics,
            'matched_baselines':baselines,'CPU_preparation_ledger':str(budget.path),
            'measured_rehydration_seconds':measured,'rehydration_cost_allowance_seconds':2*measured+5,
            'actual_neural_updates':0,'GPU_calls':0,'TEST_access':0,
            'stage_rule':'Completed current-task preparation, not revival of a stopped historical experiment'}
    proof['CPU_preparation_ledger_sha256']=sha(budget.path)
    dest=output/'PREPARED.json';write(dest,proof)
    return dest


def check_prepared(path,digest):
    if sha(path)!=digest:
        raise ValueError('Completed preparation receipt changed')
    proof=read(path)
    if proof['status']!='COMPLETE_GRAPH_AND_MATCHED_BASELINES_ONLY':
        raise ValueError('Incomplete preparation cannot be reused')
    if sha(proof['CPU_preparation_ledger'])!=proof['CPU_preparation_ledger_sha256']:
        raise ValueError('Closed CPU preparation cost ledger changed')
    ledger=read(proof['CPU_preparation_ledger'])
    if ledger['status']!='COMPLETE' or ledger['gpu']:
        raise ValueError('Preparation is not a closed CPU-only stage')
    check_plan(Path(proof['parent_plan']),proof['parent_plan_sha256'])
    for p,v in proof['source_pins'].items():
        if sha(p)!=v:
            raise ValueError('Preparation source changed')
    refs=[proof['fixed_graph_arrays'],*proof['graph_arrays'].values()]
    refs += [v['arrays'] for v in proof['matched_baselines'].values()]
    for ref in refs:
        if sha(ref['path'])!=ref['sha256']:
            raise ValueError('Completed preparation array changed')
    return proof


def derived_plan(prepared_path,digest,tag):
    proof=check_prepared(prepared_path,digest)
    plan=read(proof['parent_plan'])
    plan['completed_CPU_stage']={'path':str(prepared_path),'sha256':digest,
                                  'ledger':proof['CPU_preparation_ledger']}
    plan['source_pins']={**pins(),str(prepared_path):digest}
    plan['cost_only_preflight']=dict(plan['cost_only_preflight'])
    plan['cost_only_preflight']['original_graph_preparation_estimate_seconds']=plan['cost_only_preflight']['graph_cpu_estimated_seconds_with_allowance']
    plan['cost_only_preflight']['graph_cpu_estimated_seconds_with_allowance']=proof['rehydration_cost_allowance_seconds']
    plan['cost_only_preflight']['reuse_scope']='Immutable completed current-task CPU graph stage; global ledger still charges its full original time'
    dest=HERE/'plans'/f'E6_JENA_STAGED_{tag}.json';write(dest,plan)
    return dest


def run(path,digest,prepared_path,prepared_digest,gate_path,gate_digest,tag):
    proof=check_prepared(prepared_path,prepared_digest)
    plan=read(path)
    if plan.get('completed_CPU_stage',{}).get('sha256')!=prepared_digest:
        raise ValueError('GPU plan bound to another CPU stage')
    def selected(records,pairs,budget):
        with np.load(proof['fixed_graph_arrays']['path'],allow_pickle=False) as z:
            fixed=[{k.removeprefix(f'record{i}_'):z[k] for k in z.files if k.startswith(f'record{i}_')}
                   for i in range(len(records))]
        budget.check()
        return proof['selected_graph_lambda'],fixed,proof['TUNE_graph_grid']
    def precomputed(records,pairs,fixed,lam,cache,output,budget):
        blocks={}
        for s,ref in proof['graph_arrays'].items():
            with np.load(ref['path'],allow_pickle=False) as z:
                if not np.array_equal(z['record_origin_pairs'],pairs[s]):
                    raise ValueError('Rehydrated origin order changed')
                blocks[s]={'history':z['input_history'],
                    'messages':{mode:z[f'{mode}_neighbor_message'] for mode in impl.MODES[1:]}}
            budget.check()
        return blocks,proof['graph_arrays'],proof['graph_diagnostics']
    def baselines(*args):
        args[-1].check()
        return proof['matched_baselines']
    impl.selected_graph_lambda=selected
    impl.precompute=precomputed
    impl.matched_baselines=baselines
    impl.pins=lambda:{**pins(),str(prepared_path):prepared_digest}
    result=impl.run_balanced(path,digest,gate_path,gate_digest,tag)
    write(result.parent/'COMPLETED_CPU_STAGE.json',{'prepared_path':str(prepared_path),
        'prepared_sha256':prepared_digest,'CPU_preparation_ledger':proof['CPU_preparation_ledger'],
        'result_path':str(result),'result_sha256':sha(result),
        'reuse':'Graph inputs/fixed selection/matched baseline are immutable completed CPU stage; neural fitting new600x8',
        'no_historical_failed_result_reuse':True})
    return result


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for name in ('prepare','plan','run'):g.add_argument('--'+name,action='store_true')
    p.add_argument('--tag',required=True);p.add_argument('--frozen-plan',type=Path);p.add_argument('--plan-sha256')
    p.add_argument('--prepared',type=Path);p.add_argument('--prepared-sha256');p.add_argument('--cpu-allocation-seconds',type=float,default=230)
    p.add_argument('--gate',type=Path);p.add_argument('--gate-sha256');a=p.parse_args();access_guard()
    if a.prepare:
        dest=prepare(a.frozen_plan,a.plan_sha256,a.tag,a.cpu_allocation_seconds)
    elif a.plan:
        with Budget(2,False,'E6 Jena immutable-stage derived plan '+a.tag):
            dest=derived_plan(a.prepared,a.prepared_sha256,a.tag)
    else:
        dest=run(a.frozen_plan,a.plan_sha256,a.prepared,a.prepared_sha256,a.gate,a.gate_sha256,a.tag)
    print({'path':str(dest),'sha256':sha(dest)},flush=True)


if __name__=='__main__':main()
