"""Continue the predeclared DIAG_SUBSET on the six unexecuted frozen assets."""
import argparse
import gc
import math
import time
import psutil
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import e34 as impl
from common import *
from models_io import *

JOBS = [('jena',320,s) for s in SEEDS] + [('robin',d,s) for d in (16,320) for s in SEEDS]


def pins():
    return {str(p):sha(p) for p in (Path(__file__),HERE/'e34.py',HERE/'common.py',HERE/'models_io.py')}


def remaining():
    return 5400-sum(read(p)['wall_seconds'] for p in (HERE/'resources').glob('*.json')
                    if read(p)['gpu'] and read(p)['phase']==1)


def reserve_storage(jobs):
    current=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())
    estimate=sum(128*(2*L+H)*dim*4 + 64*1024**2 for ds,dim,seed in jobs)
    free=shutil.disk_usage(ROOT).free
    if current+estimate>4*1024**3 or free-estimate<2*1024**3:
        raise BudgetHeld('Remaining raw arrays reservation exceeds cache/disk protection')
    return {'current_cache_bytes':current,'new_uncompressed_reservation_bytes':estimate,'disk_free_bytes':free}


def exclusive(gate_path, gate_hash):
    if sha(gate_path)!=gate_hash:
        raise ValueError('Dispatch gate bytes changed')
    gate=read(gate_path)
    if gate.get('experiment')!='E34_REMAINING' or not all(gate.get(k) is True for k in
            ('execute','legacy_gpu_clear','single_gpu_dispatch_authorized','safety_pass')):
        raise ValueError('Missing explicit exclusive GPU gate')
    for record in gate['legacy_processes']:
        try:
            p=psutil.Process(record['pid'])
            if abs(p.create_time()-record['create_time'])<.01 and p.is_running():
                raise ValueError('Preserved legacy process is still running')
        except psutil.NoSuchProcess:
            pass
    for p in (HERE/'resources').glob('*.json'):
        r=read(p)
        if r['gpu'] and r['status']=='RUNNING_CONFIRMED' and psutil.pid_exists(r['pid']):
            raise ValueError('A new GPU experiment is already running')
    return gate


def timed(call, budget):
    call(); torch.cuda.synchronize()
    start=time.perf_counter(); result=call(); torch.cuda.synchronize()
    seconds=time.perf_counter()-start; budget.check()
    return result,seconds


def profile(tag,budget,bolt):
    rows=[];storage=reserve_storage(JOBS)
    for ds,dim in [('jena',320),('robin',16),('robin',320)]:
        data=load_data(ds,frozen=True,sizes=(64,32,32))
        model,record=load_ts2(ds,dim,SEEDS[0],'cuda')
        x=windows(data,'fit')[:32]
        second,second_record=load_ts2(ds,dim,SEEDS[1],'cuda')
        row={'dataset':ds,'dim':dim,'checkpoint':record,'assets':[record,second_record],
             'data_source':str(data['source']),'data_source_hash':data['source_hash'],
             'origin_hashes':{s:ahash(o) for s,o in data['origins'].items()},'samples':[]}
        del second;torch.cuda.empty_cache()
        for n in (1,32):
            full,tf=timed(lambda:encode_full(model,x[:n],'cuda',batch=32),budget)
            prefix,tp=timed(lambda:encode_prefix(model,x[:n],'cuda',batch=32),budget)
            _,tb=timed(lambda:bolt.forecast(prefix),budget)
            ext=np.concatenate((x[:n],x[:n,:48]),axis=1)
            _,tr=timed(lambda:encode_prefix(model,ext,'cuda',batch=32,first_tick=512),budget)
            row['samples'].append({'batch':n,'FULL':tf,'PREFIX':tp,'BOLT':tb,'REFERENCE48':tr})
            del full,prefix
        one,block=row['samples']
        counts={k:len(v) for k,v in data['origins'].items()}
        # Both FULL/PREFIX benchmark: 25 repeats x B1/B32. Reference covers FIT/TUNE/VAL.
        export=math.ceil(sum(counts.values())/32)*(block['FULL']+block['PREFIX']+block['REFERENCE48'])
        inference=3*math.ceil(counts['val']/32)*block['BOLT']
        bench=25*(one['FULL']+one['PREFIX']+2*one['BOLT']+block['FULL']+block['PREFIX']+2*block['BOLT'])
        row['cost_estimate_seconds']=2*(export+inference+bench+120)
        row['origin_counts']=counts
        row['uncompressed_export_bytes_upper']=sum(counts.values())*(2*L+H)*dim*4
        rows.append(row)
        del model,x;gc.collect();torch.cuda.empty_cache();budget.check()
    estimate=2*sum(r['cost_estimate_seconds'] for r in rows)
    selected=[];held=[];reserved=0
    available=remaining()
    for row in rows:
        if reserved+2*row['cost_estimate_seconds']<available*.85:
            selected.append(row);reserved+=2*row['cost_estimate_seconds']
        else:
            held.append({'dataset':row['dataset'],'dim':row['dim'],'status':'BUDGET_HELD',
                         'reason':'Measured two-seed cost plus already reserved bundles exceeds phase1 allowance'})
    chosen_jobs=[job for row in selected for job in JOBS if job[0]==row['dataset'] and job[1]==row['dim']]
    scope={'status':'COST_ONLY_SCOPE_FROZEN','scope':'DIAG_SUBSET_REMAINING6',
           'reason':'Continue the originally selected uniform32 subset; prior CPU Jena d16 pair is preserved',
           'jobs':chosen_jobs,'requested_jobs':JOBS,'sizes':[64,32,32],'device':'cuda','encoder_batch':32,
           'measurements':rows,'selected_measurements':selected,'held_bundles':held,
           'storage_reservation':storage,
           'estimated_all6_seconds':estimate,'estimated_balanced_seconds':reserved,'remaining_gpu_phase1_seconds':available,
           'pair_fits_remaining':bool(selected),'science_metrics_seen_for_these_cases':False,
           'cost_only_selection_order':'Jena320 pair, Robin16 pair, Robin320 pair; skip only a measured unaffordable pair, try independent next pair',
           'safety_gate_sha':sha(HERE/'E1_GATE.json'),'source_pins':pins(),
           'uncertainty':'One warm/one timed profile; factor2 margin +120s per case for unmeasured decoder/ridge/IO. Not an upper bound.',
           'decisions':{'better':'Retain exact-prefix extraction as a diagnostic candidate',
                        'unchanged_or_worse':'No value established in this historically exposed frozen encoder',
                        'stop':'Source/asset/label/finite/memory/exclusive GPU/global phase1 limit'}}
    path=HERE/'plans'/f'E34_REMAINING_{tag}.json';write(path,scope)
    return path,scope


def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--plan',action='store_true');g.add_argument('--run',action='store_true')
    parser.add_argument('--tag',required=True);parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--gate-sha256',required=True);parser.add_argument('--frozen-plan',type=Path)
    parser.add_argument('--plan-sha256');args=parser.parse_args()
    access_guard();exclusive(args.gate,args.gate_sha256)
    if not read(HERE/'E1_GATE.json')['safety_pass']:
        raise ValueError('E1 safety failed')
    impl.DEVICE='cuda';torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    with threadpool_limits(2),Budget(1,True,'E34 remaining6 '+args.tag) as budget:
        bolt=Bolt('cuda');parity=bolt.parity()
        if args.plan:
            path,decision=profile(args.tag,budget,bolt)
            print({'plan':str(path),'sha256':sha(path),'fits':decision['pair_fits_remaining'],
                   'estimate_seconds':decision['estimated_balanced_seconds']},flush=True)
            return
        if args.frozen_plan is None or sha(args.frozen_plan)!=args.plan_sha256:
            raise ValueError('Frozen cost plan missing or changed')
        decision=read(args.frozen_plan)
        if any(sha(p)!=v for p,v in decision['source_pins'].items()):
            raise ValueError('Frozen implementation changed')
        if not decision['pair_fits_remaining'] or decision['estimated_balanced_seconds']>=remaining()*.9:
            raise BudgetHeld('Cost forecast does not fit remaining phase1 allocation')
        startpins=pins();results=[]
        if sha(HERE/'E1_GATE.json')!=decision['safety_gate_sha']:
            raise ValueError('E1 safety evidence changed after cost profile')
        for index,row in enumerate(decision['selected_measurements']):
            suffix=[job for rr in decision['selected_measurements'][index:] for job in decision['jobs']
                    if job[0]==rr['dataset'] and job[1]==rr['dim']]
            reserve_storage(suffix);budget.check()
            forecast=2*sum(rr['cost_estimate_seconds'] for rr in decision['selected_measurements'][index:])
            if forecast>=remaining()*.9:
                raise BudgetHeld('Residual cost does not fit before next balanced two-seed bundle')
            ds,dim=row['dataset'],row['dim']
            data=load_data(ds,frozen=True,sizes=(64,32,32))
            if str(data['source'])!=row['data_source'] or data['source_hash']!=row['data_source_hash'] or any(
                    ahash(o)!=row['origin_hashes'][s] for s,o in data['origins'].items()):
                raise ValueError('Data/origins differ from cost-profile scope')
            for asset in row['assets']:
                model,current=load_ts2(ds,dim,asset['seed'],'cpu')
                if any(jsonable(current[k])!=asset[k] for k in ('checkpoint','checkpoint_sha256','result','result_sha256','SWA_state_sha256')):
                    raise ValueError('Frozen encoder replaced after cost profile')
                del model
            for seed in SEEDS:
                result=impl.run_one(ds,dim,seed,'DIAG_SUBSET_REMAINING6',bolt,budget,args.tag)
                results.append(str(HERE/'e34'/f'{args.tag}_{ds}_{dim}_{seed}.json'))
                print(f'E34 {ds} d{dim} seed={seed} COMPLETE_SAVED',flush=True)
        if startpins!=pins() or state_digest(bolt.model)!=bolt.initial:
            raise ValueError('Frozen source/Bolt changed')
        write(HERE/'e34'/f'SUMMARY_{args.tag}.json',{'status':'COMPLETE_UNVERIFIED','jobs':decision['jobs'],
            'result_paths':results,'cost_scope':decision,'Bolt_parity':parity,'Bolt_unchanged':True,
            'GPU_ledger':str(budget.path),'source_pins':pins(),'dispatch_gate_sha256':args.gate_sha256})


if __name__=='__main__':
    main()
