"""Finite raw/context experiment; all selection precedes new V evaluation."""
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import time
import uuid

from . import common


def registered_plan_path(ctx):
    for name in ('EXPERIMENT_PLAN_REPAIR_002.json','EXPERIMENT_PLAN_REPAIR_001.json','EXPERIMENT_PLAN.json'):
        path = ctx['report']/name
        if path.exists():
            return path
    return ctx['report']/'EXPERIMENT_PLAN.json'


def _execution_repair_evidence(ctx,plan):
    repair = plan.get('repair',{})
    original = ctx['cache']/'CORE_EXECUTION_START.json'
    terminal = ctx['cache']/'TERMINAL_CORE_001.json'
    if (repair.get('correction')!='LEGACY_PARENT_LOADING_SCHEMA_ONLY'
            or repair.get('preserve_execution_start_and_deadline') is not True
            or repair.get('GPU_profile_output_id')!='GPU_REPAIR_002'
            or not original.exists() or not terminal.exists()
            or repair.get('original_execution_start_binding')!=common.binding(original,ctx)
            or repair.get('failed_attempt_terminal')!=common.binding(terminal,ctx)):
        raise ValueError('The execution repair requires the original start, failed attempt and fresh profile identity')
    started,failed = common.read(original),common.read(terminal)
    if (started.get('status')!='STARTED' or started.get('plan_binding')!=plan['supersedes']
            or started.get('limits')!=plan['limits']
            or failed.get('status')!='FAILED' or failed.get('exit_code')!=1):
        raise ValueError('The execution repair must preserve the failed attempt and original execution budget')
    return started


def load_registered_plan(ctx):
    original = ctx['report']/'EXPERIMENT_PLAN.json'
    previous_path,previous = original,common.read(original)
    repair_paths = [ctx['report']/f'EXPERIMENT_PLAN_REPAIR_{number:03}.json' for number in (1,2)]
    if repair_paths[1].exists() and not repair_paths[0].exists():
        raise ValueError('The second repair requires the preserved first repair registration')
    for number,path in enumerate(repair_paths,1):
        if not path.exists():
            continue
        plan = common.read(path)
        revision_keys = {'registered_unix','source_files','supersedes','repair'}
        repair = plan.get('repair',{})
        if (plan.get('supersedes')!=common.binding(previous_path,ctx)
                or {k:v for k,v in plan.items() if k not in revision_keys}
                   !={k:v for k,v in previous.items() if k not in revision_keys}
                or set(plan['source_files'])!=set(previous['source_files'])
                or repair.get('status')!='REGISTERED_BEFORE_NEW_BODY_SEARCH'
                or repair.get('scientific_contract_changed') is not False
                or repair.get('new_body_searches_or_V_scores')!=0
                or (number==1 and repair.get('RawContext_GPU_sessions_started')!=0)):
            raise ValueError('Each repair must preserve the original scientific contract and registration')
        changed = {name for name,digest in previous['source_files'].items() if plan['source_files'][name]!=digest}
        allowed_names = ('run.py','report.py') if number==1 else ('models.py','run.py','deployment.py')
        allowed = {'research/'+common.NAME+'/'+name for name in allowed_names}
        if changed!=set(repair.get('changed_source_paths',[])) or not changed or not changed<=allowed:
            if number==1:
                raise ValueError('Only the registered reporting and plan-reader repair is admitted')
            raise ValueError('Only the registered legacy-parent schema and execution repair is admitted')
        if number==2:
            _execution_repair_evidence(ctx,plan)
        previous_path,previous = path,plan
    return previous


def make_plan(ctx):
    path = ctx['report']/'EXPERIMENT_PLAN.json'
    if path.exists():
        return load_registered_plan(ctx)
    first_audit = [json.loads(line)['time_unix'] for line in (ctx['cache']/'EVENTS.jsonl').read_text().splitlines()
                   if json.loads(line)['stage']=='audit']
    return common.sealed_json(path, dict(
        status='REGISTERED_BEFORE_NEW_V', campaign=common.NAME,
        registered_unix=time.time(), started_unix=min(first_audit),
        primary_metric='normalized original-channel observed equal-channel MSE',
        secondary_metric='MAE', seeds=list(common.SEEDS), tasks=common.EXPECTED,
        new_body_configurations=6, Jena_modes=['A','B'], PeMS_mode_selection='Jena two-seed mean A_stop; exact tie A',
        recipe=dict(common.RECIPE,optimizer='AdamW',weight_decay=0,gradient_clip=1,core_dtype='float32',outer_RevIN=False),
        search_endpoints=dict(jena_initial=[0,.25,1,3],jena_extended=[0,.25,1,3,4,5,6],
                              pems_initial=[0,.25,1,3,4,5,6],pems_extended=[0,.25,1,3,4,5,6,9,12]),
        extension='A_stop strict-cap minimum; admit entire corresponding group plus fresh refits and reporting reserve',
        limits=dict(wall_seconds=21600,GPU_reservation_seconds=14400,report_fraction=.20),
        budget_wall_clock_anchor='EXECUTION_START.started_unix after preceding dataset remote-main publication; preparation included in overall elapsed',
        tolerance=dict(atol=common.ATOL,rtol=common.RTOL,shape_index_hash='exact'),
        calibration=dict(M_fit_fraction=.60,selection_split='M_select',epochs=[0,.25,1,3],
                         parents='original completed JenaK12cap3 / PeMSK8OFFcap6; single G shared across two S seeds',
                         fresh_refit='entire M_cal at M_select selected epoch',V_used_for_selection=False),
        source_document_sha256=common.sha(Path.home()/'Downloads/CLI_Raw_Context_Multiscale_20261018_KO.md'),
        T_access=False,V_independent_test=False,prior_results_and_presentations='preserve',
        execution_precondition='One additional independently checked development example published and remote-main verified',
        source_files={p.relative_to(ctx['repo']).as_posix():common.sha(p)
                      for p in sorted((ctx['repo']/'research'/common.NAME).glob('*.py'))}))


def check_plan(ctx,path):
    path = Path(path).resolve()
    actual_path = registered_plan_path(ctx)
    allowed = {(ctx['report']/name).resolve() for name in (
        'EXPERIMENT_PLAN.json','EXPERIMENT_PLAN_REPAIR_001.json','EXPERIMENT_PLAN_REPAIR_002.json')
        if (ctx['report']/name).exists()}
    if path not in allowed:
        raise ValueError('Use the actual preregistered plan in this campaign')
    plan = load_registered_plan(ctx)
    if plan.get('status')!='REGISTERED_BEFORE_NEW_V' or plan.get('T_access') is not False:
        raise ValueError('The registered protected-data plan is required')
    for name,digest in plan['source_files'].items():
        if common.sha(ctx['repo']/name)!=digest:
            raise ValueError('Preserve the plan and register a repair before a retry: '+name)
    ctx['plan'] = plan
    ctx['plan_path'] = actual_path
    ctx['deadline'] = 1e12
    ctx['scientific_source_hashes'] = common.model_sources(ctx['repo'])
    return plan


def require_preceding_publication(ctx):
    proof = common.read(ctx['cache']/'PRECEDING_DATASET_PUBLICATION.json')
    if (proof.get('status')!='PUSHED_AND_REMOTE_VERIFIED' or proof.get('independent_positive_verified') is not True
            or not isinstance(proof.get('commit'),str) or len(proof['commit'])!=40):
        raise ValueError('Finish and verify the requested additional dataset publication first')
    return proof


def start_execution(ctx,proof):
    """Seal the ordered work package start once; resumed runs retain its deadline."""
    private = ctx['cache']/'CORE_EXECUTION_START.json'
    expected = dict(plan_binding=common.binding(registered_plan_path(ctx),ctx),
                    publication_binding=common.binding(ctx['cache']/'PRECEDING_DATASET_PUBLICATION.json',ctx),
                    preceding_commit=proof['commit'],limits=ctx['plan']['limits'])
    if private.exists():
        value = common.read(private)
        if ctx['plan'].get('repair',{}).get('correction')=='LEGACY_PARENT_LOADING_SCHEMA_ONLY':
            _execution_repair_evidence(ctx,ctx['plan'])
            expected['plan_binding'] = ctx['plan']['supersedes']
        if value.get('status')!='STARTED' or any(value.get(key)!=item for key,item in expected.items()):
            raise ValueError('The original execution start or publication identity changed')
    else:
        value = common.sealed_json(private,dict(status='STARTED',started_unix=time.time(),**expected))
    common.sealed_json(ctx['report']/'EXECUTION_START.json',value)
    ctx['execution_start'] = value
    ctx['deadline'] = value['started_unix']+value['limits']['wall_seconds']
    if time.time()>=ctx['deadline']:
        raise TimeoutError('The original core execution wall budget is exhausted; preserve partial stages')
    repair = ctx['plan'].get('repair',{})
    if repair.get('correction')=='LEGACY_PARENT_LOADING_SCHEMA_ONLY':
        common.sealed_json(ctx['report']/'EXECUTION_REPAIR_ADMISSION_002.json',dict(
            status='ADMITTED_WITH_ORIGINAL_EXECUTION_BUDGET',
            active_plan_binding=common.binding(registered_plan_path(ctx),ctx),
            original_execution_start_binding=repair['original_execution_start_binding'],
            failed_attempt_terminal=repair['failed_attempt_terminal'],
            started_unix=value['started_unix'],deadline_unix=ctx['deadline'],limits=value['limits'],
            GPU_profile_output_id=repair['GPU_profile_output_id']))
    return value


def gpu_seconds(ctx):
    journal = ctx['cache']/'EVENTS.jsonl'
    if not journal.exists():
        return 0.
    total,started,closed = 0.,{},set()
    for line in journal.read_text().splitlines():
        row = json.loads(line)
        if row.get('stage')!='GPU_SESSION':
            continue
        session = row['session_id']
        if row['status']=='STARTED':
            started[session] = row
        elif row['status'] in ('COMPLETE','FAILED'):
            if session in closed:
                raise ValueError('Duplicate GPU session completion')
            total += float(row['wall_seconds'])
            closed.add(session)
    for session,row in started.items():
        if session not in closed:
            total += max(0.,min(time.time(),row['budget_deadline_unix'])-row['time_unix'])
    return total


@contextmanager
def gpu_session(ctx,label):
    started = time.time()
    limits = ctx.get('plan',{}).get('limits',dict(GPU_reservation_seconds=14400))
    remaining = limits['GPU_reservation_seconds']-gpu_seconds(ctx)
    previous_deadline = ctx['deadline']
    deadline = min(previous_deadline,started+remaining)
    if deadline<=started:
        raise TimeoutError('The registered GPU/wall budget is exhausted')
    session_id = uuid.uuid4().hex
    common.event(ctx,'GPU_SESSION','STARTED',label=label,session_id=session_id,budget_deadline_unix=deadline)
    ctx['deadline'] = deadline
    status = 'FAILED'
    try:
        yield
        status = 'COMPLETE'
    finally:
        ctx['deadline'] = previous_deadline
        common.event(ctx,'GPU_SESSION',status,label=label,session_id=session_id,wall_seconds=time.time()-started,
                     accounting='GPU-owned process wall upper bound; not measured kernel busy time')


def _budget(ctx,seconds,*,reserve=True):
    plan = ctx['plan']
    held = plan['limits']['wall_seconds']*plan['limits']['report_fraction'] if reserve else 0.
    wall_left = ctx['deadline']-time.time()-held
    gpu_left = plan['limits']['GPU_reservation_seconds']-gpu_seconds(ctx)
    return dict(admitted=seconds<=min(wall_left,gpu_left),estimated_GPU_seconds=float(seconds),
                wall_remaining_after_reserve=wall_left,GPU_remaining=gpu_left,report_reserve_seconds=held)


def _seconds_per_update(profiles,dataset,mode):
    rows = [row for row in profiles if row['dataset']==dataset and row['mode']==mode]
    if len(rows)!=1 or rows[0]['seconds_per_update']<=0:
        raise ValueError('An actual local profile is required for each admitted model')
    return float(rows[0]['seconds_per_update'])


def estimate(ctx,tasks,profiles,requests):
    """Conservative full-origin search/refit estimate; requests are (data,mode,epochs,fit_end)."""
    total = 0.
    import math
    for dataset,mode,epochs,fit_end in requests:
        task = tasks[dataset]
        n = len(ctx['modules']['data'].origins((0,fit_end),task['L'],task['H'],1))
        updates = math.ceil(math.ceil(float(epochs)*n)/16)
        total += updates*_seconds_per_update(profiles,dataset,mode)
    return 1.25*total+120.


def _decision(ctx,dataset,cap,searches,**details):
    value = dict(status='SEALED',dataset=dataset,cap=cap,selection_split='A_stop',V_used=False,
                 searches=[result['checkpoint_binding'] for result in searches],
                 selections=[dict(run_id=r['spec']['run_id'],seed=r['spec']['seed'],
                    mode=r['spec']['scientific_identity']['raw_context_mode'],epoch=r['selected_epoch'],
                    astop_mse=r['summary']['selected_astop_mse']) for r in searches],**details)
    path = ctx['report']/'decisions'/f'{dataset}_cap{cap}.json'
    if path.exists():
        previous = common.read(path)
        # The first admission records the actual time available at that decision.
        # A resumed process must preserve that historical admission unchanged.
        if 'budget' in previous and 'budget' in value:
            value['budget'] = previous['budget']
    common.sealed_json(path,value)
    return common.binding(path,ctx)


def _require_complete(result):
    if result['status']!='COMPLETE':
        raise RuntimeError('Checkpointed partial stage; resume the same command without deleting outputs')
    return result


def _release(ctx):
    ctx['foundation'].assert_frozen_unchanged()
    del ctx['foundation']
    gc.collect()
    ctx['modules']['torch'].cuda.empty_cache()


def execute(ctx,plan_path):
    from . import audit,training,gates,pipeline
    from research.adapter_improvements_20261018_v1.audit import load_foundation
    check_plan(ctx,plan_path)
    proof = require_preceding_publication(ctx)
    start_execution(ctx,proof)
    inventory,tasks = audit.prepare(ctx)
    profile_id = ctx['plan'].get('repair',{}).get('GPU_profile_output_id','GPU')
    smoke_path = ctx['report']/f'VALIDATION_PROFILE_GPU_{profile_id}.json'
    if not smoke_path.exists():
        with gpu_session(ctx,'smoke_and_profile'):
            gates.run_gates(ctx,device='cuda',profile=True,output_id=profile_id)
    smoke = common.read(smoke_path)
    if smoke['status']!='PASS':
        raise ValueError('Actual-F alignment/initialization/gradient/parity gates must pass')
    profiles = smoke['profiles']
    queue = []
    minimum = [('jena',m,3,tasks['jena']['split_ranges']['A_fit'][1]) for m in ('A','B') for _ in common.SEEDS]
    minimum += [('jena',m,3,12579) for m in ('A','B') for _ in common.SEEDS]
    worst_pems_mode = max(('A','B'),key=lambda m:_seconds_per_update(profiles,'pems32',m))
    minimum += [('pems32',worst_pems_mode,6,8928),('pems32',worst_pems_mode,6,8928),
                ('pems32',worst_pems_mode,6,10713),('pems32',worst_pems_mode,6,10713)]
    admission = _budget(ctx,estimate(ctx,tasks,profiles,minimum)+900.)
    admission_name = 'INITIAL_ADMISSION_REPAIR_002.json' if profile_id=='GPU_REPAIR_002' else 'INITIAL_ADMISSION.json'
    admission_path = ctx['report']/admission_name
    if admission_path.exists():
        admission = common.read(admission_path)
        current = _budget(ctx,0.,reserve=False)
        if not current['admitted']:
            raise TimeoutError('The registered wall/GPU budget has expired; preserve partial stages')
    else:
        common.sealed_json(admission_path,admission)
    if not admission['admitted']:
        raise TimeoutError('Insufficient remaining budget for the complete matched initial group and reporting reserve')
    selections,all_searches = {},{}
    # Fixed original-parent calibration does not read V and does not gate A/B.
    pipeline.calibrate(ctx,tasks,inventory)
    for dataset in ('jena','pems32'):
        task = tasks[dataset]
        active = dict(ctx,task=task,device='cuda')
        with gpu_session(ctx,'search_'+dataset):
            active['deadline'] = min(active['deadline'],ctx['deadline'])
            active['foundation'] = load_foundation(active,active['modules'],task['H'],'cuda')
            group = [training.reuse_S_search(active,seed,audit.old_s_root(ctx,dataset,seed)/'SEARCH_RESULT.json')
                     for seed in common.SEEDS]
            modes = ('A','B') if dataset=='jena' else (transfer['selected_mode'],)
            endpoints = common.INITIAL_EPOCHS if dataset=='jena' else common.EXTENDED_EPOCHS
            for mode in modes:
                for seed in common.SEEDS:
                    print('SEARCH '+dataset+' '+mode+' '+str(seed),flush=True)
                    group.append(_require_complete(training.search(active,mode,seed,endpoints)))
            cap = int(endpoints[-1])
            needs = any(training.strict_last_best(result['curves'],cap) for result in group)
            extended = common.EXTENDED_EPOCHS if dataset=='jena' else common.PEMS_EXTENDED_EPOCHS
            extension = [(dataset,r['spec']['scientific_identity']['raw_context_mode'],int(extended[-1])-cap,
                          task['split_ranges']['A_fit'][1]) for r in group]
            extension += [(dataset,r['spec']['scientific_identity']['raw_context_mode'],int(extended[-1]),
                           task['split_ranges']['A_stop'][1]) for r in group]
            if dataset=='jena':
                extension += [('pems32',worst_pems_mode,6,8928),('pems32',worst_pems_mode,6,8928),
                              ('pems32',worst_pems_mode,6,10713),('pems32',worst_pems_mode,6,10713)]
            else:
                # Jena's selected fresh fits are still pending at the PeMS gate.
                extension += [('jena',r['spec']['scientific_identity']['raw_context_mode'],r['selected_epoch'],12579)
                              for r in all_searches['jena'] if not r.get('reused_parent')]
            admission = _budget(active,estimate(ctx,tasks,profiles,extension)+900.)
            will_extend = needs and admission['admitted']
            prior_decision = ctx['report']/'decisions'/f'{dataset}_cap{cap}.json'
            if prior_decision.exists():
                will_extend = common.read(prior_decision)['extend_entire_group']
            decision_binding = _decision(ctx,dataset,cap,group,needs_extension=needs,extend_entire_group=will_extend,budget=admission)
            admission = common.read(common.resolve_binding(decision_binding,ctx))['budget']
            if will_extend:
                for index,result in enumerate(group):
                    mode = result['spec']['scientific_identity']['raw_context_mode']
                    seed = result['spec']['seed']
                    if result.get('reused_parent'):
                        training.import_S_parent(active,result)
                    group[index] = _require_complete(training.search(active,mode,seed,extended))
                cap = int(extended[-1])
                _decision(ctx,dataset,cap,group,needs_extension=any(training.strict_last_best(r['curves'],cap) for r in group),
                          extend_entire_group=False,limit='Jena maximum6; PeMS maximum12')
            elif needs:
                queue.append(dict(dataset=dataset,scope='entire matched group',reason='BUDGET_LIMITED_CONVERGENCE_UNRESOLVED',
                                  requested_cap=int(extended[-1]),budget=admission))
            all_searches[dataset] = group
            selections[dataset] = dict(cap=cap,group_modes=['S',*modes],extension_required=needs,
                                       extension_executed=will_extend)
            if dataset=='jena':
                transfer = training.choose_transfer([r for r in group if r['spec']['scientific_identity']['raw_context_mode']!='S'])
                common.sealed_json(ctx['report']/'TRANSFER_SELECTION.json',dict(status='SEALED',**transfer,
                    searches=[r['checkpoint_binding'] for r in group],Jena_V_opened=False))
            _release(active)
    seal = dict(status='SEALED',selection_split='A_stop',V_used=False,transfer=transfer,
                groups=selections,searches=[r['checkpoint_binding'] for group in all_searches.values() for r in group],
                resume_queue=queue)
    decision_path = ctx['report']/'SELECTIONS_SEALED.json'
    common.sealed_json(decision_path,seal)
    selected_binding = common.binding(decision_path,ctx)
    finals = []
    for dataset,group in all_searches.items():
        active = dict(ctx,task=tasks[dataset],device='cuda')
        with gpu_session(ctx,'fresh_refit_'+dataset):
            active['deadline'] = min(active['deadline'],ctx['deadline'])
            active['foundation'] = load_foundation(active,active['modules'],tasks[dataset]['H'],'cuda')
            for result in group:
                mode = result['spec']['scientific_identity']['raw_context_mode']
                if mode=='S' and result.get('reused_parent'):
                    parent = next(p for p in inventory['parents'] if p.get('role')=='S' and p['dataset']==dataset
                                  and p['seed']==result['spec']['seed'])
                    finals.append(dict(dataset=dataset,mode=mode,seed=parent['seed'],run_id=result['spec']['run_id'],
                                       route_spec=parent['route_spec'],reused=True,search=result,parent=parent))
                else:
                    print('FRESH_REFIT '+result['spec']['run_id'],flush=True)
                    refit = _require_complete(training.refit(active,result,selected_binding))
                    path = common.resolve_binding(refit['bundle_binding'],ctx)
                    route = dict(family='B2',run_id=result['spec']['run_id'],latent_k=tasks[dataset]['K'],
                                 artifact_cache_relative_path=path.relative_to(ctx['cache']).as_posix(),
                                 artifact_sha256=common.sha(path),canonical_fit_end=tasks[dataset]['split_ranges']['A_stop'][1])
                    finals.append(dict(dataset=dataset,mode=mode,seed=result['spec']['seed'],run_id=result['spec']['run_id'],
                                       route_spec=route,reused=False,search=result,refit=refit))
            _release(active)
    common.sealed_json(ctx['cache']/'FINAL_BODY_MODELS.json',finals)
    common.sealed_json(ctx['report']/'MODELS_SEALED_BEFORE_V.json',dict(status='SEALED',selection_binding=selected_binding,
        V_used=False,finals=[dict(run_id=r['run_id'],route_spec=r['route_spec'],reused=r['reused']) for r in finals]))
    pipeline.complete(ctx,tasks,inventory,finals)
    return dict(status='COMPLETE',resume_queue=queue)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('audit','smoke','execute','verify','report'))
    parser.add_argument('--asset-root',required=True,type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args = parser.parse_args(argv)
    ctx = common.context(args.asset_root,device=args.device)
    if args.stage=='audit':
        from .audit import prepare
        prepare(ctx)
        value = make_plan(ctx)
    elif args.stage=='smoke':
        from .gates import run_gates
        if args.device=='cuda':
            check_plan(ctx,ctx['report']/'EXPERIMENT_PLAN.json')
            proof = require_preceding_publication(ctx)
            start_execution(ctx,proof)
            with gpu_session(ctx,'smoke_and_profile'):
                value = run_gates(ctx,device='cuda',profile=True,
                                  output_id=ctx['plan'].get('repair',{}).get('GPU_profile_output_id','GPU'))
        else:
            value = run_gates(ctx,device='cpu',profile=False,output_id='CPU')
    elif args.stage=='execute':
        if args.plan is None:
            parser.error('execute requires the actual --plan path')
        value = execute(ctx,args.plan)
    else:
        from . import report
        value = report.verify(ctx) if args.stage=='verify' else report.generate(ctx)
    print(json.dumps(dict(status=value.get('status','REGISTERED'),stage=args.stage),ensure_ascii=False))


if __name__=='__main__':
    main()
