"""Registered minimum campaign; no-argument invocation prints help only."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time

from . import contracts

REPO=Path(__file__).resolve().parents[2]


def sources(repo):
    base=Path(repo)/'research'/contracts.NAME
    return {path.relative_to(repo).as_posix():contracts.sha(path)
            for path in [base/name for name in ('contracts.py','revin.py','deployment.py','training_jobs.py')]}


def context(asset_root, repo=REPO):
    repo=Path(repo).resolve()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='2'
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    ctx=dict(repo=repo,asset=Path(asset_root).resolve(),cache=repo/'.cache'/contracts.NAME,
             report=repo/'reports'/contracts.NAME,deadline=1e12,scientific_source_hashes=sources(repo))
    if not (ctx['cache']/'CAMPAIGN.json').exists():
        raise ValueError('Persistent campaign ledger must exist; do not reset start/ancestry')
    from research.adapter_improvements_20261018_v1.audit import load_modules,load_task,load_foundation
    ctx['modules']=load_modules(ctx)
    ctx['task_loader']=lambda context,modules,cohort:load_task(context,cohort,modules)
    ctx['foundation_loader']=load_foundation
    return ctx


def make_plan(ctx):
    jobs=[]
    for cohort in contracts.COHORTS:
        for enabled in (False,True):
            for seed in contracts.SEEDS:
                jobs.append(dict(job_id=contracts.job_id(cohort,enabled,seed),cohort=cohort,seed=seed,
                    family='B2',revin=enabled,K=6 if cohort=='jena' else 8,device='cuda',
                    initial_action='NEW_ON' if enabled else 'EXACT_OFF_REUSE',conditional=False))
    for enabled in (False,True):
        for seed in contracts.SEEDS:
            jobs.append(dict(job_id=contracts.job_id('jena',enabled,seed,'B0'),cohort='jena',seed=seed,
                family='B0',revin=enabled,K=6,device='cuda',conditional=True,
                condition='Jena A_stop-selected common normalization; only its two seeds run'))
    value=dict(campaign=contracts.NAME,source_hashes=ctx['scientific_source_hashes'],jobs=jobs,
        initial_epochs=list(contracts.INITIAL_EPOCHS),extended_epochs=list(contracts.EXTENDED_EPOCHS),
        recipe=dict(contracts.RECIPE,optimizer='AdamW',weight_decay=0,clip=1,core_dtype='float32'),
        extension='Any group arm strictly best at cap3 triggers all four arms through epoch6',
        selection='A_fit training, strict minimum A_stop MSE and earlier tie; fresh A60 refit before repeated V',
        normalization='Two-seed minimum A_stop mean; tie OFF',
        excluded=['SOFTS','FreTS','LIFT','N-HiTS','CycleNet','TimesNet','ForeCA','full AdaPTS','K/init sweep'],
        D_seed2='NOT_APPLICABLE: D is not evidence for the final minimum composition',
        user_hard_wall_limit=None,user_hard_GPU_limit=None,T_access=False,V_independent_test=False,
        implementation_plan_sha256=contracts.sha(ctx['report']/'IMPLEMENTATION_PLAN_KO.md'))
    contracts.sealed_json(ctx['report']/'CAMPAIGN_CONTRACT.json',value)
    ctx['plan']=value
    return value


def _record(ctx, kind, started, status, **details):
    path=ctx['cache']/'CAMPAIGN.json'
    ledger=contracts.read(path)
    ledger['jobs'].append(dict(kind=kind,started_unix=started,finished_unix=time.time(),
        process_wall_seconds=time.time()-started,status=status,**details))
    contracts.atomic_json(path,ledger)


@contextmanager
def lease(ctx,kind):
    import resources
    started=time.time()
    # The inherited lease mechanism requires numerical bounds. This sentinel
    # imposes no practical time cap; the user's finite3/6epoch scope bounds work.
    with resources.GPUReservation(ctx['cache'],total_gpu_seconds=1e12,reserve_seconds=0,
                                  wall_deadline=1e12,purpose='audit' if kind=='cost' else 'training',kind=kind) as owned:
        ctx['lease']=owned
        try:
            yield owned
        finally:
            ctx.pop('lease',None)
    _record(ctx,kind,started,'COMPLETE')


def _require_complete(result):
    if result['status']!='COMPLETE':
        raise RuntimeError('Registered stage incomplete; same-command resume retains checkpoint')
    return result


def _seal_decision(ctx,cohort,cap,group,jobs):
    for result in group:_require_complete(result)
    continuing=contracts.needs_extension(group,cap)
    normalization,means=contracts.common_normalization(group)
    payload=dict(status='SEALED',cohort=cohort,cap=cap,selection_split='A_stop',V_used=False,
        matched_arms=[job['job_id'] for job in jobs],extend_all_to6=continuing if cap==3 else False,
        still_improving_at_cap=continuing,common_revin=normalization,
        astop_two_seed_means={'OFF':means[False],'ON':means[True]},
        selections={job['job_id']:dict(epoch=result['selected_epoch'],
                      astop_mse=min(row['astop_mse'] for row in result['curves'] if row.get('astop_mse') is not None),
                      checkpoint=result.get('checkpoint_binding') or contracts.binding(result['parent_checkpoint']['path'],ctx))
                    for job,result in zip(jobs,group)})
    path=ctx['report']/'decisions'/f'{cohort}_cap{cap}.json'
    contracts.sealed_json(path,payload)
    ctx['selection_decision']=dict(path=path,sha256=contracts.sha(path))
    return payload


def execute(ctx):
    from . import training_jobs as training
    from .audit import run_audit
    inventory,tasks=run_audit(ctx)
    plan=make_plan(ctx)
    decisions=[]
    pending=[]
    with lease(ctx,'B2_REVIN_MINIMUM'):
        for cohort in contracts.COHORTS:
            jobs=[job for job in plan['jobs'] if job['family']=='B2' and job['cohort']==cohort]
            from research.adapter_improvements_20261018_v1.audit import load_foundation
            foundation=load_foundation(ctx,ctx['modules'],tasks[cohort]['H'],'cuda')
            group_ctx=dict(ctx,task=tasks[cohort],foundation=foundation)
            results=[]
            for job in jobs:
                print('SEARCH3 '+job['job_id'],flush=True)
                result=training.search(job,group_ctx) if job['revin'] else inventory['controls'][job['job_id']]
                results.append(_require_complete(result))
            d3=_seal_decision(group_ctx,cohort,3,results,jobs)
            decisions.append(d3)
            pending.extend((cohort,job,result,3) for job,result in zip(jobs,results) if job['revin'])
            cap=3
            if d3['extend_all_to6']:
                expanded=[]
                for job,result in zip(jobs,results):
                    print('SEARCH6 '+job['job_id'],flush=True)
                    expanded.append(_require_complete(training.search(job,group_ctx,contracts.EXTENDED_EPOCHS,result['parent_checkpoint'])))
                d6=_seal_decision(group_ctx,cohort,6,expanded,jobs)
                decisions.append(d6)
                pending.extend((cohort,job,result,6) for job,result in zip(jobs,expanded))
                cap,results=6,expanded
            if cohort=='jena':
                norm=decisions[-1]['common_revin']
                for job in [job for job in plan['jobs'] if job['family']=='B0' and job['revin']==norm]:
                    print('COMPOSITION '+job['job_id'],flush=True)
                    result=_require_complete(training.search(job,group_ctx,contracts.INITIAL_EPOCHS if cap==3 else contracts.EXTENDED_EPOCHS))
                    pending.append((cohort,job,result,cap))
            foundation.assert_frozen_unchanged()
            del group_ctx,foundation
            import torch
            torch.cuda.empty_cache()
        contracts.sealed_json(ctx['report']/'SELECTIONS_SEALED_BEFORE_NEW_V.json',dict(
            status='SEALED',decisions=decisions,V_used=False,
            all_registered_searches_complete=True,
            pending_refits=[dict(job_id=job['job_id'],cap=cap,selected_epoch=result['selected_epoch'])
                            for cohort,job,result,cap in pending]))
        for cohort in contracts.COHORTS:
            foundation=load_foundation(ctx,ctx['modules'],tasks[cohort]['H'],'cuda')
            group_ctx=dict(ctx,task=tasks[cohort],foundation=foundation)
            for name,job,result,cap in pending:
                if name!=cohort:continue
                path=ctx['report']/'decisions'/f'{cohort}_cap{cap}.json'
                group_ctx['selection_decision']=dict(path=path,sha256=contracts.sha(path))
                print(f'REFIT{cap} '+job['job_id'],flush=True)
                _require_complete(training._refit_predict(job,group_ctx,result,cap))
            foundation.assert_frozen_unchanged()
            del group_ctx,foundation
            torch.cuda.empty_cache()
    contracts.sealed_json(ctx['report']/'DECISIONS.json',dict(status='COMPLETE',decisions=decisions,V_used_for_selection=False))
    print('EXECUTION_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',nargs='?',choices=('audit','plan','execute','costs','verify','report'))
    parser.add_argument('--asset-root',type=Path)
    args=parser.parse_args()
    if args.phase is None:parser.print_help();return
    if args.asset_root is None:parser.error('--asset-root is required')
    ctx=context(args.asset_root)
    if args.phase=='audit':
        from .audit import run_audit
        value,tasks=run_audit(ctx)
        print(json.dumps(dict(controls=len(value['controls']),baselines=len(value['baselines']),status='PASS')))
    elif args.phase=='plan':
        print(json.dumps(make_plan(ctx)))
    elif args.phase=='execute':execute(ctx)
    elif args.phase=='costs':
        from .costs import execute_costs
        execute_costs(ctx)
    elif args.phase=='verify':
        from .verification import verify
        verify(ctx)
    else:
        from .reporting import build
        build(ctx['report'])


if __name__=='__main__':main()
