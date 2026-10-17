"""Explicit audit / plan / execute / verify / report interface. Default: source only."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .core import (SUCCESS, atomic_json, campaign_ledger, dispatch_allowed, finish_job,
                   planned_jobs, read, reusable, seal_plan, sha, utc, validate_plan)

REPO = Path(__file__).resolve().parents[2]
NAME = 'adapter_diagnostics_20261017_v2'


def source_hashes(repo):
    paths = [p for p in (repo / 'research' / NAME).glob('*.py') if p.name!='reporting.py']
    base = repo / 'research/informs_broad_validation_20261015_v1'
    paths += [base / n for n in ('DATA_CONTRACT.json', 'networks.py', 'fit_base.py',
                                 'training.py', 'data_contract.py', 'ridge_baseline.py',
                                 'resources.py', 'PUBLICATION_PROTOCOL_KO.md', 'gpu_runtime.py',
                                 'evaluation.py','deployment.py','measure_costs.py')]
    paths += [repo/'research/temporal_frontend_campaign_20261015_v1/runtime.py']
    paths += [repo/'reports'/NAME/'BACKBONE_SOURCES.json']
    paths += list((repo / 'reports/informs_b2_why_where_20261017/code').glob('*.py'))
    return {p.relative_to(repo).as_posix(): sha(p) for p in sorted(paths)}


def context(args):
    repo = args.repo.resolve()
    cache = (args.cache_root or repo / '.cache' / NAME).resolve()
    report = (args.report_root or repo / 'reports' / NAME).resolve()
    if args.asset_root is None:
        raise ValueError('Explicit --asset-root required for asset operations')
    cache.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    ledger = campaign_ledger(cache / 'CAMPAIGN.json')
    return dict(repo=repo, cache=cache, report=report, asset=args.asset_root.resolve(),
                ledger=ledger, deadline=ledger['started_unix'] + ledger['limits']['wall_seconds'])


def audit(ctx):
    repo, report = ctx['repo'], ctx['report']
    def git(*command):
        return subprocess.check_output(['git', *command], cwd=repo, text=True).strip()
    state = dict(started_at=utc(), origin_main=git('rev-parse', 'origin/main'),
                 work_head=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                 origin_url=git('remote', 'get-url', 'origin'),
                 changed_files=git('status', '--short', '--untracked-files=no').splitlines(),
                 contract_source_hashes=source_hashes(repo),
                 protected_policy='Development V only; T and protected Electricity tail unopened')
    path = report / 'START_STATE.json'
    if not path.exists():
        atomic_json(path, state)
    from .data_audit import run_audit
    return run_audit(ctx['asset'], report_root=report)


def make_plan(ctx):
    hashes = source_hashes(ctx['repo'])
    output = ctx['report'] / 'EXPERIMENT_PLAN.json'
    if output.exists():
        existing = read(output)
        validate_plan(existing, ctx['repo'])
        return existing
    plan = seal_plan(dict(campaign=NAME, created_at=utc(),
                         source_hashes=hashes, limits=ctx['ledger']['limits'],
                         registered_acceptance='UNREGISTERED accuracy tolerance and speed target',
                         confirmation_seeds='Both 92601/92602 assigned before any new V results',
                         split='A_fit/A_stop/M_cal/V/T=50/10/10/10/20; M_cal and T not used',
                         evaluation='Normalized observed-channel macro MSE and MAE; stride H',
                         selection='CP0/.25/1/3 A_fit search, A_stop selection, A60 fresh refit',
                         jobs=planned_jobs(hashes)))
    atomic_json(output, plan)
    return plan


def handler(job, ctx):
    question = job['question_id']
    if question in ('A01', 'A04'):
        result = read(ctx['report'] / 'DATA_AUDIT.json')
        return dict(status='COMPLETE', reason=None,
                    artifact_paths=[str(ctx['report'] / 'DATA_AUDIT.json')], result=result)
    if question=='C00':
        from .existing_jobs import profile_job
        profiles=[profile_job(dict(job,question_id='C01',family=f),ctx) for f in ('B0','B2')]
        cost=sum(p['estimated_train_wall_seconds'] for p in profiles)*1.2+300
        result=dict(profiles=profiles,conservative_complete_pair_seconds=cost,
                    eval_serialization_load_margin='20% plus 300 seconds; estimate uncertain')
        output=ctx['report']/'profiles'/'C00_pems_pair.json'
        atomic_json(output,result)
        return dict(status='COMPLETE',reason=None,artifact_paths=[str(output)],result=result)
    module = ('existing_jobs' if question in ('A02', 'A03', 'B01', 'C01', 'E01') else
              'ridge_jobs' if question == 'B02' else
              'backbone_jobs' if question in ('D01', 'D02', 'D03') else 'new_jobs')
    return importlib.import_module('.' + module, __package__).execute_job(job, ctx)


def execute(ctx, plan_path, only=None, worker=False):
    plan = read(plan_path)
    validate_plan(plan, ctx['repo'])
    ctx['plan'] = plan
    if not worker:
        for job in plan['jobs']:
            if only and not any(job['job_id']==x or job['question_id']==x for x in only):
                continue
            command=[sys.executable,'-m','research.'+NAME+'.run','--mode','execute','--worker',
                     '--repo',str(ctx['repo']),'--asset-root',str(ctx['asset']),
                     '--cache-root',str(ctx['cache']),'--report-root',str(ctx['report']),
                     '--plan',str(Path(plan_path).resolve()),'--only',job['job_id']]
            completed=subprocess.run(command,cwd=ctx['repo'],check=False)
            if completed.returncode:
                raise RuntimeError('Isolated job worker exited '+str(completed.returncode)+': '+job['job_id'])
        return
    receipts = ctx['report'] / 'receipts'
    receipts.mkdir(parents=True, exist_ok=True)
    broad = ctx['repo'] / 'research/informs_broad_validation_20261015_v1'
    sys.path.insert(0, str(broad))
    import resources
    for job in plan['jobs']:
        if only and not any(job['job_id'] == x or job['question_id'] == x for x in only):
            continue
        path = receipts / (job['job_id'] + '.json')
        if reusable(path, job['scientific_identity'], ctx['report'], private_roots=ctx):
            print(job['job_id'], 'VERIFIED_SKIP', flush=True)
            continue
        deps = [receipts / (d + '.json') for d in job['dependency_ids']]
        if any(not p.exists() or read(p)['status'] not in SUCCESS for p in deps):
            finish_job(path, job['scientific_identity'], dict(status='NOT_RUN_DEPENDENCY',
                       reason='Required job is not complete', artifact_paths=[]), ctx['report'])
            continue
        ctx['ledger'] = campaign_ledger(ctx['cache'] / 'CAMPAIGN.json')
        cost = job['expected_cost']
        if job['question_id']=='C01':
            profile=read(ctx['report']/'profiles/C00_pems_pair.json')
            pair_cost=profile['conservative_complete_pair_seconds']
            pair_state=ctx['cache']/'PEMS_PAIR_ADMISSION.json'
            if not pair_state.exists():
                admitted=dispatch_allowed(ctx['ledger'],pair_cost,0)
                atomic_json(pair_state,dict(admitted=admitted,estimated_seconds=pair_cost,at=utc()))
            if not read(pair_state)['admitted']:
                finish_job(path,job['scientific_identity'],dict(status='SKIPPED_BUDGET',
                    reason='Actual disposable profiles exceed complete pair budget/reserve',artifact_paths=[]),ctx['report'])
                continue
        final = job['question_id'] in ('A01', 'A02', 'A03', 'A04', 'E01', 'E02')
        if not dispatch_allowed(ctx['ledger'], cost['wall_seconds'], cost['gpu_seconds'], final=final):
            finish_job(path, job['scientific_identity'], dict(status='SKIPPED_BUDGET',
                       reason='Conservative completion cost exceeds remaining time/reserve',
                       artifact_paths=[]), ctx['report'])
            continue
        resources.check_storage_budget(ctx['cache'], max_bytes=ctx['ledger']['limits']['cache_bytes'])
        started, cpu_started = time.time(), time.process_time()
        atomic_json(path, dict(status='RUNNING', scientific_identity=job['scientific_identity'],
                               started_at=utc(), artifact_hashes={}))
        print(job['job_id'], 'RUNNING', flush=True)
        gpu = job['device'] == 'cuda'
        try:
            if gpu:
                with resources.GPUReservation(ctx['cache'], total_gpu_seconds=14400,
                        reserve_seconds=0 if final else 2880, wall_deadline=ctx['deadline'],
                        purpose='audit' if final else 'training', kind=job['job_id']) as lease:
                    ctx['lease'] = lease
                    ctx['job_deadline'] = min(ctx['deadline'], lease.deadline)
                    value = handler(job, ctx)
                    lease.check()
            else:
                ctx['job_deadline'] = ctx['deadline'] - (0 if final else 4320)
                value = handler(job, ctx)
        except FileNotFoundError:
            value = dict(status='BLOCKED_MISSING_ASSET', reason='Required local asset missing', artifact_paths=[])
            (ctx['cache'] / 'errors').mkdir(exist_ok=True)
            (ctx['cache'] / 'errors' / (job['job_id'] + '.txt')).write_text(traceback.format_exc(), encoding='utf-8')
        except (resources.BudgetExceeded, TimeoutError):
            value = dict(status='SKIPPED_BUDGET', reason='Finite resource boundary reached', artifact_paths=[])
        except Exception as error:
            value = dict(status='FAILED', reason=type(error).__name__, artifact_paths=[])
            (ctx['cache'] / 'errors').mkdir(exist_ok=True)
            (ctx['cache'] / 'errors' / (job['job_id'] + '.txt')).write_text(traceback.format_exc(), encoding='utf-8')
        finally:
            ctx.pop('lease',None)
        elapsed = time.time() - started
        value.update(started_at=datetime_utc(started), consumed_resource=dict(
            wall_seconds=elapsed, process_cpu_seconds=time.process_time() - cpu_started,
            gpu_lease_seconds=elapsed if gpu else 0))
        finish_job(path, job['scientific_identity'], value, ctx['report'])
        ledger = campaign_ledger(ctx['cache'] / 'CAMPAIGN.json')
        ledger['gpu_seconds'] = resources.gpu_usage_seconds(ctx['cache'])
        ledger['cpu_worker_seconds'] += elapsed if not gpu else 0
        atomic_json(ctx['cache'] / 'CAMPAIGN.json', ledger)
        print(job['job_id'], value['status'], f'{elapsed:.2f}s', flush=True)


def datetime_utc(timestamp):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['audit', 'plan', 'execute', 'verify', 'report'])
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--report-root', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.mode is None:
        print('SOURCE_ONLY: choose --mode explicitly; no assets, model, or downloads opened')
        return
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    ctx = context(args)
    if args.mode == 'audit':
        audit(ctx)
    elif args.mode == 'plan':
        value = make_plan(ctx)
        print(value['plan_sha256'], len(value['jobs']), 'jobs')
    elif args.mode == 'execute':
        if args.plan is None:
            parser.error('--plan required for execute')
        execute(ctx, args.plan, args.only, args.worker)
    else:
        from .reporting import build_report, verify
        (verify if args.mode == 'verify' else build_report)(ctx)


if __name__ == '__main__':
    main()
