"""Sequential controller for the preregistered K-tau rule, never a K search."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from research.b2_revin_20261018_v1.contracts import read, sha, sealed_json, atomic_json


NAME = 'k_tau_rule_20261023_v1'
REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / 'reports' / NAME
CACHE = REPO / '.cache' / NAME
DATASETS = ('jena', 'ETTh1', 'ETTh2_safe17400', 'appliances')
SEEDS = tuple(range(92601, 92611))
GRID = dict(jena=(3, 6, 12), ETTh1=(1, 2, 4), ETTh2_safe17400=(1, 2, 4), appliances=(4, 7, 14))
UPPER = dict(jena=638., ETTh1=336., ETTh2_safe17400=339., appliances=311.)


def csv_read(path):
    with Path(path).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def job(dataset, k, seed, role='main'):
    if dataset not in DATASETS or seed not in SEEDS or type(k) is not int or k < 1:
        raise ValueError('Unregistered dataset, K or seed')
    return dict(job_id=f'KTAU_{role}_{dataset}_K{k}_s{seed}', cohort=dataset,
                family='B2', revin=False, init_kind='PCA', K=k, seed=seed, device='cuda')


def clock():
    return read(REPORT / 'EXECUTION_CLOCK.json')


def elapsed():
    item = clock()
    return item['charged_preparation_seconds'] + time.time() - item['resume_anchor_unix']


def deadline():
    item = clock()
    return item['resume_anchor_unix'] + item['limit_seconds'] - item['charged_preparation_seconds']


def worker(identifier, role):
    from .data import build_context
    from .runner import run_one
    from research.b2_dataset_extension_20261018_v1.train import runtime
    plans = [p for name in ('RUN_PLAN_PREFLIGHT.json', 'RUN_PLAN.json')
             if (p := REPORT / name).is_file()]
    entries = [e for path in plans for e in read(path)['jobs']
               if e['job']['job_id'] == identifier and e['role'] == role]
    if not entries or any(e != entries[0] for e in entries):
        raise ValueError('Worker is not in the byte-sealed run plan')
    registration = read(REPORT / 'PREREGISTRATION_COMMIT.json')
    if registration['preregistration_sha256'] != sha(REPORT / 'PREREGISTRATION.json'):
        raise ValueError('Preregistration changed')
    entry = entries[0]
    # The instruction allows the current fixed-cap run to finish after the
    # overall cutoff, while the controller refuses to launch another run.
    ctx = build_context(entry['job']['cohort'], deadline() + 7200)
    helper, ctx = runtime(ctx, ctx['task'], ctx['model_dir'])
    ctx['helper'] = helper
    result = run_one(ctx, entry['job'], role=role)
    destination = CACHE / 'results' / (identifier + '.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    sealed_json(destination, dict(job=entry['job'], role=role, result=result))
    print(json.dumps(dict(status='COMPLETE', job_id=identifier,
                          MSE=result['rows'][0]['MSE'], MAE=result['rows'][0]['MAE'])), flush=True)


def run_entry(entry):
    identifier = entry['job']['job_id']
    output = CACHE / 'results' / (identifier + '.json')
    if output.is_file():
        return read(output)
    for attempt in (1, 2):
        if time.time() >= deadline():
            raise TimeoutError('STOP_BUDGET: no new job may start')
        log = CACHE / 'logs' / f'{identifier}_controller{attempt}.log'
        if log.exists():
            if attempt == 2:
                raise RuntimeError('Two recorded failed attempts: ' + identifier)
            continue
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-X', 'utf8', '-m', 'research.' + NAME + '.execute',
                   '_worker', '--job-id', identifier, '--role', entry['role']]
        started = time.time()
        print(json.dumps(dict(event='START', job_id=identifier, role=entry['role'],
                              attempt=attempt, elapsed_seconds=elapsed())), flush=True)
        environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
        with log.open('x', encoding='utf8') as stream:
            process = subprocess.Popen(command, cwd=REPO, env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT)
            atomic_json(CACHE / 'ACTIVE_PROCESS.json', dict(pid=process.pid, command=command,
                        started_unix=started, job_id=identifier, log=str(log)))
            code = process.wait()
        duration = time.time() - started
        receipt = dict(job_id=identifier, role=entry['role'], attempt=attempt,
                       pid=process.pid, started_unix=started, ended_unix=time.time(),
                       seconds=duration, exit_code=code, log=str(log.relative_to(REPO)))
        sealed_json(log.with_suffix('.json'), receipt)
        atomic_json(CACHE / 'ACTIVE_PROCESS.json', dict(active=False, last=receipt))
        print(json.dumps(dict(event='EXIT', **receipt)), flush=True)
        if code == 0 and output.exists():
            return read(output)
        if time.time() >= deadline():
            raise TimeoutError('STOP_BUDGET after current run')
    raise RuntimeError('Two failed attempts: ' + identifier)


def baseline(dataset, k, family='B2'):
    values = [r for r in csv_read(REPORT / 'BASELINE_RECALCULATION.csv')
              if r['dataset'] == dataset and r['family'] == family
              and (family == 'RAW' or int(r['K']) == k)]
    if len(values) != 1:
        raise ValueError('One exact historical baseline row required')
    return values[0]


def recheck():
    plan = read(REPORT / 'RUN_PLAN_PREFLIGHT.json')
    decisions = {}
    for entry in plan['jobs']:
        record = run_entry(entry)
        item = entry['job']
        old = baseline(item['cohort'], item['K'])
        seed_values = json.loads(old['seed_MSE_json'])
        value = float(record['result']['rows'][0]['MSE'])
        difference = abs(value - float(seed_values[str(item['seed'])]))
        span = float(old['MSE_max']) - float(old['MSE_min'])
        decisions[item['cohort']] = dict(status='PASS_RECHECK' if difference < span else 'FAIL',
            reused_historical_results=difference < span, K=item['K'], seed=item['seed'],
            old_seed_mse=float(seed_values[str(item['seed'])]), new_mse=value,
            absolute_difference=difference, old_two_seed_range=span, job_id=item['job_id'])
    sealed_json(REPORT / 'G4_RECHECK_DECISIONS.json', decisions)
    return decisions


def make_plan():
    decisions = read(REPORT / 'G4_RECHECK_DECISIONS.json')
    selected = {r['dataset']: int(r['K_A_fit']) for r in csv_read(REPORT / 'K_TAU.csv')
                if float(r['tau']) == .9}
    if set(selected) != set(DATASETS):
        raise ValueError('All four A_fit rule choices required')
    entries, reused = [], []
    for dataset in DATASETS:
        k = selected[dataset]
        compatible = decisions[dataset]['reused_historical_results']
        for seed in SEEDS:
            if compatible and k in GRID[dataset] and seed <= 92602:
                old = baseline(dataset, k)
                reused.append(dict(dataset=dataset, K=k, seed=seed,
                    mse=float(json.loads(old['seed_MSE_json'])[str(seed)]),
                    mae=float(json.loads(old['seed_MAE_json'])[str(seed)]), reused=True,
                    role='main', job_id=f'HISTORICAL_{dataset}_K{k}_s{seed}',
                    search_seconds=None, refit_seconds=None, source_paths=old['source_paths_json']))
            else:
                entries.append(dict(job=job(dataset, k, seed), role='main'))
        if not compatible:
            for old_k in GRID[dataset]:
                for seed in SEEDS[:2]:
                    entries.append(dict(job=job(dataset, old_k, seed, 'comparison'), role='comparison'))
    estimated = sum(UPPER[e['job']['cohort']] for e in entries)
    plan = dict(status='ADMITTED' if elapsed() + estimated <= clock()['limit_seconds'] else 'STOP_BUDGET',
        K_main=selected, G4=decisions, jobs=entries, reused=reused,
        remaining_training_upper_seconds=estimated, charged_elapsed_seconds=elapsed(),
        limit_seconds=clock()['limit_seconds'], preregistration_sha256=sha(REPORT/'PREREGISTRATION.json'),
        preliminary_plan_sha256=sha(REPORT/'RUN_PLAN_PREFLIGHT.json'))
    sealed_json(REPORT / 'RUN_PLAN.json', plan)
    return plan


def export_runs(plan):
    rows = list(plan['reused'])
    for entry in plan['jobs']:
        output = CACHE / 'results' / (entry['job']['job_id'] + '.json')
        if not output.is_file():
            continue
        record = read(output)
        result, item = record['result'], record['job']
        row = result['rows'][0]
        rows.append(dict(dataset=item['cohort'], K=item['K'], seed=item['seed'],
            mse=row['MSE'], mae=row['MAE'], reused=False, role=entry['role'], job_id=item['job_id'],
            selected_epoch=result['fit_receipt']['selected_epoch'],
            search_seconds=result['search_summary']['seconds'], refit_seconds=result['refit_summary']['seconds'],
            bundle_sha256=result['bundle_binding']['sha256'],
            selection_sealed_before_V=result['selection_sealed_before_V']))
    if rows:
        csv_write(REPORT / 'RUNS.csv', rows)
    return rows


def train():
    plan = read(REPORT / 'RUN_PLAN.json')
    if plan['status'] != 'ADMITTED':
        raise TimeoutError('Registered run plan exceeds the fixed time budget')
    measured = {}
    for index, entry in enumerate(plan['jobs']):
        started = time.time()
        run_entry(entry)
        export_runs(plan)
        dataset, k = entry['job']['cohort'], entry['job']['K']
        if entry['role'] == 'main' and k not in GRID[dataset] and dataset not in measured:
            measured[dataset] = time.time() - started
            remaining = sum(max(UPPER[e['job']['cohort']], measured.get(e['job']['cohort'], 0))
                            for e in plan['jobs'][index+1:])
            update = dict(measured=measured, estimated_remaining_seconds=remaining,
                          charged_elapsed_seconds=elapsed(), within_budget=elapsed()+remaining <= clock()['limit_seconds'])
            atomic_json(REPORT / 'BUDGET_PROFILE_UPDATE.json', update)
            if not update['within_budget']:
                raise TimeoutError('STOP_BUDGET: new-K first-run profile exceeds the fixed budget')
        if time.time() >= deadline():
            raise TimeoutError('STOP_BUDGET after finishing the current job')
    return export_runs(plan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('_worker', 'recheck', 'spectrum', 'plan', 'train', 'report'))
    parser.add_argument('--job-id')
    parser.add_argument('--role', default='main')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.phase == '_worker':
        worker(args.job_id, args.role)
    elif args.phase == 'recheck':
        print(json.dumps(recheck()))
    elif args.phase == 'spectrum':
        from .data import build_context
        from .spectrum import compute_spectra
        contexts = {d: build_context(d, deadline(), device='cpu') for d in DATASETS}
        result = compute_spectra(contexts, REPORT, read(REPORT / 'PREREGISTRATION_COMMIT.json'))
        print(json.dumps({k:v for k,v in result.items() if k not in ('spectra_rows','k_tau_rows')}, ensure_ascii=False))
    elif args.phase == 'plan':
        result = make_plan()
        print(json.dumps(dict(status=result['status'], K_main=result['K_main'], new_jobs=len(result['jobs']), reused=len(result['reused']), estimated_seconds=result['remaining_training_upper_seconds'])))
    elif args.phase == 'train':
        try:
            train()
        except Exception as exc:
            export_runs(read(REPORT / 'RUN_PLAN.json'))
            atomic_json(REPORT / 'STOP_REASON.json', dict(reason=str(exc), elapsed_seconds=elapsed()))
            raise
    else:
        from .analysis import build_report
        stop = read(REPORT/'STOP_REASON.json')['reason'] if (REPORT/'STOP_REASON.json').exists() else None
        print(json.dumps(build_report(REPORT, stop_reason=stop, elapsed_seconds=elapsed()),ensure_ascii=False))


if __name__ == '__main__':
    main()
