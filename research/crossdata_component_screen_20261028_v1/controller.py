"""Serial fixed-cap screen with whole-group time admission and no outcome-based expansion."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from . import runtime, run_one


def stamp():
    return datetime.now(timezone.utc).isoformat()


def command(module, *args):
    return [sys.executable, '-B', '-X', 'utf8', '-m', f'research.{runtime.NAME}.{module}', *map(str, args)]


def environment():
    return dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')


def verify_result(job):
    registration = run_one.validate_registration(job)
    path = runtime.REPORT / 'jobs' / job['job_id'] / 'RESULT.json'
    result = runtime.read(path)
    if (result['status'] != 'COMPLETE' or result['job'] != job
            or result['registration'] != registration
            or result['spec'] != run_one.build_spec(job)
            or result['selection_sealed_before_V'] is not True):
        raise ValueError('Completed result differs from sealed job: ' + job['job_id'])
    for p, h in (('bundle_path', 'bundle_sha256'), ('selection_path', 'selection_sha256')):
        if runtime.sha(result[p]) != result[h]:
            raise ValueError('Completed artifact hash mismatch: ' + result[p])
    row = result['row']
    if runtime.sha(row['prediction_path']) != row['prediction_sha256']:
        raise ValueError('Completed prediction hash mismatch')
    runtime.assert_protected_T(result)
    run_one.check_selection(Path(result['selection_path']), result['selection_sha256'], job)
    expected, _ = run_one.choose_epoch(runtime.read(Path(result['selection_path']).parent / 'SEARCH_RESULT.json'),
                                       run_one.schedule(job))
    if row['selected_epoch'] != expected or row['training_cap'] != run_one.schedule(job)[-1]:
        raise ValueError('Completed row differs from A_stop selection')
    if job['method_id'] == 'E_FIXED':
        audits = result.get('E_INVARIANCE', [])
        if ([entry['phase'] for entry in audits] != ['search', 'refit']
                or any(entry['status'] != 'PASS' or entry['E_requires_grad'] is not False
                       or len({entry[key] for key in ('initial_E_sha256', 'live_E_sha256', 'selected_E_sha256')}) != 1
                       for entry in audits)):
            raise ValueError('Completed search/refit E invariance evidence differs')
        import hashlib
        import torch
        bundle = torch.load(result['bundle_path'], map_location='cpu', weights_only=False)
        E = bundle['model']['E']
        digest = hashlib.sha256(E.detach().numpy().tobytes(order='C')).hexdigest()
        if (bundle['spec'] != result['spec']
                or bundle['normalization']['fit_end'] != runtime.FIT_ENDS[job['dataset']][1]
                or not torch.equal(E, torch.as_tensor(bundle['normalization']['basis']).float())
                or digest != audits[1]['selected_E_sha256']):
            raise ValueError('Actual saved bundle E is not its fresh A60 PCA basis')
    return result


METHODS = ('OURS', 'TIME_ONLY', 'A5_LAG0', 'E_FIXED')
GROUP_SPECS = {
    'pems32_base': ('pems32', (92601, 92602)),
    'beijing_base': ('beijing', (92601, 92602)),
    'beijing_seed3': ('beijing', (92603,)),
    'pems32_seed3': ('pems32', (92603,)),
}


class Controller:
    def __init__(self, resume_stopped=False):
        self.plan = runtime.read(runtime.REPORT / 'RUN_PLAN.json')
        self.protocol = runtime.read(runtime.REPORT / 'PREREGISTRATION.json')
        self.mandatory, self.optional = self.plan['jobs'], self.plan['optional_jobs']
        self.jobs = self.mandatory + self.optional
        self.validate_plan()
        self.deadline = datetime.fromisoformat(self.protocol['training_deadline']).timestamp()
        policy = self.protocol['controller_policy']
        self.estimates = {dataset: float(policy['dataset_job_seconds'][dataset]) for dataset in ('pems32', 'beijing')}
        self.floors = {dataset: float(policy['reestimate_floor_seconds'][dataset]) for dataset in self.estimates}
        self.multiplier = float(policy['reestimate_multiplier'])
        self.margin = float(policy['admission_margin_seconds'])
        self.retry_count = int(policy['retries'])
        if (self.retry_count != 1 or self.multiplier != 1.35 or self.margin < 0
                or any(not math.isfinite(value) or value <= 0 for value in [*self.estimates.values(), *self.floors.values()])):
            raise ValueError('Time admission or one-technical-retry policy changed')
        self.progress_path = runtime.REPORT / 'PROGRESS.json'
        self.resume_stopped = resume_stopped
        self.state = runtime.read(self.progress_path) if self.progress_path.exists() else dict(
            status='READY', started_utc=stamp(), run_statuses={}, attempts={}, group_status={},
            dataset_status={}, method_status={}, active=None)
        if self.state['status'] == 'STOPPED' and not resume_stopped:
            raise ValueError('Previous scientific STOPPED requires investigation before --resume-stopped')

    def validate_plan(self):
        if len(self.mandatory) != 16 or len(self.optional) != 8:
            raise ValueError('Plan requires16 base jobs and8 time-only optional jobs')
        if self.plan.get('group_order') != list(GROUP_SPECS) or self.plan.get('extension_jobs', []) != []:
            raise ValueError('Fixed group order or no-extension policy changed')
        if len({job['job_id'] for job in self.jobs}) != 24:
            raise ValueError('Plan job identifiers must be unique')
        for entries, seeds in ((self.mandatory, [92601, 92602]), (self.optional, [92603])):
            for job in entries:
                run_one.validate_job(job)
            for dataset in ('pems32', 'beijing'):
                for method in METHODS:
                    if sorted(job['seed'] for job in entries if (job['dataset'],job['method_id']) == (dataset,method)) != seeds:
                        raise ValueError('Every registered method/seed must be present in its complete group')
        run_one.validate_registration(self.jobs[0])

    def group_jobs(self, group):
        dataset, seeds = GROUP_SPECS[group]
        return [job for job in self.jobs if job['dataset'] == dataset and job['seed'] in seeds]

    def job_estimate(self, job):
        dataset = job['dataset']
        first_full = next(j for j in self.jobs if j['dataset'] == dataset and j['method_id'] == 'OURS' and j['seed'] == 92601)
        if self.state['run_statuses'].get(first_full['job_id']) != 'COMPLETE':
            return self.estimates[dataset]
        observed = []
        for candidate in self.jobs:
            if candidate['dataset'] != dataset or self.state['run_statuses'].get(candidate['job_id']) != 'COMPLETE':
                continue
            rows = list(self.state['attempts'].get(candidate['job_id'], []))
            for attempt in (1, 2):
                path = runtime.REPORT / 'jobs' / candidate['job_id'] / f'WORKER_RECEIPT_attempt{attempt}.json'
                if path.exists():
                    rows.append(runtime.read(path))
            observed.extend(float(row['wall_seconds']) for row in rows
                            if row.get('exit_code') == 0 and math.isfinite(float(row.get('wall_seconds', -1)))
                            and float(row.get('wall_seconds', -1)) > 0)
        return max(self.floors[dataset], max(observed)*self.multiplier) if observed else self.estimates[dataset]
    def save(self):
        self.state.update(controller_pid=os.getpid(), updated_utc=stamp(), training_deadline=self.protocol['training_deadline'],
                          remaining_seconds=max(0., self.deadline - time.time()))
        runtime.atomic(self.progress_path, self.state)

    def descriptor(self, job):
        path = runtime.CACHE / 'job_inputs' / (job['job_id'] + '.json')
        if path.exists():
            if runtime.read(path) != job:
                raise ValueError('Existing job descriptor changed')
        else:
            runtime.write_new(path, job)
        return path

    def recover_worker(self):
        live_path = runtime.CACHE / 'LIVE_WORKER.json'
        live = runtime.read(live_path) if live_path.exists() else None
        active = self.state.get('active')
        if live and live.get('status') == 'RUNNING' and runtime.pid_alive(live['pid']):
            if active and active['job_id'] != live['job_id']:
                raise ValueError('Controller and live worker identities differ')
            active = dict(job_id=live['job_id'], attempt=live['attempt'], pid=live['pid'], started_unix=live['started_unix'])
            self.state['active'] = active
        if active:
            self.state['status'] = 'WAITING_EXISTING_WORKER'
            while runtime.pid_alive(active['pid']):
                self.save()
                time.sleep(10)
            self.state['active'] = None
            self.save()

    def next_attempt(self, job):
        root = runtime.REPORT / 'jobs' / job['job_id']
        used = []
        for attempt in (1, 2):
            receipt = root / f'WORKER_RECEIPT_attempt{attempt}.json'
            attempt_dir = root / f'attempt_{attempt:03d}'
            log = runtime.CACHE / 'logs' / f'{job["job_id"]}_attempt{attempt}.log'
            if receipt.exists():
                previous = runtime.read(receipt)
                if previous['exit_code'] == 20 and not self.resume_stopped:
                    raise ValueError('Scientific failure cannot be automatically retried: ' + job['job_id'])
                used.append(attempt)
            elif attempt_dir.exists() or log.exists():
                # recover_worker already established that no recorded worker is live.
                used.append(attempt)
                self.state['attempts'].setdefault(job['job_id'], []).append(
                    dict(attempt=attempt, status='INTERRUPTED_PROCESS', observed_utc=stamp()))
        return max(used, default=0) + 1

    def run(self, job):
        result_path = runtime.REPORT / 'jobs' / job['job_id'] / 'RESULT.json'
        if result_path.exists():
            verify_result(job)
            self.state['run_statuses'][job['job_id']] = 'COMPLETE'
            self.save()
            return True
        estimate = self.job_estimate(job)
        for attempt in range(self.next_attempt(job), 3):
            run_one.validate_registration(job)
            if time.time() + estimate + self.margin > self.deadline:
                self.state['run_statuses'][job['job_id']] = 'NOT_RUN_BUDGET'
                self.save()
                return False
            # A worker not yet represented in PROGRESS still holds this lock.
            with runtime.CampaignLock(runtime.SHARED_CACHE / 'worker.lock'), runtime.CampaignLock(runtime.CACHE / 'worker.lock'):
                pass
            descriptor = self.descriptor(job)
            log = runtime.CACHE / 'logs' / f'{job["job_id"]}_attempt{attempt}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('x', encoding='utf8') as stream:
                started = time.time()
                process = subprocess.Popen(command('run_one', '--job', descriptor, '--attempt', attempt,
                    '--deadline', self.deadline, '--lease-token', self.lease_token), cwd=runtime.REPO, env=environment(),
                    stdout=stream, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                active = dict(job_id=job['job_id'], attempt=attempt, pid=process.pid,
                              started_unix=started, log=str(log))
                self.state.update(status='RUNNING', active=active)
                self.state['run_statuses'][job['job_id']] = 'RUNNING'
                self.save()
                print(json.dumps(dict(event='START', **active)), flush=True)
                while process.poll() is None:
                    if time.time() >= self.deadline:
                        if os.name == 'nt':
                            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                        else:
                            process.kill()
                        process.wait(timeout=30)
                        self.state['run_statuses'][job['job_id']] = 'STOPPED_DEADLINE'
                        break
                    time.sleep(5)
                    self.save()
                return_code = process.returncode
            self.state['active'] = None
            record = dict(active, exit_code=return_code, finished_unix=time.time(), wall_seconds=time.time() - started)
            self.state['attempts'].setdefault(job['job_id'], []).append(record)
            if self.state['run_statuses'].get(job['job_id']) == 'STOPPED_DEADLINE':
                self.save()
                return False
            if return_code == 0:
                verify_result(job)
                self.state['run_statuses'][job['job_id']] = 'COMPLETE'
                self.save()
                return True
            if return_code == 20:
                raise ValueError('Scientific gate failed: ' + job['job_id'] + '; see ' + str(log))
            self.state['run_statuses'][job['job_id']] = 'FAILED_ONCE' if attempt == 1 else 'FAILED'
            self.save()
        self.state['run_statuses'][job['job_id']] = 'FAILED'
        self.save()
        return False

    def admit_remaining(self, jobs):
        remaining = [job for job in jobs if self.state['run_statuses'].get(job['job_id']) != 'COMPLETE']
        reserved = sum(self.job_estimate(job) for job in remaining) + self.margin
        return not remaining or time.time() + reserved <= self.deadline, reserved, remaining

    def run_group(self, group):
        jobs = self.group_jobs(group)
        dataset, seeds = GROUP_SPECS[group]
        for job in jobs:
            if (runtime.REPORT / 'jobs' / job['job_id'] / 'RESULT.json').exists():
                verify_result(job)
                self.state['run_statuses'][job['job_id']] = 'COMPLETE'
        if seeds == (92603,) and any(self.state['run_statuses'].get(job['job_id']) != 'COMPLETE'
                                    for job in self.group_jobs(dataset+'_base')):
            self.state['group_status'][group] = dict(status='BASE_INCOMPLETE',
                completed=sum(self.state['run_statuses'].get(job['job_id']) == 'COMPLETE' for job in jobs),
                decision_utc=stamp())
            for job in jobs:
                self.state['run_statuses'].setdefault(job['job_id'], 'BASE_INCOMPLETE')
            self.save()
            return False
        admitted, estimate, remaining = self.admit_remaining(jobs)
        decision = dict(status='ADMITTED' if admitted else 'NOT_RUN_BUDGET', estimated_seconds=estimate,
            decision_utc=stamp(), planned=len(jobs), remaining=len(remaining),
            criterion='Reserve the entire unfinished fixed group using elapsed wall time only; no V values used.')
        self.state['group_status'][group] = decision
        self.save()
        if not admitted:
            for job in remaining:
                self.state['run_statuses'][job['job_id']] = 'NOT_RUN_BUDGET'
            self.save()
            return False
        for job in jobs:
            admitted, estimate, remaining = self.admit_remaining(jobs)
            if not admitted:
                decision.update(readmission_status='NOT_RUN_BUDGET', remaining_reserved_seconds=estimate)
                for pending in remaining:
                    self.state['run_statuses'][pending['job_id']] = 'NOT_RUN_BUDGET'
                break
            self.run(job)
        complete = sum(self.state['run_statuses'].get(job['job_id']) == 'COMPLETE' for job in jobs)
        decision.update(status='COMPLETE' if complete == len(jobs) else 'INCOMPLETE', completed=complete)
        self.save()
        return complete == len(jobs)

    def summarize_groups(self):
        for dataset in ('pems32', 'beijing'):
            base = self.group_jobs(dataset+'_base')
            extra = self.group_jobs(dataset+'_seed3')
            base_complete = all(self.state['run_statuses'].get(job['job_id']) == 'COMPLETE' for job in base)
            extra_complete = all(self.state['run_statuses'].get(job['job_id']) == 'COMPLETE' for job in extra)
            primary = base+extra if base_complete and extra_complete else base if base_complete else []
            n = 3 if base_complete and extra_complete else 2 if base_complete else 0
            self.state['dataset_status'][dataset] = dict(primary_n=n, base_complete=base_complete,
                optional_seed_complete=extra_complete, pilot=n == 2,
                partial_seed3_count=sum(self.state['run_statuses'].get(job['job_id']) == 'COMPLETE' for job in extra),
                primary_job_ids=[job['job_id'] for job in primary],
                status='COMPLETE' if n == 3 else 'PILOT' if n == 2 else 'INCOMPLETE')
            for method in METHODS:
                selected_jobs = [job for job in primary if job['method_id'] == method]
                epochs = [runtime.read(runtime.REPORT/'jobs'/job['job_id']/'RESULT.json')['row']['selected_epoch'] for job in selected_jobs]
                cap = selected_jobs[0]['training_cap'] if selected_jobs else None
                count = sum(float(epoch) == cap for epoch in epochs)
                undertrained = bool(n) and count >= math.ceil(2*n/3)
                self.state['method_status'][method+':'+dataset] = dict(primary_n=n, pilot=n == 2,
                    cap_selected_count=count, cap=cap, undertrained=undertrained, provisional=(n == 2 or undertrained),
                    status='UNDERTRAINED_PROVISIONAL' if undertrained else 'PILOT' if n == 2 else 'COMPLETE' if n == 3 else 'INCOMPLETE')

    def execute(self):
        self.recover_worker()
        runtime.guard_preserved()
        for group in self.plan['group_order']:
            self.run_group(group)
        runtime.guard_preserved()
        self.summarize_groups()
        base_count = sum(self.state['run_statuses'].get(j['job_id']) == 'COMPLETE' for j in self.mandatory)
        optional_count = sum(self.state['run_statuses'].get(j['job_id']) == 'COMPLETE' for j in self.optional)
        complete = base_count + optional_count
        self.state.update(status='TRAINING_COMPLETE' if complete == len(self.jobs) else
                          'NOT_RUN_BUDGET' if complete == 0 and all(v in ('NOT_RUN_BUDGET','BASE_INCOMPLETE') for v in self.state['run_statuses'].values()) else 'TRAINING_INCOMPLETE',
                          completed_utc=stamp(), active=None, base_complete_count=base_count,
                          optional_complete_count=optional_count, completed_count=complete, extension_jobs=0)
        self.save()
        runtime.atomic(runtime.REPORT / 'EXECUTION_COMPLETION.json', self.state)
        postprocess = Path(__file__).with_name('postprocess.py')
        if postprocess.exists():
            with (runtime.CACHE / 'postprocess.log').open('a', encoding='utf8') as stream:
                result = subprocess.run(command('postprocess'), cwd=runtime.REPO, env=environment(),
                    stdout=stream, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            runtime.atomic(runtime.REPORT / 'POSTPROCESS_STATUS.json',
                           dict(status='COMPLETE' if result.returncode == 0 else 'FAILED',
                                exit_code=result.returncode, finished_utc=stamp()))
        return 0 if any(row['primary_n'] >= 2 for row in self.state['dataset_status'].values()) else 2


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-stopped', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    runtime.install_write_guard()
    controller = None
    try:
        with runtime.CampaignLock(runtime.CACHE / 'controller.lock'):
            controller = Controller(args.resume_stopped)
            if args.preflight_only:
                print(json.dumps(dict(status='PREFLIGHT_PASS', base_jobs=len(controller.mandatory), optional_jobs=len(controller.optional),
                                      extension_jobs=0, group_order=controller.plan['group_order'])), flush=True)
                return 0
            runtime.require_root_gate()
            upstream = runtime.require_upstream_finished()
            with runtime.CampaignLock(runtime.SHARED_CACHE / 'controller.lock'):
                runtime.require_upstream_finished()
                with runtime.CampaignLock(runtime.SHARED_CACHE / 'worker.lock'):
                    pass
                owner = runtime.process_snapshot()[os.getpid()]
                controller.lease_token = uuid.uuid4().hex
                runtime.atomic(runtime.CACHE / 'GPU_LEASE.json', dict(status='ACTIVE', token=controller.lease_token,
                    pid=os.getpid(), created_utc=owner['created_utc'], upstream=upstream,
                    registration_seal_sha256=runtime.sha(runtime.REPORT / 'REGISTRATION_SEAL.json'),
                    reservation='Parent holds shared controller.lock; each GPU child holds shared worker.lock and its own worker.lock. Child retains shared GPU exclusion if parent exits.'))
                try:
                    return controller.execute()
                finally:
                    lease = runtime.read(runtime.CACHE / 'GPU_LEASE.json')
                    if lease.get('pid') == os.getpid():
                        runtime.atomic(runtime.CACHE / 'GPU_LEASE.json', dict(lease, status='RELEASED', released_utc=stamp()))
    except BaseException as exc:
        if controller is not None:
            controller.state.update(status='STOPPED', stop_reason=f'{type(exc).__name__}: {exc}')
            controller.save()
            runtime.atomic(runtime.REPORT / 'EXECUTION_COMPLETION.json', controller.state)
        import traceback
        traceback.print_exc()
        return 20


if __name__ == '__main__':
    raise SystemExit(main())
