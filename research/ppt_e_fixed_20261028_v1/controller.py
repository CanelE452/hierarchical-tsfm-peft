"""One-worker campaign with sealed inputs, bounded retries, and conditional extensions."""
import argparse
from datetime import datetime, timezone
import json
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
    audits = result.get('E_INVARIANCE', [])
    if ([entry['phase'] for entry in audits] != ['search', 'refit']
            or any(entry['status'] != 'PASS' or entry['E_requires_grad'] is not False
                   or len({entry[key] for key in ('initial_E_sha256', 'live_E_sha256', 'selected_E_sha256')}) != 1
                   for entry in audits)):
        raise ValueError('Completed search/refit E invariance evidence differs')
    import torch
    from .models import tensor_sha
    bundle = torch.load(result['bundle_path'], map_location='cpu', weights_only=False)
    fitted_basis = torch.as_tensor(bundle['normalization']['basis']).float()
    if (bundle['spec'] != result['spec'] or bundle['normalization']['fit_end'] != 12579
            or not torch.equal(bundle['model']['E'], fitted_basis)
            or tensor_sha(bundle['model']['E']) != audits[1]['selected_E_sha256']):
        raise ValueError('Actual saved bundle E is not its fresh A60 PCA basis')
    return result


class Controller:
    def __init__(self, resume_stopped=False):
        self.plan = runtime.read(runtime.REPORT / 'RUN_PLAN.json')
        self.protocol = runtime.read(runtime.REPORT / 'PREREGISTRATION.json')
        self.mandatory, self.optional = self.plan['jobs'], self.plan['optional_jobs']
        self.jobs, self.extensions = self.mandatory + self.optional, self.plan['extension_jobs']
        self.validate_plan()
        self.deadline = datetime.fromisoformat(self.protocol['training_deadline']).timestamp()
        policy = self.protocol.get('controller_policy', {})
        self.base_estimate = float(policy.get('base_job_seconds', 1080))
        self.extension_estimate = float(policy.get('extension_job_seconds', 1800))
        self.margin = float(policy.get('admission_margin_seconds', 120))
        self.progress_path = runtime.REPORT / 'PROGRESS.json'
        self.resume_stopped = resume_stopped
        self.state = runtime.read(self.progress_path) if self.progress_path.exists() else dict(
            status='READY', started_utc=stamp(), run_statuses={}, attempts={}, extension_status={},
            family_status={}, active=None)
        if self.state['status'] == 'STOPPED' and not resume_stopped:
            raise ValueError('Previous scientific STOPPED requires investigation before --resume-stopped')

    def validate_plan(self):
        if (len(self.mandatory), len(self.optional), len(self.extensions)) != (5, 0, 5):
            raise ValueError('Plan must contain five E-fixed base jobs and five conditional extensions')
        for entries, extended in ((self.jobs, False), (self.extensions, True)):
            for job in entries:
                run_one.validate_job(job)
                if job['extended'] is not extended:
                    raise ValueError('Plan extension flag differs')
            for method, dataset in run_one.ALLOWED_AXES:
                if sorted(j['seed'] for j in entries if (j['method_id'], j['dataset']) == (method, dataset)) != list(range(92601, 92606)):
                    raise ValueError('Plan must contain all five seeds of each registered family')
        if self.plan['family_order'] != ['E_FIXED:jena']:
            raise ValueError('Required family order changed')
        if self.plan['optional_family_order'] != []:
            raise ValueError('Optional family order changed')
        run_one.validate_registration(self.jobs[0])

    def job_estimate(self, job):
        return self.extension_estimate if job['extended'] else self.base_estimate

    def reserve_campaign(self):
        remaining_base = [j for j in self.jobs if not (runtime.REPORT / 'jobs' / j['job_id'] / 'RESULT.json').exists()]
        for job in self.jobs:
            if job not in remaining_base:
                verify_result(job)
        extension_needed = True
        if not remaining_base:
            selected = [runtime.read(runtime.REPORT / 'jobs' / j['job_id'] / 'RESULT.json')['row']['selected_epoch'] for j in self.jobs]
            extension_needed = sum(float(value) == 3. for value in selected) >= 3
        remaining_extended = [j for j in self.extensions if extension_needed
                              and not (runtime.REPORT / 'jobs' / j['job_id'] / 'RESULT.json').exists()]
        for job in self.extensions:
            if (runtime.REPORT / 'jobs' / job['job_id'] / 'RESULT.json').exists():
                verify_result(job)
        reserved = sum(self.job_estimate(j) for j in remaining_base + remaining_extended) + self.margin
        admitted = not (remaining_base or remaining_extended) or time.time() + reserved <= self.deadline
        self.state['campaign_admission'] = dict(status='ADMITTED' if admitted else 'NOT_RUN_BUDGET',
            decision_utc=stamp(), reserved_seconds=reserved, remaining_base=len(remaining_base),
            reserved_extensions=len(remaining_extended),
            criterion='Reserve every unfinished base job plus the entire possible five-seed extension before launching; no V values used.')
        self.save()
        return admitted

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

    def family_jobs(self, family, extended=False):
        method, dataset = family.split(':')
        return [j for j in (self.extensions if extended else self.jobs)
                if (j['method_id'], j['dataset']) == (method, dataset)]

    def run_family(self, family):
        jobs = self.family_jobs(family)
        remaining = [j for j in jobs if not (runtime.REPORT / 'jobs' / j['job_id'] / 'RESULT.json').exists()]
        estimate = sum(self.job_estimate(j) for j in remaining) + self.margin
        admitted = not remaining or time.time() + estimate <= self.deadline
        decision = dict(status='ADMITTED' if admitted else 'NOT_RUN_BUDGET',
                        estimated_seconds=estimate, decision_utc=stamp(),
                        criterion='Whole five-seed family admission using elapsed time only')
        self.state['family_status'][family] = decision
        self.save()
        if not admitted:
            for job in jobs:
                self.state['run_statuses'].setdefault(job['job_id'], 'NOT_RUN_BUDGET')
            self.save()
            return False
        for job in jobs:
            self.run(job)
        complete = all(self.state['run_statuses'].get(j['job_id']) == 'COMPLETE' for j in jobs)
        decision.update(status='COMPLETE' if complete else 'INCOMPLETE',
                        completed=sum(self.state['run_statuses'].get(j['job_id']) == 'COMPLETE' for j in jobs))
        self.save()
        return complete

    def extension_decision(self, family):
        basic = self.family_jobs(family)
        if any(self.state['run_statuses'].get(j['job_id']) != 'COMPLETE' for j in basic):
            return dict(status='BASE_INCOMPLETE', performed=False)
        selections = [runtime.read(runtime.REPORT / 'jobs' / j['job_id'] / 'RESULT.json')['row']['selected_epoch'] for j in basic]
        count = sum(float(e) == 3. for e in selections)
        if count < 3:
            return dict(status='NOT_REQUIRED', cap_selected_count=count, performed=False)
        remaining = [j for j in self.family_jobs(family, True)
                     if not (runtime.REPORT / 'jobs' / j['job_id'] / 'RESULT.json').exists()]
        estimate = sum(self.job_estimate(j) for j in remaining) + self.margin
        admitted = not remaining or time.time() + estimate <= self.deadline
        return dict(status='ADMITTED' if admitted else 'NOT_RUN_BUDGET', cap_selected_count=count,
                    estimated_seconds=estimate, performed=False)

    def extend_family(self, family):
        decision = self.extension_decision(family)
        self.state['extension_status'][family] = decision
        self.save()
        jobs = self.family_jobs(family, True)
        if decision['status'] == 'ADMITTED':
            for job in jobs:
                self.run(job)
            completed = all(self.state['run_statuses'].get(j['job_id']) == 'COMPLETE' for j in jobs)
            decision.update(status='COMPLETE' if completed else 'INCOMPLETE', performed=completed)
            if completed:
                selected = [runtime.read(runtime.REPORT / 'jobs' / j['job_id'] / 'RESULT.json')['row']['selected_epoch'] for j in jobs]
                decision.update(cap10_selected_count=sum(float(x) == 10. for x in selected),
                                undertrained_at_10=sum(float(x) == 10. for x in selected) >= 3)
        else:
            for job in jobs:
                self.state['run_statuses'].setdefault(job['job_id'], decision['status'])
        self.save()

    def execute(self):
        self.recover_worker()
        runtime.guard_preserved()
        if not self.reserve_campaign():
            for job in self.jobs + self.extensions:
                self.state['run_statuses'].setdefault(job['job_id'], 'NOT_RUN_BUDGET')
            self.state.update(status='NOT_RUN_BUDGET', completed_utc=stamp(), active=None,
                              base_complete_count=sum(v == 'COMPLETE' for k, v in self.state['run_statuses'].items()
                                                      if not k.endswith('_ext10')), optional_complete_count=0)
            self.save()
            runtime.atomic(runtime.REPORT / 'EXECUTION_COMPLETION.json', self.state)
            return 2
        for family in self.plan['family_order']:
            self.run_family(family)
        for family in self.plan['family_order']:
            self.extend_family(family)
        for family in self.plan['optional_family_order']:
            self.run_family(family)
            self.extend_family(family)
        runtime.guard_preserved()
        base_count = sum(self.state['run_statuses'].get(j['job_id']) == 'COMPLETE' for j in self.mandatory)
        optional_count = sum(self.state['run_statuses'].get(j['job_id']) == 'COMPLETE' for j in self.optional)
        self.state.update(status='TRAINING_COMPLETE' if base_count == len(self.mandatory) else 'TRAINING_INCOMPLETE',
                          completed_utc=stamp(), active=None, base_complete_count=base_count,
                          optional_complete_count=optional_count)
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
        return 0 if base_count == len(self.mandatory) else 1


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
                                      extension_jobs=len(controller.extensions))), flush=True)
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
