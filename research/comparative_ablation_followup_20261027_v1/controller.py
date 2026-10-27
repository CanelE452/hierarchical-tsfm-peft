"""Serial, resumable orchestration for the already registered follow-up scope."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from . import common


GATE_EXIT = 20
BUDGET_EXIT = 21
TERMINAL = {'FAILED', 'NOT_RUN_BUDGET', 'NOT_RUN_FAMILY_FAILURE',
            'FAILED_ONCE_RETRY_NOT_RUN_BUDGET', 'FAILED_ONCE_CONTROLLER_STOPPED'}


class CampaignLock:
    """An OS lock, released on process exit; the file's mere presence is irrelevant."""
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.stream.write(b'0')
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError(f'Another live campaign process holds {self.path.name}') from exc
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            self.stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()


def pid_alive(pid):
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x100000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied cannot establish absence.
        try:
            return kernel.WaitForSingleObject(handle, 0) != 0
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stamp(seconds=None):
    return datetime.fromtimestamp(time.time() if seconds is None else seconds, timezone.utc).isoformat()


def environment():
    return dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')


def command(module, *args):
    return [str(getattr(common, 'PYTHON', sys.executable)), '-B', '-X', 'utf8', '-m',
            f'research.{common.NAME}.{module}', *map(str, args)]


class Controller:
    def __init__(self, resume_stopped=False):
        self.resume_stopped = resume_stopped
        self.jobs = common.selected_jobs()
        self.budget = common.read(common.REPORT / 'BUDGET.json')
        self.clock_path = common.REPORT / 'EXECUTION_CLOCK.json'
        self.progress_path = common.REPORT / 'PROGRESS.json'
        self.active = None
        self.reports = {}
        self.post_path = common.REPORT / 'POSTPROCESS_STATUS.json'
        self.post = common.read(self.post_path) if self.post_path.exists() else {}
        if self.progress_path.exists():
            self.state = common.read(self.progress_path)
        else:
            old = common.read(common.OLD_REPORT / 'EXECUTION_COMPLETION.json')
            plan = common.read(common.OLD_REPORT / 'RUN_PLAN.json')
            selected = {job['job_id'] for job in self.jobs}
            self.state = dict(status='RUNNING', started_utc=stamp(), stop_reason=None,
                run_statuses={job['job_id']: 'NOT_RUN_SCOPE' for job in plan['jobs'] + plan['extension_jobs']
                              if job['job_id'] not in selected and old['run_statuses'].get(job['job_id']) != 'COMPLETE'},
                prior_run_statuses=old['run_statuses'],
                prior_complete_job_ids=[key for key, value in old['run_statuses'].items() if value == 'COMPLETE'],
                extension_status={}, attempts={}, active={}, first_a4_wall_seconds={},
                limit_seconds=common.LIMIT_SECONDS, last_start_cutoff=common.LAST_START)
            for job in plan['extension_jobs']:
                key = job['method_id'] + '/' + job['dataset']
                if key != 'A1/jena':
                    self.state['extension_status'][key] = dict(status='NOT_RUN_SCOPE', performed=False,
                        prior=old.get('extension_status', {}).get(key))

    def end_time(self):
        end = datetime.fromisoformat(common.LAST_START).timestamp()
        if self.clock_path.exists():
            clock = common.read(self.clock_path)
            if clock['limit_seconds'] != common.LIMIT_SECONDS or clock['latest_start'] != common.LAST_START:
                raise ValueError('Follow-up clock limits changed')
            end = min(end, clock['training_started_unix'] + common.LIMIT_SECONDS)
        return end

    def progress(self):
        now = time.time()
        started = common.read(self.clock_path)['training_started_unix'] if self.clock_path.exists() else None
        self.state.update(updated_utc=stamp(now), controller_pid=os.getpid(),
                          elapsed_seconds=0. if started is None else max(0., now - started),
                          remaining_seconds=max(0., self.end_time() - now))
        self.state['active'] = {} if self.active is None else {
            self.active['job']['job_id']: {key: self.active[key] for key in ('pid', 'attempt', 'started_unix', 'log')}}
        common.atomic(self.progress_path, self.state)

    def gate(self):
        try:
            if common.validate_admission().get('status') != 'PASS':
                raise ValueError('Follow-up admission gate did not pass')
            common.guard_preserved()
        except Exception as exc:
            raise ValueError('Follow-up scientific gate: ' + str(exc)) from exc

    def result_path(self, job):
        return common.OLD_REPORT / 'jobs' / job['job_id'] / 'RESULT.json'

    def descriptor(self, job):
        path = common.CACHE / 'job_inputs' / (job['job_id'] + '.json')
        if path.exists():
            if common.read(path) != job:
                raise ValueError('Existing six-field job descriptor changed')
        else:
            common.write_new(path, job)
        return path

    def verify_complete(self, job):
        self.gate()
        descriptor = self.descriptor(job)
        log = common.CACHE / 'verification' / f'{job["job_id"]}_{uuid.uuid4().hex}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x', encoding='utf8') as stream:
            completed = subprocess.run(command('job_worker', '--job', descriptor, '--verify-only'),
                cwd=common.REPO, env=environment(), stdout=stream, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if completed.returncode != 0:
            raise ValueError('Existing COMPLETE failed registered run_job verification: ' + job['job_id'])
        self.state['run_statuses'][job['job_id']] = 'COMPLETE'

    def failures(self, method):
        return sum(self.state['run_statuses'].get(job['job_id']) == 'FAILED'
                   for job in self.jobs if job['method_id'] == method)

    def next_attempt(self, job):
        existing = []
        for attempt in (1, 2):
            log = common.CACHE / 'logs' / f'{job["job_id"]}_attempt{attempt}.log'
            receipt = log.with_suffix('.json')
            worker = common.CACHE / 'worker_receipts' / f'{job["job_id"]}_attempt{attempt}.json'
            if log.exists():
                if not receipt.exists() and not worker.exists():
                    raise RuntimeError('Unresolved previous worker log: ' + str(log))
                saved = common.read(receipt if receipt.exists() else worker)
                if (saved.get('exit_code') == GATE_EXIT or saved.get('status') == 'STOPPED') and not self.resume_stopped:
                    raise RuntimeError('Prior worker scientific gate requires explicit investigation')
                existing.append(attempt)
        return max(existing, default=0) + 1

    def start(self, job, attempt):
        if self.active is not None:
            raise RuntimeError('Serial invariant violated')
        if time.time() >= self.end_time():
            return False
        self.gate()
        descriptor = self.descriptor(job)
        if job['method_id'] == 'A1' and job['extended']:
            extension = [item for item in self.jobs if item['method_id'] == 'A1']
            if not self.extension_allowed(extension):
                return False
        if time.time() >= self.end_time():
            return False
        log = common.CACHE / 'logs' / f'{job["job_id"]}_attempt{attempt}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open('x', encoding='utf8')
        if time.time() >= self.end_time():
            stream.close()
            common.write_new(log.with_suffix('.json'), dict(job_id=job['job_id'], attempt=attempt,
                exit_code=BUDGET_EXIT, status='NOT_RUN_BUDGET', process_started=False))
            return False
        try:
            process = subprocess.Popen(command('job_worker', '--job', descriptor, '--attempt', attempt),
                cwd=common.REPO, env=environment(), stdout=stream, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except BaseException:
            stream.close()
            raise
        self.active = dict(job=job, attempt=attempt, process=process, pid=process.pid,
                           started_unix=time.time(), log=str(log), stream=stream)
        self.state['run_statuses'][job['job_id']] = 'RUNNING'
        print(json.dumps(dict(event='START', job_id=job['job_id'], attempt=attempt, pid=process.pid)), flush=True)
        self.progress()
        return True

    def finish(self, exit_code):
        item, self.active = self.active, None
        item['stream'].close()
        job = item['job']
        worker_path = common.CACHE / 'worker_receipts' / f'{job["job_id"]}_attempt{item["attempt"]}.json'
        worker = common.read(worker_path) if worker_path.exists() else {}
        receipt = dict(job_id=job['job_id'], attempt=item['attempt'], pid=item['pid'], exit_code=exit_code,
            started_unix=item['started_unix'], finished_unix=time.time(),
            wall_seconds=time.time() - item['started_unix'], log=item['log'],
            result_present=self.result_path(job).is_file(), worker=worker)
        common.write_new(Path(item['log']).with_suffix('.json'), receipt)
        self.state['attempts'].setdefault(job['job_id'], []).append(receipt)
        if exit_code == 0 and receipt['result_present']:
            self.state['run_statuses'][job['job_id']] = 'COMPLETE'
            if job['method_id'] == 'A4' and job['dataset'] not in self.state['first_a4_wall_seconds']:
                self.state['first_a4_wall_seconds'][job['dataset']] = receipt['wall_seconds']
                self.reestimate_a4()
        elif exit_code == BUDGET_EXIT:
            self.state['run_statuses'][job['job_id']] = 'NOT_RUN_BUDGET'
        elif item['attempt'] == 2:
            self.state['run_statuses'][job['job_id']] = 'FAILED'
        else:
            self.state['run_statuses'][job['job_id']] = 'FAILED_ONCE'
        print(json.dumps(dict(event='EXIT', **receipt)), flush=True)
        self.progress()
        if exit_code == GATE_EXIT or worker.get('status') == 'STOPPED':
            raise ValueError('Worker gate stopped: ' + job['job_id'])
        common.guard_preserved()
        return self.state['run_statuses'][job['job_id']]

    def reestimate_a4(self):
        measured = self.state['first_a4_wall_seconds']
        pending = [job for job in self.jobs if job['method_id'] == 'A4'
                   and self.state['run_statuses'].get(job['job_id']) not in TERMINAL | {'COMPLETE'}]
        seconds = sum(measured.get(job['dataset'], self.budget['per_dataset_a4_seconds'][job['dataset']])
                      for job in pending)
        self.state['budget'] = dict(initial=self.budget, measured_a4_per_dataset_seconds=dict(measured),
            a4_jobs_remaining=len(pending), estimated_a4_seconds_remaining=seconds,
            estimated_a4_finish_utc=stamp(time.time() + seconds),
            admission_remaining_seconds=max(0., self.end_time() - time.time()))
        print(json.dumps(dict(event='A4_ETA_UPDATED', **self.state['budget'])), flush=True)

    def run_job(self, job):
        self.poll_reports()
        if self.result_path(job).exists():
            self.verify_complete(job)
            self.progress()
            return
        if self.state['run_statuses'].get(job['job_id']) in TERMINAL:
            return
        if self.failures(job['method_id']) >= 3:
            self.state['run_statuses'][job['job_id']] = 'NOT_RUN_FAMILY_FAILURE'
            self.progress()
            return
        attempt = self.next_attempt(job)
        if attempt > 2:
            self.state['run_statuses'][job['job_id']] = 'FAILED'
            self.progress()
            return
        while attempt <= 2:
            if not self.start(job, attempt):
                self.state['run_statuses'][job['job_id']] = (
                    'FAILED_ONCE_RETRY_NOT_RUN_BUDGET' if attempt == 2 else 'NOT_RUN_BUDGET')
                self.progress()
                return
            while True:
                self.poll_reports()
                code = self.active['process'].poll()
                if code is not None:
                    status = self.finish(code)
                    break
                self.progress()
                time.sleep(2)
            if status != 'FAILED_ONCE':
                return
            attempt += 1

    def launch_report(self, stage):
        if stage in self.reports:
            return
        previous = self.post.get(stage, {})
        if previous.get('status') == 'RUNNING' and pid_alive(previous['pid']):
            raise RuntimeError('A previous reporting process is still active')
        number = int(previous.get('attempt', 0)) + 1
        log = common.CACHE / 'logs' / f'reporting_{stage}_attempt{number}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open('x', encoding='utf8')
        try:
            args = command('bundle') if stage == 'bundle' else command('reporting', '--stage', stage)
            process = subprocess.Popen(args, cwd=common.REPO,
                env=environment(), stdout=stream, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except BaseException:
            stream.close()
            raise
        self.reports[stage] = dict(process=process, stream=stream)
        self.post[stage] = dict(status='RUNNING', pid=process.pid, attempt=number,
                                started_utc=stamp(), log=str(log))
        common.atomic(self.post_path, self.post)

    def poll_reports(self):
        for stage, item in list(self.reports.items()):
            code = item['process'].poll()
            if code is None:
                continue
            item['stream'].close()
            del self.reports[stage]
            directory = common.REPORT / 'interim_A4' if stage == 'interim' else common.REPORT
            failure_path = directory / 'REPORTING_FAILURE.json'
            failure = common.read(failure_path) if code != 0 and failure_path.exists() else {}
            self.post[stage].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code,
                                    finished_utc=stamp(), failure=failure)
            common.atomic(self.post_path, self.post)
            print(json.dumps(dict(event='REPORT_EXIT', stage=stage, **self.post[stage])), flush=True)
            if code != 0 and failure.get('classification') == 'SCIENTIFIC_GATE':
                raise ValueError('Scientific reporting gate failed: ' + str(failure))

    def wait_reports(self):
        while self.reports:
            self.poll_reports()
            if self.reports:
                time.sleep(2)

    def a4_extension_status(self):
        datasets = dict.fromkeys(job['dataset'] for job in self.jobs if job['method_id'] == 'A4')
        for dataset in datasets:
            results = [common.read(self.result_path(job)) for job in self.jobs
                       if job['method_id'] == 'A4' and job['dataset'] == dataset and self.result_path(job).exists()]
            caps = sum(float(result['row']['selected_epoch']) == 3 for result in results)
            self.state['extension_status']['A4/' + dataset] = dict(triggered=len(results) == 10 and caps >= 5,
                performed=False, n=len(results), cap_count=caps, status='NOT_RUN_SCOPE',
                provisional=len(results) == 10 and caps >= 5)

    def extension_started(self, jobs):
        for job in jobs:
            if self.result_path(job).is_file():
                return True
            for attempt in (1, 2):
                path = common.CACHE / 'worker_admissions' / f'{job["job_id"]}_attempt{attempt}.json'
                if path.exists():
                    evidence = common.read(path)
                    if evidence['job'] != job or evidence['attempt'] != attempt:
                        raise ValueError('Extension worker admission evidence differs from registered job')
                    return True
        return False

    def extension_allowed(self, jobs):
        path = common.REPORT / 'A1_JENA_EXTENSION_ADMISSION.json'
        previous = ([path] if path.exists() else []) + sorted(
            common.REPORT.glob('A1_JENA_EXTENSION_ADMISSION_[0-9][0-9][0-9].json'))
        if previous:
            latest = common.read(previous[-1])
            if not latest['allowed']:
                return False
            if self.extension_started(jobs):
                return True
        elif self.extension_started(jobs):
            raise ValueError('Extension worker started without a recorded all-seed admission')
        self.gate()
        required = len(jobs) * self.budget['a1_jena_extension_seconds_per_job']
        remaining = max(0., self.end_time() - time.time())
        allowed = time.time() < self.end_time() and required <= remaining
        number = len(previous) + 1
        output = path if not previous else path.with_name(f'A1_JENA_EXTENSION_ADMISSION_{number:03d}.json')
        common.write_new(output, dict(allowed=allowed, n=len(jobs), required_seconds=required,
            remaining_seconds=remaining, checked_utc=stamp(), status='ADMITTED' if allowed else 'NOT_RUN_BUDGET',
            decision_number=number, first_worker_admitted=False,
            previous=None if not previous else dict(path=previous[-1].name, sha256=common.sha(previous[-1]))))
        return allowed

    def recover(self):
        # Lock proves no active wrapper even when a saved PID was recycled.
        with CampaignLock(common.CACHE / 'worker.lock'):
            for item in self.state.get('active', {}).values():
                if pid_alive(item['pid']):
                    raise RuntimeError('Saved worker PID is still alive; refusing another admission')
        for stage, item in self.post.items():
            if item.get('status') == 'RUNNING' and pid_alive(item['pid']):
                raise RuntimeError('Saved reporting PID is still alive; await it before controller resume')
        self.state['active'] = {}

    def finalize(self, retry_interim=False):
        try:
            self.wait_reports()
            self.gate()
            a4_complete = all(self.state['run_statuses'].get(job['job_id']) == 'COMPLETE'
                              for job in self.jobs if job['method_id'] == 'A4')
            if retry_interim and a4_complete and self.post.get('interim', {}).get('status') != 'COMPLETE':
                self.launch_report('interim')
                self.wait_reports()
            self.launch_report('final')
            self.wait_reports()
            if self.post['final']['status'] != 'COMPLETE' or (a4_complete and self.post.get('interim', {}).get('status') != 'COMPLETE'):
                raise RuntimeError('Postprocessing failed; resume with --postprocess-only after repair')
            self.gate()
            self.launch_report('bundle')
            self.wait_reports()
            if self.post['bundle']['status'] != 'COMPLETE':
                raise RuntimeError('Bundle postprocessing failed; resume with --postprocess-only after repair')
        except ValueError as exc:
            self.state.update(status='STOPPED', stop_reason=f'{type(exc).__name__}: {exc}')
            self.progress()
            common.atomic(common.REPORT / 'EXECUTION_COMPLETION.json', self.state)
            common.atomic(common.REPORT / 'CONTROLLER_STOP.json', self.state)
            raise

    def execute(self):
        try:
            self.gate()
            self.recover()
            self.state.update(status='RUNNING', stop_reason=None)
            self.progress()
            a4 = [job for job in self.jobs if job['method_id'] == 'A4']
            extension = [job for job in self.jobs if job['method_id'] == 'A1']
            for job in a4:
                self.run_job(job)
            self.a4_extension_status()
            self.progress()
            if all(self.state['run_statuses'].get(job['job_id']) == 'COMPLETE' for job in a4):
                self.launch_report('interim')
            if self.extension_allowed(extension):
                for job in extension:
                    self.run_job(job)
            else:
                for job in extension:
                    self.state['run_statuses'][job['job_id']] = 'NOT_RUN_BUDGET'
            results = [common.read(self.result_path(job)) for job in extension if self.result_path(job).exists()]
            cap10 = sum(float(result['row']['selected_epoch']) == 10 for result in results)
            self.state['extension_status']['A1/jena'] = dict(triggered=True, cap_count=10, n=10,
                performed=len(results) == 10, cap10_count=cap10,
                status=('UNDERTRAINED_AT_10' if cap10 >= 5 else 'EXTENDED') if len(results) == 10 else
                       ('INCOMPLETE_EXTENSION' if results else
                        ('NOT_RUN_BUDGET' if all(self.state['run_statuses'].get(job['job_id']) == 'NOT_RUN_BUDGET'
                                               for job in extension) else 'FAILED_OR_STOPPED')))
            self.wait_reports()
            self.gate()
            complete = all(self.state['run_statuses'].get(job['job_id']) == 'COMPLETE' for job in self.jobs)
            self.state['status'] = 'COMPLETE' if complete else 'PARTIAL'
            self.state['stop_reason'] = None if complete else 'Budget, cutoff or registered job/family failure; see run_statuses.'
        except BaseException as exc:
            self.state.update(status='STOPPED', stop_reason=f'{type(exc).__name__}: {exc}')
            # A failed gate never terminates the already admitted scientific job.
            if self.active is not None:
                while self.active['process'].poll() is None:
                    time.sleep(2)
                try:
                    self.finish(self.active['process'].poll())
                except BaseException as drain_error:
                    self.state['drain_error'] = str(drain_error)
            try:
                self.wait_reports()
            except BaseException as report_error:
                self.state['report_error'] = str(report_error)
        self.progress()
        common.atomic(common.REPORT / 'EXECUTION_COMPLETION.json', self.state)
        if self.state['status'] == 'STOPPED':
            common.atomic(common.REPORT / 'CONTROLLER_STOP.json', self.state)
            return GATE_EXIT
        self.finalize()
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--postprocess-only', action='store_true')
    parser.add_argument('--resume-stopped', action='store_true',
                        help='Explicit operator acknowledgement after resolving a recorded gate stop')
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    common.install_write_guard()
    with CampaignLock(common.CACHE / 'controller.lock'):
        completion = common.REPORT / 'EXECUTION_COMPLETION.json'
        finished = common.read(completion) if completion.exists() else None
        progress = common.REPORT / 'PROGRESS.json'
        prior_state = finished or (common.read(progress) if progress.exists() else {})
        if prior_state.get('status') == 'STOPPED' and not args.resume_stopped:
            raise RuntimeError('Prior gate stop requires explicit --resume-stopped after resolution')
        if args.postprocess_only and finished is None:
            raise RuntimeError('Postprocess-only requires a persisted execution completion')
        runner = Controller(resume_stopped=args.resume_stopped)
        if args.postprocess_only or (finished and finished['status'] != 'STOPPED'):
            runner.recover()
            runner.finalize(retry_interim=True)
            return 0
        return runner.execute()


if __name__ == '__main__':
    raise SystemExit(main())
