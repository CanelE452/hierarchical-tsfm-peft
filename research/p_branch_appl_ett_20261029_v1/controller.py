"""One immutable P-branch job at a time, with durable recovery and typed failures."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from . import runtime, run_one

INITIAL_ESTIMATES = {'appliances': 647.449, 'ETTh1': 852.185, 'ETTh2_safe17400': 873.753}
HARD_WALLS = {'appliances': 2700., 'ETTh1': 3408.740, 'ETTh2_safe17400': 3495.012}
TERMINAL_STATES = {'TRAINING_COMPLETE', 'TRAINING_INCOMPLETE', 'SCIENTIFIC_SUSPEND',
                   'UNKNOWN_SUSPEND', 'USER_CANCELLED'}
SHARED_CACHE = runtime.REPO / '.cache/ppt_justification_20261028_v1'


class ScientificSuspend(RuntimeError):
    pass


class UnknownSuspend(RuntimeError):
    pass


class UserCancelled(RuntimeError):
    pass


def stamp():
    return datetime.now(timezone.utc).isoformat()


def command(module, *args):
    return [sys.executable, '-B', '-X', 'utf8', '-m', f'research.{runtime.NAME}.{module}', *map(str, args)]


def environment():
    return dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')


def process_record(pid):
    try:
        process = psutil.Process(int(pid))
        return dict(pid=process.pid, ppid=process.ppid(), created=process.create_time(),
                    command=process.cmdline(), name=process.name(), cwd=process.cwd())
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return None


def module_name(record):
    argv = record.get('command', [])
    return argv[argv.index('-m') + 1] if '-m' in argv and argv.index('-m') + 1 < len(argv) else None


def argument(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def same_identity(record, pid, created):
    return bool(record and record['pid'] == pid and created is not None and record['created'] == created)


def matches_worker(record, descriptor, attempt):
    if not record or module_name(record) != f'research.{runtime.NAME}.run_one':
        return False
    argv = record['command']
    path, number = argument(argv, '--job'), argument(argv, '--attempt')
    return bool(path and Path(path).resolve() == Path(descriptor).resolve() and number == str(attempt))


def worker_candidates(descriptor, attempt, not_before):
    found = []
    for process in psutil.process_iter(['pid']):
        record = process_record(process.pid)
        if (matches_worker(record, descriptor, attempt)
                and record['created'] >= not_before - 2):
            found.append(record)
    return sorted(found, key=lambda r: (r['created'], r['pid']))


def foreign_workloads(allowed_pids=()):
    """Recognize actual experiment owners/compute; ordinary desktop contexts are not conflicts."""
    allowed = set(allowed_pids) | {os.getpid()}
    try:
        for parent in psutil.Process().parents():
            record = process_record(parent.pid)
            if (record and module_name(record) == f'research.{runtime.NAME}.controller'
                    and Path(record['cwd']).resolve() == runtime.REPO.resolve()):
                allowed.add(parent.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    conflicts = []
    for process in psutil.process_iter(['pid']):
        if process.pid in allowed:
            continue
        record = process_record(process.pid)
        if not record:
            continue
        module = module_name(record) or ''
        if module.startswith('research.') and module.rsplit('.', 1)[-1] in ('controller', 'run_one', 'cost_profile'):
            conflicts.append(dict(pid=record['pid'], created=record['created'], kind='LIVE_EXPERIMENT', module=module))
    try:
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True,
                               timeout=15, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return conflicts + [dict(kind='GPU_QUERY_UNAVAILABLE', error=type(exc).__name__)]
    if query.returncode:
        return conflicts + [dict(kind='GPU_QUERY_FAILED', exit_code=query.returncode)]
    for line in query.stdout.splitlines():
        cells = line.split(',', 1)
        if len(cells) != 2 or not cells[0].strip().isdigit():
            continue
        pid, image = int(cells[0].strip()), Path(cells[1].strip()).name.lower()
        if pid in allowed:
            continue
        if any(token in image for token in ('python', 'wsl', 'vmmem', 'jupyter', 'ollama', 'llama', 'cuda')):
            conflicts.append(dict(pid=pid, kind='FOREIGN_GPU_COMPUTE', image=image))
    return conflicts


def checked_registration(job):
    try:
        return run_one.validate_registration(job)
    except (ValueError, AssertionError, PermissionError, FileNotFoundError) as exc:
        raise ScientificSuspend('Registration/preservation verification failed: '+str(exc)) from exc


def verify_result(job):
    registration = checked_registration(job)
    path = runtime.REPORT / 'jobs' / job['job_id'] / 'RESULT.json'
    result = runtime.read(path)
    if (result['status'] != 'COMPLETE' or result['job'] != job
            or result['registration'] != registration
            or result['spec'] != run_one.build_spec(job)
            or result['selection_sealed_before_V'] is not True):
        raise ScientificSuspend('Completed result differs from sealed job: ' + job['job_id'])
    receipt_path = path.parent / f'WORKER_RECEIPT_attempt{result["attempt"]}.json'
    if not receipt_path.exists():
        raise UnknownSuspend('Completed artifact has no final worker receipt: ' + job['job_id'])
    receipt = runtime.read(receipt_path)
    if (receipt.get('status') != 'COMPLETE' or receipt.get('exit_code') != 0
            or receipt.get('job_id') != job['job_id'] or receipt.get('attempt') != result['attempt']):
        kind, reason = receipt_failure(receipt, receipt.get('exit_code'))
        if kind == 'SCIENTIFIC':
            raise ScientificSuspend(reason)
        raise UnknownSuspend('Completed artifact has no verified successful worker receipt: ' + job['job_id'])
    for p, h in (('bundle_path', 'bundle_sha256'), ('selection_path', 'selection_sha256')):
        if runtime.sha(result[p]) != result[h]:
            raise ScientificSuspend('Completed artifact hash mismatch: ' + result[p])
    row = result['row']
    if runtime.sha(row['prediction_path']) != row['prediction_sha256']:
        raise ScientificSuspend('Completed prediction hash mismatch')
    runtime.assert_protected_T(result)
    run_one.check_selection(Path(result['selection_path']), result['selection_sha256'], job)
    expected, _ = run_one.choose_epoch(runtime.read(Path(result['selection_path']).parent / 'SEARCH_RESULT.json'),
                                      run_one.schedule(job))
    if row['selected_epoch'] != expected or row['training_cap'] != 3:
        raise ScientificSuspend('Completed row differs from A_stop selection')
    return result


def admission_allowed(deadline, estimate, margin, now):
    return deadline is None or now + estimate + margin <= deadline


def receipt_failure(receipt, return_code, timed_out=False):
    if timed_out:
        return 'TECHNICAL', 'FIXED_WALL_TIMEOUT'
    if receipt is not None:
        if receipt.get('status')=='USER_CANCELLED' or receipt.get('failure_kind')=='USER_CANCELLED':
            return 'USER_CANCELLED', 'Worker received a user cancellation'
        kind = receipt.get('failure_kind')
        if kind in ('TECHNICAL', 'SCIENTIFIC', 'UNKNOWN'):
            return kind, receipt.get('error', 'Worker receipt failure')
        if receipt.get('status') in ('SCIENTIFIC_SUSPEND', 'STOPPED'):
            return 'UNKNOWN', 'Unclassified stopped receipt; exit code is not a scientific classification'
    if return_code in (-9, -15, 137, 143, 3221225786, 3221226505):
        return 'TECHNICAL', 'UNEXPECTED_PROCESS_LOSS'
    return 'UNKNOWN', 'Missing typed worker failure receipt'


def terminate_owned(descriptor, attempt, not_before):
    """Only exact current job entries and their verified child processes may be terminated."""
    records = worker_candidates(descriptor, attempt, not_before)
    killed = []
    for record in reversed(records):
        current = process_record(record['pid'])
        if not same_identity(current, record['pid'], record['created']) or not matches_worker(current, descriptor, attempt):
            continue
        try:
            process = psutil.Process(record['pid'])
            children = process.children(recursive=True)
            for child in reversed(children):
                child_record = process_record(child.pid)
                if child_record and matches_worker(child_record, descriptor, attempt):
                    child.terminate()
                    killed.append(dict(pid=child.pid, created=child_record['created']))
            process.terminate()
            killed.append(dict(pid=process.pid, created=record['created']))
        except psutil.NoSuchProcess:
            pass
    deadline = time.monotonic() + 10
    while worker_candidates(descriptor, attempt, not_before) and time.monotonic() < deadline:
        time.sleep(.2)
    survivors = worker_candidates(descriptor, attempt, not_before)
    for record in survivors:
        current = process_record(record['pid'])
        if same_identity(current, record['pid'], record['created']) and matches_worker(current, descriptor, attempt):
            try:
                psutil.Process(record['pid']).kill()
            except psutil.NoSuchProcess:
                pass
    if worker_candidates(descriptor, attempt, not_before):
        raise UnknownSuspend('Owned worker could not be stopped; no duplicate job is allowed')
    return killed


class Controller:
    def __init__(self, resume_stopped=False):
        self.plan = runtime.read(runtime.REPORT / 'RUN_PLAN.json')
        self.protocol = runtime.read(runtime.REPORT / 'PREREGISTRATION.json')
        self.jobs = self.plan['jobs']
        self.extensions = self.plan.get('extension_jobs', [])
        self.validate_plan()
        policy = self.protocol.get('execution_policy', {})
        self.policy = policy
        deadline = policy.get('admission_deadline')
        self.deadline = datetime.fromisoformat(deadline).timestamp() if deadline else None
        self.margin = float(policy.get('admission_margin_seconds', 60))
        self.wait_seconds = float(policy.get('resource_wait_seconds', 60))
        self.poll_seconds = float(policy.get('heartbeat_seconds', 10))
        self.hard_walls = dict(policy.get('hard_wall_seconds', HARD_WALLS))
        if (policy.get('retries', 1) != 1 or self.margin < 0 or self.wait_seconds <= 0
                or self.hard_walls != HARD_WALLS):
            raise ScientificSuspend('Registered retry/watchdog policy changed')
        self.progress_path = runtime.REPORT / 'PROGRESS.json'
        self.state = runtime.read(self.progress_path) if self.progress_path.exists() else dict(
            status='READY', started_utc=stamp(), run_statuses={}, attempts={}, extension_status={},
            active=None, wait_events=[], estimates=dict(INITIAL_ESTIMATES))
        self.state.setdefault('estimates', dict(INITIAL_ESTIMATES))
        for key, default in (('attempts', {}), ('run_statuses', {}), ('extension_status', {}), ('wait_events', []), ('active', None)):
            self.state.setdefault(key, default)
        if self.state['status'] in ('SCIENTIFIC_SUSPEND', 'UNKNOWN_SUSPEND', 'USER_CANCELLED', 'STOPPED'):
            raise ScientificSuspend('A suspended/cancelled campaign requires explicit diagnosed recovery')

    def validate_plan(self):
        if len(self.jobs) != 60 or self.extensions:
            raise ScientificSuspend('Plan must contain sixty base jobs and no extension jobs')
        expected = [(s, d, m) for s in range(92601, 92611)
                    for d in ('appliances', 'ETTh1', 'ETTh2_safe17400')
                    for m in ('P_TIME_ONLY', 'P_CHANNEL_ONLY')]
        actual = [(j['seed'], j['dataset'], j['method_id']) for j in self.jobs]
        if actual != expected or len({j['job_id'] for j in self.jobs}) != 60:
            raise ScientificSuspend('Registered seed/dataset/method job order changed')
        for job in self.jobs:
            run_one.validate_job(job)
            if job['extended']:
                raise ScientificSuspend('Extension jobs are outside this campaign')
        checked_registration(self.jobs[0])

    def save(self):
        self.state.update(controller_pid=os.getpid(), controller_creation_time=psutil.Process().create_time(),
                          updated_utc=stamp(), admission_deadline=self.policy.get('admission_deadline'),
                          deadline_override=self.deadline is None)
        runtime.atomic(self.progress_path, self.state)

    def cancelled(self):
        if (runtime.CACHE / 'USER_STOP_REQUEST.json').exists() or (runtime.REPORT / 'USER_STOP_REQUEST.json').exists():
            raise UserCancelled('User requested this campaign to stop')

    def descriptor(self, job):
        path = runtime.CACHE / 'job_inputs' / (job['job_id'] + '.json')
        if path.exists():
            if runtime.read(path) != job:
                raise ScientificSuspend('Existing job descriptor changed')
        else:
            runtime.write_new(path, job)
        return path

    def active_workers(self, active):
        descriptor = active.get('descriptor') or self.descriptor(next(j for j in self.jobs if j['job_id'] == active['job_id']))
        return worker_candidates(descriptor, active['attempt'], active['started_unix'])

    def wait_active(self, active, process=None):
        descriptor = active['descriptor']
        timed_out = False
        while True:
            self.cancelled()
            workers = self.active_workers(active)
            if workers:
                actual = workers[-1]
                active.update(actual_pid=actual['pid'], actual_creation_time=actual['created'])
                live_path = runtime.CACHE / 'LIVE_WORKER.json'
                live = runtime.read(live_path) if live_path.exists() else {}
                if (live.get('pid') == actual['pid'] and live.get('process_creation_time') == actual['created']
                        and live.get('job_id') == active['job_id'] and live.get('attempt') == active['attempt']):
                    active.update(phase=live.get('phase'), phase_updated_utc=live.get('phase_updated_utc'))
            if not workers and (process is None or process.poll() is not None):
                break
            wall = time.time() - active['started_unix']
            if wall >= self.hard_walls[active['dataset']]:
                active['terminated_owned'] = terminate_owned(descriptor, active['attempt'], active['started_unix'])
                timed_out = True
                if process is not None:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        raise UnknownSuspend('Launcher remains after owned worker termination')
                break
            active.update(elapsed_seconds=wall, warning_slow=wall >= 120)
            self.state.update(status='RUNNING', active=active)
            self.save()
            time.sleep(self.poll_seconds)
        return timed_out

    def recover_worker(self):
        active = self.state.get('active')
        live_path = runtime.CACHE / 'LIVE_WORKER.json'
        live = runtime.read(live_path) if live_path.exists() else None
        if live and live.get('status') == 'RUNNING':
            job = next((j for j in self.jobs if j['job_id'] == live.get('job_id')), None)
            if job is None:
                raise ScientificSuspend('Unregistered live worker job')
            descriptor = self.descriptor(job)
            created = live.get('process_creation_time')
            record = process_record(live['pid'])
            if same_identity(record, live['pid'], created) and matches_worker(record, descriptor, live['attempt']):
                if active and (active['job_id'], active['attempt']) != (live['job_id'], live['attempt']):
                    raise ScientificSuspend('Progress and actual live worker identities differ')
                original_start = min(live['started_unix'], active['started_unix']) if active else live['started_unix']
                active = dict(job_id=job['job_id'], dataset=job['dataset'], attempt=live['attempt'],
                              seed=job['seed'], method_id=job['method_id'], phase=live.get('phase'),
                              pid=live['pid'], process_creation_time=created, started_unix=original_start,
                              descriptor=str(descriptor), log=live.get('log'))
                self.state['active'] = active
        if active:
            self.state.update(status='WAITING_EXISTING_WORKER', active=active)
            self.save()
            timed_out = self.wait_active(active)
            self.state['active'] = None
            self.finish_attempt(next(j for j in self.jobs if j['job_id'] == active['job_id']), active,
                                return_code=None, timed_out=timed_out)
            self.save()

    def next_attempt(self, job):
        public = runtime.REPORT / 'jobs' / job['job_id']
        used = []
        permitted = run_one.allowed_attempts(job)
        for attempt in permitted:
            path = public / f'WORKER_RECEIPT_attempt{attempt}.json'
            if path.exists():
                previous = runtime.read(path)
                if previous.get('status') == 'COMPLETE':
                    used.append(attempt)
                    continue
                kind, _ = receipt_failure(previous, previous.get('exit_code'))
                if kind == 'USER_CANCELLED':
                    raise UserCancelled('Previous worker was cancelled by the user')
                if kind == 'SCIENTIFIC':
                    raise ScientificSuspend('Previous scientific worker failure: ' + job['job_id'])
                if kind == 'UNKNOWN':
                    raise UnknownSuspend('Previous unclassified worker failure: ' + job['job_id'])
                used.append(attempt)
            elif ((public / f'attempt_{attempt:03d}').exists()
                  or any(a.get('attempt') == attempt for a in self.state['attempts'].get(job['job_id'], []))
                  or (runtime.CACHE / 'logs' / f'{job["job_id"]}_attempt{attempt}.log').exists()):
                used.append(attempt)
        return next((attempt for attempt in permitted if attempt not in used), max(permitted)+1)

    def resource_wait(self, job):
        started = time.time()
        while True:
            self.cancelled()
            estimate = self.state['estimates'][job['dataset']]
            if not admission_allowed(self.deadline, estimate, self.margin, time.time()):
                self.state['run_statuses'][job['job_id']] = 'NOT_RUN_BUDGET'
                self.save()
                return False
            reasons = foreign_workloads()
            if not reasons:
                try:
                    with ExitStack() as stack:
                        stack.enter_context(runtime.CampaignLock(runtime.CACHE / 'worker.lock'))
                        stack.enter_context(runtime.CampaignLock(SHARED_CACHE / 'worker.lock'))
                    return True
                except RuntimeError as exc:
                    if 'Another live process holds' not in str(exc):
                        raise
                    reasons = [dict(kind='WORKER_LOCK_BUSY')]
            event = dict(job_id=job['job_id'], checked_utc=stamp(), wait_seconds=time.time()-started,
                         next_check_utc=datetime.fromtimestamp(time.time()+self.wait_seconds, timezone.utc).isoformat(),
                         reasons=reasons)
            self.state['wait_events'].append(event)
            self.state.update(status='WAIT_RESOURCE', waiting=event)
            self.save()
            time.sleep(self.wait_seconds)

    def finish_attempt(self, job, active, return_code, timed_out=False):
        path = runtime.REPORT / 'jobs' / job['job_id'] / f'WORKER_RECEIPT_attempt{active["attempt"]}.json'
        receipt = runtime.read(path) if path.exists() else None
        active = dict(active)
        if receipt:
            active.update(actual_pid=receipt.get('pid'), actual_creation_time=receipt.get('process_creation_time'),
                          phase=receipt.get('phase'))
        if active.get('log') and Path(active['log']).exists():
            active['log_sha256'] = runtime.sha(active['log'])
        result_path = runtime.REPORT / 'jobs' / job['job_id'] / 'RESULT.json'
        if result_path.exists() and not timed_out:
            verify_result(job)
            if not receipt or receipt.get('status') != 'COMPLETE':
                raise ScientificSuspend('Result exists without a complete worker receipt')
            result = dict(active, exit_code=0, finished_unix=receipt.get('finished_unix', time.time()),
                          wall_seconds=receipt.get('wall_seconds', time.time()-active['started_unix']), failure_kind=None)
            self.state['run_statuses'][job['job_id']] = 'COMPLETE'
            self.state['estimates'][job['dataset']] = max(self.state['estimates'][job['dataset']], 1.35*result['wall_seconds'])
        else:
            if return_code is None and receipt is not None:
                return_code = receipt.get('exit_code')
            if timed_out:
                kind, reason = receipt_failure(receipt, return_code, True)
            elif return_code is None and receipt is None:
                kind, reason = ('TECHNICAL', 'INTERRUPTED_PROCESS_WITHOUT_RESULT')
            else:
                kind, reason = receipt_failure(receipt, return_code, timed_out)
            result = dict(active, exit_code=return_code, finished_unix=time.time(),
                          wall_seconds=time.time()-active['started_unix'], failure_kind=kind,
                          error=reason, timed_out=timed_out)
            self.state['attempts'].setdefault(job['job_id'], []).append(result)
            self.state['run_statuses'][job['job_id']] = 'FAILED_ONCE' if any(a > active['attempt'] for a in run_one.allowed_attempts(job)) else 'FAILED_TERMINAL'
            self.save()
            if kind == 'USER_CANCELLED':
                raise UserCancelled(reason)
            if kind == 'SCIENTIFIC':
                raise ScientificSuspend(reason)
            if kind == 'UNKNOWN':
                raise UnknownSuspend(reason)
            return False
        previous = self.state['attempts'].setdefault(job['job_id'], [])
        if not any(a.get('attempt') == active['attempt'] and a.get('exit_code') == 0 for a in previous):
            previous.append(result)
        return True

    def spawn_attempt(self, job, attempt):
        while True:
            checked_registration(job)
            if not self.resource_wait(job):
                return None, None
            descriptor = self.descriptor(job)
            log = runtime.CACHE / 'logs' / f'{job["job_id"]}_attempt{attempt}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            started = time.time()
            self.state.update(status='SPAWNING', active=dict(job_id=job['job_id'], dataset=job['dataset'],
                seed=job['seed'], method_id=job['method_id'], phase='SPAWNING',
                module=f'research.{runtime.NAME}.run_one', argv=command('run_one','--job',descriptor,'--attempt',attempt),
                attempt=attempt, started_unix=started, descriptor=str(descriptor), log=str(log)))
            self.save()
            try:
                with log.open('x', encoding='utf8') as stream:
                    process = subprocess.Popen(command('run_one', '--job', descriptor, '--attempt', attempt),
                        cwd=runtime.REPO, env=environment(), stdout=stream, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            except OSError as exc:
                destination = runtime.CACHE / 'launch_failures' / f'{job["job_id"]}_{time.time_ns()}.log'
                destination.parent.mkdir(parents=True, exist_ok=True)
                if log.exists():
                    log.replace(destination)
                self.state.update(status='WAIT_RESOURCE', active=None,
                    last_launch_failure=dict(job_id=job['job_id'], error=type(exc).__name__+': '+str(exc),
                                             log=str(destination), utc=stamp()))
                self.save()
                time.sleep(self.wait_seconds)
                continue
            record = process_record(process.pid)
            active = dict(self.state['active'], pid=process.pid,
                          process_creation_time=record['created'] if record else None)
            self.state.update(status='RUNNING', active=active)
            self.save()
            print(json.dumps(dict(event='START', job_id=job['job_id'], attempt=attempt,
                                  pid=process.pid, started_utc=stamp())), flush=True)
            return process, active

    def run(self, job):
        result_path = runtime.REPORT / 'jobs' / job['job_id'] / 'RESULT.json'
        if result_path.exists():
            verify_result(job)
            self.state['run_statuses'][job['job_id']] = 'COMPLETE'
            self.save()
            return True
        first = self.next_attempt(job)
        for attempt in (a for a in run_one.allowed_attempts(job) if a >= first):
            process, active = self.spawn_attempt(job, attempt)
            if process is None:
                return False
            try:
                timed_out = self.wait_active(active, process)
            except UserCancelled:
                terminate_owned(active['descriptor'], attempt, active['started_unix'])
                raise
            return_code = process.poll()
            self.state['active'] = None
            success = self.finish_attempt(job, active, return_code, timed_out)
            self.save()
            if success:
                return True
        self.state['run_statuses'][job['job_id']] = 'FAILED_TERMINAL'
        self.save()
        return False

    def first_seed_smoke(self):
        import numpy as np
        references = runtime.read(runtime.REPORT / 'REFERENCE_RUNS.json')['rows']
        canonical = {d: next(r for r in references if r['dataset'] == d and r['method_id'] == 'OURS'
                            and int(r['seed']) == 92603) for d in runtime.DATASETS}
        shapes = runtime.V_SHAPES
        checked, missing = [], []
        for job in self.jobs[:6]:
            path = runtime.REPORT / 'jobs' / job['job_id'] / 'RESULT.json'
            if not path.exists():
                missing.append(job['job_id'])
                continue
            result = verify_result(job)
            row = result['row']
            if row['trainable_parameters'] != runtime.EXPECTED_PARAMETERS[job['dataset']][job['method_id']]:
                raise ScientificSuspend('First-seed trainable parameter count differs')
            base = canonical[job['dataset']]
            arrays = []
            for archive_path, digest in ((row['prediction_path'], row['prediction_sha256']),
                                         (base['prediction_path'], base['prediction_sha256'])):
                if runtime.sha(archive_path) != digest:
                    raise ScientificSuspend('First-seed prediction SHA differs')
                with np.load(archive_path, allow_pickle=False) as archive:
                    for key in ('pred_normalized', 'target_normalized', 'observed'):
                        with archive.zip.open(key+'.npy') as stream:
                            version = np.lib.format.read_magic(stream)
                            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                            shape, _, dtype = reader(stream)
                        if tuple(shape) != tuple(shapes[job['dataset']]) or dtype.hasobject:
                            raise ScientificSuspend('First-seed numeric shape/header differs')
                    origins = archive['origins']
                    if int(origins.max()) + 48 > runtime.T_STARTS[job['dataset']]:
                        raise ScientificSuspend('First-seed target reaches protected T')
                    arrays.append({key: archive[key].copy() for key in
                        ('pred_normalized', 'target_normalized', 'observed', 'origins', 'channel_ids', 'mean', 'std')})
            new, ref = arrays
            for key in ('target_normalized', 'observed', 'origins', 'channel_ids', 'mean', 'std'):
                if not np.array_equal(new[key], ref[key], equal_nan=new[key].dtype.kind in 'fc'):
                    raise ScientificSuspend('First-seed cohort/scaler differs: '+key)
            error = new['pred_normalized'].astype(np.float64)-new['target_normalized'].astype(np.float64)
            mask = new['observed'].astype(bool)
            if (not np.isfinite(new['pred_normalized']).all() or not np.isfinite(error[mask]).all()
                    or not mask.any(axis=(0,1)).all()):
                raise ScientificSuspend('First-seed prediction/observed target is nonfinite or empty')
            mse = math.fsum(float(np.mean(error[:,:,ch][mask[:,:,ch]]**2)) for ch in range(error.shape[2]))/error.shape[2]
            mae = math.fsum(float(np.mean(np.abs(error[:,:,ch][mask[:,:,ch]]))) for ch in range(error.shape[2]))/error.shape[2]
            if max(abs(mse-float(row['MSE'])),abs(mae-float(row['MAE']))) > 1e-9:
                raise ScientificSuspend('First-seed independent FP64 metrics differ')
            checked.append(dict(job_id=job['job_id'], status='PASS', MSE=mse, MAE=mae,
                                trainable_parameters=row['trainable_parameters'], T_numeric_values_parsed=0))
        record = dict(status='PASS' if len(checked)==6 else 'PARTIAL_VALIDATED', checked_utc=stamp(),
                      completed=len(checked), missing=missing, jobs=checked, T_numeric_values_parsed=0,
                      initialization_scope='Retained branch initialization is verified by registered CPU fixtures; numerical factory unchanged.')
        path = runtime.REPORT/'FIRST_SEED_SMOKE_VALIDATION.json'
        if path.exists():
            previous=runtime.read(path)
            if previous['completed'] > len(checked):
                raise ScientificSuspend('Previously verified smoke results disappeared')
        runtime.atomic(path,record)

    def update_extension_status(self):
        result = {}
        for dataset in runtime.DATASETS:
            result[dataset] = {}
            for method in run_one.METHOD_FAMILY:
                jobs=[j for j in self.jobs if j['dataset']==dataset and j['method_id']==method]
                complete=[j for j in jobs if self.state['run_statuses'].get(j['job_id'])=='COMPLETE']
                selected=[runtime.read(runtime.REPORT/'jobs'/j['job_id']/'RESULT.json')['row']['selected_epoch'] for j in complete]
                count=sum(float(e)==3. for e in selected)
                status='BASE_INCOMPLETE' if len(complete)<10 else 'NOT_RUN_SCOPE' if count>=5 else 'NOT_REQUIRED'
                result[dataset][method]=dict(status=status,n=len(complete),cap_selected_count=count,performed=False)
        self.state['extension_status']=result

    def terminalize(self,status,reason):
        active=self.state.get('active')
        if active and active.get('descriptor'):
            terminate_owned(active['descriptor'],active['attempt'],active['started_unix'])
            if self.active_workers(active):
                raise UnknownSuspend('Cannot finalize while the owned worker remains alive')
        self.state['active']=None
        statuses=self.state['run_statuses']
        for job in self.jobs:
            path=runtime.REPORT/'jobs'/job['job_id']/'RESULT.json'
            if path.exists():
                try:
                    verify_result(job)
                    statuses[job['job_id']]='COMPLETE'
                except (ScientificSuspend,UnknownSuspend,ValueError,AssertionError,KeyError,FileNotFoundError) as exc:
                    digest=runtime.sha(path)
                    attempt=active['attempt'] if active and active['job_id']==job['job_id'] else 0
                    try:
                        attempt=runtime.read(path).get('attempt',attempt)
                    except (ValueError,KeyError):
                        pass
                    destination=path.parent/'rejected'/f'RESULT_attempt{int(attempt):03d}_{digest[:12]}.json'
                    if destination.exists():
                        raise ScientificSuspend('Rejected evidence destination already exists')
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    path.rename(destination)
                    rejection=dict(status='REJECTED_SCIENTIFIC_EVIDENCE',
                        job=job,run_status=status,rejected_result_path=destination.relative_to(runtime.REPO).as_posix(),
                        rejected_result_sha256=digest,reason=type(exc).__name__+': '+str(exc),created_utc=stamp(),
                        trigger='RESULT_WRITTEN_THEN_TERMINAL_VALIDATION_REJECTED',attempt=int(attempt),
                        original_result_path=path.relative_to(runtime.REPO).as_posix())
                    worker_receipt=path.parent/f'WORKER_RECEIPT_attempt{int(attempt)}.json'
                    if worker_receipt.exists():
                        rejection.update(worker_receipt_path=worker_receipt.relative_to(runtime.REPO).as_posix(),
                                         worker_receipt_sha256=runtime.sha(worker_receipt))
                    else:
                        rejection.update(worker_receipt_missing=True)
                    runtime.write_new(path.parent/'RESULT_REJECTION.json',rejection)
                    statuses[job['job_id']]=status
            elif job['job_id'] not in statuses:
                statuses[job['job_id']]='NOT_RUN_'+status
            elif statuses[job['job_id']]=='COMPLETE':
                raise ScientificSuspend('Previously completed result disappeared during terminalization')
        self.update_extension_status()
        self.state.update(status=status,stop_reason=reason,completed_utc=stamp(),planned_count=60,
                          completed_count=sum(value=='COMPLETE' for value in statuses.values()))
        self.save()
        runtime.atomic(runtime.REPORT/'EXECUTION_COMPLETION.json',self.state)

    def postprocess(self):
        source=Path(__file__).with_name('postprocess.py')
        if not source.exists():
            return
        with (runtime.CACHE/'postprocess.log').open('a',encoding='utf8') as stream:
            result=subprocess.run(command('postprocess'),cwd=runtime.REPO,env=environment(),stdout=stream,
                stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        runtime.atomic(runtime.REPORT/'POSTPROCESS_STATUS.json',dict(status='COMPLETE' if result.returncode==0 else 'FAILED',
                       exit_code=result.returncode,finished_utc=stamp()))

    def execute(self):
        if self.state.get('status')=='TRAINING_COMPLETE':
            for job in self.jobs:
                verify_result(job)
            return 0
        self.recover_worker()
        runtime.guard_preserved(full=True)
        for index,job in enumerate(self.jobs):
            self.run(job)
            if index==5:
                self.first_seed_smoke()
            if (index+1)%6==0:
                self.update_extension_status()
                self.save()
                print(json.dumps(dict(event='SEED_COMPLETE',seed=job['seed'],completed=sum(v=='COMPLETE' for v in self.state['run_statuses'].values()),
                                      planned=60,estimates=self.state['estimates'],utc=stamp())),flush=True)
        runtime.guard_preserved(full=True)
        self.update_extension_status()
        complete=sum(self.state['run_statuses'].get(j['job_id'])=='COMPLETE' for j in self.jobs)
        self.state.update(status='TRAINING_COMPLETE' if complete==60 else 'TRAINING_INCOMPLETE',
                          completed_utc=stamp(),active=None,completed_count=complete,planned_count=60)
        self.save()
        runtime.atomic(runtime.REPORT/'EXECUTION_COMPLETION.json',self.state)
        self.postprocess()
        return 0 if complete==60 else 1


def prevent_sleep(active):
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if active else 0))


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--resume-stopped',action='store_true',help='Compatibility flag; cannot bypass suspended/cancelled state')
    args=parser.parse_args(argv)
    runtime.install_write_guard()
    controller=None
    try:
        with ExitStack() as stack:
            stack.enter_context(runtime.CampaignLock(runtime.CACHE/'controller.lock'))
            controller=Controller(args.resume_stopped)
            while True:
                try:
                    shared=runtime.CampaignLock(SHARED_CACHE/'controller.lock')
                    shared.__enter__()
                    stack.callback(shared.__exit__,None,None,None)
                    break
                except RuntimeError as exc:
                    if 'Another live process holds' not in str(exc):
                        raise
                    controller.cancelled()
                    controller.state.update(status='WAIT_RESOURCE',waiting=dict(kind='SHARED_CONTROLLER_LOCK_BUSY',checked_utc=stamp()))
                    controller.save()
                    time.sleep(controller.wait_seconds)
            if args.preflight_only:
                print(json.dumps(dict(status='PREFLIGHT_PASS',base_jobs=60,extension_jobs=0,admission_deadline=controller.policy.get('admission_deadline'))),flush=True)
                return 0
            prevent_sleep(True)
            return controller.execute()
    except (ScientificSuspend,UnknownSuspend,UserCancelled,KeyboardInterrupt) as exc:
        status='USER_CANCELLED' if isinstance(exc,(UserCancelled,KeyboardInterrupt)) else 'UNKNOWN_SUSPEND' if isinstance(exc,UnknownSuspend) else 'SCIENTIFIC_SUSPEND'
        if controller is not None:
            controller.terminalize(status,str(exc))
            controller.postprocess()
        import traceback
        traceback.print_exc()
        return 20 if status!='USER_CANCELLED' else 2
    except BaseException as exc:
        if controller is not None:
            controller.state.update(status='CONTROLLER_CRASHED',controller_error=type(exc).__name__+': '+str(exc))
            controller.save()
        import traceback
        traceback.print_exc()
        return 70
    finally:
        prevent_sleep(False)


if __name__=='__main__':
    raise SystemExit(main())
