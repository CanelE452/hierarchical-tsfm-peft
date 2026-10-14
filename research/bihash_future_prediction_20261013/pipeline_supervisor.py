"""Continue this live study sequentially after its verified existing DEV process.

This supervisor never restarts or terminates the observed DEV job. Each child
owns a fresh tag and enforces the study's existing source, data and resource
gates. Final interpretation and completion approval remain separate audits.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

import psutil
import runtime as rt


def emit(value):
    print(rt.json_line(value), flush=True)


def wait_existing(pid, created, folder):
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - created) > .001:
            raise RuntimeError('OBSERVED_PID_IDENTITY_CHANGED')
        arguments = process.cmdline()
        if not any(Path(arg).name == 'run.py' for arg in arguments) or 'dev' not in arguments:
            raise RuntimeError('OBSERVED_PROCESS_IS_NOT_EXPECTED_DEV')
    except psutil.NoSuchProcess:
        emit({'existing_DEV_process': 'already terminal; authoritative result required', 'pid': pid})
        return
    started = time.time()
    last = -60.
    while process.is_running():
        try:
            if abs(process.create_time() - created) > .001:
                raise RuntimeError('OBSERVED_PID_REUSED')
            if process.status() == psutil.STATUS_ZOMBIE: break
        except psutil.NoSuchProcess:
            break
        elapsed = time.time() - started
        if elapsed - last >= 45:
            emit({'verified_wait': 'existing DEV process', 'pid': pid, 'create_time': created,
                  'wait_seconds': elapsed, 'supervisor_pid': os.getpid()})
            last = elapsed
        if (folder / 'HALT_BEFORE_NEXT_PHASE').exists():
            raise RuntimeError('USER_REQUESTED_QUEUE_HALT_PRESERVE_EXISTING_DEV')
        time.sleep(2)


def launch(label, arguments, folder, stages):
    if (folder / 'HALT_BEFORE_NEXT_PHASE').exists():
        raise RuntimeError('USER_REQUESTED_QUEUE_HALT')
    if time.time() - rt.GOAL_STARTED_UNIX > rt.CAMPAIGN_SECONDS:
        raise rt.BudgetHeld('CAMPAIGN_WALL_BEFORE_NEXT_PHASE')
    log_path = folder / (label + '_stdout.log')
    started = time.time()
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2',
           'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    command = [sys.executable, '-u', str(rt.HERE / arguments[0]), *arguments[1:]]
    emit({'launch_phase': label, 'started_unix': started, 'arguments': arguments,
          'log': log_path.relative_to(rt.ROOT).as_posix()})
    with log_path.open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=rt.ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, text=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        birth = psutil.Process(process.pid).create_time()
        last = -60.
        while process.poll() is None:
            elapsed = time.time() - started
            if elapsed - last >= 45:
                emit({'verified_phase_live': label, 'pid': process.pid, 'create_time': birth,
                      'elapsed_seconds': elapsed, 'log': log_path.relative_to(rt.ROOT).as_posix()})
                last = elapsed
            time.sleep(2)
        code = process.returncode
    receipt = {'phase': label, 'pid': process.pid, 'create_time': birth,
               'arguments': arguments, 'started_unix': started, 'finished_unix': time.time(),
               'exit_code': code, 'stdout_log': log_path.relative_to(rt.ROOT).as_posix(),
               'stdout_sha256': rt.sha(log_path), 'status': 'COMPLETE' if code == 0 else 'FAILED_PRESERVED',
               'resource_note': 'Child terminal resource receipts own cost; supervisor wait is not duplicated GPU/CPU execution usage.'}
    rt.write(folder / (label + '_TERMINAL.json'), receipt)
    stages.append(receipt)
    emit({'phase_terminal': label, 'exit_code': code, 'log': receipt['stdout_log']})
    return code


def require_complete(path, stage):
    value = rt.read(path)
    if value.get('stage') != stage or value.get('status') != 'COMPLETE':
        raise RuntimeError('REQUIRED_TERMINAL_NOT_COMPLETE: ' + str(path))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-pid', type=int, required=True)
    parser.add_argument('--dev-created', type=float, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum(): raise ValueError('INVALID_FRESH_TAG')
    folder = rt.HERE / ('pipeline_' + args.tag)
    folder.mkdir(parents=True, exist_ok=False)
    stages = []
    output = {'started_unix': time.time(), 'supervisor_pid': os.getpid(),
              'observed_DEV_pid': args.dev_pid, 'observed_DEV_created': args.dev_created,
              'supervisor_source_sha256': rt.sha(__file__), 'stages': stages,
              'no_existing_process_restart_or_termination': True, 'goal_complete': False}
    status, error = 'COMPLETE_GPU_PHASES_AWAITING_INDEPENDENT_AUDIT_AND_REPORT', None
    method = rt.HERE / 'METHOD_LOCK.json'
    dev = rt.HERE / 'run_dev_v1/RESULT.json'
    fresh = rt.HERE / 'run_fresh_v1/RESULT.json'
    probes = rt.HERE / 'run_probes_v1/RESULT.json'
    evaluation = rt.HERE / 'run_evaluate_v1/RESULT.json'
    cost = rt.HERE / 'run_cost_v1/RESULT.json'
    try:
        wait_existing(args.dev_pid, args.dev_created, folder)
        require_complete(dev, 'dev')
        if not method.is_file(): raise RuntimeError('METHOD_LOCK_MISSING_AFTER_DEV')
        if not rt.read(method).get('locked'): raise RuntimeError('METHOD_LOCK_NOT_LOCKED')
        for future in (fresh, probes, evaluation, cost):
            if future.parent.exists(): raise FileExistsError('FUTURE_TAG_ALREADY_EXISTS: ' + str(future.parent))
        # The download command preserves failures and tries both independent sources.
        launch('raw_download_v1', ['download_new.py', 'download', 'all', '--method-lock', str(method)], folder, stages)
        if launch('fresh_v1', ['run.py', '--stage', 'fresh', '--tag', 'fresh_v1', '--method-lock', str(method)], folder, stages):
            raise RuntimeError('FRESH_TERMINAL_FAILED_PRESERVE_PATH_RESULTS')
        require_complete(fresh, 'fresh')
        # This FIT-memory-only CPU diagnostic adds no model selection or EVAL access.
        diagnostic = ['hash_memory_diagnostics.py', '--result', str(dev), '--result', str(fresh),
                      '--tag', 'future_keys_full_v1', '--expected-cases', '174']
        launch('future_keys_full_v1', diagnostic, folder, stages)
        common = ['--dev-result', str(dev), '--fresh-result', str(fresh), '--method-lock', str(method)]
        if launch('probes_v1', ['run.py', '--stage', 'probes', '--tag', 'probes_v1', *common], folder, stages):
            raise RuntimeError('PROBE_TERMINAL_FAILED_EVAL_REMAINS_CLOSED')
        require_complete(probes, 'probes')
        if launch('evaluate_v1', ['run.py', '--stage', 'evaluate', '--tag', 'evaluate_v1', *common,
                                   '--probe-result', str(probes)], folder, stages):
            raise RuntimeError('EVALUATION_TERMINAL_FAILED_PRESERVE_PARTIAL_INFERENCE')
        require_complete(evaluation, 'evaluate')
        if launch('cost_v1', ['run.py', '--stage', 'cost', '--tag', 'cost_v1', *common,
                             '--probe-result', str(probes), '--evaluation-result', str(evaluation)], folder, stages):
            raise RuntimeError('COST_TERMINAL_FAILED_PRESERVE_MEASUREMENTS')
        require_complete(cost, 'cost')
        proof = rt.HERE / 'verify_deployment_outputs.py'
        if proof.is_file():
            arguments = ['verify_deployment_outputs.py', *common, '--probe-result', str(probes),
                         '--evaluation-result', str(evaluation), '--cost-result', str(cost), '--tag', 'outputs_v1']
            if launch('deployment_outputs_v1', arguments, folder, stages):
                raise RuntimeError('DEPLOYMENT_OUTPUT_AUDIT_FAILED_PRESERVE_ARRAYS')
        else:
            output['deployment_output_audit'] = 'NOT_RUN_SCRIPT_NOT_READY; root must execute separately'
    except Exception as exc:
        status = 'BUDGET_HELD_PRESERVED' if isinstance(exc, rt.BudgetHeld) else 'FAILED_PRESERVED'
        error = str(exc)
        output['failure_type'] = type(exc).__name__
    output.update(status=status, error=error, finished_unix=time.time(),
                  supervisor_CPU_seconds=sum(psutil.Process().cpu_times()[:2]),
                  resource_note='Supervisor wait wall is orchestration time, not extra GPU work; child receipts are counted once.')
    rt.write(folder / 'RESULT.json', output)
    emit({'pipeline_terminal': str(folder / 'RESULT.json'), 'status': status, 'error': error,
          'goal_complete': False, 'remaining': 'independent full audit, visualization review, Korean interpretation and completion audit'})
    return 0 if error is None else 2


if __name__ == '__main__': raise SystemExit(main())
