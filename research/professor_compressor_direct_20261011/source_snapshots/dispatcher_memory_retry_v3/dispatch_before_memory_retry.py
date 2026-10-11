"""Sequential actual fits and reports; existing results and failed attempts stay intact."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import psutil
import torch

from common_direct import HERE, ROOT, Resources, digest, read_json, write_json


def command_for(job):
    command = [sys.executable, str(HERE / ('runner_gt.py' if job['family'] == 'GT' else 'runner_ts2.py')),
               '--dataset', job['dataset'], '--seed', str(job['seed'])]
    if job['family'] == 'GT' and job['no_spatial']:
        command.append('--no-spatial')
    if job['family'] == 'TS2':
        command += ['--dim', str(job['dim'])]
        if job.get('retry_tag'):
            command += ['--retry-tag', job['retry_tag']]
    return command


def complete(path):
    if not path.exists():
        return False
    value = read_json(path)
    if value.get('status') != 'complete':
        return False
    prediction = ROOT / value['prediction_path']
    if not prediction.is_file() or digest(prediction) != value.get('prediction_sha256'):
        raise RuntimeError(f'Existing actual result has invalid prediction receipt: {path}')
    return True


def other_gpu_runners():
    found = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            args = proc.info['cmdline'] or []
            script = Path(args[1]).name if len(args) > 1 else ''
            if (proc.pid != os.getpid() and proc.info['name'].lower().startswith('python')
                    and script in ('runner_gt.py', 'runner_ts2.py', 'runner_baselines.py')
                    and HERE.name in args[1]):
                found.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def run_process(command, log, resources):
    with log.open('x', encoding='utf-8') as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        while process.poll() is None:
            resources.check(stage='sequential_dispatch_wait', child_pid=process.pid,
                            child_command=command)
            time.sleep(1)
    return process.returncode


def main(wait_pid):
    resources = Resources()
    archive = HERE / 'dispatch_attempts' / str(time.time_ns())
    archive.mkdir(parents=True)
    for name in ('EXECUTION_PREFLIGHT.json', 'RUN_QUEUE.json'):
        if (HERE / name).exists():
            shutil.copy2(HERE / name, archive / name)
    protocol = read_json(HERE / 'protocol.json')
    plans = []
    for dataset in protocol['datasets']:
        plans.append({'id': f'{dataset}_BASELINES', 'result': HERE / 'baselines' / dataset / 'result.json',
                      'command': [sys.executable, str(HERE / 'runner_baselines.py'), '--dataset', dataset]})
    for job in protocol['jobs']:
        plans.append({'id': job['id'], 'result': HERE / 'runs' / job['id'] / 'result.json',
                      'command': command_for(job)})
    inventory = {'status': 'PREFLIGHT', 'started_unix': time.time(),
                 'environment': {'python': sys.version, 'executable': sys.executable,
                                 'platform': platform.platform(), 'torch': torch.__version__,
                                 'cuda_build': torch.version.cuda,
                                 'gpu': torch.cuda.get_device_name(0)},
                 'protocol_sha256': digest(HERE / 'protocol.json'),
                 'gpu_concurrency': 1, 'old_jobs_auto_resume': False,
                 'planned_representation_fits': 16, 'extra_preserved_pilots': 1,
                 'previous_A_H_results_reused_as_new_method_results': 0,
                 'initial_timing_estimates_seconds': {'GT_robin': [15, 30], 'GT_jena': [90, 180],
                       'TS2_robin_d16': [25, 50], 'TS2_robin_d320': [60, 180],
                       'TS2_jena_d16': [90, 240], 'TS2_jena_d320': [150, 480]},
                 'estimate_note': 'Robin measurements and Jena GT running throughput; remaining TS2 timings extrapolated, not convergence estimates.',
                 'runs': [{'id': item['id'], 'command': item['command'],
                           'status': 'REUSE_ACTUAL_NEW_SCOPE' if complete(item['result']) else 'PLANNED_NEW',
                           'result_path': item['result']} for item in plans]}
    write_json(HERE / 'EXECUTION_PREFLIGHT.json', inventory)
    if wait_pid:
        try:
            proc = psutil.Process(wait_pid)
            while proc.is_running():
                resources.check(stage='wait_current_new_scope_fit', pid_to_wait=wait_pid)
                time.sleep(1)
        except psutil.NoSuchProcess:
            pass
    inventory['status'] = 'RUNNING'
    queue_path = HERE / 'RUN_QUEUE.json'
    try:
        for plan, row in zip(plans, inventory['runs']):
            resources.check()
            if complete(plan['result']):
                row.update(status='REUSE_ACTUAL_NEW_SCOPE', result_sha256=digest(plan['result']))
                write_json(queue_path, inventory)
                continue
            if other_gpu_runners():
                raise RuntimeError(f'GPU concurrency gate: live runners {other_gpu_runners()}')
            row.update(status='RUNNING', started_unix=time.time())
            log = HERE / f"dispatch_{plan['id']}_{time.time_ns()}.log"
            row['log'] = log
            write_json(queue_path, inventory)
            print(f"START {plan['id']}", flush=True)
            code = run_process(plan['command'], log, resources)
            row.update(exit_code=code, ended_unix=time.time())
            row['elapsed_seconds'] = row['ended_unix']-row['started_unix']
            if code == 0 and complete(plan['result']):
                row.update(status='ACTUAL_COMPLETE', result_sha256=digest(plan['result']))
            else:
                row['status'] = 'FAILED_PRESERVED_PATH'
                row['resume'] = 'Inspect actual log and partial checkpoint. Use new retry-tag/attempt after correcting only this path.'
            write_json(queue_path, inventory)
            print(f"END {plan['id']} {row['status']} {row['elapsed_seconds']:.1f}s", flush=True)
            report_log = HERE / f"analysis_after_{plan['id']}_{time.time_ns()}.log"
            report_code = run_process([sys.executable, str(HERE / 'analyze_direct.py')], report_log, resources)
            row.update(analysis_exit_code=report_code, analysis_log=report_log)
            write_json(queue_path, inventory)
            # Bad scientific scores never abort other methods. Global caps still apply.
            resources.check()
        inventory['status'] = 'ALL_COMMANDS_EXECUTED_PENDING_FULL_AUDIT'
    except BaseException as error:
        inventory.update(status='INCOMPLETE', error=repr(error))
        raise
    finally:
        inventory['ended_unix'] = time.time()
        write_json(queue_path, inventory)
        resources.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int)
    main(parser.parse_args().wait_pid)
